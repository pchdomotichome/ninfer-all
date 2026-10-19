// ninfer::ops - Qwen3.8-Flash-Next MoE experts over GGUF block banks (contract in
// include/ninfer/ops/moe_experts.h). The router's pairs are grouped by expert on the device; the
// gate/up products then run once per active expert over all of its tokens, the down products
// accumulate weighted into a fixed-point plane, and the shared expert takes the same path as a
// one-expert bank that every token selects. Up to eight tokens run the vector kernel's MoE form
// (ops/linear/gguf); wider calls over device-resident banks run all three projections through the
// integer tensor-core matrix kernel over the routed pairs, where the block type has one.
#include "ninfer/ops/moe_experts.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

// Wider calls over device-resident banks take the matrix kernel.
constexpr int kVectorTokens = 8;

// The (hidden, expert width, max experts, top-k) pairs this Op serves. The constants above are the
// Flash-Next geometry's; `experts` is a bound, not part of the key, because an expert-pruned release
// keeps fewer entries of the same geometry.
struct MoeGeometry {
    std::int32_t hidden;
    std::int32_t width;
    std::int32_t experts;
    std::int32_t top_k;
};

constexpr MoeGeometry kGeometries[]{
    {kHidden, kWidth, kExperts, kTopK}, // Qwen3.8-Flash-Next (qwen4_exp)
    {2048, 512, 256, 8},                // Qwen3.6-35B-A3B (qwen3_5_moe)
};

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("moe_experts_gguf: ") + message); }
}

// `m` carries the hidden width and `ids` the top-k; the expert width rides on the banks because no
// tensor of the call states it.
const MoeGeometry* geometry_for(std::int32_t hidden, std::int32_t width, std::int32_t top_k) {
    for (const MoeGeometry& geometry : kGeometries) {
        if (geometry.hidden == hidden && geometry.width == width && geometry.top_k == top_k) {
            return &geometry;
        }
    }
    return nullptr;
}

bool shaped(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1) {
    return tensor.dtype == dtype && tensor.is_contiguous() && tensor.data != nullptr &&
           tensor.ne[0] == n0 && tensor.ne[1] == n1 && tensor.ne[2] == 1 && tensor.ne[3] == 1;
}

// How many of one expert's columns share a pass over its rows, from the average it gets.
int chunk_for(std::int64_t pairs, std::int64_t experts) {
    const std::int64_t average = (pairs + experts - 1) / experts;
    return average >= 6 ? 8 : average >= 3 ? 4 : average >= 2 ? 2 : 1;
}

gguf::MoeTable table(const GgufExpertTable& bank, int rows, int k) {
    require(bank.experts != nullptr && bank.row_bytes > 0, "an expert table is empty");
    return gguf::MoeTable{bank.experts, bank.row_bytes, rows, k};
}

bool fusable(const GgufExpertTable& gate, const GgufExpertTable& up) {
    return gate.format == up.format && gate.row_bytes == up.row_bytes;
}

// out = silu(gate) * up, rounded to BF16, over [pairs][rows] planes.
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                              __nv_bfloat16* __restrict__ out, std::int64_t count) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const float g = gate[i];
    out[i]        = __float2bfloat16(g / (1.0f + expf(-g)) * up[i]);
}

// fixed[t][r] += each of token t's pairs' contribution weights[p] * down[p][r] in 2^-32 fixed
// point, rounded per pair as the vector products round it. One thread per (token, row).
__global__ void accumulate_fixed_kernel(const float* __restrict__ down,
                                        const float* __restrict__ weights, int per_token, int rows,
                                        std::int64_t count,
                                        unsigned long long* __restrict__ fixed) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const std::int64_t t = i / rows, r = i % rows;
    long long sum = 0;
    for (int j = 0; j < per_token; ++j) {
        const std::int64_t p = t * per_token + j;
        const double scaled  = fmin(
            fmax(double(down[p * rows + r]) * double(weights[p]) * 4294967296.0, -4.0e18), 4.0e18);
        sum += __double2ll_rn(scaled);
    }
    fixed[i] += static_cast<unsigned long long>(sum);
}

__global__ void store_fixed_kernel(const unsigned long long* __restrict__ fixed,
                                   float* __restrict__ y, std::int64_t count) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    y[i] = static_cast<float>(static_cast<double>(static_cast<long long>(fixed[i])) *
                              (1.0 / 4294967296.0));
}

struct Plan {
    std::int32_t tokens = 0;
    std::int32_t pairs  = 0;
};

bool matrix_path(std::int32_t tokens, bool device_resident, const GgufExpertTable& table, int rows,
                 int k) {
    return tokens > kVectorTokens && device_resident &&
           gguf::moe_matrix_supported(detail::gguf_type(table.format), rows, k, true);
}

// One bank's pass: the middle of every pair, then its weighted down product into `fixed`. `hidden`
// and `width` are the call's geometry, read from `m` and the banks.
void run_bank(const Tensor& routing_ids, std::int32_t pairs, std::int32_t experts, int per_token,
              const float* pair_weights, const __nv_bfloat16* m, const void* activation,
              std::int32_t tokens, std::int32_t hidden, std::int32_t width,
              const GgufExpertTable& gate, const GgufExpertTable& up,
              const GgufExpertTable& down, bool device_resident, WorkspaceArena& workspace,
              unsigned long long* fixed, cudaStream_t stream) {
    auto scope           = workspace.scope();
    Tensor routing_bytes = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::moe_routing_bytes(experts, pairs))});
    const gguf::MoeRouting routing =
        gguf::moe_sort_routes(static_cast<const std::int32_t*>(routing_ids.data), pairs, experts,
                              routing_bytes.data, stream);
    const int active                = std::min(pairs, experts);
    const int chunk                 = chunk_for(pairs, active);
    Tensor middle                   = workspace.alloc(DType::BF16, {width, pairs});
    auto* middle_p                  = static_cast<__nv_bfloat16*>(middle.data);
    const gguf::MoeTable gate_table = table(gate, width, hidden);
    const gguf::MoeTable up_table   = table(up, width, hidden);
    if (matrix_path(tokens, device_resident, gate, width, hidden) &&
        matrix_path(tokens, device_resident, up, width, hidden)) {
        // Each projection's pairs gathered in routing order and quantized for its schedule; an
        // expert receives at most one pair per token.
        auto planes          = workspace.scope();
        const auto gate_type = detail::gguf_type(gate.format),
                   up_type   = detail::gguf_type(up.format);
        const auto activation_bytes =
            static_cast<std::int32_t>(gguf::moe_matrix_activation_bytes(hidden, pairs));
        Tensor gate_x = workspace.alloc(DType::U8, {activation_bytes});
        gguf::quantize_moe_matrix_activation(gate_type, m, hidden, routing, pairs, per_token,
                                             gate_x.data, stream);
        Tensor up_x = gate_x;
        if (gguf::matrix_activation_layout(up_type) != gguf::matrix_activation_layout(gate_type)) {
            up_x = workspace.alloc(DType::U8, {activation_bytes});
            gguf::quantize_moe_matrix_activation(up_type, m, hidden, routing, pairs, per_token,
                                                 up_x.data, stream);
        }
        Tensor gate_plane = workspace.alloc(DType::FP32, {width, pairs});
        Tensor up_plane   = workspace.alloc(DType::FP32, {width, pairs});
        gguf::moe_matrix_product(gate_type, gate_table, routing, active, pairs, tokens, true,
                                 gate_x.data, static_cast<float*>(gate_plane.data), stream);
        gguf::moe_matrix_product(up_type, up_table, routing, active, pairs, tokens, true, up_x.data,
                                 static_cast<float*>(up_plane.data), stream);
        const std::int64_t count = std::int64_t(width) * pairs;
        swiglu_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
            static_cast<const float*>(gate_plane.data), static_cast<const float*>(up_plane.data),
            middle_p, count);
        CUDA_CHECK(cudaGetLastError());
    } else if (fusable(gate, up)) {
        gguf::moe_vector_swiglu(detail::gguf_type(gate.format), gate_table, up_table, routing,
                                active, per_token, activation, tokens, middle_p, chunk, stream);
    } else {
        Tensor gate_plane = workspace.alloc(DType::FP32, {width, pairs});
        auto* gate_p      = static_cast<float*>(gate_plane.data);
        gguf::moe_vector_product(detail::gguf_type(gate.format), gate_table, routing, active,
                                 per_token, per_token, activation, tokens,
                                 gguf::MoeOutput{.f32 = gate_p}, chunk, stream);
        gguf::moe_vector_product(detail::gguf_type(up.format), up_table, routing, active, per_token,
                                 per_token, activation, tokens,
                                 gguf::MoeOutput{.bf16 = middle_p, .gate = gate_p}, chunk, stream);
    }
    const auto down_type = detail::gguf_type(down.format);
    if (matrix_path(tokens, device_resident, down, hidden, width)) {
        // The middle in routing order, its `width` values padded to whole 256-value steps.
        Tensor middle_x = workspace.alloc(
            DType::U8,
            {static_cast<std::int32_t>(gguf::moe_matrix_activation_bytes(width, pairs))});
        gguf::quantize_moe_matrix_activation(down_type, middle_p, width, routing, pairs, 1,
                                             middle_x.data, stream);
        Tensor down_plane = workspace.alloc(DType::FP32, {hidden, pairs});
        auto* down_p      = static_cast<float*>(down_plane.data);
        gguf::moe_matrix_product(down_type, table(down, hidden, width), routing, active, pairs,
                                 tokens, true, middle_x.data, down_p, stream);
        const std::int64_t count = std::int64_t(hidden) * tokens;
        accumulate_fixed_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
            down_p, pair_weights, static_cast<const std::int32_t*>(routing_ids.data), per_token,
            hidden, count, fixed);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    Tensor middle_q = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(width, pairs))});
    gguf::quantize_vector_activation(middle_p, width, pairs, nullptr, middle_q.data, stream);
    gguf::moe_vector_product(
        down_type, table(down, hidden, width), routing, active, 1, per_token, middle_q.data,
        pairs, gguf::MoeOutput{.weighted = fixed, .weights = pair_weights}, chunk, stream);
}

// The routing of a one-expert bank every token selects, laid out as moe_sort_routes lays it out
// (bounds [2], sorted [tokens], active [1], active count): no sort needed.
__global__ void shared_routing_kernel(std::int32_t* __restrict__ base, int tokens) {
    std::int32_t* bounds = base;
    std::int32_t* sorted = bounds + 2;
    std::int32_t* active = sorted + tokens;
    if (threadIdx.x == 0) {
        bounds[0] = 0;
        bounds[1] = tokens;
        active[0] = 0;
        active[1] = 1;
    }
    if (int(threadIdx.x) < tokens) { sorted[threadIdx.x] = int(threadIdx.x); }
}

// run_bank's vector path for the shared expert of up to eight tokens, weighted into `fixed`.
// Its allocations stay in the caller's scope: the work may still run on a side stream after the
// call returns.
void run_shared_vector(const float* shared_weights, const void* activation, std::int32_t tokens,
                       std::int32_t hidden, std::int32_t width, const GgufMoeWeights& banks,
                       WorkspaceArena& workspace, unsigned long long* fixed, cudaStream_t stream) {
    Tensor routing_bytes = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::moe_routing_bytes(1, tokens))});
    auto* base = static_cast<std::int32_t*>(routing_bytes.data);
    shared_routing_kernel<<<1, 32, 0, stream>>>(base, tokens);
    CUDA_CHECK(cudaGetLastError());
    const gguf::MoeRouting routing{base, base + 2, base + 2 + tokens, base + 3 + tokens};
    const int chunk                 = chunk_for(tokens, 1);
    Tensor middle                   = workspace.alloc(DType::BF16, {width, tokens});
    auto* middle_p                  = static_cast<__nv_bfloat16*>(middle.data);
    const gguf::MoeTable gate_table = table(banks.shared_gate, width, hidden);
    const gguf::MoeTable up_table   = table(banks.shared_up, width, hidden);
    if (fusable(banks.shared_gate, banks.shared_up)) {
        gguf::moe_vector_swiglu(detail::gguf_type(banks.shared_gate.format), gate_table, up_table,
                                routing, 1, 1, activation, tokens, middle_p, chunk, stream);
    } else {
        Tensor gate_plane = workspace.alloc(DType::FP32, {width, tokens});
        auto* gate_p      = static_cast<float*>(gate_plane.data);
        gguf::moe_vector_product(detail::gguf_type(banks.shared_gate.format), gate_table, routing,
                                 1, 1, 1, activation, tokens, gguf::MoeOutput{.f32 = gate_p}, chunk,
                                 stream);
        gguf::moe_vector_product(detail::gguf_type(banks.shared_up.format), up_table, routing, 1, 1,
                                 1, activation, tokens,
                                 gguf::MoeOutput{.bf16 = middle_p, .gate = gate_p}, chunk, stream);
    }
    Tensor middle_q = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(width, tokens))});
    gguf::quantize_vector_activation(middle_p, width, tokens, nullptr, middle_q.data, stream);
    gguf::moe_vector_product(detail::gguf_type(banks.shared_down.format),
                             table(banks.shared_down, hidden, width), routing, 1, 1, 1,
                             middle_q.data, tokens,
                             gguf::MoeOutput{.weighted = fixed, .weights = shared_weights}, chunk,
                             stream);
}

} // namespace

std::size_t moe_experts_gguf_workspace_bytes(std::int32_t tokens) {
    require(tokens > 0, "tokens must be positive");
    // Every geometry this Op serves, then the decode form's own sizing: the caller's arena is one
    // size for all of them, so the largest wins.
    std::size_t bytes = 0;
    for (const MoeGeometry& geometry : kGeometries) {
        const std::int32_t pairs = tokens * geometry.top_k;
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc(
            DType::U8,
            {static_cast<std::int32_t>(gguf::vector_activation_bytes(geometry.hidden, tokens))});
        (void)layout.alloc(DType::I64, {geometry.hidden, tokens});
        (void)layout.alloc(DType::I32, {tokens});
        // The larger of the two banks' passes, which run one after the other in the same scope: the
        // routing and the middle, then gate and up (the vector path's gate plane, or the matrix
        // path's activations and planes, released before the down product), then the down product's
        // input (and the matrix path's output plane).
        (void)layout.alloc(
            DType::U8, {static_cast<std::int32_t>(gguf::moe_routing_bytes(geometry.experts, pairs))});
        (void)layout.alloc(DType::BF16, {geometry.width, pairs});
        {
            auto planes = layout.scope();
            if (tokens > kVectorTokens) {
                const auto activation_bytes = static_cast<std::int32_t>(
                    gguf::moe_matrix_activation_bytes(geometry.hidden, pairs));
                (void)layout.alloc(DType::U8, {activation_bytes});
                (void)layout.alloc(DType::U8, {activation_bytes});
                (void)layout.alloc(DType::FP32, {geometry.width, pairs});
            }
            (void)layout.alloc(DType::FP32, {geometry.width, pairs});
        }
        (void)layout.alloc(
            DType::U8,
            {static_cast<std::int32_t>(gguf::vector_activation_bytes(geometry.width, pairs))});
        if (tokens > kVectorTokens) {
            (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(
                                              gguf::moe_matrix_activation_bytes(geometry.width, pairs))});
            (void)layout.alloc(DType::FP32, {geometry.hidden, pairs});
        }
        bytes = std::max(bytes, layout.peak_bytes(1));
    }
    return std::max(bytes, moe_experts_gguf_decode_workspace_bytes(tokens));
}

namespace {
void check_stage_inputs(const Tensor& m, const Tensor& ids, const Tensor& weights,
                        std::int32_t top_k) {
    const std::int32_t tokens = m.ne[1];
    require(tokens > 0, "tokens must be positive");
    require(tokens * top_k <= 65535, "too many tokens per call");
    require(shaped(ids, DType::I32, top_k, tokens), "ids must be I32 [top-k, tokens]");
    require(shaped(weights, DType::FP32, top_k, tokens), "weights must be FP32 [top-k, tokens]");
}

template <int Hidden, int Width, int Experts, int TopK>
void run(const Tensor& m, const Tensor& ids, const Tensor& weights, const Tensor& shared,
         const GgufMoeWeights& banks, WorkspaceArena& workspace, Tensor& y, cudaStream_t stream) {
    const std::int32_t tokens = m.ne[1];
    const std::int32_t hidden = m.ne[0];
    require(tokens > 0 && shaped(m, DType::BF16, hidden, tokens), "m must be BF16 [hidden, tokens]");
    GgufMoeStage stage;
    stage.tokens      = tokens;
    Tensor activation = workspace.alloc(
        DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(hidden, tokens))});
    gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(m.data), hidden, tokens,
                                     nullptr, activation.data, stream);
    Tensor fixed = workspace.alloc(DType::I64, {hidden, tokens});
    CUDA_CHECK(cudaMemsetAsync(fixed.data, 0, fixed.bytes(), stream));
    stage.activation = activation.data;
    stage.fixed      = static_cast<unsigned long long*>(fixed.data);
    return stage;
}

void moe_experts_gguf_add(const GgufMoeStage& stage, const Tensor& m, const Tensor& ids,
                          const Tensor& weights, const Tensor* shared,
                          const GgufMoeWeights& banks, WorkspaceArena& workspace,
                          cudaStream_t stream) {
    const std::int32_t hidden = m.ne[0];
    const std::int32_t top_k  = ids.ne[0];
    const std::int32_t width  = banks.width;
    const MoeGeometry* geometry = geometry_for(hidden, width, top_k);
    require(geometry != nullptr,
            "no instantiation for this hidden width, expert width and top-k");
    check_stage_inputs(m, ids, weights, top_k);
    require(m.ne[1] == stage.tokens && stage.fixed != nullptr, "the stage began with other tokens");
    require(banks.experts >= top_k && banks.experts <= geometry->experts,
            "the expert tables hold neither the top-k nor the geometry's maximum");
    const std::int32_t tokens = stage.tokens;
    const auto* m_p           = static_cast<const __nv_bfloat16*>(m.data);
    run_bank(ids, tokens * top_k, banks.experts, top_k, static_cast<const float*>(weights.data),
             m_p, stage.activation, tokens, hidden, width, banks.gate, banks.up, banks.down,
             banks.device_resident, workspace, stage.fixed, stream);
    if (shared != nullptr) {
        require(shared->dtype == DType::FP32 && shared->is_contiguous() &&
                    shared->numel() == tokens,
                "shared must be FP32 [tokens]");
        auto scope        = workspace.scope();
        Tensor shared_ids = workspace.alloc(DType::I32, {tokens});
        CUDA_CHECK(cudaMemsetAsync(shared_ids.data, 0, shared_ids.bytes(), stream));
        run_bank(shared_ids, tokens, 1, 1, static_cast<const float*>(shared->data), m_p,
                 stage.activation, tokens, hidden, width, banks.shared_gate, banks.shared_up,
                 banks.shared_down, banks.device_resident, workspace, stage.fixed, stream);
    }
}

void moe_experts_gguf_add_products(const GgufMoeStage& stage, const float* values,
                                   const std::int32_t* pairs, const std::int32_t* count,
                                   std::int32_t max_count, const Tensor& weights,
                                   cudaStream_t stream) {
    require(stage.fixed != nullptr && max_count >= 0 && max_count <= stage.tokens * kTopK,
            "invalid product count");
    if (max_count == 0) { return; }
    add_products_kernel<<<dim3(5, static_cast<unsigned>(max_count)), 512, 0, stream>>>(
        values, pairs, count, kTopK, static_cast<const float*>(weights.data), stage.fixed);
    CUDA_CHECK(cudaGetLastError());
}

void moe_experts_gguf_finish(const GgufMoeStage& stage, Tensor& y, cudaStream_t stream) {
    const std::int32_t hidden = y.ne[0];
    require(stage.fixed != nullptr && shaped(y, DType::FP32, hidden, stage.tokens),
            "y must be FP32 [hidden, tokens]");
    const std::int64_t count = std::int64_t(hidden) * stage.tokens;
    store_fixed_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        fixed_p, static_cast<float*>(y.data), count);
    CUDA_CHECK(cudaGetLastError());
}

// One row per instantiation, so the geometry a call selects is readable here and nowhere else.
// `experts` is the maximum the routed tables may hold: an expert-pruned release keeps fewer entries
// of the same geometry, which is why it is a bound and not part of the key.
struct MoeGeometry {
    std::int32_t hidden;
    std::int32_t width;
    std::int32_t experts;
    std::int32_t top_k;
    std::size_t (*workspace)(std::int32_t);
    void (*run)(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const GgufMoeWeights&,
                WorkspaceArena&, Tensor&, cudaStream_t);
};

Prepared prepared_at(void* base, std::int32_t k, std::int32_t columns) {
    auto* qs  = static_cast<std::int8_t*>(base);
    auto* ds  = reinterpret_cast<half2*>(qs + std::size_t(columns) * k);
    auto* sum = reinterpret_cast<int*>(ds + std::size_t(columns) * (k / 32));
    return {qs, ds, sum};
}
} // namespace

bool moe_experts_gguf_decode_supported(const GgufMoeWeights& banks, std::int32_t tokens) {
    // The decode form's kernels and preparation carry the Flash-Next geometry's constants; another
    // geometry takes the staged path, which reads its widths from the call.
    return tokens > 0 && tokens <= kVectorTokens && banks.width == kWidth &&
           fusable(banks.gate, banks.up) && down_fits(banks.down);
}

std::size_t moe_experts_gguf_decode_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {kWidth, tokens * kTopK});
    (void)layout.alloc(DType::BF16, {kWidth, tokens});
    (void)layout.alloc(DType::I64, {kHidden * tokens + 1});
    (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(sizeof(gguf::MoeDecodeRecord)) *
                                   tokens * kTopK});
    (void)layout.alloc(DType::U8, {prepared_bytes(kHidden, tokens)});
    (void)layout.alloc(DType::U8, {prepared_bytes(kWidth, tokens * kTopK)});
    // A shared expert on the generic path: its activation, ids and the bank pass's workspace.
    (void)layout.alloc(DType::U8,
                       {static_cast<std::int32_t>(gguf::vector_activation_bytes(kHidden, tokens))});
    (void)layout.alloc(DType::I32, {tokens});
    (void)layout.alloc(DType::U8, {static_cast<std::int32_t>(gguf::moe_routing_bytes(1, tokens))});
    (void)layout.alloc(DType::BF16, {kWidth, tokens});
    (void)layout.alloc(DType::FP32, {kWidth, tokens});
    (void)layout.alloc(DType::U8,
                       {static_cast<std::int32_t>(gguf::vector_activation_bytes(kWidth, tokens))});
    return layout.peak_bytes(1);
}

namespace {
// The shared expert of a decode call, weighted into its fixed-point sum: the fused decode kernels
// when its banks allow them, else the generic vector path.
void run_shared(const Tensor& m, const Tensor& shared, const GgufMoeWeights& banks,
                WorkspaceArena& workspace, const GgufMoeDecode& out, int chunk,
                cudaStream_t stream) {
    const std::int32_t tokens = out.tokens;
    if (!shared_decodes(banks)) {
        Tensor activation = workspace.alloc(
            DType::U8, {static_cast<std::int32_t>(gguf::vector_activation_bytes(kHidden, tokens))});
        gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(m.data), kHidden,
                                         tokens, nullptr, activation.data, stream);
        run_shared_vector(static_cast<const float*>(shared.data), activation.data, tokens, kHidden,
                          kWidth, banks, workspace, out.fixed, stream);
        return;
    }
    return nullptr;
}

} // namespace

std::size_t moe_experts_gguf_workspace_bytes(std::int32_t tokens) {
    require(tokens > 0, "tokens must be positive");
    std::size_t bytes = 0;
    for (const MoeGeometry& geometry : kGeometries) {
        bytes = std::max(bytes, geometry.workspace(tokens));
    }
    return bytes;
}

void moe_experts_gguf(const Tensor& m, const Tensor& ids, const Tensor& weights,
                      const Tensor& shared, const GgufMoeWeights& banks, WorkspaceArena& workspace,
                      Tensor& y, cudaStream_t stream) {
    const std::int32_t hidden = m.ne[0];
    const std::int32_t top_k  = ids.ne[0];
    require(geometry_for(hidden, banks.width, top_k) != nullptr,
            "no instantiation for this hidden width, expert width and top-k");
    check_stage_inputs(m, ids, weights, top_k);
    require(shaped(y, DType::FP32, hidden, m.ne[1]), "y must be FP32 [hidden, tokens]");
    auto scope = workspace.scope();
    if (moe_experts_gguf_decode_supported(banks, m.ne[1])) {
        const auto state = moe_experts_gguf_decode_begin(m, ids, shared, banks, workspace, stream);
        moe_experts_gguf_decode_finish(state, ids, nullptr, weights, banks, nullptr, nullptr,
                                       nullptr, nullptr, y, stream);
        return;
    }
    const auto stage  = moe_experts_gguf_begin(m, workspace, stream);
    moe_experts_gguf_add(stage, m, ids, weights, &shared, banks, workspace, stream);
    moe_experts_gguf_finish(stage, y, stream);
}

} // namespace ninfer::ops
