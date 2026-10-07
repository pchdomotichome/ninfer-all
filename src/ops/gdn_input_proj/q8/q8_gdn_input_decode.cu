#include "core/weight.h"
#include "ops/gdn_input_proj/q8/q8_gdn_input_kernels.h"

#include "ops/gdn_input_proj/q8/q8_gdn_input_kernels_unified.h"
#include "ops/linear/common/route_table.h"
#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/linear/q8/q8_decode.cuh"

namespace ninfer::ops::detail {
namespace {

using Output = Q8SplitOutput2<8192, 4096>;

struct Q8GdnDecodeConvEpilogue {
    GdnConvEpilogue<SnapshotHistoryPublish> conv;
    __nv_bfloat16* z;

    template <class IgnoredOutput>
    __device__ __forceinline__ void operator()(const IgnoredOutput&, std::int32_t, std::int32_t row,
                                               float accumulator) const {
        if (row < 8192) {
            const float projected[1]{accumulator};
            conv.store(row, projected);
        } else {
            z[row - 8192] = __float2bfloat16_rn(accumulator);
        }
    }
};

GdnConvEpilogue<SnapshotHistoryPublish>
make_conv_epilogue(const Tensor& conv_weight, Tensor& conv_states, const Tensor& valid_columns,
                   const Tensor& initial_slot, const Tensor& snapshot_base_slot, Tensor& query,
                   Tensor& key, Tensor& value) {
    return {
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_slot.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        8192,
        2048,
        2048,
        4096,
        0,
        1,
        0,
        SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                               static_cast<const std::int32_t*>(snapshot_base_slot.data), 8192},
    };
}

// The decode kernel is templated on K so one instantiation serves a width rather
// than one being compiled per model; the runtime width selects between them.
template <class Epilogue>
void launch_decode_for_width(int hidden, const Tensor& x, const Weight& weight, Output output,
                             Epilogue epilogue, cudaStream_t stream) {
    constexpr int kRows       = 12288;
    constexpr int kRowsPerCta = 8;
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const auto* x_data     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* code_data  = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scale_data = static_cast<const std::uint8_t*>(weight.scales);
    switch (hidden) {
    case 2048:
        q8_decode_kernel<kRows, 2048, kRowsPerCta, Output, Epilogue>
            <<<kRows / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(x_data, code_data, scale_data,
                                                                   output, epilogue);
        break;
    case 4096:
        q8_decode_kernel<kRows, 4096, kRowsPerCta, Output, Epilogue>
            <<<kRows / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(x_data, code_data, scale_data,
                                                                   output, epilogue);
        break;
    default:
        throw std::invalid_argument("Q8 GDN input decode: unsupported input width");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
    if (fused_route_table("unified/q8_gdn_input", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    }
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    launch_decode_for_width(x.ne[0], x, weight, output, Q8DecodeStoreEpilogue{}, stream);
}

void q8_gdn_input_decode_conv_snapshot_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states,
    const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot,
    Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    if (fused_route_table("unified/q8_gdn_input", x.ne[1]) == LinearRouteTable::Unified) {
        unified::q8_gdn_input_decode_conv_snapshot_launch(
            x, weight, conv_weight, conv_states, valid_columns, initial_slot, snapshot_base_slot,
            query, key, value, z, stream);
        return;
    }
    const Output ignored_output{static_cast<__nv_bfloat16*>(query.data),
                                static_cast<__nv_bfloat16*>(z.data)};
    const Q8GdnDecodeConvEpilogue epilogue{
        make_conv_epilogue(conv_weight, conv_states, valid_columns, initial_slot,
                           snapshot_base_slot, query, key, value),
        static_cast<__nv_bfloat16*>(z.data),
    };
    launch_decode_for_width(x.ne[0], x, weight, ignored_output, epilogue, stream);
}

} // namespace ninfer::ops::detail
