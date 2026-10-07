#include "core/weight.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/q4/q4_sliced_k_launch.cuh"

#include <cuda_bf16.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kN            = 34816;
constexpr int kK            = 5120;
constexpr int kIntermediate = kN / 2;

struct Q4SwiGluSmallTRows {
    static constexpr int kOutputRowsPerCta = 8;

    __host__ __device__ int output_rows(int rows) const { return rows / 2; }

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & 7) + (local_row >= 8 ? kIntermediate : 0);
    }
};

struct Q4SwiGluSmallTEpilogue {
    template <class Output>
    __device__ __forceinline__ void store_fragment(const Output& output, int row, int token,
                                                   float4 projected, int rows,
                                                   int token_end) const {
        if (row >= rows / 2) return;
        if (token < token_end) output.store(row, token, silu(projected.x) * projected.z);
        if (token + 1 < token_end) output.store(row, token + 1, silu(projected.y) * projected.w);
    }
};


template <int K, int Intermediate, int ActiveCols>
void launch_small_t_active(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    // Wider tiles are shared-memory limited before six CTAs can be resident.
    // Match launch bounds to that limit instead of forcing register spills.
    constexpr int kMinBlocks = ActiveCols <= 8    ? 6
                               : ActiveCols <= 16 ? 4
                               : ActiveCols <= 24 ? 3
                                                  : 2;
    using Schedule =
        Q4A16SlicedKMmaSchedule<16, (ActiveCols + 7) / 8 * 8, 8, 1, Cache::cg, Cache::ca,
                                kMinBlocks, K, ActiveCols, Q4SlicedKReduction::Pairwise>;
    launch_q4_a16_sliced_k_mma<Schedule>(
        q4_linear_operands(x, w),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), Intermediate},
        Q4SwiGluSmallTEpilogue{}, stream, Q4SwiGluSmallTRows{});
}


} // namespace

using SmallTLauncher = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int K, int Intermediate, std::size_t... Offsets>
constexpr auto make_small_t_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallTLauncher, sizeof...(Offsets)>{
        &launch_small_t_active<K, Intermediate, 8 * (1 + static_cast<int>(Offsets))>...};
}

void q4_linear_swiglu_small_t_unified_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > 32) {
        throw std::invalid_argument("Q4 LinearSwiGLU exact small-T requires T=2..32");
    }
    const std::size_t index = static_cast<std::size_t>((x.ne[1] - 1) / 8);
    if (w.n == 34816 && w.k == 5120) {
        make_small_t_launchers<5120, 17408>(std::make_index_sequence<4>{})[index](x, w, out, stream);
        return;
    }
    if (w.n == 24576 && w.k == 4096) {
        make_small_t_launchers<4096, 12288>(std::make_index_sequence<4>{})[index](x, w, out, stream);
        return;
    }
    throw std::invalid_argument("Q4 LinearSwiGLU exact small-T: no instantiation for this shape");
}

} // namespace ninfer::ops::detail
