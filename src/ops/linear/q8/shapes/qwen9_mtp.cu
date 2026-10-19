#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

// Generated from the n<rows>K<k>.cu shape table, one per linear the Qwen3.5-9B MTP layer needs at
// its own widths. Every schedule named here takes its width from the Geometry template argument, so
// this file selects on the token count alone; the tile and block choices are the ones measured for
// the nearest registered shape of the same role, not a measurement at this width.
//
//   mtp/input_projection            (4096,  8192)
//   mtp/layers/0/attention/query    (10240, 4096)  parent, split into q and k
//   mtp/layers/0/attention/output   (4096,  4096)
//   mtp/layers/0/mlp/gate           (24576, 4096)  gate/up parent
//   mtp/layers/0/mlp/down           (4096,  12288)

using Access = Q8KSplitScaleAccess;
using Stage  = Q8KSplitActivationStage;

template <int KWarps, int TileTokens, int MinBlocks, Access A, Cache AC, Cache WC, Stage S>
using Tile = Q8KSplitSm8xFourWarpSchedule<KWarps, TileTokens, MinBlocks, A, AC, WC, S>;

template <class Geometry>
struct Tiles {
    using C4   = Tile<16, 8, 1, Access::Direct, Cache::cg, Cache::cg, Stage::RuntimeActive>;
    using C8   = Tile<16, 8, 1, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
    using C16  = Tile<16, 16, 1, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
    using C24  = Tile<8, 24, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
    using C32  = Tile<8, 32, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
    using C40  = Tile<8, 40, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
    using C48  = Tile<8, 48, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
    using C56  = Tile<8, 56, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
    using C64  = Tile<8, 64, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
};

} // namespace

Q8Launch select_q8_n4096_k8192(std::int32_t tokens) {
    using G = Q8N4096K8192;
    using C = Tiles<G>;
    if (tokens <= 4) return launch_q8_ksplit<G, 4, C::C4>;
    if (tokens <= 8) return launch_q8_ksplit<G, 8, C::C8>;
    if (tokens <= 16) return launch_q8_ksplit<G, 16, C::C16>;
    if (tokens <= 24) return launch_q8_ksplit<G, 24, C::C24>;
    if (tokens <= 32) return launch_q8_ksplit<G, 32, C::C32>;
    if (tokens <= 40) return launch_q8_ksplit<G, 40, C::C40>;
    if (tokens <= 48) return launch_q8_ksplit<G, 48, C::C48>;
    if (tokens <= 56) return launch_q8_ksplit<G, 56, C::C56>;
    if (tokens <= 64) return launch_q8_ksplit<G, 64, C::C64>;
    if (tokens < 896) return launch_q8_mma_r32_c128;
    return launch_q8_mma_r64_c128;
}

Q8Launch select_q8_n10240_k4096(std::int32_t tokens) {
    using G = Q8N10240K4096;
    using C = Tiles<G>;
    if (tokens <= 4) return launch_q8_ksplit<G, 4, C::C4>;
    if (tokens <= 8) return launch_q8_ksplit<G, 8, C::C8>;
    if (tokens <= 16) return launch_q8_ksplit<G, 16, C::C16>;
    if (tokens <= 24) return launch_q8_ksplit<G, 24, C::C24>;
    if (tokens <= 32) return launch_q8_ksplit<G, 32, C::C32>;
    if (tokens <= 40) return launch_q8_ksplit<G, 40, C::C40>;
    if (tokens <= 48) return launch_q8_ksplit<G, 48, C::C48>;
    if (tokens <= 56) return launch_q8_ksplit<G, 56, C::C56>;
    if (tokens <= 64) return launch_q8_ksplit<G, 64, C::C64>;
    if (tokens < 896) return launch_q8_mma_r32_c128;
    return launch_q8_mma_r64_c128;
}

Q8Launch select_q8_n4096_k4096(std::int32_t tokens) {
    using G = Q8N4096K4096;
    using C = Tiles<G>;
    if (tokens <= 4) return launch_q8_ksplit<G, 4, C::C4>;
    if (tokens <= 8) return launch_q8_ksplit<G, 8, C::C8>;
    if (tokens <= 16) return launch_q8_ksplit<G, 16, C::C16>;
    if (tokens <= 24) return launch_q8_ksplit<G, 24, C::C24>;
    if (tokens <= 32) return launch_q8_ksplit<G, 32, C::C32>;
    if (tokens <= 40) return launch_q8_ksplit<G, 40, C::C40>;
    if (tokens <= 48) return launch_q8_ksplit<G, 48, C::C48>;
    if (tokens <= 56) return launch_q8_ksplit<G, 56, C::C56>;
    if (tokens <= 64) return launch_q8_ksplit<G, 64, C::C64>;
    if (tokens < 896) return launch_q8_mma_r32_c128;
    return launch_q8_mma_r64_c128;
}

Q8Launch select_q8_n24576_k4096(std::int32_t tokens) {
    using G = Q8N24576K4096;
    using C = Tiles<G>;
    if (tokens <= 4) return launch_q8_ksplit<G, 4, C::C4>;
    if (tokens <= 8) return launch_q8_ksplit<G, 8, C::C8>;
    if (tokens <= 16) return launch_q8_ksplit<G, 16, C::C16>;
    if (tokens <= 24) return launch_q8_ksplit<G, 24, C::C24>;
    if (tokens <= 32) return launch_q8_ksplit<G, 32, C::C32>;
    if (tokens <= 40) return launch_q8_ksplit<G, 40, C::C40>;
    if (tokens <= 48) return launch_q8_ksplit<G, 48, C::C48>;
    if (tokens <= 56) return launch_q8_ksplit<G, 56, C::C56>;
    if (tokens <= 64) return launch_q8_ksplit<G, 64, C::C64>;
    if (tokens < 896) return launch_q8_mma_r32_c128;
    return launch_q8_mma_r64_c128;
}

Q8Launch select_q8_n4096_k12288(std::int32_t tokens) {
    using G = Q8N4096K12288;
    using C = Tiles<G>;
    if (tokens <= 4) return launch_q8_ksplit<G, 4, C::C4>;
    if (tokens <= 8) return launch_q8_ksplit<G, 8, C::C8>;
    if (tokens <= 16) return launch_q8_ksplit<G, 16, C::C16>;
    if (tokens <= 24) return launch_q8_ksplit<G, 24, C::C24>;
    if (tokens <= 32) return launch_q8_ksplit<G, 32, C::C32>;
    if (tokens <= 40) return launch_q8_ksplit<G, 40, C::C40>;
    if (tokens <= 48) return launch_q8_ksplit<G, 48, C::C48>;
    if (tokens <= 56) return launch_q8_ksplit<G, 56, C::C56>;
    if (tokens <= 64) return launch_q8_ksplit<G, 64, C::C64>;
    if (tokens < 896) return launch_q8_mma_r32_c128;
    return launch_q8_mma_r64_c128;
}

} // namespace ninfer::ops::detail