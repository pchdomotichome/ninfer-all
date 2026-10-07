#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

// The 4096-hidden model's vocabulary head: the same 248320 rows as the 2048-hidden
// model's, reducing over 4096 columns instead of 2048. Every kernel these name reads
// the input width at runtime and the tile widths are column counts, so the token
// thresholds below are the 2048-hidden shape's own -- a wider reduction amortises the
// weight read over more of it, which moves a boundary, but which boundary is a
// measurement on this card and not something this registration can claim.
Q6Launch select_q6_n248320_k4096(std::int32_t tokens) {
    if (tokens <= 3) return launch_q6_a16_simt_r8_t4_cg;
    if (tokens <= 16) return launch_q6_a16_mma_r64_t16_k128;
    if (tokens <= 24) return launch_q6_a16_mma_r64_t24_k128;
    if (tokens <= 32) return launch_q6_a16_mma_r64_t32_k128;
    if (tokens <= 40) return launch_q6_a16_mma_r64_t40_k128;
    if (tokens <= 48) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 56) return launch_q6_a16_mma_r64_t56_k128;
    if (tokens <= 64) return launch_q6_a16_mma_r64_t64_k128;
    if (tokens <= 72) return launch_q6_a16_mma_r64_t72_k128;
    if (tokens <= 80) return launch_q6_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q6_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q6_a16_mma_r64_t112;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail