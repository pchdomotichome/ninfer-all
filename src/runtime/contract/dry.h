#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::runtime {

// The DRY repetition penalty. A candidate token is penalised when emitting it would continue a
// stretch of the recent output that the model already wrote verbatim, instead of penalising tokens
// for being frequent (presence and frequency do that, and for code they punish the language).
//
// A stretch counts when it reaches `allowed_length` tokens, and the penalty grows with the stretch
// as base^(stretch - allowed_length) times the multiplier. The window is the last `last_n` generated
// tokens, which is what llama.cpp's DRY sampler keeps: the prompt is not part of it.
//
// This is the host half. It produces the sparse set of penalised candidates; the engine turns that
// set into a per-lane penalty vector and the sampling kernel subtracts it from the logits.
struct DryParams {
    float multiplier = 0.0F;         // 0 disables the penalty
    float base = 1.0F;               // exponential base, must be >= 1
    std::int32_t allowed_length = 2; // shortest stretch, in tokens, that counts
    std::int32_t last_n = 0;         // window size in generated tokens, 0 disables
    // Tokens that end a stretch: a stretch that reaches one is capped at the breaker, and a
    // candidate that is itself a breaker is never penalised. Empty means no cap.
    std::vector<std::int32_t> breakers;
};

struct DryCandidate {
    std::int32_t token = 0;
    float penalty = 0.0F;
};

class DryPenalty {
  public:
    // Recomputes the penalised candidates for the given window, oldest token first. The window may
    // be longer than last_n; only its tail is read.
    void compute(std::span<const TokenId> window, const DryParams& params);
    void reset() { candidates_.clear(); }

    [[nodiscard]] bool empty() const noexcept { return candidates_.empty(); }
    [[nodiscard]] const std::vector<DryCandidate>& candidates() const noexcept { return candidates_; }

  private:
    std::vector<DryCandidate> candidates_;
    std::vector<std::int32_t> repeat_count_;
    std::vector<TokenId> reversed_;
};

} // namespace ninfer::runtime
