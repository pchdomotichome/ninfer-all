#include "runtime/contract/dry.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace ninfer::runtime {
namespace {

// log of the largest finite float, used to cap the exponent before base^exponent overflows.
constexpr float kFloatMaxLog = 88.7228391F;

bool is_breaker(const std::vector<std::int32_t>& breakers, TokenId token) {
    return std::find(breakers.begin(), breakers.end(), token) != breakers.end();
}

} // namespace

void DryPenalty::compute(std::span<const TokenId> window, const DryParams& params) {
    candidates_.clear();
    if (params.multiplier <= 0.0F || params.base < 1.0F || params.last_n <= 0) { return; }
    const std::int32_t allowed = std::max<std::int32_t>(1, params.allowed_length);
    const std::size_t n = std::min<std::size_t>(static_cast<std::size_t>(params.last_n), window.size());
    if (n <= static_cast<std::size_t>(allowed)) { return; }
    const std::int32_t count = static_cast<std::int32_t>(n);

    // The algorithm reads the window from the end, so keep it reversed: reversed_[0] is the newest
    // token. This is the Z-algorithm over the reversed window, as llama.cpp's DRY sampler does.
    reversed_.resize(n);
    for (std::size_t i = 0; i < n; ++i) { reversed_[i] = window[window.size() - 1 - i]; }

    // A breaker caps how long a stretch may be: the stretch that reaches a breaker stops there.
    std::int32_t rep_limit = count;
    if (!params.breakers.empty()) {
        for (std::int32_t i = 0; i < count; ++i) {
            if (is_breaker(params.breakers, reversed_[i])) {
                rep_limit = i;
                break;
            }
        }
    }
    if (rep_limit < allowed) { return; }

    // For every position, the length of the longest suffix that the end of the window also ends
    // with. The Z-algorithm computes it in linear time over the reversed window.
    repeat_count_.assign(n, 0);
    const std::int32_t last = count - 1;
    std::int32_t rt         = 0;
    std::int32_t lt         = 0;
    for (std::int32_t k = 1; k < count; ++k) {
        if (k > rt) {
            std::int32_t m = 0;
            while (m + k < count && reversed_[m] == reversed_[m + k]) { ++m; }
            repeat_count_[last - k] = std::min(m, rep_limit);
            if (m > 0) {
                lt = k;
                rt = k + m - 1;
            }
        } else {
            const std::int32_t pair            = k - lt;
            const std::int32_t right_part_len  = rt - k + 1;
            if (repeat_count_[last - pair] < right_part_len) {
                repeat_count_[last - k] = std::min(repeat_count_[last - pair], rep_limit);
            } else {
                std::int32_t i = rt + 1;
                while (i < count && reversed_[i] == reversed_[i - k]) { ++i; }
                repeat_count_[last - k] = std::min(i - k, rep_limit);
                lt                    = k;
                rt                    = i - 1;
            }
        }
    }

    // The token that follows a stretch would extend it, so it is the one to penalise. The longest
    // stretch wins when a token closes more than one. The map keeps the candidates in token order,
    // so the result does not depend on the hash order.
    std::map<TokenId, std::int32_t> max_stretch;
    for (std::int32_t i = 0; i < count - 1; ++i) {
        const std::int32_t stretch = repeat_count_[i];
        if (stretch < allowed) { continue; }
        const TokenId token = reversed_[count - 2 - i];
        auto [it, inserted] = max_stretch.emplace(token, stretch);
        if (!inserted && it->second < stretch) { it->second = stretch; }
    }
    if (max_stretch.empty()) { return; }

    std::int32_t max_exponent = 0;
    if (params.base > 1.000001F) { max_exponent = static_cast<std::int32_t>(kFloatMaxLog / std::log(params.base)); }
    candidates_.reserve(max_stretch.size());
    for (const auto& [token, stretch] : max_stretch) {
        if (is_breaker(params.breakers, token)) { continue; }
        std::int32_t exponent = stretch - allowed;
        if (max_exponent > 0 && exponent > max_exponent) { exponent = max_exponent; }
        const float penalty = params.multiplier * std::pow(params.base, static_cast<float>(exponent));
        if (penalty > 0.0F) { candidates_.push_back(DryCandidate{token, penalty}); }
    }
}

} // namespace ninfer::runtime
