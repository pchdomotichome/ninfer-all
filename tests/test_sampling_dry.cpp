#include "runtime/contract/dry.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

using ninfer::runtime::DryCandidate;
using ninfer::runtime::DryParams;
using ninfer::runtime::DryPenalty;
using TokenId = ninfer::TokenId;

// The reference the fast implementation must agree with: for every position, the length of the
// longest suffix that the end of the window also ends with, capped at the breaker limit.
std::vector<int> brute_repeat(const std::vector<TokenId>& tail, int rep_limit) {
    const int n = static_cast<int>(tail.size());
    std::vector<int> counts(n, 0);
    for (int j = 0; j < n; ++j) {
        int len = 0;
        while (j - len >= 0 && n - 1 - len >= 0 && tail[j - len] == tail[n - 1 - len]) { ++len; }
        counts[j] = std::min(len, rep_limit);
    }
    return counts;
}

std::vector<std::pair<TokenId, float>> brute_candidates(const std::vector<TokenId>& window,
                                                         const DryParams& params) {
    std::vector<std::pair<TokenId, float>> out;
    if (params.multiplier <= 0.0F || params.base < 1.0F || params.last_n <= 0) { return out; }
    const int allowed = std::max(1, params.allowed_length);
    const int n       = static_cast<int>(std::min<std::size_t>(params.last_n, window.size()));
    if (n <= allowed) { return out; }
    const std::vector<TokenId> tail(window.end() - n, window.end());
    int rep_limit = n;
    for (int i = 0; i < n; ++i) {
        if (std::find(params.breakers.begin(), params.breakers.end(), tail[n - 1 - i]) !=
            params.breakers.end()) {
            rep_limit = i;
            break;
        }
    }
    if (rep_limit < allowed) { return out; }
    const std::vector<int> counts = brute_repeat(tail, rep_limit);
    std::map<TokenId, int> stretches;
    for (int i = 0; i < n - 1; ++i) {
        if (counts[i] < allowed) { continue; }
        const TokenId token = tail[i + 1];
        auto [it, inserted] = stretches.emplace(token, counts[i]);
        if (!inserted && it->second < counts[i]) { it->second = counts[i]; }
    }
    for (const auto& [token, stretch] : stretches) {
        if (std::find(params.breakers.begin(), params.breakers.end(), token) != params.breakers.end()) {
            continue;
        }
        out.emplace_back(token, params.multiplier * std::pow(params.base, stretch - allowed));
    }
    return out;
}

bool same_candidates(const std::vector<DryCandidate>& actual,
                     const std::vector<std::pair<TokenId, float>>& expected) {
    if (actual.size() != expected.size()) { return false; }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i].token != expected[i].first) { return false; }
        if (std::fabs(actual[i].penalty - expected[i].second) > 1e-5F) { return false; }
    }
    return true;
}

} // namespace

int main() {
    int failures = 0;

    // The worked example from llama.cpp's DRY sampler: "a b c c b c y a b c" repeats the suffix
    // "a b c" three tokens long at the first c, so the token that would extend it (the second c)
    // takes the penalty, and "b c" twice makes y the candidate for a shorter stretch.
    {
        const std::vector<TokenId> window = {1, 2, 3, 3, 2, 3, 4, 1, 2, 3};
        const DryParams params{.multiplier = 1.0F, .base = 2.0F, .allowed_length = 2, .last_n = 256};
        DryPenalty dry;
        dry.compute(window, params);
        const std::vector<std::pair<TokenId, float>> expected = {{3, 2.0F}, {4, 1.0F}};
        failures += check(same_candidates(dry.candidates(), expected),
                          "DRY: the worked example must penalise c (stretch 3) and y (stretch 2)");
    }

    // The window is only its tail: "1 2 3 1 2 3" repeats the suffix "1 2 3", so the token that
    // would extend it is the second 1. A window of three tokens cannot hold a stretch of three, so
    // the same text with a shorter window earns no penalty.
    {
        const std::vector<TokenId> window = {9, 9, 1, 2, 3, 1, 2, 3};
        DryPenalty dry;
        dry.compute(window, DryParams{.multiplier = 1.0F, .base = 1.5F, .allowed_length = 3, .last_n = 6});
        failures += check(dry.candidates().size() == 1 && dry.candidates().front().token == 1,
                          "DRY: the window is read from its tail");
        dry.compute(window, DryParams{.multiplier = 1.0F, .base = 1.5F, .allowed_length = 3, .last_n = 3});
        failures += check(dry.empty(), "DRY: only the last_n tokens of the window count");
    }

    // Disabled states produce nothing.
    {
        DryPenalty dry;
        const std::vector<TokenId> window = {1, 2, 3, 1, 2, 3};
        dry.compute(window, DryParams{.multiplier = 0.0F, .base = 1.5F, .allowed_length = 2, .last_n = 16});
        failures += check(dry.empty(), "DRY: a zero multiplier disables the penalty");
        dry.compute(window, DryParams{.multiplier = 1.0F, .base = 0.5F, .allowed_length = 2, .last_n = 16});
        failures += check(dry.empty(), "DRY: a base below one disables the penalty");
        dry.compute(window, DryParams{.multiplier = 1.0F, .base = 1.5F, .allowed_length = 2, .last_n = 0});
        failures += check(dry.empty(), "DRY: a zero window disables the penalty");
        const std::vector<TokenId> short_window = {1, 2};
        dry.compute(short_window,
                   DryParams{.multiplier = 1.0F, .base = 1.5F, .allowed_length = 2, .last_n = 16});
        failures += check(dry.empty(), "DRY: a window no longer than allowed_length does nothing");
    }

    // A breaker caps the stretch, and a candidate that is itself a breaker is never penalised.
    // "1 2 3 5 1 2 3" ends in a three-token repeat of "1 2 3", so the token that would extend it
    // is 5: the penalty is 2^(3-2) = 2. A breaker on the 1 stops the stretch two tokens short, so
    // the same text earns 2^(2-2) = 1, and a breaker on the 5 exempts the candidate itself.
    {
        const std::vector<TokenId> window = {1, 2, 3, 5, 1, 2, 3};
        DryPenalty dry;
        dry.compute(window, DryParams{.multiplier = 1.0F, .base = 2.0F, .allowed_length = 2, .last_n = 16});
        failures += check(dry.candidates().size() == 1 && dry.candidates().front().token == 5 &&
                              std::fabs(dry.candidates().front().penalty - 2.0F) < 1e-5F,
                          "DRY: the token that would extend the repeat takes the penalty");
        dry.compute(window,
                    DryParams{.multiplier = 1.0F, .base = 2.0F, .allowed_length = 2, .last_n = 16,
                              .breakers = {1}});
        failures += check(dry.candidates().size() == 1 && dry.candidates().front().token == 5 &&
                              std::fabs(dry.candidates().front().penalty - 1.0F) < 1e-5F,
                          "DRY: a breaker caps the stretch");
        dry.compute(window,
                    DryParams{.multiplier = 1.0F, .base = 2.0F, .allowed_length = 2, .last_n = 16,
                              .breakers = {5}});
        failures += check(dry.empty(), "DRY: a candidate that is a breaker is never penalised");
    }

    // The fast implementation must agree with the reference on random windows.
    {
        std::mt19937 rng(1234);
        for (int trial = 0; trial < 400; ++trial) {
            const int n = 4 + static_cast<int>(rng() % 40);
            std::vector<TokenId> window(static_cast<std::size_t>(n));
            for (auto& token : window) { token = static_cast<TokenId>(rng() % 5); }
            DryParams params{.multiplier = 0.5F,
                             .base = 1.05F + static_cast<float>(rng() % 30) / 100.0F,
                             .allowed_length = 1 + static_cast<std::int32_t>(rng() % 4),
                             .last_n = 1 + static_cast<std::int32_t>(rng() % 64)};
            if (trial % 3 == 0) { params.breakers = {static_cast<std::int32_t>(rng() % 5)}; }
            DryPenalty dry;
            dry.compute(window, params);
            const std::vector<std::pair<TokenId, float>> expected = brute_candidates(window, params);
            if (!same_candidates(dry.candidates(), expected)) {
                failures += check(false, "DRY: the fast implementation disagrees with the reference");
                break;
            }
        }
    }

    if (failures == 0) { std::cout << "sampling_dry: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
