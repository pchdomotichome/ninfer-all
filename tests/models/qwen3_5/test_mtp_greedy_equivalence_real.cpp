#include "guarded_main.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Greedy speculative decoding is a resampling of the target distribution, not a different model:
// with temperature 0 the licensed stream must equal the stream the same engine emits with no
// speculative backend, token for token. That makes an independent oracle available without
// reaching inside the engine -- the ordinary decode path *is* the reference. A divergence here is
// a defect in the verify path (or in the state it carries forward), not in greedy sampling.
//
// The defect this test witnesses lives in the multi-column commit: a round that licenses more than
// one token leaves a recurrent state the next round does not reproduce. Measured on the 4B GGUF
// artifact, forcing a round to zero columns makes every case below pass, so the anchor path, the
// ledger, greedy sampling and the single-column fold are all correct on their own.

namespace {

ninfer::EngineOptions engine_options(const char* artifact,
                                     ninfer::SpeculativeBackend backend,
                                     std::uint32_t draft_tokens, std::uint32_t max_context) {
    ninfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.max_context    = max_context;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(max_context);
    options.prefill_chunk  = max_context;
    options.kv_cache       = ninfer::KvCacheStorage::BFloat16;
    options.use_cuda_graph = false;
    options.enable_vision  = false;
    options.speculative.backend = backend;
    if (backend != ninfer::SpeculativeBackend::None) {
        options.speculative.draft_tokens = draft_tokens;
    }
    return options;
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

// Two prompts, both short enough that several verify rounds run and at least one draft is
// accepted. The first acceptance is where a committed draft's state stops being scratch, so a
// prompt that never accepts one would pass even with the defect present.
std::vector<std::string> prompts() {
    return {
        "Name the three primary colors in one short sentence.",
        "Count from one to ten.",
    };
}

std::vector<std::int32_t> generate(ninfer::Engine& engine, const std::string& prompt,
                                    std::uint32_t outputs) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = prompt, .media = {}});
    input.messages.push_back(std::move(user));
    const ninfer::GenerationResult result = engine.generate(engine.prepare(std::move(input)),
                                                             greedy_options(outputs));
    return result.generated_token_ids;
}

int compare_widths(const char* artifact, std::uint32_t draft_tokens, std::uint32_t max_context,
                   std::uint32_t outputs) {
    int failures = 0;

    ninfer::Engine reference(engine_options(artifact, ninfer::SpeculativeBackend::None, 0,
                                            max_context));
    // A second ordinary engine stands in for a second reading of the oracle. Greedy decode is
    // deterministic, so if these two disagree the comparison below cannot be attributed to the
    // speculative path and the test would be reporting its own harness.
    ninfer::Engine reference_again(engine_options(artifact, ninfer::SpeculativeBackend::None, 0,
                                                 max_context));
    ninfer::Engine verify(engine_options(artifact, ninfer::SpeculativeBackend::Mtp, draft_tokens,
                                         max_context));

    for (const std::string& prompt : prompts()) {
        const std::vector<std::int32_t> expected = generate(reference, prompt, outputs);
        if (generate(reference_again, prompt, outputs) != expected) {
            std::cerr << "mtp_greedy_matches_ordinary: ordinary decode is not deterministic; the "
                         "comparison cannot be attributed to the speculative path\n";
            ++failures;
            continue;
        }
        const std::vector<std::int32_t> actual = generate(verify, prompt, outputs);
        if (expected.empty()) {
            std::cerr << "mtp_greedy_matches_ordinary: reference produced no tokens\n";
            ++failures;
            continue;
        }
        if (actual != expected) {
            const std::size_t limit = std::min(actual.size(), expected.size());
            std::size_t at          = 0;
            while (at < limit && actual[at] == expected[at]) { ++at; }
            std::cerr << "mtp_greedy_matches_ordinary: K=" << draft_tokens
                      << " diverges at emitted index " << at << " of " << expected.size()
                      << " (got " << (at < actual.size() ? actual[at] : -1) << ", expected "
                      << (at < expected.size() ? expected[at] : -1) << ")\n";
            std::cerr << "  expected:";
            for (std::size_t i = 0; i < std::min<std::size_t>(expected.size(), 16); ++i) {
                std::cerr << ' ' << expected[i];
            }
            std::cerr << "\n  actual:  ";
            for (std::size_t i = 0; i < std::min<std::size_t>(actual.size(), 16); ++i) {
                std::cerr << ' ' << actual[i];
            }
            std::cerr << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    // Widths 1..3 sweep the draft window. Note that these do not currently produce three distinct
    // verify widths: mtp_draft_count is 1 for these prompts, so the round's column count is
    // min(mtp_draft_count, verify_window, ...) == 1 whatever the window. The sweep is kept because
    // that is an accident of these prompts, not a guarantee, and a width that did change must not
    // go unchecked -- but a divergence here is not evidence of width-dependent behaviour.
    int failures = 0;
    for (std::uint32_t draft_tokens = 1; draft_tokens <= 3; ++draft_tokens) {
        failures += compare_widths(artifact, draft_tokens, 256, 40);
    }
    if (failures != 0) {
        std::cerr << failures << " MTP greedy width(s) diverged from ordinary decode\n";
        return 1;
    }
    std::cout << "mtp_greedy_matches_ordinary: all widths reproduced ordinary decode\n";
    return 0;
}