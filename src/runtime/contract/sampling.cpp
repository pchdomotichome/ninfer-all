#include "runtime/contract/sampling.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::runtime {
namespace {

void validate(const ResolvedSamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty) || !std::isfinite(sampling.dry_multiplier) ||
        !std::isfinite(sampling.dry_base)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.temperature < 0.0F || sampling.temperature > 2.0F) {
        throw std::invalid_argument("temperature must be in [0,2]");
    }
    if (sampling.top_k < 1 || sampling.top_k > 20) {
        throw std::invalid_argument("resolved top_k must be in [1,20]");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
    if (sampling.presence_penalty < -2.0F || sampling.presence_penalty > 2.0F) {
        throw std::invalid_argument("presence_penalty must be in [-2,2]");
    }
    if (sampling.frequency_penalty < -2.0F || sampling.frequency_penalty > 2.0F) {
        throw std::invalid_argument("frequency_penalty must be in [-2,2]");
    }
    if (sampling.dry_multiplier < 0.0F || sampling.dry_multiplier > 2.0F) {
        throw std::invalid_argument("dry_multiplier must be in [0,2]");
    }
    if (sampling.dry_base < 1.0F || sampling.dry_base > 100.0F) {
        throw std::invalid_argument("dry_base must be in [1,100]");
    }
    if (sampling.dry_allowed_length < 1 || sampling.dry_allowed_length > 64) {
        throw std::invalid_argument("dry_allowed_length must be in [1,64]");
    }
    if (sampling.dry_penalty_last_n < 0 || sampling.dry_penalty_last_n > 4096) {
        throw std::invalid_argument("dry_penalty_last_n must be in [0,4096]");
    }
}

} // namespace

ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                            SamplingMode mode, const SamplingOverrides& overrides) {
    return resolve_sampling(defaults.for_mode(mode), overrides);
}

ResolvedSamplingParameters resolve_sampling(const SamplingPreset& preset,
                                            const SamplingOverrides& overrides) {
    ResolvedSamplingParameters resolved{
        .temperature       = overrides.temperature.value_or(preset.temperature),
        .top_k             = overrides.top_k.value_or(preset.top_k),
        .top_p             = overrides.top_p.value_or(preset.top_p),
        .min_p             = overrides.min_p.value_or(preset.min_p),
        .presence_penalty  = overrides.presence_penalty.value_or(preset.presence_penalty),
        .frequency_penalty = overrides.frequency_penalty.value_or(preset.frequency_penalty),
        .dry_multiplier    = overrides.dry_multiplier.value_or(preset.dry_multiplier),
        .dry_base          = overrides.dry_base.value_or(preset.dry_base),
        .dry_allowed_length =
            overrides.dry_allowed_length.value_or(preset.dry_allowed_length),
        .dry_penalty_last_n = overrides.dry_penalty_last_n.value_or(preset.dry_penalty_last_n),
        .seed              = overrides.seed.value_or(0),
    };
    // The registered sampling pipeline has an exact top-20 candidate domain. Preserve the
    // existing public meaning of zero (use the target cap), then expose only concrete values to
    // runtimes.
    if (resolved.top_k == 0) { resolved.top_k = 20; }
    validate(resolved);
    return resolved;
}

} // namespace ninfer::runtime
