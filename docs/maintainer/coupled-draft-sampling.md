# Coupled draft sampling (design)

How to make speculative decoding stop throwing away drafts when the request samples.

## The problem

A draft is kept only when it equals the row's own pick. Under greedy decoding the pick
is the argmax, so a greedy draft matches; under sampling (temperature above 0) the pick
is a draw, and a greedy draft matches it only with probability `p_i(draft)`. The MTP
head proposes its argmax, so at `temp 1.0 / top-p 0.95 / top-k 20` most drafted tokens
are discarded even when the draft's distribution is nearly the target's.

Two independent wastes sit in that sentence:

1. The draft picks the *most likely* token; the target draws one. A draw from almost the
   same distribution disagrees with an argmax far more often than two draws from the
   same distribution disagree with each other.
2. The two sides use different random numbers, so even when both prefer the same
   token, a difference of one token in the candidate lists shifts every later pick.

## What Strata does

Strata's engine has two opt-in switches for this (`include/strata/core/coupled_draft.hpp`,
`include/strata/kernels/sampler.hpp`):

- `STRATA_SPEC_COUPLED=1`: the draft layer *samples* instead of taking its argmax, with
  the target's own chain (the request's `top_k / top_p / min_p / temperature` and the same
  penalties over the same history) and **the same uniform** the target will draw for the
  row that verifies that draft. Two inverse-CDF picks from one uniform agree far more
  often than an argmax agrees with a sample.
- `STRATA_SPEC_GUMBEL=1`: replaces the inverse-CDF pick on *both* sides with a Gumbel-max
  pick, `argmax_i p_i / E_i` with `E_i ~ Exp(1)` keyed by `(seed, counter, token id)`
  (`sampler.cu: gumbel_exp`). The noise is per token, so both sides apply the *same* noise
  to a token whichever other tokens survived either cut.

Measured by Strata on an RTX 3060 (IQ3_XXS, `--spec 4 --mtp`, `temp 1.0 / top_p 0.95 /
top_k 20`, 12 interleaved pairs): accepted drafts 62.1% -> 66.9% (coupled) -> 68.0%
(coupled + Gumbel). On a Ryzen AI Max+ 395 the same switch took 41.1 -> 44.9 tok/s.

The output text does not change: verification is still an exact match against the
target's own sample, so the target's draw decides the text and the drafts only decide
how many tokens one window yields.

### The counter arithmetic

All positions are sequence indices. A window that starts at `p` with `a` drafts accepted
continues at `p' = p + a + 1`; chain step `j` runs at cell `p + a + j` and predicts the
token at `p + a + j + 2`, which is row `j + 1` of the next window and is verified by row
`j`'s pick, whose counter is `p' + j`. So **a draft made at cell `c` is verified with
counter `c + 1`**, whatever the window size, the accepted count or a suffix drafter in
between. The penalty history mirrors the target's: the last `h` tokens of
`(consumed', x', drafts 0..j-1)`.

## What NInfer has today

Two pieces make the port small.

**The uniform is already position-keyed and graph-safe.** `sampling_uniform(seed,
position, purpose, sub)` (`src/ops/kernel/sampling_device.cuh`) is a pure hash of its
inputs, with no mutable RNG state, so it replays identically inside a captured CUDA
graph. It plays the role of Strata's `Philox(seed, counter)`: the same call with the same
`(seed, position)` yields the same number on both sides of the round.

**The acceptance rule already exists.** `speculative_accept_coupled_drafts_kernel`
(`src/ops/kernel/speculative_round.cuh`) keeps the longest draft prefix whose target
argmax matches, then takes the target's own sample whatever was drafted; a greedy draft
survives with probability `p_i(draft)`. That is the rule for an *argmax* draft. Under a
coupled draft the two sides draw from the same uniform, so the acceptance becomes an
exact comparison of two draws of the same number, and the probability rule stops being
needed for the sampling case.

**What is missing is the draft's pick.** `src/models/qwen3_5/execution/draft.cpp` takes
the MTP head's logits and calls `ops::argmax(logits, flat_drafts, ...)` (two call sites:
the full proposal head and the optimized DFlash proposal head). That argmax is the line
the change lands on.

## The port

### 1. The draft picks a sample, not the argmax

In `draft.cpp`, replace the argmax with a pick that:

1. runs the request's chain over the draft's logits: temperature, `top_k`, `top_p`,
   `min_p`, and the same penalties the target applies over the same history;
2. draws with the uniform the verify will use for that draft: the draft made at chain
   cell `c` is verified at position `c + 1`, so the pick is
   `sampling_uniform(seed, c + 1, kPurposePick, 0)` -- the same call the verify row
   makes.

The candidate support of the draft is the draft head's own logits sorted descending, so
the chain is the same code the target's sampler runs
(`sampling_normalize_support` in `sampling_device.cuh`), with the draft's probabilities
instead of the target's.

### 2. The penalty history mirrors the target's

The draft's chain must see the same penalties the target will apply to that position.
NInfer keeps the lane's recent tokens (`token_counts` in `sampling.cuh`, the request's
generated history in the program path), so the draft's penalty rows are the same rows the
verify row will use -- no ring buffer needed, as long as the draft and the verify read
the same history at the same position. Strata keeps a ring because its drafter runs
ahead of the verify; NInfer's drafter runs inside the same round, so the histories
coincide.

### 3. Gumbel-max (a second, separate change)

A Gumbel pick needs a per-token noise: `E_i = -log(u_i)` with
`u_i = sampling_uniform(seed, counter, kPurposeGumbel, token_id)`, and the pick becomes
`argmax_i p_i / E_i` over the support, on **both** the draft and the verify side. The
verify side's change is local to the sampler; the draft side reuses the same function.
Greedy requests are unaffected (`p_i` is a point mass, the argmax wins either way).

The noise key must include the token id, so `sampling_uniform`'s `sub` parameter is the
natural carrier: `sub = token_id`.

### 4. What must not change

The target's pick. Verification stays an exact match against the target's own sample, so
the generated text is a function of the seed alone, exactly as it is today. The draft
only changes how many tokens one window yields. The existing determinism guarantee
(every configuration reproduces its own output exactly) has to survive the change: a
different draft count is already a different output, and coupling must not make the same
draft count produce two different outputs across runs.

## Measuring it

The switch should be a **per-request field** (like `dry_multiplier`), not a process flag: the
draft path is graph-captured, but the request's sampler parameters are refreshed into the
device at the start of every round, so one loaded model can serve both arms and the
interleaved A/B needs no reload.

The router's request log already carries the numbers:

- `speculative.accepted_per_position` -- the acceptance per draft position, and
  `speculative.accepted_tokens / drafted_tokens` for the rate.
- `timings_seconds.decode` and `throughput_tokens_per_second.decode` for the speed.
- the output hash, to prove the text did not change.

A measurement is a pair of runs with the same seed and the same prompt, one with the
argmax draft and one with the coupled draft, over at least twelve interleaved pairs
(the acceptance rate moves by a few points, and one run shows nothing).

## Upstream

Two pull requests, in this order:

1. *Coupled draft sampling* -- the draft samples with the target's chain and the shared
   uniform. It needs no new sampler primitive, only the draft's pick.
2. *Gumbel-max pick* -- the per-token noise, on both sides, as an option.

Both are opt-in flags, off by default, with the greedy path untouched.
