# Third-party engines: what was measured elsewhere, and what it changes here

This reference records measurements taken in other engines that bear on NInfer's design, so that a
proposal here can be answered with a number instead of an argument. It is a reference, not a
backlog: the plan built on it is tracked with the work that carries it.

The engines compared here share this repository's subject. [Strata](https://github.com/Niko1221/Strata)
is a native implementation of `Qwen3.8-Flash-Next` with its memory tiers (VRAM expert cache / pinned
host / mapped SSD) and its speculation stack. [Project Maya](https://github.com/mw00/project-maya)
(MIT) is a fork of Strata whose engine was rewritten for GLM-5.3-Flash and DeepSeek-V4.1-Flash-1017
and whose server and dashboard grew from Strata's. This repository implements the same model family
in `src/models/qwen4_exp/`, independently of both.

Maya's code lives under the `strata` namespace and credits Strata, ggml/llama.cpp, exllamav3 and
beellama.cpp (`README.md:255-267`). Its `docs/GLM5-FLASH.md:7-13` states that about two thirds of
Strata's machinery applied to GLM unchanged.

## What NInfer already holds that these engines do not

Before the transfers below, the parts of the design where this repository is ahead, so that they are
not "improved" by copying:

- **Numerical qualification.** Every floating-point Op against an independent FP64 oracle; the GGUF
  CPU expert products reproduce the GPU's decode products bit for bit
  (`docs/qwen3-8-flash-next.md`, "The CPU share"); perplexity agreeing with llama.cpp window by
  window to ±0.0004 nats over five 65,536-token windows.
- **The artifact contract.** The v3 container, its framing, codecs and binding, and the Op
  development contract ([Op development](op-development.md)).
- **Device route profiles.** 47 GPU models measured, with calibration for a card that has none and a
  digest that refuses a stale entry ([Device profiles](../device-profiles.md)).
- **The serving surface.** OpenAI chat/completions/responses, Anthropic messages, the llama.cpp
  completion and rerank routes, a model registry with load/unload/sleep, slot files, structured
  output, token log probabilities and an MCP proxy.
- **The n-gram subsystem.** A hot-row profile counted from a corpus, `ram-hot` residency, io_uring
  and IOCP readers, a bounded CLOCK row cache and demand-plus-hint reads with
  `POSIX_FADV_WILLNEED` (`src/models/qwen4_exp/ngram_table.cpp:250-254`).
- **The hybrid prefix cache.** Content-addressed KV blocks and sparse state snapshots over a device
  and a pinned host slab pool, with GDSF eviction ([Hybrid prefix cache](hybrid-prefix-cache-spec.md)).
- **The stage solver.** `solve_stage_plan` (`src/core/stage_plan.h:240`) balances the stages
  exactly and is verified against a brute-force partition oracle.

## Measurements that closed proposals

Maya records what it tried and discarded in `bench/results/PROGRESS.md`, under "Tried and measured,
not kept". Three of its results are the reason two plausible proposals are not pursued here.

**Prefetching experts by predicting the next layer's routing.** Measured three times, and it lost
three times:

- *"Reading predicted disk experts ahead: exact but slower — only ~45 % of disk misses are
  predictable a layer or two ahead, and wrong reads compete for the one NVMe."*
- *"Prefetching the next layer's predicted experts into VRAM: exact, 7 % slower on Uranus."*
- *"Next-layer prefetch again, now that the CPU lane leaves PCIe idle: still a loss (52.4 → 65.7
  ms/token on a fixed text) — the VRAM hit rises 72 → 77 % but the wrong guesses cost more than the
  right ones save."*

The engine keeps the switch off for that reason: `STRATA_GLM_PREFETCH_N` defaults to 0 with the note
*"measured a net LOSS on Mercury (the window between routes is shorter than one fetch)"*
(`src/core/glm_fast_state.hpp:368`).

**Keeping a prompt's disk reads in host memory for the answer.** *"More disk reads, not fewer (a
long prompt touches every expert)."*

**A smarter eviction policy.** *"LRU 67 %, LFU (ours) 77.5 %, decayed-frequency 78 %, a small learned
model 77 %; the theoretical optimum (knowing the future) is 87 % — the gap needs the future text,
not a better history model."* On another machine the same replay reached *"LRU = LFU (97.7 / 96.4 %
per GPU) and even Belady's optimum only reaches 98.6 / 98.0 % — the misses are capacity (30 GB
RAM), not policy."*

The consequence is a standing rule for the expert cache: a policy change is worth a few points at
most, and the misses are a capacity problem. Measure before changing the policy, and prefer the
capacity lever.

## What transfers: instruments, not features

**An offline tier replay.** `tools/glm_tier_replay.py` replays recorded routes
(`STRATA_GLM_ROUTE_LOG`) through a model of the engine's own tier rules and compares policies:
`evict: lfu | lru | slru | slfu | protect<N> | reuse | soft | nlru | belady`, plus `pick:` and
`promote:`. Belady is the ceiling. The tool validates itself first: replay a run with the prefetch
off and compare the replay's hits with the live run's own masks, which agree within a few points
under the engine's policy. This turns "would a different eviction policy help?" into a question with
an answer before any engine change is written.

**A warm-start cache simulator.** `tools/glm_cache_sim.py` compares a single pool against two pools
split at a layer boundary, cold against warm, with statically pinned experts on top of an LFU and a
host tier demoting from VRAM. Its metric is steady-state NVMe reads and megabytes per token on a
held-out trace set.

**A calibration discipline.** `tools/calibrate_glm.py` and `tools/calibrate_draft.py` measure on the
machine that will run the engine:

- One running engine for every setting, because *"an engine start reads the whole expert set from
  disk (minutes)"*.
- Sixty fresh prompts in three families (code, explain, lists), and a new group of three answered per
  measurement, because *"the expert tiers follow the text they serve: three prompts answered over
  and over left them holding just those answers' experts"*.
- The engine runs against a copy of the usage file with the writes stripped, so its own answers do
  not leak into the next warm-up.
- A setting is kept only when it beats the engine's own by more than 3 % in an interleaved
  re-measurement, and is then confirmed against the incumbent in three interleaved rounds.
- For a drafter: a warm-up that is not counted, at least 1,536 tokens written, a 5 % margin, and a
  tie that keeps the model's own MTP block.

**Quantization metrics per release.** Maya publishes, against the FP8 model: the KL over the
reference's top 64 plus one bucket, the share of positions where the argument maximum agrees, the
top-1 next-token share, and the zero-shot recovery as the quant's score over the FP8 model's, over
five tasks of 400 questions each (ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, PIQA) scored as
lm-evaluation-harness does. Its releases report 0.428 / 83.3 % / 68.8 % / 97.9 % at 96.5 GB and
0.188 / 90.0 % / 71.2 % / 99.2 % at 156.3 GB, with the worst domains the tool calls (KL 0.850) and
wikitext-2 (0.934). This repository measures loss instead — perplexity against llama.cpp and
against BF16 — which is finer, and does not publish agreement or task retention per release.

**A build guard for a compatibility layer.** `tools/check_hip_compat.py` scans every source file for
CUDA runtime names and fails the build when one has no mapping in the compatibility headers, on CI
and without a GPU. It caught three unmapped device queries that broke every AMD build.

## The route that survives: prediction for policy, not for bytes

The two measurements above are not the same thing, and the difference decides where a prediction is
worth using.

| Question | Maya's measurement |
|---|---|
| Which experts will the next layer route to? | the next layer uses ranks 0/1/2/3 in **96 / 88 / 75 / 62 %** of cases; ranks 4–7 only 52 → 24 % |
| Which of this layer's misses will it take from disk? | only **~45 %** are predictable one or two layers ahead |

Predicting the routing is easy. Predicting the misses is hard, because a miss depends on what the
cache already holds, which depends on the past. That is why the byte prefetch fails and a policy
does not.

Maya's lookahead carries the next layers' top-k in the route and has the service thread queue reads
for the predicted experts into free host slots, with reader threads of its own
(`STRATA_GLM_AHEAD`, `src/core/glm_fast_path.cu:3618-3627`). It also ships the telemetry that
decides the question: experts predicted per depth, disk misses named per depth, reads issued, used
and waited on, slots stolen, queue full, no free slot.

Here, the mechanics for a predictive admission policy already exist. `ops::moe_route` computes the
softmax over **all** experts into an FP32 workspace and exports only the top ten
(`src/ops/moe_route/moe_route.cu:258`); keeping a wider slice gives the ranks below the top ten.
The expert cache's `rebalance()` chooses its candidates by a decayed score — history only
(`src/models/qwen4_exp/expert_cache.cpp:133-219`) — so a predictive policy has somewhere to land.

Instrument before implementing. The instrumentation above is what turned Maya's prefetch from an
assumption into a measurement.

## The staged path of Flash-Next

The staged miss path is where the transfer above has the most to say, because it is the path that
moves experts on every decode.

How it runs:

- `src/models/qwen4_exp/executor.cpp:1544-1546` calls `cache->hand_over(i)` for each text layer
  (the MTP block is outside it) and marks the layer as admitting. The guard `!stream` means this is
  the **host** path: the disk path has its own, `ExpertStream::prepare()`
  (`src/models/qwen4_exp/expert_stream.h`), which reads on demand and has no admission at all.
- `src/models/qwen4_exp/expert_misses.cu` `serve_request` calls `ExpertCache::admit(...)`; on a
  success the copy engine fills the slot and the table is patched to it, and the service thread
  admits the experts **behind** their call (`admit_behind`, `expert_misses.cu:707`).

How the victim is chosen (`src/models/qwen4_exp/expert_cache.cpp:300-353`): a free slot first; among
the experts the call does not use, the one of the **lowest frequency tier** (0 below 1.5, 1 below
6.0, 2 at or above 6.0) and, within a tier, the one stamped least recently.

Two things follow.

**The staged path has no promotion threshold.** `rebalance()`, the profile-seeded route, has one:
`score[candidate] > 1.25 * score[victim] + 0.5` (`expert_cache.cpp:24-25`). `admit()`, the
miss-driven route of the staged host path, does not: an expert routed once has score 0, tier 0, and
displaces any other tier-0 expert. Maya's equivalent guard is a promotion only when
`hot >= 2 * cold + 4`, and a host tier that keeps an expert whose count has not caught up for 64
routes × 42 layers = 2,688 ("*a newly hot expert keeps its place while its count catches up*",
measured at 7–11 disk reads a token without it).

**Recency wins the decode; frequency wins the hit rate.** Maya's own policy makes the same split by
the number of conversations: one conversation uses plain LRU, several use aged counts with recency as
the tie-break (`src/core/glm_fast_path.cu:4771-4784`). On a 3090 + 3060 split the aged counts raised
the smaller half's VRAM hits from 55.7 % to 65.2 % (off-card experts a token 57.7 → 45.4 on the 3060,
71.8 → 64.6 on the 3090), while **recency won the decode there (+2.6 %)**; over 3× V100-16 GB with
an NVMe tier, recency moved Maya-S from 24.8 to 26.1 tokens/s. `admit()` here orders the tiers
first and recency second, and `admit()` is the decode path.

The margin is bounded by the ceiling above: Belady reaches 87 % where LFU reaches 77.5 %, so the
victim rule is worth a few points and no more. Measure it on the card; do not assume it.

Two facts about the path that constrain any change:

- `--expert-tail-skip` exists only for staged GGUF host experts (`executor.cpp:739-757`); the disk
  path has no lossy skip.
- The wide calls already prefetch: `kPrefetchTokens = 256` copies the layer's whole bank on the
  transfer stream while the earlier layers run (`executor.cpp:637-640`). What the wide path has and
  the narrow path does not is that overlap.

## Not carried over

- **An explicit host tier of experts.** The host residency mode already is that tier (the whole
  bank, page-locked), and disk residency's middle tier is the operating system's page cache. What
  disk residency lacks is the hint that makes the page cache effective, not the tier.
- **Tensor parallelism.** Maya has it for its DeepSeek model. The transport probe, the projected
  payoff and the prerequisites are recorded in [Pipeline stages](pipeline-stages.md).
- **Other platforms.** Maya's AMD, Windows, Intel and macOS support is a product of its own.
- **An environment-variable surface.** Maya keeps roughly ninety `STRATA_*` and `MAYA_*` variables
  as a debug surface. This repository's contract is the executable `--help`.

## Where the rest of the analysis lives

The full comparison, the negative results with their sources, the calibration and quantization
methodology, and the prioritized plan are in the shared knowledge base under
`comparisons/project-maya-vs-ninfer.md`.
