# Accepted geometries and formats

Six places enumerate, by literal, what they accept. Five are predicates; the
sixth is compiled. Everything else in the engine derives its widths from the
artifact's own `TextConfig` — the frontend included — so these are where a model
outside the registered set stops. This file records what each admits, where the
enumeration lives, and where the boundary between a predicate and a kernel sits.
It is the reference to read before changing any of them, and before concluding
that a model is unsupported.

Measured on 2026-10-07 by walking Qwen3.5-9B (hidden 4096, 32 layers, 16/4 heads)
from a working conversion down to the first failure. The first five were admitted
with 29 lines across four files; the sixth needed the width turned from a constant
into a parameter, which was a larger change than it looked because the split-K
translation unit held it in template position across five launchers and three
function-pointer tables.

| # | validator | file | admits | nature |
|---|---|---|---|---|
| 1 | `input_projection` | `src/ops/weight_input.cpp` | two geometries | predicate |
| 2 | `prepare_gdn_gating_proj_weights` | `src/ops/weight_input.cpp` | two A/B shapes | predicate |
| 3 | `require_causal_geometry` | `.../dense/causal_cache/causal_softmax_attention.cpp` | two head pairs | predicate over kernels |
| 4 | `bf16_gdn_gating_admits` | `.../bf16_gdn_gating_proj_plan.cpp` | `is_27` \|\| `is_35` | predicate over a schedule table |
| 5 | `gdn_input_proj_workspace_capacity_bytes` | `src/ops/wrapper/gdn_input_proj.cpp` | one Q8 profile | predicate |
| 6 | `q8_gdn_input_admits` | `.../q8/q8_gdn_input_plan.cpp` | every instantiated width | **was a compiled `kHidden`; now a parameter** |

## `input_projection` — attention and GDN input projections

`src/ops/weight_input.cpp`, `input_projection(std::span<const WeightInput, 4>, bool)`.

| accepted | hidden | attention q / k | GDN q / value | model |
|---|---:|---|---|---|
| `dense` | 5120 | 6144 / 1024 | 2048 / 6144 | Qwen3.5-27B |
| `moe` | 2048 | 4096 / 512 | 2048 / 4096 | Qwen3.6-35B-A3B |
| `qwen9` | 4096 | 4096 / 1024 | 2048 / 4096 | Qwen3.5-9B |
| `qwen4` | 2560 | 4096 / 1024 | 2048 / 4096 | Qwen3.5-4B |

The check is `require(admitted, "input projection: unsupported logical projection
geometry")`, so a Qwen3.5 model at any other hidden width is refused before its
formats are examined. **The shape patterns are not what varies** — the third and
fourth matrices equal the first two respectively in every case, and only `K`
differs. The ratios do not generalise: attention `q / k` is 6 on the 5120 model, 8
on the 2048 one, and 4 on both dense Qwen3.5 models, which therefore share one
pattern and differ only in hidden width.

The GGUF branch below that check asks for `admitted`, not for `dense`. It is
geometry-agnostic by construction — it lays the parts out from the row counts it
reads off the matrices — so the only thing it needs is a geometry that was already
accepted above.

Reached from `prepare_gdn_input_proj_weights`, so a failure surfaces as
`startup failed | initializing frontend` with no message on the phase line;
`gdb -ex 'catch throw' -ex bt` names `input_projection` directly.

**The binding is not part of this problem.**
`src/models/qwen3_5/load/text.cpp` derives every width from `TextConfig`:
`bind_attention`, `bind_dense` and `bind_gdn` read `hidden_size`,
`query_width`, `key_width` and `intermediate_size` from the config the artifact
carries. `qwen3_5.Builder` walks `config["layer_types"]` and selects the mixer
per layer, so depth and the full-attention/linear-attention split are already
generic. The checkpoint's fused rows are already split into the separate
logicals this engine binds, through `builder.add(..., source_shape=..., rows=...)`
over `select_rows`.

Formats, once a geometry is admitted, by branch:

| condition | admits |
|---|---|
| contiguous, `moe` | `Q8_G32_FP16` |
| contiguous, `dense` | `NVFP4`, `FP8_E4M3FN_ROW_BF16`, and `BF16` when attention |
| non-contiguous | `(Q4_G64, Q5_G64)` or `(T2_G128, T2_G128)`, `dense` only |
| any GGUF part | `dense` only, remapped by `gguf_projection` |

So the four GDN mixer projections must share one parent format on the contiguous
path. `qwen3_5_9b` carries its mixer's `query_key` at Q4 and `value_z` at Q5
because that model was converted against a frontend with an `input_columns`
auxiliary; here it is all `Q8_G32_FP16`, as `qwen3_6_35b_a3b` has it.

## `prepare_sparse_moe_weights` — routed expert profiles

`src/ops/weight_input.cpp`. Also treats the four expert matrices as one profile
rather than four choices, because the prefill route quantises the activation and
its gate/up epilogue emits the encoded intermediate `down` consumes, so a mixture
with a groupwise codec has no route at any token count.

| admitted | gate/up | down | shared |
|---|---|---|---|
| groupwise | `Q4_G64_FP16` | `Q5_G64_FP16` or `Q6_G64_FP16` | `Q8_G32_FP16` both |
| MTP | `Q8_G32_FP16` | `Q8_G32_FP16` | — |
| NVFP4 | `NVFP4` | `NVFP4` | `NVFP4` both |

The router must be `BF16`. **There is no profile with a gate or up at Q6**, and
`src/ops/sparse_moe/{decode,small_t}/…_plan.cpp` reject the pair a second time
through `sparse_moe: unsupported routed codec profile`.

The dispatch that would carry a new width is not free of charge on the prefill
side: `sparse_moe_prefill_kernels.cu` specialises gate/up per format
(`sparse_moe_prefill_q4_gate_up_kernel`, `..._q8_gate_up_kernel`), while decode
and small_t dispatch on a codec struct and `Q6Codec` is already defined and
instantiated for `down`. Adding a `(Q6, Q6)` profile is two `case` labels in the
decode and small_t switches plus one new prefill kernel.

## Formats that reach an Op without a route

`prepare_input_projection_weight` admits `Q8_G32_FP16` for the models in this
family and nothing else. On `qwen3_6_35b_a3b` that is why attention, SSM and the
input projections sit at Q8 while the routed experts do not: **the Q8 there is a
route, not a preference.**

Groupwise formats are `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`,
`q8_g32_fp16` and `t2_g128_fp16`; block-scaled is `nvfp4`; row-scaled is
`fp8_e4m3fn_row_bf16`. [Tensor formats](tensor-formats.md) defines each and
notes that layout and Op support are separate questions.

## Searched and not found upstream

Across the 27 GitHub forks carrying this code, as of 2026-10-06:

- the `input_projection` geometry literals appear in six forks and **all six are
  identical**; the differing blob shas come from other work in the same file
  (KVMem ring, Windows paths, a fusion engine tree);
- `sparse_moe: unsupported routed codec profile` appears under seven distinct
  shas and **all seven admit the same pairs**, differing only in enum spelling
  from an earlier generation;
- `gguf_blocks_moe.py` and `moe_experts_gguf` have **no hits outside this
  repository**. The dense `gguf_blocks.py` is upstream.

One fork does resolve the first validator by structure rather than by widening
it: `usertomlin/ninfer-5090-laptop` has no `src/ops/weight_input.cpp`. It has
`attn_input_proj/`, `gdn_input_proj/`, `gdn_gating_proj/` and `linear_pair/`,
each holding one subdirectory per format family, and no geometry literal
remains in the projection path. It runs the 9B-class models because the check is
gone rather than loosened. Porting that is a larger change than deriving the
widths from the config, and it also opens the mixed `(Q4, Q5)` GDN mixer that the
contiguous path refuses; it is not compatible with this fork's merge history or
its sm_86 tuning.

## The GDN control and the head geometry

`prepare_gdn_gating_proj_weights` takes the A/B pair and admits
`{48, 5120}`, `{32, 2048}`, `{32, 4096}` or `{32, 2560}`. The first is the
5120-hidden model's 48 control rows, the other three the three 32-control-row
models at their own input widths. The wrapper looks these up in
`kGdnControlGeometries` in `src/ops/wrapper/gdn_gating_proj.cpp`, and the pair's
row count is twice the control-row count — which is why 64 recurs across three
rows rather than being a coincidence to notice.

The Op side keeps the same widths in `is_27` / `is_qwen9` / `is_qwen4` and in
`kGdnControlGeometries` in
`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp`; the two are separate
tables because the wrapper validates what the artifact bound and the Op validates
what it compiled.

`require_causal_geometry` admits `(24 query, 4 kv)`, `(16 query, 2 kv)` or
`(16 query, 4 kv)` at `kHeadDim`. Note the ratio is not constant across them —
6, 8 and 4 — so a width derived from the other members of a group is not
sufficient; the pair has to be enumerated.

The `(16, 4)` pair is what both dense Qwen3.5 artifacts carry
(`num_key_value_heads = 4` on the 4B GGUF and the 9B). It was admitted by the
predicate but had no kernel: the small-T and prompt dispatchers fell through to
`CausalD256H16Kv2`, whose kernel uses the compile-time 2 KV heads as the input
token stride. Token 0 is unaffected (stride times zero), so single-column decode
looked right while every later column of a multi-column speculative verify read
the wrong K/V planes. `CausalD256H16Kv4` and its dispatch branches in
`small_t.cu`, `prompt.cu` and the fp8/nvfp4/k8v4 launchers fix it. A geometry
that the predicate admits but no dispatcher names is a silent wrong-kernel bug,
not an unsupported model: enumerate the pair in a `switch`/`if` chain rather
than only in the predicate.

## `bf16_gdn_gating_admits` and the schedule table

`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp` selects among
`MmaCooperativeSplit{32,16,8,4,2}`, `MmaUnsplit`, `GemvPairedRows`,
`SmallTSplit10` and `SimtWarpRowC{4,8}`. The MMA tile width is
`is_35(problem) ? 64 : 128`, so a new geometry lands on the 128 path unless it
is added to `is_35` deliberately. Which schedule wins also depends on `sm_count`,
because a cooperative schedule needs its whole grid resident — so a width added
here is re-measured per device rather than inherited.

## `q8_gdn_input_admits` — the width is a parameter now, not a constant

`src/ops/gdn_input_proj/q8/q8_gdn_input_plan.cpp` admits a shape through
`supported_shape`, which took an exact `input_rows == 2048` and
`padded_k == 2048`. It now admits every width the decode kernel is instantiated
for and requires `padded_k == input_rows`. The `kRoutes` table selects on the
column count, never on `K`, so admitting a width adds no route.

The schedules were named for the width they were written against
(`gdn_input_proj.q8.decode.r8.direct.k2048.split2` and siblings) and the kernels
matched them (`q8_k2048_decode.cuh`, `q8_k2048_decode_kernel<…>`, plus a
file-scope `constexpr int kHidden = 2048` in both `q8_gdn_input_gemm_mma.cu` and
`q8_gdn_input_gemm_splitk.cu`). **That is gone.** The decode kernel is
`q8_decode_kernel<Rows, K, RowsPerCta, …>` in `ops/linear/q8/q8_decode.cuh`, where
`K` is a template parameter and every quantity the kernel derives from it —
`kGroupsPerRow`, `kValuesPerPhase`, `kPhases` — follows; it reads global memory and
has no shared-memory or register footprint that depends on the width. The MMA
launcher passes `x.ne[0]` as the runtime argument the kernel already took. The
split-K translation unit carries `Hidden` through its kernel, its five launchers and
its three function-pointer tables, one table per width in `SplitKLaunchers<Hidden>`,
selected by a runtime `switch` that rejects any other width by name.

`q8_gdn_input_schedule_name` no longer carries a width: the schedule is the same
one for every admitted width, so a name naming one would be wrong for the other.
The width belongs to the problem, not the schedule.

**Two paths reach the same op.** `unified::q8_gdn_input_decode_launch` and its
siblings pass the width as the `StaticK` argument of `Q8A16GemvSchedule`, and
`q8_a16_gemv_kernel` falls back to `operands.k` when `StaticK` is zero — so the
unified path was always width-generic. It is not the path that runs on this fork:
`device_profiles.json` gives the sm_86 profile no `unified/q8_gdn_input` key, so
`fused_route_table` resolves to `Legacy` and the legacy route is the live one.
Check which path a device actually takes before concluding a kernel is dead; the
RTX 5090 entry has the key and the RTX 3090 entry does not.

`catalog_is_closed()` guards the *column* domain — the ranges must tile
`[1, kAnyCols]` contiguously — not `K`, so a new width does not disturb it.

A new width is therefore a new instantiation and a new entry in each dispatch
switch, plus a qualification against the op oracle: `tests/ops/test_gdn_input_proj.cpp`
runs every admitted width against `projection_oracle` in FP64.

That is the same wall a fork adding Qwen3.5-0.8B describes: its record of the
work lists new projection kernels selecting GEMV, small-batch SIMT or MMA from
the matrix shape and token count, wider prefill MMA tiles with matrix-geometry
and column-tail selection, and a normalized-control op qualified for specific
input widths, heads and token counts.

## `linear_swiglu` — the compiled profile table

`src/ops/wrapper/linear_swiglu.cpp`, `kSwiGluShapes`. A profile is named by the
pair (hidden width, FFN intermediate); the gate/up parent is twice the intermediate
and reduces over the hidden width, so the pair fixes every operand's shape. The
registered rows are `(5120, 17408)`, `(2048, 6144)` and `(4096, 12288)`. A weight
outside every row is rejected rather than executed on an uncompiled shape.

Two flags ride on the row because they are bound to one model's release rather than
to the shape, and a new row states explicitly whether they apply rather than
inheriting them:

- `native_formats` — that release also carries NVFP4, FP8 and T2 weights.
- `a8_int_decode` — the A8 integer decode mainloop, measured at one width.

They are separate because reusing one flag for both would claim the i8 route exists
because the release has FP8 weights, which is a different fact.

`linear_swiglu_compiles_a8_int_decode` is how a caller asks instead of naming a
width. `parameters.cpp` used to test `n == 34816 && k == 5120` to pick the
verify-phase policy, which meant the next profile added here would silently lose the
decode route. `tests/models/qwen3_5/test_mlp_a8_decode_wiring.cpp` covers it, and
that test used to restate the same comparison — a mirror that would have kept
passing once the source stopped doing it.

A GGUF-sourced FFN does not pass through here: `linear_swiglu` returns to
`detail::gguf_swiglu` before the table is consulted, which is why a model converted
from GGUF needs no row. A model converted from safetensors does need one.

## Fixed: Qwen3.5-9B's ordinary decode path (Q4 LinearSwiGLU GEMV)

A Qwen3.5-9B artifact built from a safetensors recipe started, generated with
`--spec mtp`, and failed its warmup without it: the target returned a token outside
the public token domain, which is what a non-finite logit set produces.

It was two file-level constants in `src/ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu`
fixed at the 5120-hidden model's values and reached from the decode path's T=1 GEMV:
`kIntermediate` (17408, used as the gate-to-up row offset and the launch grid) and
`kTiles`, which the kernel body read instead of its own `Tiles` template parameter.
Both are now derived from the template's `GateUpRows`.

**This section replaces an earlier claim that native-quant formats were at fault.**
That was wrong: a control across six artifacts showed the 5120-hidden model's native
formats passing the same warmup, so the format was not the discriminator. What
separated the failing artifact from the passing one was the *shape* -- the 9B's
`{24576, 4096, 12288}` was the only Q4 LinearSwiGLU profile in use that the Op's
numerical test did not cover, and the constants that disagreed with it were the
5120-hidden model's. The GGUF 9B passed because a GGUF-sourced FFN never reaches
this kernel, not because GGUF is correct.

The reproduction is now the Op oracle rather than a model:
`tests/ops/linear_swiglu/test_q4_a16.cpp` runs both registered Q4 profiles.

Two diagnostics that misled first, recorded so they are not retried: the
linear-route family (`NINFER_LINEAR_ROUTES=unified` reports the same fault) and the
activation-compute policy (all five gate_up policy flags report it). And
`compute-sanitizer --tool memcheck` reports zero errors on the failing case, because
the wrong bytes read are inside the allocation -- an in-bounds read of the wrong
region is not a memory error.

Still open: the routed candidate key in `q4_linear_swiglu_plan.cpp` is the literal
`"q4_linear_swiglu/34816x17408x5120"`, so every profile consults the 5120-hidden
model's device route table.

## Related

- [Engine architecture](engine-architecture.md) — ownership between Models, Ops
  and artifact.
- [Op development](op-development.md) — the qualification contract for Op work.
- `docs/weight-conversion.md` — the recipes, and which formats each assigns.
