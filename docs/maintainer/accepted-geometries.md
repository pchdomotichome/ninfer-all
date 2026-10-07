# Accepted geometries and formats

Three validators enumerate, by literal, what they accept. Everything else in the
engine derives its widths from the artifact's own `TextConfig` — the frontend
included — so these three are where a model outside the registered set stops.
This file records what each admits, where the enumeration lives, and what the
rest of the engine already handles. It is the reference to read before changing
any of the three, and before concluding that a model is unsupported.

## `input_projection` — attention and GDN input projections

`src/ops/weight_input.cpp`, `input_projection(std::span<const WeightInput, 4>, bool)`.

| accepted | hidden | attention q / k | GDN q / value |
|---|---:|---|---|
| `dense` | 5120 | 6144 / 1024 | 2048 / 6144 |
| `moe` | 2048 | 4096 / 512 | 2048 / 4096 |

The check is `require(dense || moe, "input projection: unsupported logical
projection geometry")`, so a Qwen3.5 model at any other hidden width is refused
before its formats are examined. **The shape patterns are not what varies** — the
third and fourth matrices equal the first two respectively in every case, and
only `K` differs. The ratios do not generalise either: attention `q / k` is 6 on
the 5120 model, 8 on the 2048 one, and 4 on Qwen3.5-9B.

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

## Related

- [Engine architecture](engine-architecture.md) — ownership between Models, Ops
  and artifact.
- [Op development](op-development.md) — the qualification contract for Op work.
- `docs/weight-conversion.md` — the recipes, and which formats each assigns.
