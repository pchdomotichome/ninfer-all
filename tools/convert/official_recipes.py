"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from .methods import (
    cast_direct,
    fp8_row_maxabs,
    grouped_absmax,
    grouped_search,
    import_encoded,
)
from .sources.compressed_tensors import compressed_matrix_source

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None, method=grouped_absmax):
    method = method if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


def _optional(model, recipe, *, method=grouped_absmax):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(
                ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
            ):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format, method=method)
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8, method=method)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dense_groupwise(model, recipe, vocabulary, gate_up=Q4, *, method=grouped_absmax):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe, method=method)
    _assign(recipe, "text/token_embedding", vocabulary, method=method)
    _assign(recipe, "text/output_head", vocabulary, method=method)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(("/mlp/gate", "/mlp/up")):
            format = gate_up
        elif name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format, method=method)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_8_27b_q6(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8, gate_up=Q6)


def qwen3_8_27b_imatrix(model, recipe, sources):
    """Qwen3.8-27B sized for one 24 GB card: ``grouped_search`` with signed scales over every
    text matrix, weighted by ``--source imatrix=PATH`` (a file from ``tools.convert.imatrix``);
    a Q4 embedding, a Q6 head, and Q4 mixer outputs and MLP down in layers 36-63. Vision, MTP
    and draft components keep the ``grouped_absmax`` formats of ``_optional``.

    The Q4 MLP down (5120x17408) needs the Q4 ``linear_add`` route of that shape."""

    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    search = {
        "method": grouped_search,
        "parameters": {"imatrix": str(sources["imatrix"].path), "negative_scales": True},
    }
    recipe.assign("text/token_embedding", format=Q4, **search)
    recipe.assign("text/output_head", format=Q6, **search)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        layer = int(name.split("/")[2])
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        elif name.endswith(("/attention/output", "/gdn/output", "/mlp/down")) and layer >= 36:
            format = Q4
        else:
            format = Q5
        recipe.assign(name, format=format, **search)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_5_9b(model, recipe, sources):
    """Qwen3.5-9B: a thirty-two layer dense hybrid, eight full-attention and twenty-four GDN.

    The smallest routed-free model in this family that still carries Vision and MTP, and the
    one that fits a 24 GB card several times over. Nothing about it needs a different binding:
    `bind_attention`, `bind_dense` and `bind_gdn` read every width from the artifact's own
    `TextConfig`, and `qwen3_5.Builder` walks `layer_types` and already slices the checkpoint's
    fused `in_proj_qkv` and `qkv` rows into the separate logicals this engine binds. So the
    recipe is formats only.

    The widths follow the artifact inventory of that model: 4096 hidden, a 12288 MLP
    intermediate, sixteen query heads over a 256 head dim, four key-value heads, 248320
    vocabulary rows, and full attention on layers 3, 7, 11, ... 31. The inventory's fused
    objects are what set the split below -- `query_key` at Q4 and `gate_value` at Q5 for
    attention, `query_key` at Q4 and `value_z` at Q5 for GDN, `gate_up` at Q4 -- so the
    separate logicals each inherit their parent's width.

    `token_embedding` and `output_head` sit at Q6 rather than the Q8 the 27B recipes use. Both
    endpoints are 248320 by 4096, which is most of what a Q8 pair costs on a model this small.

    The GDN mixer is the one place this recipe cannot follow the inventory. That inventory
    grades `query_key` at Q4 and `value_z` at Q5, but it does so against a frontend that
    splits the mixer projection with an `input_columns` auxiliary. Here the four mixer
    projections reach one op together -- `prepare_gdn_input_proj_weights` hands them to
    `input_projection` as a `std::span<WeightInput const, 4>` and that validates a single
    parent format -- so a Q4/Q5 mix throws "input projection: unsupported single-parent
    format" at load. They all sit at Q8_G32, which is what `qwen3_6_35b_a3b` does for the
    same tensors and packs them into one object.
    """
    if "num_experts" in model.config:
        raise ValueError("this recipe requires the Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q6)
    _assign(recipe, "text/output_head", Q6)
    mixer = ("/gdn/query", "/gdn/key", "/gdn/value", "/gdn/z")
    q4 = ("/attention/query", "/attention/key", "/mlp/gate", "/mlp/up")
    q5 = ("/attention/gate", "/attention/value", "/attention/output",
          "/gdn/output", "/mlp/down")
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            continue
        if name.endswith(mixer):
            _assign(recipe, name, Q8)
        elif name.endswith(q4):
            _assign(recipe, name, Q4)
        elif name.endswith(q5):
            _assign(recipe, name, Q5)
        else:
            raise ValueError(f"{name}: qwen3_5_9b has no format for this projection")
    return


def qwen3_6_35b_a3b_apex(model, recipe, sources):
    """`qwen3_6_35b_a3b` with APEX's edge widened across the routed experts.

    APEX (https://github.com/localai-org/apex-quant) measures that a uniform width
    over the routed experts leaves quality on the table: the edge layers align the
    embedding and build the output logits, the middle layers are largely redundant,
    and grading between them matches Q8_0 quality where a uniform Q5_K does not. On
    a forty-layer routed MoE it grades the experts Q6_K on the ten edge layers,
    Q5_K on the ten next to them and Q4_K through the middle.

    One bit of that is expressible. `qwen3_6_35b_a3b` already lifts the expert `down`
    to Q6, but only at layers 34, 38 and 39; this recipe gives the ten edge layers
    that width and takes the ad-hoc exceptions with it. Everything else in APEX's
    gradient is pinned by the engine rather than chosen:

    - `gate` and `up` cannot be graded. `prepare_sparse_moe_weights` treats the four
      expert matrices as one profile, because the prefill route quantises the
      activation and its gate/up epilogue emits the encoded intermediate `down`
      consumes, so only (Q4, Q5) and (Q4, Q6) exist and a gate at Q6 has no route at
      any token count.
    - The shared expert cannot move off Q8; the same check requires it at Q8_G32.
    - Attention, SSM and the input projections cannot move off Q8 either.
      `prepare_input_projection_weight` admits Q8_G32 for this model and nothing
      else, so the Q8 there in `qwen3_6_35b_a3b` is a route, not a preference.

    Adding a (Q6, Q6) expert profile is what would recover the `gate` and `up` half,
    and that is an engine change rather than a recipe one.

    The boundary is APEX's own on a forty-layer routed MoE of 256 experts with 8
    active per token, which is this model, derived from `num_hidden_layers` at an
    eighth of the depth. APEX reports the ratios as architecture specific and
    re-measures per model, so treat the boundary as a starting point.

    Reference: https://github.com/localai-org/apex-quant
    """
    if "num_experts" not in model.config:
        raise ValueError("this recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    layers = model.config["num_hidden_layers"]
    edge = max(1, layers // 8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            rank = min(layer, layers - 1 - layer)
            format = (Q6 if rank < edge else Q5) if name.endswith("/down") else Q4
        else:
            format = Q8
        _assign(recipe, name, format)


def _flash_next_expert_values(parameter):
    """Keep fused HF expert banks from treating compressed codes as unscaled values."""
    import torch

    from .sources.logical import LogicalSource

    source = parameter.source

    def read(begin, end):
        values = source.values(begin, end)
        if values.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            raise ValueError(f"{source.label}: HF expert banks require decoded floating-point "
                             "values; fused FP8 source scales are not supported yet")
        return values

    return LogicalSource(source.shape, source.label, read)


def flash_next_mtp_formats(model, recipe):
    """The HF MTP block: Q4 routed experts, BF16 dense and shared projections."""
    formats = {}
    for name, parameter in model.parameters.items():
        if not name.startswith("mtp/"):
            continue
        format = Q4 if "/moe/experts/" in name else parameter.direct_format
        source = _flash_next_expert_values(parameter) if format == Q4 else None
        _assign(recipe, name, format, source=source)
        if format == Q4:
            recipe.assign(name, activation_policy="AllowA8")
        formats[name] = format
    return formats


def qwen3_8_flash_next(model, recipe, sources):
    """HF BF16/block-FP8 quality candidate: Q4 gate/up, Q5 down; no sensitive-layer mask.

    Dense GDN/QSA and shared experts use Q8, embedding Q8 and head Q6. HC, router,
    indexer, PLE, vision and small GDN gates retain their direct representation.
    The table comes from sharded HF BF16 rows (written as FP8) or an IQ4_NL GGUF.
    """
    from .qwen4_exp_ngram import configure_ngram
    from .sources.safetensors import SafetensorsSource

    if "text" in model.components and model.config.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("qwen3_8_flash_next requires Qwen3.8-Flash-Next mathematics")
    if "mtp" in model.components and "mtp" in sources and not isinstance(sources["mtp"], SafetensorsSource):
        raise ValueError("qwen3_8_flash_next requires an HF safetensors MTP source; "
                         "use qwen3_8_flash_next_gguf to import an MTP GGUF")
    for name, parameter in model.parameters.items():
        if not name.startswith("text/"):
            continue
        if name == "text/token_embedding":
            _assign(recipe, name, Q8)
        elif name == "text/output_head":
            _assign(recipe, name, Q6)
        elif parameter.projection:
            if "/moe/experts/" in name:
                format = Q5 if name.endswith("/down") else Q4
                source = _flash_next_expert_values(parameter)
            elif "/moe/shared/" in name or "/attention/" in name or (
                "/gdn/" in name and not name.endswith(("/a_projection", "/b_projection"))
            ):
                format = Q8
                source = None
            else:
                continue
            _assign(recipe, name, format, source=source)
            recipe.assign(name, activation_policy="AllowA8")
    flash_next_mtp_formats(model, recipe)
    if "ngram" in model.components:
        configure_ngram(model, recipe, sources["ngram"])


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


def qwen3_8_27b_nvfp4_nvidia(model, recipe, sources):
    """nvidia/Qwen3.8-27B-NVFP4 (ModelOpt AutoQuant) layout: every text MLP
    projection is NVFP4; attention and GDN projections are per-row FP8. The
    source output head is NVFP4, but the runtime registers the vocabulary
    projection only for FP8, so it is dequantised and re-quantised there."""
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        if name == "text/output_head":
            recipe.assign(
                name,
                format=FP8,
                method=fp8_row_maxabs,
                source=source,
                activation_policy="AllowA8",
            )
            continue
        format = (
            "nvfp4"
            if name.startswith("text/layers/") and "/mlp/" in name
            else FP8
        )
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


def qwen3_8_27b_nvfp4_orcarouter(model, recipe, sources):
    """orcarouter/Qwen3.8-27B-Uncensored-NVFP4 (GPTQ compressed-tensors)
    layout: MLP projections of layers 0..55 are NVFP4 packed; the last eight
    layers' MLP projections and every attention/GDN projection are per-row
    FP8; the embedding and output head remain BF16 and are re-quantised."""
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    recipe.assign("text/output_head", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if (
            not name.startswith("text/layers/")
            or not parameter.projection
        ):
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.assign(name, source=model.source(name, quantized))
            continue
        layer = int(name.split("/")[2])
        format = "nvfp4" if ("/mlp/" in name and layer < 56) else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


_MOE_EXPERT_SOURCE = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}


def _moe_encoded_source(store, parameter, name):
    """Map one routed or shared expert matrix onto its per-expert NVFP4 prefix.

    The MoE description reads routed experts out of a single fused `experts.gate_up_proj`
    tensor with an offset, so the generic factory cannot reach an encoded source for them
    and says so. The compressed-tensors checkpoint stores every expert as its own prefix,
    which is a plain matrix, so the mapping is stated here once instead.
    """
    head, _, tail = name.partition("/moe/")
    layer = head.rsplit("/", 1)[-1]
    if tail.startswith("experts/"):
        expert, _, role = tail[len("experts/") :].partition("/")
        prefix = f"mlp.experts.{expert}."
    else:
        prefix, role = "mlp.shared_expert.", tail.rsplit("/", 1)[-1]
    if role not in _MOE_EXPERT_SOURCE:
        raise ValueError(f"{name}: unknown expert matrix {role!r}")
    leaf = prefix + _MOE_EXPERT_SOURCE[role]
    for root in ("model.language_model.layers.", "model.layers."):
        prefix = f"{root}{layer}.{leaf}"
        if store.has(prefix + ".weight_packed"):
            return compressed_matrix_source(store, prefix, parameter.shape, "nvfp4")
    raise ValueError(f"{name}: no NVFP4 source for {leaf} in {store.path}")


def qwen3_6_35b_a3b_nvfp4(model, recipe, sources):
    """`qwen3_6_35b_a3b` with routed and shared experts imported as NVFP4 instead of Q4/Q5.

    Everything outside the experts keeps the representation of the groupwise recipe, so the
    two artifacts differ in exactly one mechanism and are comparable to each other.
    """
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    quantized = sources["quantized"]
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name or "/moe/shared/" in name:
            recipe.assign(
                name,
                format="nvfp4",
                method=import_encoded,
                source=_moe_encoded_source(quantized, parameter, name),
                activation_policy="AllowA4",
            )
        else:
            _assign(recipe, name, Q8)
    # The op takes the routed experts as one plane per bank, so the banks the model declares have
    # to become one parent each. Default packing keeps parents apart when their sources were
    # quantised against different divisors, which is right where the consumer reads one divisor and
    # wrong here: this plane is a stack by construction, and its kernels take the divisor from the
    # row they are reading.
    for names in model.packing_groups:
        # Text layers only: the MTP component declares the same role names and keeps its groupwise
        # Q8 experts, which this recipe does not assign and must not group.
        if all(
            name.startswith("text/layers/")
            and ("/moe/experts/" in name or "/moe/shared/" in name)
            for name in names
        ):
            recipe.group(names)


RECIPES = {
    "qwen3_8_flash_next": qwen3_8_flash_next,
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_q6": qwen3_8_27b_q6,
    "qwen3_8_27b_imatrix": qwen3_8_27b_imatrix,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_8_27b_nvfp4_nvidia": qwen3_8_27b_nvfp4_nvidia,
    "qwen3_8_27b_nvfp4_orcarouter": qwen3_8_27b_nvfp4_orcarouter,
    "qwen3_5_9b": qwen3_5_9b,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "qwen3_6_35b_a3b_apex": qwen3_6_35b_a3b_apex,
    "qwen3_6_35b_a3b_nvfp4": qwen3_6_35b_a3b_nvfp4,
}
