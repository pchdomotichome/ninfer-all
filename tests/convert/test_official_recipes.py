from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.convert.methods import (
    cast_direct,
    fp8_row_maxabs,
    grouped_search,
    import_encoded,
)
from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import (
    RECIPES,
    qwen3_5_9b,
    qwen3_6_35b_a3b,
    qwen3_6_35b_a3b_apex,
    qwen3_8_27b,
    qwen3_8_27b_imatrix,
    qwen3_8_27b_nvfp4_nvidia,
    qwen3_8_27b_q6,
)
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"

LAYER = "text/layers/0"


def _dense_model() -> Model:
    model = Model({"text": {"config": {}}})
    names = (
        "text/token_embedding",
        "text/output_head",
        f"{LAYER}/gdn/query",
        f"{LAYER}/gdn/output",
        f"{LAYER}/attention/query",
        f"{LAYER}/attention/key",
        f"{LAYER}/attention/output",
        f"{LAYER}/mlp/gate",
        f"{LAYER}/mlp/up",
        f"{LAYER}/mlp/down",
    )
    for name in names:
        inputs = () if name in ("text/token_embedding", "text/output_head") else ("input",)
        source = array_source(torch.ones((4, 8), dtype=torch.bfloat16), name)
        model.add(Parameter(name, (4, 8), source, inputs=inputs))
    return model


def _formats(recipe_function) -> dict[str, set[str]]:
    model = _dense_model()
    recipe = Recipe(model)
    recipe_function(model, recipe, {})
    return {
        name: {selection.format for selection in selections}
        for name, selections in recipe.selections.items()
    }


def _single(formats: dict[str, set[str]], name: str) -> str:
    values = formats[name]
    assert len(values) == 1, f"{name} is split across formats {values}"
    return next(iter(values))


def test_q6_recipe_is_registered() -> None:
    assert RECIPES["qwen3_8_27b_q6"] is qwen3_8_27b_q6


def test_registered_recipe_gives_the_mlp_pair_q4() -> None:
    formats = _formats(qwen3_8_27b)
    assert _single(formats, f"{LAYER}/mlp/gate") == Q4
    assert _single(formats, f"{LAYER}/mlp/up") == Q4
    assert _single(formats, f"{LAYER}/mlp/down") == Q5
    assert _single(formats, "text/token_embedding") == Q8
    assert _single(formats, "text/output_head") == Q8


def test_q6_recipe_moves_only_the_mlp_pair() -> None:
    registered = _formats(qwen3_8_27b)
    q6 = _formats(qwen3_8_27b_q6)

    assert set(registered) == set(q6)
    moved = {name for name in q6 if q6[name] != registered[name]}
    assert moved == {f"{LAYER}/mlp/gate", f"{LAYER}/mlp/up"}

    for name in moved:
        assert registered[name] == {Q4}
        assert q6[name] == {Q6}


def test_q6_recipe_keeps_the_vocabulary_at_q8() -> None:
    """Q8 is 8.5 bits per weight, so the vocabulary endpoints already outrank Q6."""
    formats = _formats(qwen3_8_27b_q6)
    assert _single(formats, "text/token_embedding") == Q8
    assert _single(formats, "text/output_head") == Q8


def test_q6_recipe_leaves_the_other_projections_alone() -> None:
    formats = _formats(qwen3_8_27b_q6)
    assert _single(formats, f"{LAYER}/attention/query") == Q4
    assert _single(formats, f"{LAYER}/attention/key") == Q4
    assert _single(formats, f"{LAYER}/gdn/query") == Q4
    assert _single(formats, f"{LAYER}/attention/output") == Q5
    assert _single(formats, f"{LAYER}/gdn/output") == Q5
    assert _single(formats, f"{LAYER}/mlp/down") == Q5


def _nvidia_model() -> Model:
    model = Model({"text": {"config": {}}})
    names = (
        "text/token_embedding",
        "text/output_head",
        f"{LAYER}/gdn/a_projection",
        f"{LAYER}/gdn/b_projection",
        f"{LAYER}/gdn/query",
        f"{LAYER}/gdn/output",
        f"{LAYER}/attention/query",
        f"{LAYER}/attention/output",
        f"{LAYER}/mlp/gate",
        f"{LAYER}/mlp/up",
        f"{LAYER}/mlp/down",
    )
    for name in names:
        inputs = () if name == "text/token_embedding" else ("input",)
        source = array_source(torch.ones((4, 8), dtype=torch.bfloat16), name)
        model.add(
            Parameter(
                name,
                (4, 8),
                source,
                source_factory=lambda store, fmt, _source=source: _source,
                inputs=inputs,
            )
        )
    return model


def test_nvidia_recipe_stores_the_head_as_fp8() -> None:
    """The runtime's nvfp4 linear op has no vocabulary-shape kernel, so the
    source's NVFP4 head must be dequantised and re-quantised to FP8."""
    model = _nvidia_model()
    recipe = Recipe(model)
    qwen3_8_27b_nvfp4_nvidia(model, recipe, {"quantized": object()})

    head = recipe.selections["text/output_head"]
    assert len(head) == 1
    assert head[0].format == "fp8_e4m3fn_row_bf16"
    assert head[0].method is fp8_row_maxabs

    assert _single(_selection_formats(recipe), f"{LAYER}/mlp/gate") == "nvfp4"
    assert _single(_selection_formats(recipe), f"{LAYER}/mlp/down") == "nvfp4"
    mlp_gate = recipe.selections[f"{LAYER}/mlp/gate"][0]
    assert mlp_gate.method is import_encoded
    assert _single(_selection_formats(recipe), f"{LAYER}/attention/query") == (
        "fp8_e4m3fn_row_bf16"
    )
    assert _single(_selection_formats(recipe), f"{LAYER}/gdn/query") == (
        "fp8_e4m3fn_row_bf16"
    )


def _selection_formats(recipe: Recipe) -> dict[str, set[str]]:
    return {
        name: {selection.format for selection in selections}
        for name, selections in recipe.selections.items()
    }


# APEX grades a forty-layer routed MoE on these boundaries: its six-bit edge tier
# covers the ten edge layers. Only `down` can carry it, because the engine admits
# (Q4, Q5) and (Q4, Q6) expert profiles and no gate at Q6.
APEX_EDGE_40 = frozenset({0, 1, 2, 3, 4, 35, 36, 37, 38, 39})
APEX_MIDDLE_40 = (5, 9, 10, 29, 30, 34)

MOE_ROLES = (
    "moe/experts/0/gate",
    "moe/experts/0/up",
    "moe/experts/0/down",
    "moe/shared/gate",
    "moe/shared/down",
    "gdn/query",
    "gdn/output",
)


# Qwen3.5-9B is dense: eight full-attention layers on range(3, 32, 4) and twenty-four
# GDN ones. The inventory stores the mixer projections fused, so each pair below is one
# object in that inventory and two logicals here.
NINEB_FULL = 3
NINEB_GDN = 4

NINEB_ATTENTION = ("query", "key", "gate", "value", "output")
NINEB_MLP = ("gate", "up", "down")
NINEB_GDN_MIXER = ("query", "key", "value", "z", "output")
# 1-D tensors: the builder gives them no inputs, which is what makes `projection` false.
NINEB_DIRECT = (
    "input_norm", "post_attention_norm",
    "attention/query_norm", "attention/key_norm",
    "gdn/norm", "gdn/convolution", "gdn/a_projection", "gdn/b_projection",
)


def _dense_9b_model() -> Model:
    model = Model({"text": {"config": {"hidden_size": 4096, "vocab_size": 248320,
                                       "num_hidden_layers": 32}}})
    matrices = ["text/token_embedding", "text/output_head"]
    for layer, mixer in ((NINEB_FULL, "attention"), (NINEB_GDN, "gdn")):
        p = f"text/layers/{layer}/"
        matrices += [p + "attention/" + r for r in NINEB_ATTENTION]
        matrices += [p + "mlp/" + r for r in NINEB_MLP]
        if mixer == "gdn":
            matrices += [p + "gdn/" + r for r in NINEB_GDN_MIXER]
    for name in ["text/final_norm"] + [
        f"text/layers/{layer}/{role}"
        for layer in (NINEB_FULL, NINEB_GDN)
        for role in NINEB_DIRECT
    ]:
        vectors = (256,)
        model.add(Parameter(
            name, vectors, array_source(torch.ones(vectors, dtype=torch.bfloat16), name)
        ))
    for name in matrices:
        shape = (256, 64)
        inputs = () if name.startswith("text/") and name.count("/") == 1 else ("input",)
        model.add(Parameter(
            name, shape, array_source(torch.ones(shape, dtype=torch.bfloat16), name),
            inputs=inputs,
        ))
    return model


def _nineb_formats() -> dict[str, set[str]]:
    model = _dense_9b_model()
    recipe = Recipe(model)
    qwen3_5_9b(model, recipe, {})
    return _selection_formats(recipe)


def test_nineb_recipe_is_registered() -> None:
    assert RECIPES["qwen3_5_9b"] is qwen3_5_9b


def test_nineb_recipe_splits_the_fused_pairs_onto_one_width() -> None:
    """The inventory stores `query_key` at Q4 and `gate_value` at Q5 for attention, so
    each pair's halves match, and it puts `gate_up` at Q4 with `down` at Q5."""
    formats = _nineb_formats()
    for role in ("query", "key"):
        assert _single(formats, f"text/layers/{NINEB_FULL}/attention/{role}") == Q4
    for role in ("gate", "value"):
        assert _single(formats, f"text/layers/{NINEB_FULL}/attention/{role}") == Q5
    assert _single(formats, f"text/layers/{NINEB_FULL}/attention/output") == Q5
    assert _single(formats, f"text/layers/{NINEB_GDN}/gdn/output") == Q5
    for role in ("gate", "up"):
        assert _single(formats, f"text/layers/{NINEB_FULL}/mlp/{role}") == Q4
    assert _single(formats, f"text/layers/{NINEB_FULL}/mlp/down") == Q5


def test_nineb_recipe_holds_the_gdn_mixer_on_one_width() -> None:
    """The inventory grades GDN `query_key` at Q4 and `value_z` at Q5, but those four
    projections reach one op as a `std::span<WeightInput const, 4>` that validates a
    single parent format, so a mix fails at load. All four sit at Q8_G32, which is what
    `qwen3_6_35b_a3b` does and packs them into one stored object."""
    formats = _nineb_formats()
    for role in ("query", "key", "value", "z"):
        name = f"text/layers/{NINEB_GDN}/gdn/{role}"
        assert _single(formats, name) == Q8, f"{name} must share the mixer's width"


def test_nineb_recipe_leaves_the_vectors_bf16() -> None:
    """`a_projection` and `b_projection` are 32 by 4096 and stay BF16 in the inventory,
    as do the norms and the convolution. The vectors carry no inputs, so the projection
    walk skips them, and whatever assigns them lands on `cast_direct` at BF16 rather
    than on a groupwise codec."""
    model = _dense_9b_model()
    recipe = Recipe(model)
    qwen3_5_9b(model, recipe, {})
    for layer in (NINEB_FULL, NINEB_GDN):
        for role in NINEB_DIRECT:
            selections = recipe.selections[f"text/layers/{layer}/{role}"]
            assert [s.format for s in selections] == ["bf16"]
            assert selections[0].method is cast_direct
    for name in ("text/token_embedding", "text/output_head"):
        assert [s.format for s in recipe.selections[name]] == [Q6]
    assert [s.format for s in recipe.selections["text/final_norm"]] == ["bf16"]


def test_nineb_recipe_sits_the_vocabulary_endpoints_at_q6() -> None:
    """Both are 248320 by 4096, which is most of what a Q8 pair costs on a model this
    small; the inventory has them at Q6, unlike the 27B recipes' Q8."""
    formats = _nineb_formats()
    assert _single(formats, "text/token_embedding") == Q6
    assert _single(formats, "text/output_head") == Q6


def test_nineb_recipe_refuses_a_moe_checkpoint() -> None:
    model = _moe_model()
    with pytest.raises(ValueError, match="Dense"):
        qwen3_5_9b(model, Recipe(model), {})



def _moe_model(layers: int = 40) -> Model:
    model = Model({"text": {"config": {"num_experts": 256, "num_hidden_layers": layers}}})
    names = ["text/token_embedding", "text/output_head"]
    for layer in sorted(APEX_EDGE_40 | set(APEX_MIDDLE_40)) if layers == 40 else range(layers):
        names.extend(f"text/layers/{layer}/{role}" for role in MOE_ROLES)
    for name in names:
        inputs = () if name == "text/token_embedding" else ("input",)
        source = array_source(torch.ones((4, 8), dtype=torch.bfloat16), name)
        model.add(Parameter(name, (4, 8), source, inputs=inputs))
    return model


def _apex_formats(recipe_function) -> dict[str, set[str]]:
    model = _moe_model()
    recipe = Recipe(model)
    recipe_function(model, recipe, {})
    return _selection_formats(recipe)


def test_apex_recipe_is_registered() -> None:
    assert RECIPES["qwen3_6_35b_a3b_apex"] is qwen3_6_35b_a3b_apex


@pytest.mark.parametrize("layer", sorted(APEX_EDGE_40))
def test_apex_recipe_lifts_the_expert_down_across_the_whole_edge(layer) -> None:
    formats = _apex_formats(qwen3_6_35b_a3b_apex)
    assert _single(formats, f"text/layers/{layer}/moe/experts/0/down") == Q6


@pytest.mark.parametrize("layer", APEX_MIDDLE_40)
def test_apex_recipe_leaves_the_expert_down_at_q5_away_from_the_edge(layer) -> None:
    formats = _apex_formats(qwen3_6_35b_a3b_apex)
    assert _single(formats, f"text/layers/{layer}/moe/experts/0/down") == Q5


def test_apex_recipe_holds_gate_and_up_at_the_only_admitted_width() -> None:
    """`prepare_sparse_moe_weights` admits (Q4, Q5) and (Q4, Q6) only, so the
    gradient cannot reach gate and up. Q4 is the gate of both profiles."""
    formats = _apex_formats(qwen3_6_35b_a3b_apex)
    for layer in sorted(APEX_EDGE_40 | set(APEX_MIDDLE_40)):
        for role in ("moe/experts/0/gate", "moe/experts/0/up"):
            assert _single(formats, f"text/layers/{layer}/{role}") == Q4


def test_apex_recipe_only_uses_profiles_the_loader_admits() -> None:
    """Every layer must land on (Q4, Q5) or (Q4, Q6) with a Q8 shared expert."""
    formats = _apex_formats(qwen3_6_35b_a3b_apex)
    admitted = {(Q4, Q5), (Q4, Q6)}
    for layer in sorted(APEX_EDGE_40 | set(APEX_MIDDLE_40)):
        gate = _single(formats, f"text/layers/{layer}/moe/experts/0/gate")
        down = _single(formats, f"text/layers/{layer}/moe/experts/0/down")
        assert (gate, down) in admitted, f"layer {layer} would not bind"
        for role in ("moe/shared/gate", "moe/shared/down"):
            assert _single(formats, f"text/layers/{layer}/{role}") == Q8


def test_apex_recipe_holds_the_shared_expert_and_attention_at_their_only_route() -> None:
    """Neither can move off Q8. The groupwise check requires it on the shared
    expert, and `prepare_input_projection_weight` admits Q8_G32 for this model and
    nothing else, so the Q8 there is a route rather than a preference."""
    formats = _apex_formats(qwen3_6_35b_a3b_apex)
    for layer in sorted(APEX_EDGE_40 | set(APEX_MIDDLE_40)):
        for role in ("moe/shared/gate", "moe/shared/down", "gdn/query", "gdn/output"):
            assert _single(formats, f"text/layers/{layer}/{role}") == Q8
    assert _single(formats, "text/token_embedding") == Q8
    assert _single(formats, "text/output_head") == Q6


def test_apex_recipe_widens_the_edge_the_official_recipe_lifts() -> None:
    """`qwen3_6_35b_a3b` lifts `down` at three layers, 34, 38 and 39. This one
    covers the ten edge layers, so it has to move at least the seven it did not."""
    official = _apex_formats(qwen3_6_35b_a3b)
    apex = _apex_formats(qwen3_6_35b_a3b_apex)
    for already in (34, 38, 39):
        assert _single(official, f"text/layers/{already}/moe/experts/0/down") == Q6
    moved = [
        layer
        for layer in sorted(APEX_EDGE_40)
        if _single(official, f"text/layers/{layer}/moe/experts/0/down") == Q5
    ]
    assert moved == [0, 1, 2, 3, 4, 35, 36, 37]
    for layer in moved:
        assert _single(apex, f"text/layers/{layer}/moe/experts/0/down") == Q6
    # Layer 34 is the official recipe's third lift but APEX's near-edge, so a
    # symmetric edge boundary gives it back rather than keeping the exception.
    assert _single(apex, "text/layers/34/moe/experts/0/down") == Q5


def test_apex_recipe_needs_moe_mathematics() -> None:
    model = _dense_model()
    with pytest.raises(ValueError, match="MoE"):
        qwen3_6_35b_a3b_apex(model, Recipe(model), {})



def _imatrix_recipe() -> Recipe:
    model = Model({"text": {"config": {}}})
    names = ["text/token_embedding", "text/output_head"]
    for layer in (35, 36):
        for role in (
            "gdn/a_projection", "gdn/query", "gdn/value", "gdn/output", "attention/query",
            "attention/gate", "attention/output", "mlp/gate", "mlp/up", "mlp/down",
        ):
            names.append(f"text/layers/{layer}/{role}")
    for name in names:
        inputs = () if name == "text/token_embedding" else ("input",)
        source = array_source(torch.ones((4, 8), dtype=torch.bfloat16), name)
        model.add(Parameter(name, (4, 8), source, inputs=inputs))
    recipe = Recipe(model)
    qwen3_8_27b_imatrix(model, recipe, {"imatrix": SimpleNamespace(path=Path("im.safetensors"))})
    return recipe


def test_imatrix_recipe_searches_every_text_matrix_with_signed_scales() -> None:
    recipe = _imatrix_recipe()
    assert RECIPES["qwen3_8_27b_imatrix"] is qwen3_8_27b_imatrix
    for name, selections in recipe.selections.items():
        if name.endswith("/gdn/a_projection"):
            assert name in recipe.separate_parameters
            continue
        (selection,) = selections
        assert selection.method is grouped_search, name
        assert selection.parameters == {"imatrix": "im.safetensors", "negative_scales": True}


@pytest.mark.parametrize(
    "name, format",
    [
        ("text/token_embedding", Q4),
        ("text/output_head", Q6),
        ("text/layers/35/gdn/query", Q4),
        ("text/layers/35/gdn/value", Q5),
        ("text/layers/35/attention/gate", Q5),
        ("text/layers/35/mlp/up", Q4),
        ("text/layers/35/gdn/output", Q5),
        ("text/layers/35/attention/output", Q5),
        ("text/layers/35/mlp/down", Q5),
        ("text/layers/36/gdn/output", Q4),
        ("text/layers/36/attention/output", Q4),
        ("text/layers/36/mlp/down", Q4),
        ("text/layers/36/gdn/value", Q5),
    ],
)
def test_imatrix_recipe_formats(name, format) -> None:
    assert _single(_selection_formats(_imatrix_recipe()), name) == format


def test_imatrix_recipe_requires_the_imatrix_source() -> None:
    model = _dense_model()
    with pytest.raises(KeyError):
        qwen3_8_27b_imatrix(model, Recipe(model), {})
