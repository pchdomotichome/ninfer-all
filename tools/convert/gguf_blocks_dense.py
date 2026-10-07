"""A Qwen3.5 dense ``qwen35`` GGUF release, converted in its own block formats.

Unlike ``gguf_blocks_moe``, every dimension here comes from the file's header
rather than from module constants: the same module takes any dense Qwen3.5
release the exporter wrote, and a release whose geometry does not match what the
tensor map derives fails in :func:`validate` instead of producing a wrong
artifact.

The structural differences from the MoE release this mirrors:

* the FFN is dense (``ffn_gate``/``ffn_up``/``ffn_down``), not a routed bank;
* the full-attention layer keeps ``attn_k`` and ``attn_v`` as separate tensors
  where the MoE release fuses them into one ``attn_kv``;
* ``attn_q`` still carries the query and its output gate head-interleaved, so the
  two are separated on read exactly as the MoE release does;
* the output head is tied to ``token_embd`` in releases that carry no
  ``output.weight``, and a release that has one uses it.

The GDN block is structurally identical to the MoE model's, including the tiled
value heads, so the untiling helpers are the same with the head counts taken from
the header.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .gguf_blocks import _dequantize, block_source
from .methods import AuxiliaryValue, cast_direct, import_encoded
from .sources.gguf import GGUFFile
from .sources.logical import LogicalSource, array_source
from .ternary import RowMap, _direct, _norm, rows

# Header keys, all under the dense model's own namespace (the MoE release uses
# ``qwen35moe.*`` for the same fields).
_HIDDEN = "qwen35.embedding_length"
_LAYERS = "qwen35.block_count"
_NEXTN = "qwen35.nextn_predict_layers"
_INTERVAL = "qwen35.full_attention_interval"
_HEADS = "qwen35.attention.head_count"
_KV_HEADS = "qwen35.attention.head_count_kv"
_HEAD_DIM = "qwen35.attention.key_length"
_VALUE_LENGTH = "qwen35.attention.value_length"
_INTERMEDIATE = "qwen35.feed_forward_length"
_SSM_STATE = "qwen35.ssm.state_size"
_SSM_GROUPS = "qwen35.ssm.group_count"
_SSM_RANK = "qwen35.ssm.time_step_rank"
_SSM_INNER = "qwen35.ssm.inner_size"
_SSM_CONV = "qwen35.ssm.conv_kernel"


@dataclass(frozen=True, slots=True)
class Geometry:
    """Every dimension the tensor map needs, taken from the file's header."""

    hidden: int
    layers: int
    attention_heads: int
    head_dim: int
    kv_heads: int
    intermediate: int
    gdn_key_heads: int
    gdn_value_heads: int
    gdn_head_dim: int
    gdn_taps: int
    full_attention_interval: int
    mtp: bool

    @property
    def query_rows(self) -> int:
        return self.attention_heads * self.head_dim

    @property
    def kv_rows(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def gdn_key_dim(self) -> int:
        return self.gdn_key_heads * self.gdn_head_dim

    @property
    def gdn_value_dim(self) -> int:
        return self.gdn_value_heads * self.gdn_head_dim

    @property
    def gdn_channels(self) -> int:
        return 2 * self.gdn_key_dim + self.gdn_value_dim

    @property
    def mtp_block(self) -> int:
        """The nextn layer follows the target layers, so it is the last block."""
        return self.layers

    def full_attention(self, layer: int) -> bool:
        return (layer + 1) % self.full_attention_interval == 0


def _require(gguf: GGUFFile, key: str) -> int:
    value = gguf.kv.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{gguf.path}: {key} is {value!r}")
    return value


def geometry(gguf: GGUFFile) -> Geometry:
    """Read the model's geometry out of the header.

    ``block_count`` counts the nextn layer when the file carries one, so the
    target layers are that minus ``nextn_predict_layers``.
    """

    blocks = _require(gguf, _LAYERS)
    nextn = gguf.kv.get(_NEXTN, 0)
    if not isinstance(nextn, int) or nextn < 0:
        raise ValueError(f"{gguf.path}: {_NEXTN} is {nextn!r}")
    key_dim = _require(gguf, _SSM_GROUPS) * _require(gguf, _SSM_STATE)
    value_dim = _require(gguf, _SSM_RANK) * _require(gguf, _SSM_STATE)
    if _require(gguf, _SSM_INNER) != value_dim:
        raise ValueError(
            f"{gguf.path}: {_SSM_INNER} is not time_step_rank * state_size ({value_dim})"
        )
    if _require(gguf, _HEAD_DIM) != _require(gguf, _VALUE_LENGTH):
        raise ValueError(f"{gguf.path}: key_length and value_length differ")
    return Geometry(
        hidden=_require(gguf, _HIDDEN),
        layers=blocks - nextn,
        attention_heads=_require(gguf, _HEADS),
        head_dim=_require(gguf, _HEAD_DIM),
        kv_heads=_require(gguf, _KV_HEADS),
        intermediate=_require(gguf, _INTERMEDIATE),
        gdn_key_heads=_require(gguf, _SSM_GROUPS),
        gdn_value_heads=_require(gguf, _SSM_RANK),
        gdn_head_dim=_require(gguf, _SSM_STATE),
        gdn_taps=_require(gguf, _SSM_CONV),
        full_attention_interval=_require(gguf, _INTERVAL),
        mtp=nextn > 0,
    )


# ── linear attention (GDN) tiling ────────────────────────────────────────────
# The exporter stores value heads tiled so a plain repeat broadcasts the key
# heads; NInfer pairs value head `h` with key head `h // per_key` and wants the
# grouped order. The head counts come from the header, so a model with a
# different ratio gets its own permutation rather than the MoE model's 32/16.


def _tiled_to_grouped(geom: Geometry) -> np.ndarray:
    heads = np.arange(geom.gdn_value_heads)
    per_key = geom.gdn_value_heads // geom.gdn_key_heads
    return (heads % per_key) * geom.gdn_key_heads + heads // per_key


def _untile(geom: Geometry, array: np.ndarray, head_dim: int) -> np.ndarray:
    if array.shape[0] != geom.gdn_value_heads * head_dim:
        raise ValueError(
            f"axis 0 has {array.shape[0]} entries, expected "
            f"{geom.gdn_value_heads * head_dim}"
        )
    view = array.reshape(geom.gdn_value_heads, head_dim, *array.shape[1:])
    return np.ascontiguousarray(view[_tiled_to_grouped(geom)].reshape(array.shape))


def _untiled_rows(geom: Geometry, base: int) -> RowMap:
    permutation = _tiled_to_grouped(geom)
    head_dim = geom.gdn_head_dim

    def select(begin: int, end: int) -> np.ndarray:
        r = np.arange(begin, end, dtype=np.int64)
        return base + permutation[r // head_dim] * head_dim + r % head_dim

    return select


def _attention_rows(geom: Geometry, gate: bool) -> RowMap:
    """The query (or output gate) rows of the head-interleaved ``attn_q``."""
    d = geom.head_dim

    def select(begin: int, end: int) -> np.ndarray:
        r = np.arange(begin, end, dtype=np.int64)
        return (r // d) * 2 * d + (d if gate else 0) + r % d

    return select


def tiled_input_columns(geom: Geometry) -> np.ndarray:
    """For each stored ``ssm_out`` column (tiled value heads), its grouped activation element."""

    columns = np.arange(geom.gdn_value_dim, dtype=np.int64)
    head_dim = geom.gdn_head_dim
    tiled_head, lane = columns // head_dim, columns % head_dim
    per_key = geom.gdn_value_heads // geom.gdn_key_heads
    grouped_head = (tiled_head % geom.gdn_key_heads) * per_key + tiled_head // geom.gdn_key_heads
    return (grouped_head * head_dim + lane).astype(np.int32)


# ── the tensor map ───────────────────────────────────────────────────────────


def _attention_tensors(prefix: str, geom: Geometry) -> dict[str, tuple[tuple[int, ...], str]]:
    h = geom.hidden
    return {
        prefix + "attn_q.weight": ((2 * geom.query_rows, h), "blocks"),
        prefix + "attn_k.weight": ((geom.kv_rows, h), "blocks"),
        prefix + "attn_v.weight": ((geom.kv_rows, h), "blocks"),
        prefix + "attn_output.weight": ((h, geom.query_rows), "blocks"),
        prefix + "attn_q_norm.weight": ((geom.head_dim,), "F32"),
        prefix + "attn_k_norm.weight": ((geom.head_dim,), "F32"),
    }


def _gdn_tensors(prefix: str, geom: Geometry) -> dict[str, tuple[tuple[int, ...], str]]:
    h = geom.hidden
    return {
        prefix + "attn_qkv.weight": ((geom.gdn_channels, h), "blocks"),
        prefix + "attn_gate.weight": ((geom.gdn_value_dim, h), "blocks"),
        prefix + "ssm_out.weight": ((h, geom.gdn_value_dim), "blocks"),
        prefix + "ssm_alpha.weight": ((geom.gdn_value_heads, h), "ssm_pair"),
        prefix + "ssm_beta.weight": ((geom.gdn_value_heads, h), "ssm_pair"),
        prefix + "ssm_a": ((geom.gdn_value_heads,), "F32"),
        prefix + "ssm_dt.bias": ((geom.gdn_value_heads,), "F32"),
        prefix + "ssm_conv1d.weight": ((geom.gdn_channels, geom.gdn_taps), "F32"),
        prefix + "ssm_norm.weight": ((geom.gdn_head_dim,), "F32"),
    }


def _mlp_tensors(prefix: str, geom: Geometry) -> dict[str, tuple[tuple[int, ...], str]]:
    h, i = geom.hidden, geom.intermediate
    return {
        prefix + "ffn_gate.weight": ((i, h), "blocks"),
        prefix + "ffn_up.weight": ((i, h), "blocks"),
        prefix + "ffn_down.weight": ((h, i), "blocks"),
    }


def expected_tensors(geom: Geometry, vocabulary: int) -> dict[str, tuple[tuple[int, ...], str]]:
    """Row-major shape and kind of every tensor the file must carry."""

    out: dict[str, tuple[tuple[int, ...], str]] = {
        "token_embd.weight": ((vocabulary, geom.hidden), "blocks"),
        "output_norm.weight": ((geom.hidden,), "F32"),
    }
    for layer in range(geom.layers):
        p = f"blk.{layer}."
        out |= {
            p + "attn_norm.weight": ((geom.hidden,), "F32"),
            p + "post_attention_norm.weight": ((geom.hidden,), "F32"),
        }
        out |= _mlp_tensors(p, geom)
        out |= (
            _attention_tensors(p, geom)
            if geom.full_attention(layer)
            else _gdn_tensors(p, geom)
        )
    if geom.mtp:
        p = f"blk.{geom.mtp_block}."
        out |= {
            p + "attn_norm.weight": ((geom.hidden,), "F32"),
            p + "post_attention_norm.weight": ((geom.hidden,), "F32"),
            p + "nextn.eh_proj.weight": ((geom.hidden, 2 * geom.hidden), "blocks"),
            p + "nextn.enorm.weight": ((geom.hidden,), "F32"),
            p + "nextn.hnorm.weight": ((geom.hidden,), "F32"),
            p + "nextn.shared_head_norm.weight": ((geom.hidden,), "F32"),
        }
        out |= _attention_tensors(p, geom)
        out |= _mlp_tensors(p, geom)
    return out


def validate(gguf: GGUFFile) -> Geometry:
    """Check the file against the geometry its own header declares.

    The header is the source of truth for the dimensions, so this compares the
    file's tensor set and shapes against what those dimensions imply. Raises on
    any mismatch, so a wrong reading fails here rather than producing a silently
    wrong artifact.
    """

    from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

    geom = geometry(gguf)
    # The vocabulary is not a geometry key; the embedding declares it.
    expected = expected_tensors(geom, gguf.tensors["token_embd.weight"].shape[0])
    present = set(gguf.tensors)
    # A release may or may not carry a separate output head; both are legal.
    if "output.weight" in present:
        expected["output.weight"] = (
            (gguf.tensors["output.weight"].shape[0], geom.hidden),
            "blocks",
        )
    if present != set(expected):
        missing = sorted(set(expected) - present)[:5]
        extra = sorted(present - set(expected))[:5]
        raise ValueError(
            f"{gguf.path}: tensor set mismatch (missing {missing}, extra {extra})"
        )

    for name, (shape, kind) in expected.items():
        info = gguf.tensors[name]
        stored = info.type_id in GGUF_FORMATS_BY_TYPE
        if info.shape != shape:
            raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
        if kind == "ssm_pair":
            # Releases disagree on this pair: a stored block type, BF16, or plain
            # F32 as this one ships it.
            if not stored and info.type_name not in ("BF16", "F32"):
                raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
        elif kind == "blocks":
            if not stored:
                raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
        elif info.type_name != kind:
            raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")

    end = max(info.offset + info.nbytes for info in gguf.tensors.values())
    if gguf.data_bytes_available < end:
        raise ValueError(f"{gguf.path}: the data section is truncated")
    return geom


# ── sources ──────────────────────────────────────────────────────────────────


def _attention(gguf: GGUFFile, g: str, a: str, geom: Geometry):
    q_shape = (geom.query_rows, geom.hidden)
    kv_shape = (geom.kv_rows, geom.hidden)
    return {
        a + "query": (
            block_source(gguf, g + "attn_q.weight", q_shape, _attention_rows(geom, False)),
            True,
        ),
        a + "gate": (
            block_source(gguf, g + "attn_q.weight", q_shape, _attention_rows(geom, True)),
            True,
        ),
        a + "key": (block_source(gguf, g + "attn_k.weight", kv_shape, rows()), True),
        a + "value": (block_source(gguf, g + "attn_v.weight", kv_shape, rows()), True),
        a + "output": (
            block_source(gguf, g + "attn_output.weight", (geom.hidden, geom.query_rows), rows()),
            True,
        ),
        a + "query_norm": (_norm(gguf, g + "attn_q_norm.weight", True), False),
        a + "key_norm": (_norm(gguf, g + "attn_k_norm.weight", True), False),
    }


def _gdn(gguf: GGUFFile, g: str, n: str, geom: Geometry):
    key_dim, value_dim = geom.gdn_key_dim, geom.gdn_value_dim
    h = geom.hidden
    out: dict[str, tuple[LogicalSource, bool]] = {}
    qkv = g + "attn_qkv.weight"
    out[n + "query"] = (block_source(gguf, qkv, (key_dim, h), rows()), True)
    out[n + "key"] = (block_source(gguf, qkv, (key_dim, h), rows(key_dim)), True)
    out[n + "value"] = (
        block_source(gguf, qkv, (value_dim, h), _untiled_rows(geom, 2 * key_dim)),
        True,
    )
    out[n + "z"] = (
        block_source(gguf, g + "attn_gate.weight", (value_dim, h), _untiled_rows(geom, 0)),
        True,
    )
    out[n + "output"] = (block_source(gguf, g + "ssm_out.weight", (h, value_dim), rows()), True)
    for role, tensor in (
        ("a_projection", "ssm_alpha.weight"),
        ("b_projection", "ssm_beta.weight"),
    ):
        stored = gguf.info(g + tensor).type_name
        if stored == "BF16":
            words = _untile(geom, gguf.read_bf16_words(g + tensor), 1)
        elif stored == "F32":
            # Unquantized: read it and narrow, as the direct-format norms do.
            values = gguf.read_direct(g + tensor).astype(np.float32)
            words = _untile(
                geom, torch.from_numpy(values).to(torch.bfloat16).view(torch.int16).numpy(), 1
            )
        else:
            values = _dequantize(gguf, g + tensor, 0, geom.gdn_value_heads)
            words = _untile(geom, values.to(torch.bfloat16).view(torch.int16).numpy(), 1)
        out[n + role] = (
            array_source(
                torch.from_numpy(np.ascontiguousarray(words.view(np.int16))).view(torch.bfloat16),
                g + tensor,
            ),
            False,
        )
    ssm_a = _untile(geom, gguf.read_direct(g + "ssm_a"), 1).astype(np.float64)
    if not np.all(ssm_a < 0):
        raise ValueError(f"{g}ssm_a must be strictly negative (-exp(A_log))")
    out[n + "a_log"] = (
        _direct(np.log(-ssm_a).astype(np.float32), torch.float32, g + "ssm_a"),
        False,
    )
    out[n + "dt_bias"] = (
        _direct(
            _untile(geom, gguf.read_direct(g + "ssm_dt.bias"), 1),
            torch.float32,
            g + "ssm_dt.bias",
        ),
        False,
    )
    taps = gguf.read_direct(g + "ssm_conv1d.weight")
    channels = np.concatenate(
        [taps[: 2 * key_dim], _untile(geom, taps[2 * key_dim :], geom.gdn_head_dim)]
    )
    out[n + "convolution"] = (
        _direct(channels.T, torch.bfloat16, g + "ssm_conv1d.weight"),
        False,
    )
    out[n + "norm"] = (_norm(gguf, g + "ssm_norm.weight", False), False)
    return out


def _mlp(gguf: GGUFFile, g: str, p: str, geom: Geometry):
    h, i = geom.hidden, geom.intermediate
    return {
        p + "input_norm": (_norm(gguf, g + "attn_norm.weight", True), False),
        p + "post_attention_norm": (_norm(gguf, g + "post_attention_norm.weight", True), False),
        p + "mlp/gate": (block_source(gguf, g + "ffn_gate.weight", (i, h), rows()), True),
        p + "mlp/up": (block_source(gguf, g + "ffn_up.weight", (i, h), rows()), True),
        p + "mlp/down": (block_source(gguf, g + "ffn_down.weight", (h, i), rows()), True),
    }


def text_sources(gguf: GGUFFile, geom: Geometry):
    """Every text parameter's GGUF source; True marks encoded block rows."""

    h = geom.hidden
    vocabulary = gguf.tensors["token_embd.weight"].shape[0]
    # Releases without a separate head tie it to the embedding, so both names read
    # the same tensor; one that carries `output.weight` uses it.
    head = "output.weight" if "output.weight" in gguf.tensors else "token_embd.weight"
    out: dict[str, tuple[LogicalSource, bool]] = {
        "text/token_embedding": (
            block_source(gguf, "token_embd.weight", (vocabulary, h), rows()),
            True,
        ),
        "text/output_head": (
            block_source(gguf, head, (vocabulary, h), rows()),
            True,
        ),
        "text/final_norm": (_norm(gguf, "output_norm.weight", True), False),
    }
    for layer in range(geom.layers):
        g, p = f"blk.{layer}.", f"text/layers/{layer}/"
        out |= _mlp(gguf, g, p, geom)
        out |= (
            _attention(gguf, g, p + "attention/", geom)
            if geom.full_attention(layer)
            else _gdn(gguf, g, p + "gdn/", geom)
        )
    if geom.mtp:
        # The MTP block is full attention whatever the interval says.
        g, p = f"blk.{geom.mtp_block}.", "mtp/layers/0/"
        out |= _mlp(gguf, g, p, geom)
        out |= _attention(gguf, g, p + "attention/", geom)
        out["mtp/input_projection"] = (
            block_source(gguf, g + "nextn.eh_proj.weight", (h, 2 * h), rows()),
            True,
        )
        for role, tensor in (
            ("embedding_norm", "nextn.enorm.weight"),
            ("hidden_norm", "nextn.hnorm.weight"),
            ("final_norm", "nextn.shared_head_norm.weight"),
        ):
            out["mtp/" + role] = (_norm(gguf, g + tensor, True), False)
    return out


# ── recipe ───────────────────────────────────────────────────────────────────


def _assign(recipe, name: str, source: LogicalSource, encoded: bool, model) -> str:
    if encoded:
        format = source.read_encoded(0, 1).format
        recipe.assign(
            name, format=format, method=import_encoded, source=source, activation_policy="AllowA8"
        )
        return format
    format = model.parameters[name].direct_format
    recipe.assign(name, format=format, method=cast_direct, source=source)
    return format


def _group_same_format(recipe, names: list[str], formats: dict[str, str]) -> None:
    if len({formats[name] for name in names}) == 1:
        recipe.group(names)


def qwen3_5_dense_gguf(model, recipe, sources):
    """A Qwen3.5 dense ``qwen35`` GGUF release in its own block formats.

    Point ``--source gguf`` at the model's ``.gguf``. Every quantized tensor keeps
    the exporter's blocks, so no weight is re-quantised: a row of the artifact is
    byte for byte a row of the GGUF. The geometry -- hidden size, layer count,
    head counts, FFN width and the GDN's own dimensions -- comes from the file's
    header, so any dense Qwen3.5 release the exporter wrote is taken, not just
    one size.

    Vision comes from the release's ``mmproj`` GGUF (``--source vision=mmproj.gguf``),
    as the MoE recipe takes it.
    """

    gguf = sources["gguf"]
    if not isinstance(gguf, GGUFFile):
        raise ValueError("--source gguf must name the model's .gguf file")
    geom = validate(gguf)
    # validate() reads the FILE; what the groupings need is whether the MODEL
    # carries the component, which --components decides.
    mtp = "mtp" in model.components
    if mtp and not geom.mtp:
        raise ValueError(f"{gguf.path}: the MTP component needs a file with blk.{geom.mtp_block}.nextn.*")

    formats: dict[str, str] = {}
    for name, (source, encoded) in text_sources(gguf, geom).items():
        formats[name] = _assign(recipe, name, source, encoded, model)

    if "vision" in model.components:
        from .gguf_blocks import _vision_source

        vision = _vision_source(model, sources)
        for name, parameter in model.parameters.items():
            if name.startswith("vision/"):
                formats[name] = parameter.direct_format
                recipe.assign(
                    name, format=parameter.direct_format, method=cast_direct,
                    source=model.source(name, vision),
                )

    if set(formats) != set(model.parameters):
        missing = sorted(set(model.parameters) - set(formats))[:5]
        raise ValueError(f"the GGUF leaves logical parameters without a source: {missing}")

    for layer in range(geom.layers):
        p = f"text/layers/{layer}/"
        if geom.full_attention(layer):
            a = p + "attention/"
            _group_same_format(recipe, [a + "query", a + "gate"], formats)
            _group_same_format(recipe, [a + "key", a + "value"], formats)
        else:
            n = p + "gdn/"
            qkv = [n + "query", n + "key", n + "value"]
            recipe.group(qkv + [n + "z"] if formats[n + "z"] == formats[n + "query"] else qkv)
            recipe.group([n + "a_projection", n + "b_projection"])
        _group_same_format(recipe, [p + "mlp/gate", p + "mlp/up"], formats)
    if mtp:
        p = "mtp/layers/0/"
        a = p + "attention/"
        _group_same_format(recipe, [a + "query", a + "gate"], formats)
        _group_same_format(recipe, [a + "key", a + "value"], formats)
        _group_same_format(recipe, [p + "mlp/gate", p + "mlp/up"], formats)

    columns = AuxiliaryValue(
        "int32", (geom.gdn_value_dim,), tiled_input_columns(geom).astype("<i4").tobytes()
    )
    for layer in range(geom.layers):
        if geom.full_attention(layer):
            continue
        name = f"text/layers/{layer}/gdn/output"
        for input_name in model.parameters[name].inputs:
            recipe.use(name, input_name, auxiliaries={"input_columns": columns})


RECIPES = {
    "qwen3_5_dense_gguf": qwen3_5_dense_gguf,
}

__all__ = [
    "RECIPES",
    "Geometry",
    "expected_tensors",
    "geometry",
    "text_sources",
    "tiled_input_columns",
    "validate",
]
