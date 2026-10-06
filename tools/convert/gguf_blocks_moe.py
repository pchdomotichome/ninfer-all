"""A Qwen3.6-35B-A3B ``qwen35moe`` GGUF kept in its own ggml block formats.

Sibling of `gguf_blocks`: same contract (a row of the artifact is byte for byte
a row of the GGUF, no weight is re-quantised), different geometry. The dense
module's constants are module-level and describe the 27B, so the MoE model needs
its own rather than a parameterised shared table.

What differs beyond the numbers:

* Routed experts are **fused** in the GGUF. ``blk.N.ffn_up_exps.weight`` is
  ``ne = (HIDDEN, MOE_INTER, EXPERTS)`` -- hidden, intermediate, expert -- and
  ``TensorInfo.shape`` reports it reversed as ``(EXPERTS, MOE_INTER, HIDDEN)``.
  Quantization runs along the contiguous axis, so one tensor is ``EXPERTS *
  MOE_INTER`` rows of ``HIDDEN``, and expert ``e`` is the **contiguous** row
  range ``[e * MOE_INTER, (e + 1) * MOE_INTER)``. A row copy therefore moves an
  expert without touching its blocks, which is what makes this recipe possible
  without a dequantise/re-quantise pass over the 33 GB of experts.
* The artifact stores those experts as one fused object per layer too
  (``gate_e, up_e, gate_e + 1, up_e + 1, ...``), but that packing is the
  writer's business: this module assigns each routed-expert path its own
  ``LogicalSource`` and lets the artifact layer decide how they cohabit.
* There is no dense MLP: every layer routes to ``EXPERTS`` of
  ``MOE_INTER`` and to one shared expert of ``SHARED_INTER``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .methods import cast_direct, import_encoded
from .sources.gguf import GGUFFile
from .sources.logical import LogicalSource, array_source
from .gguf_blocks import _dequantize, block_source
from .ternary import RowMap, _direct, _norm, attention_rows, full_attention, rows

# ── geometry (qwen35moe, Qwen3.6-35B-A3B) ────────────────────────────────────
HIDDEN = 2048
LAYERS = 40
VOCABULARY = 248320

EXPERTS = 256
EXPERTS_PER_TOKEN = 8
MOE_INTERMEDIATE = 512
SHARED_INTERMEDIATE = 512

ATTENTION_HEADS = 16
ATTENTION_HEAD_DIM = 256
ATTENTION_QUERY_ROWS = ATTENTION_HEADS * ATTENTION_HEAD_DIM  # 4096
ATTENTION_KV_ROWS = 2 * ATTENTION_HEAD_DIM  # 512

GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 32
GDN_HEAD_DIM = 128
GDN_KEY_DIM = GDN_KEY_HEADS * GDN_HEAD_DIM  # 2048
GDN_VALUE_DIM = GDN_VALUE_HEADS * GDN_HEAD_DIM  # 4096
GDN_CHANNELS = 2 * GDN_KEY_DIM + GDN_VALUE_DIM  # 8192
GDN_TAPS = 4

MTP_BLOCK = LAYERS  # 40: the nextn layer follows the 40 target layers

EXPECTED_HEADER = {
    "general.architecture": "qwen35moe",
    "qwen35moe.embedding_length": HIDDEN,
    "qwen35moe.block_count": LAYERS + 1,
    "qwen35moe.expert_count": EXPERTS,
    "qwen35moe.expert_feed_forward_length": MOE_INTERMEDIATE,
    "qwen35moe.expert_shared_feed_forward_length": SHARED_INTERMEDIATE,
    "qwen35moe.attention.head_count": ATTENTION_HEADS,
    "qwen35moe.attention.head_count_kv": 2,
    "qwen35moe.attention.key_length": ATTENTION_HEAD_DIM,
    "qwen35moe.attention.value_length": ATTENTION_HEAD_DIM,
    "qwen35moe.ssm.conv_kernel": GDN_TAPS,
    "qwen35moe.ssm.state_size": GDN_HEAD_DIM,
    "qwen35moe.ssm.group_count": GDN_KEY_HEADS,
    "qwen35moe.ssm.time_step_rank": GDN_VALUE_HEADS,
    "qwen35moe.ssm.inner_size": GDN_VALUE_DIM,
    "qwen35moe.full_attention_interval": 4,
}

Rows = Callable[[int, int], np.ndarray]


def _moe_tensors(prefix: str) -> dict[str, tuple[tuple[int, ...], str]]:
    """Row-major shape and kind of one layer's MoE tensors, experts excluded."""

    return {
        prefix + "ffn_gate_inp.weight": ((EXPERTS, HIDDEN), "F32"),
        prefix + "ffn_gate_inp_shexp.weight": ((HIDDEN,), "F32"),
        prefix + "ffn_gate_shexp.weight": ((SHARED_INTERMEDIATE, HIDDEN), "blocks"),
        prefix + "ffn_up_shexp.weight": ((SHARED_INTERMEDIATE, HIDDEN), "blocks"),
        prefix + "ffn_down_shexp.weight": ((HIDDEN, SHARED_INTERMEDIATE), "blocks"),
    }


def _routed_experts(prefix: str) -> dict[str, tuple[tuple[int, ...], str]]:
    """The fused routed-expert tensors, with the expert axis first in ``info.shape``."""

    return {
        prefix + "ffn_gate_exps.weight": ((EXPERTS, MOE_INTERMEDIATE, HIDDEN), "blocks"),
        prefix + "ffn_up_exps.weight": ((EXPERTS, MOE_INTERMEDIATE, HIDDEN), "blocks"),
        prefix + "ffn_down_exps.weight": ((EXPERTS, HIDDEN, MOE_INTERMEDIATE), "blocks"),
    }


def _attention_tensors(prefix: str) -> dict[str, tuple[tuple[int, ...], str]]:
    """A full-attention layer.

    ``attn_q`` is ``2 * ATTENTION_QUERY_ROWS`` rows because the exporter fuses
    the query with its output gate head-interleaved, exactly as the dense model
    does; the two are separated on read with :func:`attention_rows`. The linear
    (GDN) layers fuse query/key/value into a single ``attn_qkv`` instead, so a
    layer has either this set or :func:`_gdn_tensors`, never both.
    """

    return {
        prefix + "attn_q.weight": ((2 * ATTENTION_QUERY_ROWS, HIDDEN), "blocks"),
        prefix + "attn_k.weight": ((ATTENTION_KV_ROWS, HIDDEN), "blocks"),
        prefix + "attn_v.weight": ((ATTENTION_KV_ROWS, HIDDEN), "blocks"),
        prefix + "attn_output.weight": ((HIDDEN, ATTENTION_QUERY_ROWS), "blocks"),
        prefix + "attn_q_norm.weight": ((ATTENTION_HEAD_DIM,), "F32"),
        prefix + "attn_k_norm.weight": ((ATTENTION_HEAD_DIM,), "F32"),
    }


def _gdn_tensors(prefix: str) -> dict[str, tuple[tuple[int, ...], str]]:
    """A linear-attention (GDN) layer.

    ``ssm_alpha``/``ssm_beta`` are ``(GDN_VALUE_HEADS, HIDDEN)`` -- the stored
    ``ne`` is ``(HIDDEN, GDN_VALUE_HEADS)``, and ``info.shape`` reverses it.
    """

    return {
        prefix + "attn_qkv.weight": ((GDN_CHANNELS, HIDDEN), "blocks"),
        prefix + "attn_gate.weight": ((GDN_VALUE_DIM, HIDDEN), "blocks"),
        prefix + "ssm_out.weight": ((HIDDEN, GDN_VALUE_DIM), "blocks"),
        prefix + "ssm_alpha.weight": ((GDN_VALUE_HEADS, HIDDEN), "ssm_pair"),
        prefix + "ssm_beta.weight": ((GDN_VALUE_HEADS, HIDDEN), "ssm_pair"),
        prefix + "ssm_a": ((GDN_VALUE_HEADS,), "F32"),
        prefix + "ssm_dt.bias": ((GDN_VALUE_HEADS,), "F32"),
        prefix + "ssm_conv1d.weight": ((GDN_CHANNELS, GDN_TAPS), "F32"),
        prefix + "ssm_norm.weight": ((GDN_HEAD_DIM,), "F32"),
    }


def expected_tensors(mtp: bool) -> dict[str, tuple[tuple[int, ...], str]]:
    """Row-major shape and kind ("blocks" = any stored ggml block type) of every tensor.

    Kind ``"ssm_pair"`` marks the two GDN gate tensors community releases store
    at different precisions, like the dense module's ``ssm_alpha``/``ssm_beta``.
    """

    out: dict[str, tuple[tuple[int, ...], str]] = {
        "token_embd.weight": ((VOCABULARY, HIDDEN), "blocks"),
        "output.weight": ((VOCABULARY, HIDDEN), "blocks"),
        "output_norm.weight": ((HIDDEN,), "F32"),
    }
    for layer in range(LAYERS):
        p = f"blk.{layer}."
        out |= {
            p + "attn_norm.weight": ((HIDDEN,), "F32"),
            p + "post_attention_norm.weight": ((HIDDEN,), "F32"),
        }
        out |= _moe_tensors(p)
        out |= _routed_experts(p)
        out |= _attention_tensors(p) if full_attention(layer) else _gdn_tensors(p)
    if mtp:
        # The MTP block is a full-attention layer whatever the interval says:
        # full_attention(40) is False, but the file carries attn_q/k/v, not the
        # fused GDN attn_qkv.
        p = f"blk.{MTP_BLOCK}."
        out |= {
            p + "attn_norm.weight": ((HIDDEN,), "F32"),
            p + "post_attention_norm.weight": ((HIDDEN,), "F32"),
            p + "nextn.eh_proj.weight": ((HIDDEN, 2 * HIDDEN), "blocks"),
            p + "nextn.enorm.weight": ((HIDDEN,), "F32"),
            p + "nextn.hnorm.weight": ((HIDDEN,), "F32"),
            p + "nextn.shared_head_norm.weight": ((HIDDEN,), "F32"),
        }
        out |= _attention_tensors(p)
        out |= _moe_tensors(p)
        out |= _routed_experts(p)
    return out


def has_mtp(gguf: GGUFFile) -> bool:
    return f"blk.{MTP_BLOCK}.nextn.eh_proj.weight" in gguf.tensors


def validate(gguf: GGUFFile) -> bool:
    """Check the file against the geometry this module describes.

    Returns whether an MTP layer is present. Raises on any mismatch, so a wrong
    constant fails here instead of producing a silently wrong artifact.
    """

    from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

    for key, expected in EXPECTED_HEADER.items():
        if gguf.kv.get(key) != expected:
            raise ValueError(
                f"{gguf.path}: {key} = {gguf.kv.get(key)!r}, expected {expected!r}"
            )

    mtp = has_mtp(gguf)
    blocks = gguf.kv.get("qwen35moe.block_count")
    if blocks != LAYERS + (1 if mtp else 0):
        raise ValueError(f"{gguf.path}: {blocks} blocks, expected {LAYERS} (+1 for MTP)")

    expected = expected_tensors(mtp)
    if set(gguf.tensors) != set(expected):
        missing = sorted(set(expected) - set(gguf.tensors))[:5]
        extra = sorted(set(gguf.tensors) - set(expected))[:5]
        raise ValueError(
            f"{gguf.path}: tensor set mismatch (missing {missing}, extra {extra})"
        )

    for name, (shape, kind) in expected.items():
        info = gguf.tensors[name]
        stored = info.type_id in GGUF_FORMATS_BY_TYPE
        if info.shape != shape:
            raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
        if kind == "ssm_pair":
            # GGUF releases disagree on this pair: BF16, Q8_0 or a k-quant.
            if not stored and info.type_name != "BF16":
                raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
        elif kind == "blocks":
            if not stored:
                raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
        elif kind in ("F32", "BF16"):
            # the paired MoE routers ship as either unquantized float width
            if info.type_name not in ("F32", "BF16"):
                raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
        elif info.type_name != kind:
            raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")

    end = max(info.offset + info.nbytes for info in gguf.tensors.values())
    if gguf.data_bytes_available < end:
        raise ValueError(f"{gguf.path}: the data section is truncated")
    return mtp

# ── linear attention (GDN) tiling ────────────────────────────────────────────
# The exporter stores value heads tiled so a plain repeat broadcasts the key
# heads; NInfer pairs value head `h` with key head `h // per_key` and wants the
# grouped order. The dense module's helpers hardcode its own 48/16 ratio, so the
# MoE model needs its own with per_key = 32 // 16 = 2.


def _tiled_to_grouped_permutation() -> np.ndarray:
    heads = np.arange(GDN_VALUE_HEADS)
    per_key = GDN_VALUE_HEADS // GDN_KEY_HEADS
    return (heads % per_key) * GDN_KEY_HEADS + heads // per_key


def _untile(array: np.ndarray, head_dim: int) -> np.ndarray:
    if array.shape[0] != GDN_VALUE_HEADS * head_dim:
        raise ValueError(
            f"axis 0 has {array.shape[0]} entries, expected {GDN_VALUE_HEADS * head_dim}"
        )
    view = array.reshape(GDN_VALUE_HEADS, head_dim, *array.shape[1:])
    return np.ascontiguousarray(view[_tiled_to_grouped_permutation()].reshape(array.shape))


def _untiled_rows(base: int) -> RowMap:
    permutation = _tiled_to_grouped_permutation()

    def select(begin: int, end: int) -> np.ndarray:
        r = np.arange(begin, end, dtype=np.int64)
        return base + permutation[r // GDN_HEAD_DIM] * GDN_HEAD_DIM + r % GDN_HEAD_DIM

    return select


def tiled_input_columns() -> np.ndarray:
    """For each stored ``ssm_out`` column (tiled value heads), its grouped activation element."""

    columns = np.arange(GDN_VALUE_DIM, dtype=np.int64)
    tiled_head, lane = columns // GDN_HEAD_DIM, columns % GDN_HEAD_DIM
    per_key = GDN_VALUE_HEADS // GDN_KEY_HEADS
    grouped_head = (tiled_head % GDN_KEY_HEADS) * per_key + tiled_head // GDN_KEY_HEADS
    return (grouped_head * GDN_HEAD_DIM + lane).astype(np.int32)


# ── sources ──────────────────────────────────────────────────────────────────


def _read_float_words(gguf: GGUFFile, tensor: str) -> LogicalSource:
    """An unquantized tensor read straight through, widened to bfloat16.

    ``_norm`` is this minus the exporter's stored ``1 + w``; the MoE routers are
    plain matrices, so they take this path instead.
    """

    values = gguf.read_direct(tensor).astype(np.float32)
    return _direct(values, torch.bfloat16, tensor)


def _attention(
    gguf: GGUFFile, g: str, a: str
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    q_shape, kv_shape = (ATTENTION_QUERY_ROWS, HIDDEN), (ATTENTION_KV_ROWS, HIDDEN)
    encoded = {
        a + "query": block_source(gguf, g + "attn_q.weight", q_shape, attention_rows(False)),
        a + "gate": block_source(gguf, g + "attn_q.weight", q_shape, attention_rows(True)),
        a + "key": block_source(gguf, g + "attn_k.weight", kv_shape, rows()),
        a + "value": block_source(gguf, g + "attn_v.weight", kv_shape, rows()),
        a + "output": block_source(
            gguf, g + "attn_output.weight", (HIDDEN, ATTENTION_QUERY_ROWS), rows()
        ),
    }
    direct = {
        a + "query_norm": _norm(gguf, g + "attn_q_norm.weight", True),
        a + "key_norm": _norm(gguf, g + "attn_k_norm.weight", True),
    }
    return encoded, direct


def _gdn(
    gguf: GGUFFile, g: str, n: str
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    encoded = {}
    direct: dict[str, LogicalSource] = {}
    qkv = g + "attn_qkv.weight"
    encoded[n + "query"] = block_source(gguf, qkv, (GDN_KEY_DIM, HIDDEN), rows())
    encoded[n + "key"] = block_source(gguf, qkv, (GDN_KEY_DIM, HIDDEN), rows(GDN_KEY_DIM))
    encoded[n + "value"] = block_source(
        gguf, qkv, (GDN_VALUE_DIM, HIDDEN), _untiled_rows(2 * GDN_KEY_DIM)
    )
    encoded[n + "z"] = block_source(
        gguf, g + "attn_gate.weight", (GDN_VALUE_DIM, HIDDEN), _untiled_rows(0)
    )
    encoded[n + "output"] = block_source(
        gguf, g + "ssm_out.weight", (HIDDEN, GDN_VALUE_DIM), rows()
    )
    for role, tensor in (
        ("a_projection", "ssm_alpha.weight"),
        ("b_projection", "ssm_beta.weight"),
    ):
        if gguf.info(g + tensor).type_name == "BF16":
            words = _untile(gguf.read_bf16_words(g + tensor), 1)
        else:
            values = _dequantize(gguf, g + tensor, 0, GDN_VALUE_HEADS)
            words = _untile(values.to(torch.bfloat16).view(torch.int16).numpy(), 1)
        direct[n + role] = array_source(
            torch.from_numpy(np.ascontiguousarray(words.view(np.int16))).view(torch.bfloat16),
            g + tensor,
        )
    ssm_a = _untile(gguf.read_direct(g + "ssm_a"), 1).astype(np.float64)
    if not np.all(ssm_a < 0):
        raise ValueError(f"{g}ssm_a must be strictly negative (-exp(A_log))")
    direct[n + "a_log"] = _direct(np.log(-ssm_a).astype(np.float32), torch.float32, g + "ssm_a")
    direct[n + "dt_bias"] = _direct(
        _untile(gguf.read_direct(g + "ssm_dt.bias"), 1), torch.float32, g + "ssm_dt.bias"
    )
    taps = gguf.read_direct(g + "ssm_conv1d.weight")
    channels = np.concatenate(
        [taps[: 2 * GDN_KEY_DIM], _untile(taps[2 * GDN_KEY_DIM :], GDN_HEAD_DIM)]
    )
    direct[n + "convolution"] = _direct(channels.T, torch.bfloat16, g + "ssm_conv1d.weight")
    direct[n + "norm"] = _norm(gguf, g + "ssm_norm.weight", False)
    return encoded, direct


def _moe_layer(
    gguf: GGUFFile, g: str, p: str
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    """One layer's MoE block: the routed experts, the shared expert, the routers.

    Each routed expert is a contiguous row range of its fused tensor, so the
    blocks move without re-quantising. ``gate``/``up`` have ``MOE_INTERMEDIATE``
    rows of ``HIDDEN`` per expert; ``down`` has ``HIDDEN`` rows of
    ``MOE_INTERMEDIATE``.
    """

    encoded: dict[str, LogicalSource] = {}
    gate_tensor, up_tensor = g + "ffn_gate_exps.weight", g + "ffn_up_exps.weight"
    down_tensor = g + "ffn_down_exps.weight"
    for expert in range(EXPERTS):
        gate_base = expert * MOE_INTERMEDIATE
        down_base = expert * HIDDEN
        prefix = p + f"moe/experts/{expert}/"
        encoded[prefix + "gate"] = block_source(
            gguf, gate_tensor, (MOE_INTERMEDIATE, HIDDEN), rows(gate_base)
        )
        encoded[prefix + "up"] = block_source(
            gguf, up_tensor, (MOE_INTERMEDIATE, HIDDEN), rows(gate_base)
        )
        encoded[prefix + "down"] = block_source(
            gguf, down_tensor, (HIDDEN, MOE_INTERMEDIATE), rows(down_base)
        )
    encoded[p + "moe/shared/gate"] = block_source(
        gguf, g + "ffn_gate_shexp.weight", (SHARED_INTERMEDIATE, HIDDEN), rows()
    )
    encoded[p + "moe/shared/up"] = block_source(
        gguf, g + "ffn_up_shexp.weight", (SHARED_INTERMEDIATE, HIDDEN), rows()
    )
    encoded[p + "moe/shared/down"] = block_source(
        gguf, g + "ffn_down_shexp.weight", (HIDDEN, SHARED_INTERMEDIATE), rows()
    )
    direct = {
        p + "input_norm": _norm(gguf, g + "attn_norm.weight", True),
        p + "post_attention_norm": _norm(gguf, g + "post_attention_norm.weight", True),
        p + "moe/router": _read_float_words(gguf, g + "ffn_gate_inp.weight"),
        p + "moe/shared_score": _read_float_words(gguf, g + "ffn_gate_inp_shexp.weight"),
    }
    return encoded, direct


def text_sources(
    gguf: GGUFFile, *, mtp: bool
) -> tuple[dict[str, LogicalSource], dict[str, LogicalSource]]:
    """Encoded block sources and direct sources of every GGUF-backed parameter."""

    encoded = {
        "text/token_embedding": block_source(
            gguf, "token_embd.weight", (VOCABULARY, HIDDEN), rows()
        ),
        "text/output_head": block_source(gguf, "output.weight", (VOCABULARY, HIDDEN), rows()),
    }
    direct = {"text/final_norm": _norm(gguf, "output_norm.weight", True)}
    for layer in range(LAYERS):
        g, p = f"blk.{layer}.", f"text/layers/{layer}/"
        e, d = _moe_layer(gguf, g, p)
        encoded |= e
        direct |= d
        if full_attention(layer):
            e, d = _attention(gguf, g, p + "attention/")
        else:
            e, d = _gdn(gguf, g, p + "gdn/")
        encoded |= e
        direct |= d
    if mtp:
        # The MTP block is full attention regardless of the interval.
        g, p = f"blk.{MTP_BLOCK}.", "mtp/layers/0/"
        e, d = _moe_layer(gguf, g, p)
        encoded |= e
        direct |= d
        e, d = _attention(gguf, g, p + "attention/")
        encoded |= e
        direct |= d
    return encoded, direct
