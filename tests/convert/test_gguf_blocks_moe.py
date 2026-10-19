from __future__ import annotations

import numpy as np
import pytest

from tools.convert import gguf_blocks_moe
from tools.convert.sources.gguf import GGUFFile, write_gguf

TYPE_Q8_0 = 8
TYPE_F32 = 0


def _q8_0_blocks(generator, rows: int, columns: int) -> np.ndarray:
    blocks = generator.integers(0, 256, size=(rows, columns // 32, 34), dtype=np.uint8)
    scales = generator.uniform(0.01, 0.02, size=(rows, columns // 32)).astype(np.float16)
    blocks[:, :, :2] = scales.view(np.uint8).reshape(rows, columns // 32, 2)
    return blocks


def _header(experts: int = 256, mtp: bool = False, **overrides) -> dict:
    header = dict(gguf_blocks_moe.EXPECTED_HEADER)
    header["qwen35moe.expert_count"] = experts
    # block_count counts the MTP layer only when the file carries one
    header["qwen35moe.block_count"] = gguf_blocks_moe.LAYERS + (1 if mtp else 0)
    header.update(overrides)
    return header


def test_expert_count_comes_from_the_header(tmp_path):
    """A pruned release sizes every downstream structure, so the count is read, not assumed."""

    path = tmp_path / "moe.gguf"
    write_gguf(path, _header(experts=8), [("w", (4, 32), TYPE_Q8_0, b"\x00" * (4 * 34))])
    with GGUFFile(path) as gguf:
        assert gguf_blocks_moe.expert_count(gguf) == 8


def test_expert_count_refuses_a_header_without_one(tmp_path):
    path = tmp_path / "moe.gguf"
    header = _header()
    del header["qwen35moe.expert_count"]
    write_gguf(path, header, [("w", (4, 32), TYPE_Q8_0, b"\x00" * (4 * 34))])
    with GGUFFile(path) as gguf:
        with pytest.raises(ValueError, match="expert_count"):
            gguf_blocks_moe.expert_count(gguf)


def test_expected_tensors_sizes_the_expert_axis():
    """The routed tensors carry the expert axis first, at the header's count."""

    for experts in (256, 32):
        tensors = gguf_blocks_moe.expected_tensors(mtp=False, experts=experts)
        assert tensors["blk.0.ffn_gate_exps.weight"][0] == (experts, 512, 2048)
        assert tensors["blk.0.ffn_up_exps.weight"][0] == (experts, 512, 2048)
        assert tensors["blk.0.ffn_down_exps.weight"][0] == (experts, 2048, 512)
        assert tensors["blk.0.ffn_gate_inp.weight"][0] == (experts, 2048)
        # The GGUF fuses the routed experts: one tensor per layer and role, with
        # the expert axis inside it. The per-expert paths are ninfer's, and come
        # from text_sources, not from this map.
        routed = [n for n in tensors if n.endswith(("_gate_exps.weight", "_up_exps.weight", "_down_exps.weight"))]
        assert len(routed) == gguf_blocks_moe.LAYERS * 3


def test_expected_tensors_gives_a_layer_one_set_of_attention_tensors():
    """A layer is full attention or GDN, never both: layer 0 is GDN, layer 3 is full."""

    tensors = gguf_blocks_moe.expected_tensors(mtp=False)
    assert "blk.0.attn_qkv.weight" in tensors
    assert "blk.0.attn_q.weight" not in tensors
    assert "blk.3.attn_q.weight" in tensors
    assert "blk.3.attn_qkv.weight" not in tensors


def test_the_mtp_block_is_full_attention_whatever_the_interval_says():
    """full_attention(40) is False, but the file carries attn_q/k/v, not the fused attn_qkv."""

    assert not gguf_blocks_moe.full_attention(gguf_blocks_moe.MTP_BLOCK)
    tensors = gguf_blocks_moe.expected_tensors(mtp=True)
    p = f"blk.{gguf_blocks_moe.MTP_BLOCK}."
    assert p + "attn_q.weight" in tensors
    assert p + "attn_qkv.weight" not in tensors
    assert p + "nextn.eh_proj.weight" in tensors
    # and the MTP layer still carries its own MoE block
    assert p + "ffn_gate_exps.weight" in tensors
    assert p + "ffn_up_shexp.weight" in tensors


def test_validate_refuses_a_wrong_geometry(tmp_path):
    """A wrong constant fails at validate instead of producing a silently wrong artifact."""

    path = tmp_path / "moe.gguf"
    write_gguf(
        path,
        _header(**{"qwen35moe.embedding_length": 1024}),
        [("w", (4, 32), TYPE_Q8_0, b"\x00" * (4 * 34))],
    )
    with GGUFFile(path) as gguf:
        with pytest.raises(ValueError, match="embedding_length"):
            gguf_blocks_moe.validate(gguf)


def test_validate_refuses_a_missing_tensor(tmp_path):
    path = tmp_path / "moe.gguf"
    write_gguf(path, _header(), [("w", (4, 32), TYPE_Q8_0, b"\x00" * (4 * 34))])
    with GGUFFile(path) as gguf:
        with pytest.raises(ValueError, match="tensor set mismatch"):
            gguf_blocks_moe.validate(gguf)


def test_the_gdn_tiling_uses_this_models_head_ratio():
    """The dense helpers hardcode 48/16; this model is 32/16, so the permutation differs."""

    permutation = gguf_blocks_moe._tiled_to_grouped_permutation()
    assert gguf_blocks_moe.GDN_VALUE_HEADS == 32
    assert gguf_blocks_moe.GDN_KEY_HEADS == 16
    # per_key is 2 here against the dense module's 3
    assert sorted(permutation.tolist()) == list(range(32))

    # value head h pairs with key head h // 2, and the exporter's tiled position is
    # (h % 2) * 16 + h // 2
    for head in (0, 1, 15, 16, 31):
        assert permutation[head] == (head % 2) * 16 + head // 2


def test_tiled_input_columns_covers_every_grouped_element():
    columns = gguf_blocks_moe.tiled_input_columns()
    assert columns.shape == (gguf_blocks_moe.GDN_VALUE_DIM,)
    assert sorted(columns.tolist()) == list(range(gguf_blocks_moe.GDN_VALUE_DIM))


def test_recipe_is_registered_under_its_name():
    from tools.convert.__main__ import _function

    assert _function("qwen3_6_35b_a3b_gguf") is gguf_blocks_moe.qwen3_6_35b_a3b_gguf
