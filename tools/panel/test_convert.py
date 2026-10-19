"""Self-check for the GGUF listing behind the Convert downloader and the Fit probér.

The quantise name is the only label a file gets, and both views group and compare by it, so a
file the parser cannot name ends up lumped into "__otro": unsorted, unlabelled, and skipped by
anyone reading the list. That is what happened to Unsloth's dynamic quants (UD-Q2_K_XL), which
the old matcher did not know, and it is why the matcher reads a token out of the name instead of
looking for known substrings.

    python3 test_convert.py
"""

from server import API


class Stub:
    """A fake repo: the tree hf_details would have returned, and nothing else."""

    GGUF_QUANT = API.GGUF_QUANT
    GGUF_PREFIXES = API.GGUF_PREFIXES
    _quant_of = API._quant_of
    hf_ggufs = API.hf_ggufs

    def __init__(self, tree):
        self.tree = tree

    def hf_details(self, model):
        return self.tree


def repo(*paths):
    return Stub([{"path": p, "size": 1024} for p in paths])


def quants(files):
    return [f["quant"] for f in files]


def test_dynamic_quants_keep_their_own_name():
    """The case that was reported: UD- is Unsloth's dynamic quant, an unlisted prefix, and the
    file was filed as "otro" instead of keeping the name the publisher gave it."""
    got = quants(repo("Dirk-Qwen3.8-27B-UD-Q4_K_XL.gguf",
                      "Qwen3.8-27B-UD-Q2_K_XL.gguf").hf_ggufs("x"))
    assert got == ["UD-Q2_K_XL", "UD-Q4_K_XL"], got
    assert "__otro" not in got


def test_the_quant_is_the_last_token_of_the_name():
    """A publisher prefixes the quant on every file (Q4_K_M/...-f16.gguf); the trailing token is
    the one that describes that file. A sharded name must not read "of-00002" as a quant."""
    assert repo("Q4_K_M/model-f16.gguf").hf_ggufs("x")[0]["quant"] == "F16"
    assert repo("deepseek-v3-q4_k_m-00001-of-00002.gguf").hf_ggufs("x")[0]["quant"] == "Q4_K_M"


def test_a_directory_names_the_file_when_the_file_has_none():
    assert repo("Q4_K_M/model.gguf").hf_ggufs("x")[0]["quant"] == "Q4_K_M"


def test_mmproj_and_mtp_are_kept_out_of_the_quants():
    got = quants(repo("model-Q4_K_M.gguf", "mmproj-model-f16.gguf", "model-mtp.gguf").hf_ggufs("x"))
    assert got == ["Q4_K_M", "__mmproj", "__mtp"], got


def test_known_quants_come_first_in_the_lab_order():
    """The list reads best big to small, which is the order GGUF_PREFIXES already carries."""
    got = quants(repo("m-Q2_K.gguf", "m-IQ3_XXS.gguf", "m-Q6_K.gguf").hf_ggufs("x"))
    assert got == ["IQ3_XXS", "Q6_K", "Q2_K"], got


def test_artifacts_are_listed_only_when_asked_for():
    """A .ninfer sizes exactly like a GGUF for the Fit view, but the Convert tab offers these as
    conversion sources and a .ninfer cannot be one, so it must stay out by default."""
    tree = repo("model-Q4_K_M.gguf", "qwen38-27b-ud-q4_k_xl.ninfer")
    assert "kind" not in tree.hf_ggufs("x")[0]
    assert len(tree.hf_ggufs("x")) == 1
    with_art = tree.hf_ggufs("x", with_artifacts=True)
    assert len(with_art) == 2
    assert with_art[-1]["kind"] == "ninfer"
    # No parsed quant: the artifact's own name is what a recipe points at.
    assert with_art[-1]["quant"] is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("  ok ", name)
    print("convert: todo ok")
