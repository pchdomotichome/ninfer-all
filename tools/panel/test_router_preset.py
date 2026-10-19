"""Self-check for the router preset writer.

Editing router-preset.ini in place is the one place the panel writes the
router's own config, so the round-trip and the "other sections survive"
guarantee are checked here instead of by restarting the router and hoping.

    python3 test_router_preset.py
"""

import tempfile
from pathlib import Path

from server import API

PRESET = """# Router ninfer — catálogo del lab
# A comment that must survive.

[*]
host = 127.0.0.1
port = 9081
device = 1

[alpha]
artifact = /models/alpha.ninfer
alias = first
spec = mtp
draft-tokens = 3

[beta]
artifact = /models/beta.ninfer
"""


class Stub:
    """Just the two statics plus a preset path: no config, no server."""

    def __init__(self, preset):
        self.preset = preset

    def _router_paths(self):
        return Path(self.preset).parent, Path(self.preset)

    _parse_preset = staticmethod(API._parse_preset)
    _split_preset = staticmethod(API._split_preset)
    _join_preset = staticmethod(API._join_preset)
    _router_write_model = API._router_write_model


def test_round_trip_is_byte_identical():
    chunks = API._split_preset(PRESET)
    assert API._join_preset(chunks) == PRESET, "a no-op rewrite must not touch the file"


def test_edit_alias_keeps_every_other_line():
    with tempfile.TemporaryDirectory() as tmp:
        preset = Path(tmp) / "router-preset.ini"
        preset.write_text(PRESET)
        Stub(preset)._router_write_model("alpha", "/models/alpha.ninfer", "renamed",
                                         {"spec": "mtp", "draft-tokens": "4"})
        text = preset.read_text()
        assert "alias = renamed" in text
        assert "draft-tokens = 4" in text
        assert "# A comment that must survive." in text
        assert "[*]" in text and "device = 1" in text
        assert "[beta]" in text and "artifact = /models/beta.ninfer" in text
        assert "alias = first" not in text


def test_flag_order_is_preserved():
    """A save that changes nothing must produce the same lines, in the same order.

    The file is edited by hand too; reshuffling it makes every real diff noisy.
    """
    with tempfile.TemporaryDirectory() as tmp:
        preset = Path(tmp) / "router-preset.ini"
        preset.write_text(PRESET)
        stub = Stub(preset)
        stub._router_write_model("alpha", "/models/alpha.ninfer", "first",
                             {"spec": "mtp", "draft-tokens": "3"})
        assert preset.read_text() == PRESET, "an unchanged save must be byte-identical"


def test_new_key_lands_at_the_end():
    with tempfile.TemporaryDirectory() as tmp:
        preset = Path(tmp) / "router-preset.ini"
        preset.write_text(PRESET)
        Stub(preset)._router_write_model("alpha", "/models/alpha.ninfer", "first",
                                         {"spec": "mtp", "draft-tokens": "3",
                                          "vision": "true"})
        body = preset.read_text().split("[alpha]\n", 1)[1].split("[", 1)[0]
        keys = [ln.split("=", 1)[0].strip() for ln in body.splitlines() if "=" in ln]
        assert keys.index("vision") == len(keys) - 1, keys


def test_new_section_is_appended():
    with tempfile.TemporaryDirectory() as tmp:
        preset = Path(tmp) / "router-preset.ini"
        preset.write_text(PRESET)
        Stub(preset)._router_write_model("gamma", "/models/gamma.ninfer", "g", {"vision": "true"})
        text = preset.read_text()
        assert text.index("[gamma]") > text.index("[beta]")
        assert API._join_preset(API._split_preset(text)) == text


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("  ok ", name)
    print("router preset: todo ok")
