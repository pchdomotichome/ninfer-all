"""Self-check for the Fit view.

Two things can go wrong quietly here.

fit-data.json is a copy of measurements that live in docs/config-calculator.html. A rerun of
those sweeps updates the page and not the copy, and the panel would keep answering with
yesterday's constants. The first test therefore re-reads the page and compares.

The arithmetic that combines them lives in js/fit.js. A wrong constant there does not raise --
it produces a confident wrong verdict, which is the failure mode this whole view exists to
avoid. The rest of the tests drive that arithmetic through node, including the two details that
already burned us once: the cache is reserved by the 64-token physical page rather than by the
context asked for, and DFlash2's CUDA graph allowance steps up with depth.

    python3 test_fit.py
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
PAGE = HERE.parent.parent / "docs" / "config-calculator.html"

# The DATA block is JavaScript, so JavaScript is what parses it. Keys the view does not read are
# not compared: the page's decode-vs-depth sweeps stay where they are, in the page.
EXTRACT = r"""
const fs = require('node:fs');
const [, , page] = process.argv;   // run as a file: argv[0] is node, [1] the script
const src = fs.readFileSync(page, 'utf8');
const at = src.indexOf('const DATA = {');
const body = src.slice(at, src.indexOf('\n};', at) + 3);
const DATA = new Function(body + '; return DATA;')();
const conv = o => Array.isArray(o) ? o.map(conv)
  : (o && typeof o === 'object') ? Object.fromEntries(Object.entries(o).map(([k, v]) => [k, conv(v)]))
  : (o === Infinity ? null : o);
const d = conv(DATA);
const models = {};
for (const [k, m] of Object.entries(d.models)) {
  models[k] = {label: m.label, weightsBytes: m.weightsBytes, workspaceBytes: m.workspaceBytes,
               nativeMaxContext: m.nativeMaxContext, kv: m.kv, spec: m.spec};
}
console.log(JSON.stringify({cardMiB: d.cardMiB, kvLabels: d.kvLabels, kvNotes: d.kvNotes,
                            perplexity: d.perplexity, models}));
"""

# The arithmetic, driven where it lives. Every case below is a number the calculator's method
# notes predict, not a golden value copied out of a run.
CHECKS = r"""
import fs from 'node:fs';
const [dataPath, jsUrl] = process.argv.slice(1);   // -e: argv[0] is node, [1] the first arg
const data = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
const F = await import(jsUrl);
const GIB = 1024 ** 3;
const bad = [];
const eq = (what, got, want) => { if (got !== want) bad.push(`${what}: got ${got}, want ${want}`); };
const ok = (what, cond) => { if (!cond) bad.push(`${what}: not true`); };
const p27 = data.models['27b'], p35 = data.models['35b'];

// The cache is reserved by the whole physical page the context rounds up to.
eq('pageRoundUp(1)', F.pageRoundUp(1, 64), 64);
eq('pageRoundUp(64)', F.pageRoundUp(64, 64), 64);
eq('pageRoundUp(65)', F.pageRoundUp(65, 64), 128);

// A flat allowance, and the stepped one: DFlash2 asks for 288 MiB up to 8,192 and 384 MiB past
// it, so a value between measured points takes the higher of the two.
eq('graph flat', F.graphBytes(p27.spec.mtp3, 131072), 90177536);
eq('graph at its step', F.graphBytes(p27.spec['dflash2-7'], 8192), 301989888);
eq('graph past its step', F.graphBytes(p27.spec['dflash2-7'], 8193), 402653184);
eq('graph far past it', F.graphBytes(p27.spec['dflash2-7'], 1000000), 402653184);

// Per-token cost: MTP's extra draft tokens are buffered in the KV format, so its term is a
// ratio; DFlash pays a fixed 4,096 B/token regardless of the format.
ok('mtp3 ratio scales the format', F.kvPerToken(p27, 'bf16', 'mtp3') > F.kvPerToken(p27, 'int8', 'mtp3'));
eq('dflash adds a flat scratch', F.kvPerToken(p35, 'int8', 'dflash-3'), p35.kv['int8'] + 4096);
eq('none is the format alone', F.kvPerToken(p27, 'int8', 'none'), p27.kv['int8']);

const usable = 23 * GIB, wb = p27.weightsBytes;
const b = F.budget(data, p27, 'mtp3', 'int8', 32768, wb, usable);
eq('the parts are the total', b.total,
   b.weightsBytes + b.kvBytes + b.seqBytes + b.workspaceBytes);
eq('room is what is left', b.free, usable - b.total);

// The reported context has to be achievable and the largest one that is: it lands on a page
// boundary, it fits, and one page more does not.
const quiet = F.maxContext(data, p27, 'mtp3', 'int8', wb, usable);
ok('max context is a page boundary', quiet % 64 === 0);
const at = F.budget(data, p27, 'mtp3', 'int8', quiet, wb, usable);
ok('max context fits', at.total <= usable);
ok('and one page more does not',
   F.budget(data, p27, 'mtp3', 'int8', quiet + 64, wb, usable).total > usable);

// Room is what changes the answer, and the model's own ceiling caps it.
ok('a shallower format reaches further',
   F.maxContext(data, p27, 'mtp3', 'rk4v4', wb, usable) >= quiet);
eq('a huge card hits the native ceiling',
   F.maxContext(data, p27, 'mtp3', 'int8', wb, 4096 * GIB), p27.nativeMaxContext);

// Weights that do not fit cannot be rescued by any context.
const over = F.budget(data, p27, 'none', 'int8', 32768, 200 * GIB, usable);
ok('an oversized artifact is over budget', over.free < 0);

// The profile decides every term above, so a name that matches two measured profiles, or none,
// must not silently resolve to one: the view says so instead.
eq('a named profile is used', F.profileFor(data, {id: 'qwen38-27b-iq3s'}), '27b');
eq('an unmeasured model resolves to nothing',
   F.profileFor(data, {id: 'qwen38-flash-next-128k'}), null);
eq('an ambiguous name resolves to nothing', F.profileFor(data, {id: 'a27bb35b'}), null);
eq('no model resolves to nothing', F.profileFor(data, null), null);

console.log(JSON.stringify(bad));
"""


def _node(script: str, *args: str) -> str:
    out = subprocess.run(["node", "--input-type=module", "-e", script, *args],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, f"node failed:\n{out.stderr[:2000]}"
    return out.stdout.strip()


def test_constants_still_match_the_calculator():
    """fit-data.json is a copy; this is the copy's whole guarantee."""
    data = json.loads((HERE / "fit-data.json").read_text())
    assert "config-calculator.html" in data["source"]["file"], "provenance must name the source"
    assert "ninfer" in data["source"]["project"], "provenance must name the project"

    with tempfile.TemporaryDirectory() as tmp:
        extract = Path(tmp) / "extract.cjs"
        extract.write_text(EXTRACT)
        out = subprocess.run(["node", str(extract), str(PAGE)],
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, f"node failed:\n{out.stderr[:2000]}"
        live = json.loads(out.stdout)

    for key in ("cardMiB", "kvLabels", "kvNotes", "perplexity"):
        assert data[key] == live[key], f"{key} drifted from {PAGE.name}"
    assert data["pageTokens"] == 64, "the page size comes from kPagedKVPageSize"
    assert data["kvOrder"] == ["bf16", "int8", "fp8", "rk8v4", "rk4v4", "k8v4", "nvfp4"]

    assert set(data["models"]) == set(live["models"]), "a measured profile appeared or vanished"
    for name, lived in live["models"].items():
        mine = data["models"][name]
        assert mine == lived, f"profile {name} drifted from {PAGE.name}"
        # The view picks its formats from kvOrder and its specs from the profile, so a hole in
        # either shows as a missing option rather than as a wrong number.
        assert set(mine["kv"]) <= set(data["kvOrder"]), f"{name} has a format kvOrder lacks"
        assert mine["spec"]["none"], f"{name} needs a 'none' spec to measure the heads against"


def test_the_arithmetic():
    with tempfile.TemporaryDirectory() as tmp:
        check = Path(tmp) / "check.mjs"
        check.write_text(CHECKS)
        # -e rather than the file: the checks are ESM and this way they need no extension dance.
        out = _node(check.read_text(), str(HERE / "fit-data.json"),
                    (HERE / "js" / "fit.js").as_uri())
    bad = json.loads(out)
    assert not bad, "js/fit.js disagrees with its method notes:\n  " + "\n  ".join(bad)


def test_every_control_it_binds_exists():
    """A renamed id makes the view render nothing at all, and nowhere does it say why: the
    first `$('#...')` that comes back null throws, the click handler's promise is dropped, and
    the section just stays empty. Cheaper to catch here."""
    js = (HERE / "js" / "fit.js").read_text()
    html = (HERE / "index.html").read_text()
    ids = set(re.findall(r"\$\('#([a-z0-9-]+)'\)", js))
    assert len(ids) > 8, f"only found {sorted(ids)}: the selector convention changed"
    missing = sorted(i for i in ids if f'id="{i}"' not in html)
    assert not missing, f"index.html has no id for {missing}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("  ok ", name)
    print("fit: todo ok")
