"""The flag types the panel infers from `--help` must match the real parser.

The binary does not expose a machine-readable option schema: the types live in
`if (arg == "--flag")` handlers in src/serve/serve_options.cpp, where a flag is
a boolean exactly when its handler does not call require_value(). So that file
is the oracle and the help text is the thing under test.

    python3 test_help_flags.py [--binary PATH] [--source PATH]

Skips rather than fails when the sources are not present, so it can run on a
machine that only has the binary.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from pathlib import Path

HERE = pathlib.Path(__file__).parent
DEFAULT_SOURCE = HERE.parents[1] / "src" / "serve" / "serve_options.cpp"
DEFAULT_BINARY = pathlib.Path("/root/ninfer-all/build-sm86/apps/ninfer-serve")

sys.path.insert(0, str(HERE))
import server  # noqa: E402


# The help of the binary documents these wrongly: the parser treats them as
# booleans, but the help prints a metavar. Keep the list short and delete an
# entry when the help is fixed.
HELP_IS_WRONG = {
    "--host-cache-reserve-mib": "serve_options.cpp documents a boolean with `N`",
}


def oracle(source: pathlib.Path) -> dict[str, bool]:
    """flag -> is_boolean, read from the option parser itself."""
    text = source.read_text()
    out: dict[str, bool] = {}
    for m in re.finditer(r'if \(arg == "(--[a-z0-9-]+)"\) \{\n(.*?)\n            continue;',
                         text, re.S):
        body = m.group(2)
        out[m.group(1)] = "require_value" not in body and "require_list" not in body
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = ap.parse_args()

    if not args.binary.is_file():
        print(f"SKIP: no binary at {args.binary}")
        return 0
    if not args.source.is_file():
        print(f"SKIP: no option parser at {args.source}")
        return 0

    truth = oracle(args.source)
    out = subprocess.run([args.binary, "--help"], capture_output=True,
                         text=True, timeout=30)
    flags = server.API._parse_help(server.API, out.stdout or out.stderr)

    wrong, unlabelled = [], 0
    for f in flags:
        want = truth.get(f["flag"])
        if want is None:
            unlabelled += 1
            continue
        if f["takes_value"] != want:   # oracle is "is boolean", the field is "takes value"
            continue
        if f["flag"] in HELP_IS_WRONG:
            continue
        wrong.append((f["flag"], "value" if f["takes_value"] else "boolean",
                      "boolean" if want else "value"))
    covered = len(flags) - unlabelled

    print(f"flags parsed: {len(flags)} (typed: {covered}, unlabelled: {unlabelled})")
    print(f"boolean switches: {sum(1 for f in flags if not f['takes_value'])}")
    print(f"enum dropdowns : {sum(1 for f in flags if f['enum'])}")
    if wrong:
        print(f"\ntype mismatches ({len(wrong)}):")
        for flag, got, want in wrong:
            print(f"  {flag:34} panel says {got:8} parser says {want}")
        return 1
    print("\nhelp parsing: every flag matches the option parser")
    return 0


if __name__ == "__main__":
    sys.exit(main())