#!/usr/bin/env python3
"""
Benchmark engine: starts AIPerf, follows its progress, archives and records.

No dependencies. Reads config.toml with tomllib, runs AIPerf's venv as a
subprocess and follows its exports. The registry is an appended JSONL: every row
carries the profile it ran under, so the ranking never mixes incomparable runs
(comparing requires the same profile, or an explicit crossover).

    bench.py run --profile spec --dataset spec_al_gsm8k --concurrency 2
    bench.py runs --profile spec
    bench.py profiles
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.toml"

# The spec-decode metrics the `ninfer` adapter leaves behind. The names carry
# AIPerf's own spec_decode_ prefix; when the engine does not emit
# metrics.speculative_decoding (llama.cpp, or ninfer without the patch) they are
# absent and stay null instead of breaking the ingest.
SPEC_FIELDS = ("spec_decode_acceptance_length", "spec_decode_draft_acceptance_rate",
               "spec_decode_accepted_per_verified",
               "spec_decode_token_weighted_acceptance_length",
               "spec_decode_draft_tokens", "spec_decode_accepted_draft_tokens",
               "spec_decode_steps")


def scalar(value) -> float | None:
    """
    Flatten the three shapes AIPerf uses for a number.

    In the summary export a metric is {"avg":..,"p50":..}; in a per-request record it
    is {"value":..,"unit":..}; some fields come flat.
    """
    if isinstance(value, dict):
        for key in ("value", "avg"):
            if value.get(key) is not None:
                try:
                    return float(value[key])
                except (TypeError, ValueError):
                    return None
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- config

def load_config(path: Path = CONFIG) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def aiperf_ready(cfg: dict) -> tuple[bool, str]:
    """(ready, reason). A broken venv is reported, not hidden."""
    if not cfg.get("aiperf", {}).get("enabled", False):
        return False, "disabled in config.toml"
    venv = Path(cfg["aiperf"]["venv"])
    exe = venv / "bin" / "aiperf"
    if not exe.is_file():
        return False, f"{exe} does not exist"
    try:
        out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=30)
        return out.returncode == 0, (out.stdout or out.stderr).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError) as err:
        return False, str(err)


def git_rev(repo: str | None) -> str:
    if not repo or not Path(repo).is_dir():
        return ""
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


# ------------------------------------------------------------------------- server

def pid_on_port(port: int) -> int | None:
    try:
        out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"pid=(\d+)", out.stdout)
    return int(match.group(1)) if match else None


def process_argv(port: int) -> list[str]:
    """
    The server's real argv. /props does not expose the spec-decode config in
    either engine, so the command line is the source.
    """
    pid = pid_on_port(port)
    if pid is None:
        return []
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")[:-1]
    except OSError:
        return []


def spec_config(argv: list[str]) -> dict:
    """
    Normalise any engine's spec-decode config into common fields.

    Without this a llama.cpp run with --spec-draft-n-max 5 and an ninfer run with
    --draft-tokens 3 look identical in the table, and they are not: MTP depth is
    exactly the variable that explains the acceptance difference.
    """
    def value(*flags: str) -> str | None:
        for i, arg in enumerate(argv):
            if arg in flags and i + 1 < len(argv):
                return argv[i + 1]
        return None

    def has(*flags: str) -> bool:
        return any(arg in flags for arg in argv)

    backend = value("--spec")
    draft = value("--draft-tokens")
    if backend is None and (has("--spec-draft-n-max") or has("--spec-draft-p-min")):
        backend = "llama.cpp-draft"          # llama.cpp does not name the backend
        draft = value("--spec-draft-n-max")
    return {
        "spec_backend": backend or "none",
        "draft_depth": int(draft) if draft and draft.isdigit() else None,
        "lm_head_draft": has("--lm-head-draft") or None,
        "adaptive_mtp": has("--adaptive-mtp") or None,
        "draft_p_min": value("--spec-draft-p-min"),
        "spec_poll": value("--spec-draft-poll"),
    }


def server_snapshot(url: str) -> dict:
    """A snapshot of the server at run time, engine-neutral."""
    snap: dict = {"url": url}
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/props", timeout=5) as fh:
            props = json.loads(fh.read())
    except (OSError, ValueError, urllib.error.URLError):
        return snap
    params = (props.get("default_generation_settings") or {}).get("params") or {}
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or params.get("n_ctx")
    snap.update({
        "model": Path(props.get("model_path", "")).name or props.get("model_alias", ""),
        "model_path": props.get("model_path", ""),
        "alias": props.get("model_alias", ""),
        "slots": props.get("total_slots"),
        "max_context": n_ctx,
        "modalities": props.get("modalities", {}),
        "quant": props.get("model_ftype"),
        "build": props.get("build_info"),
    })
    port = int(url.rsplit(":", 1)[-1].split("/")[0]) if ":" in url else 0
    argv = process_argv(port)
    if argv:
        snap["binary"] = Path(argv[0]).name
        snap["argv"] = argv
    # In router mode the process argv has no --spec: those flags live in the
    # preset section of the model being run, and only the loaded model's own
    # argument vector carries them. Reading the process argv alone labels every
    # router run as "no spec-decode" even when the engine was drafting.
    model_argv = loaded_model_argv(url, props.get("model_alias") or
                                   Path(props.get("model_path", "")).name)
    if model_argv:
        snap["model_argv"] = model_argv
        snap.update(spec_config(model_argv))
    elif argv:
        snap.update(spec_config(argv))
    return snap


def loaded_model_argv(url: str, model: str) -> list[str]:
    """
    The argument vector the router built for one model, or [] when it is not a router.

    GET /models answers with the effective argv per catalog entry, which is where
    a router keeps --spec and --draft-tokens.
    """
    if not model:
        return []
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as fh:
            models = json.loads(fh.read()).get("data") or []
    except (OSError, ValueError, urllib.error.URLError):
        return []
    for entry in models:
        if (entry.get("id") or "") == model:
            return (entry.get("status") or {}).get("args") or []
    return []


# --------------------------------------------------------------------------- run spec

class RunSpec:
    """One run: what is measured, against which dataset, at which concurrency."""

    def __init__(self, profile: str, dataset: str, concurrency: int,
                 model: str, requests: int = 50, output_tokens: int = 512,
                 url: str = "", extra: list[str] | None = None,
                 tokenizer: str | None = None):
        self.profile = profile
        self.dataset = dataset
        self.concurrency = concurrency
        self.model = model
        self.requests = requests
        self.output_tokens = output_tokens
        self.url = url
        self.extra = extra or []
        self.tokenizer = tokenizer  # None → argv usa el del cfg
        # The run_id is frozen here: as a property re-stamping now() on every access,
        # the directory and the registry ended up with different run_ids (bug
        # 2026-10-11: the UI showed T054148 and the ranking stored T054206).
        self._run_id = (f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
                        f"-{self.profile}-{self.dataset}-c{self.concurrency}")

    def resolved_url(self, cfg: dict) -> str:
        return self.url or cfg["ninfer"]["url"]

    def capture_server(self, cfg: dict) -> dict:
        """
        A snapshot of the server taken BEFORE the run, stored beside the artifacts.

        It goes into the run directory and not only into the registry: that way
        `reingest` can rebuild an old row even after the server has changed.
        """
        return server_snapshot(self.resolved_url(cfg))

    @property
    def run_id(self) -> str:
        return self._run_id

    def argv(self, cfg: dict) -> list[str]:
        ap = cfg["aiperf"]
        # The run's own tokenizer, falling back to the one from the config.
        tok = self.tokenizer or ap["tokenizer"]
        argv = [
            str(Path(ap["venv"]) / "bin" / "aiperf"), "profile",
            "--model", self.model,
            # AIPerf requires an explicit, local tokenizer; without this it aborts.
            "--tokenizer", tok,
            "--endpoint-type", "chat",
            "--url", self.resolved_url(cfg),
            # Without --streaming there is no time_to_first_token: the ranking would lose
            # the TTFT column.
            "--streaming",
            "--request-count", str(self.requests),
            "--concurrency", str(self.concurrency),
            "--output-tokens-mean", str(self.output_tokens),
            "--public-dataset", self.dataset,
            "--export-level", "records",
            # Fixed arrival grid: with poisson/gamma the effective concurrency varies
            # between runs and two records stop being comparable row by row.
            "--request-rate-mode", "constant",
        ]
        argv += self.extra
        return argv


# --------------------------------------------------------------------------- runner

class Runner:
    """The AIPerf subprocess with a queryable state. The panel reads it over SSE."""

    def __init__(self, spec: RunSpec, cfg: dict, run_dir: Path):
        self.spec = spec
        self.cfg = cfg
        self.run_dir = run_dir
        self.proc: subprocess.Popen | None = None
        self.log_lines: list[str] = []
        self.error = ""
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        argv = self.spec.argv(self.cfg) + ["--output-artifact-dir", str(self.run_dir)]
        (self.run_dir / "cmd.txt").write_text(" ".join(argv) + "\n")
        self.proc = subprocess.Popen(
            argv, cwd=self.run_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self.run_dir.as_posix()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            with self._lock:
                self.log_lines.append(line.rstrip("\n"))
                del self.log_lines[:-500]
        self.proc.wait()

    def log(self) -> list[str]:
        with self._lock:
            return list(self.log_lines)

    def progress(self) -> dict:
        """Partials from the jsonl AIPerf writes during the run."""
        partial = {"records": 0, "latency_avg": None, "out_tps_avg": None, "done": False,
                   "phase": "starting", "log_tail": ""}
        # Before the first request there is no jsonl to read (dataset downloading,
        # warming up). Without this the UI silently shows "0 requests" and there is no
        # way to tell whether it downloaded, hung or is failing.
        with self._lock:
            if self.log_lines:
                last = [ln for ln in self.log_lines if ln.strip()]
                if last:
                    partial["log_tail"] = last[-1].strip()[:200]
        path = self.run_dir / "profile_export.jsonl"
        if path.exists():
            lat, tps, n = 0.0, 0.0, 0
            try:
                with open(path) as fh:
                    for raw in fh:
                        try:
                            metrics = json.loads(raw).get("metrics", {})
                        except json.JSONDecodeError:
                            continue  # a half-written line
                        lat += scalar(metrics.get("request_latency")) or 0.0
                        # output_token_throughput does not exist per request; the per-request
                        # equivalent is e2e_output_token_throughput.
                        tps += (scalar(metrics.get("output_token_throughput"))
                                or scalar(metrics.get("e2e_output_token_throughput")) or 0.0)
                        n += 1
            except OSError:
                pass
            if n:
                partial.update(records=n, latency_avg=lat / n, out_tps_avg=tps / n)
        if not partial["records"] and partial["log_tail"]:
            # There is activity (dataset downloading, warming up) but no measured request
            # yet: this is not "starting", it is "preparing".
            partial["phase"] = "preparing"
        elif partial["records"]:
            partial["phase"] = "measuring"
        with self._lock:
            partial["done"] = self.proc is not None and self.proc.poll() is not None
            if partial["done"]:
                partial["phase"] = "finished" if partial["records"] else "failed"
        return partial

    def stop(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        os.killpg(os.getpgid(self.proc.pid), 15)  # grupo entero: aiperf usa ZMQ
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.proc.pid), 9)
            self.proc.wait(timeout=10)


# --------------------------------------------------------------------------- ingest

def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def ingest(run_dir: Path, spec: RunSpec, cfg: dict) -> dict | None:
    """Turn AIPerf's exports into one registry row."""
    export = run_dir / "profile_export_aiperf.json"
    if not export.exists():
        return None
    try:
        data = json.loads(export.read_text())
    except json.JSONDecodeError:
        return None
    # The server snapshot is taken before the run and lives beside the artifacts,
    # so it survives a later configuration change.
    try:
        server = json.loads((run_dir / "server.json").read_text())
    except (OSError, json.JSONDecodeError):
        server = {}

    def avg(block: str, key: str = "avg"):
        return (data.get(block) or {}).get(key)

    # AIPerf already pools the spec-decode metrics into the summary export, with
    # the right weighting. Only when the summary lacks them (cancelled run, partial
    # export) are the individual records averaged, as a safety net.
    spec_pool: dict[str, float | None] = {f: avg(f) for f in SPEC_FIELDS}
    records = run_dir / "profile_export.jsonl"
    if all(v is None for v in spec_pool.values()) and records.exists():
        pool: dict[str, list[float]] = {f: [] for f in SPEC_FIELDS}
        with open(records) as fh:
            for raw in fh:
                try:
                    metrics = json.loads(raw).get("metrics", {})
                except json.JSONDecodeError:
                    continue
                for field in SPEC_FIELDS:
                    value = scalar(metrics.get(field))
                    if value is not None:
                        pool[field].append(value)
        spec_pool = {f: _mean(v) for f, v in pool.items()}

    # The exact command stays on the row: it is what makes a run reproducible.
    try:
        aiperf_cmd = (run_dir / "cmd.txt").read_text().strip()
    except OSError:
        aiperf_cmd = ""

    entry = {
        "run_id": spec.run_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": spec.profile,          # ← la llave de comparabilidad
        "dataset": spec.dataset,
        "model": spec.model,
        "concurrency": spec.concurrency,
        # Everything that makes two runs incomparable travels with the row.
        # The ranking groups by profile; within a profile, if these three differ the
        # comparison is misleading and has to be said so.
        "request_count_target": spec.requests,
        "output_tokens_mean": spec.output_tokens,
        "request_rate_mode": "constant",
        "request_count": avg("request_count"),
        "completed": avg("completed_request_count"),
        "error_rate": avg("request_error_rate"),
        "duration_s": avg("benchmark_duration"),
        "isl_avg": avg("input_sequence_length"),
        "osl_avg": avg("output_sequence_length"),
        "latency_avg": avg("request_latency"),
        "latency_p50": avg("request_latency", "p50"),
        "latency_p95": avg("request_latency", "p95"),
        "latency_p99": avg("request_latency", "p99"),
        "ttft_avg": avg("time_to_first_token"),
        "ttft_p99": avg("time_to_first_token", "p99"),
        "out_tps_avg": avg("output_token_throughput"),
        # Prefill and decode are different bottlenecks: an artifact with slow prefill and
        # fast decode is not the same problem as the reverse.
        "prefill_tps": avg("effective_prefill_throughput"),
        "total_tps": data.get("total_token_throughput"),
        "req_per_s": avg("request_throughput"),
        # effective_concurrency arrives as {avg, p50, ...}: without flattening it the
        # UI prints [object Object].
        "effective_concurrency": scalar(data.get("effective_concurrency")),
        "aiperf_version": data.get("aiperf_version"),
        "ninfer_git": git_rev(cfg.get("ninfer", {}).get("repo")),
        "ninfer_binary": cfg.get("ninfer", {}).get("binary", ""),
        "url": spec.resolved_url(cfg),
        # What the server was doing: engine, slots, context and the normalised
        # spec-decode config. Without it MTP5 and MTP3 look identical.
        "server": server,
        "tokenizer": spec.tokenizer or cfg["aiperf"]["tokenizer"],
        "artifacts_dir": run_dir.as_posix(),
        # Histogram pooled by position: {"0": n, "1": n, ...} over the acceptance
        # positions. It is what shows the per-position decay in the UI; without it only
        # the average survives.
        "acceptance_histogram": data.get("pooled_spec_decode_acceptance_histogram"),
        "aiperf_cmd": aiperf_cmd,
        **spec_pool,
    }
    return entry


def append_registry(registry: Path, entry: dict) -> None:
    registry.parent.mkdir(parents=True, exist_ok=True)
    with open(registry, "a") as fh:          # append-only: nunca se reescribe
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def list_runs(registry: Path, profile: str | None = None) -> list[dict]:
    if not registry.exists():
        return []
    runs = []
    for raw in registry.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue                      # a half-written line of a live run
        if profile and entry.get("profile") != profile:
            continue
        runs.append(entry)
    return runs


# --------------------------------------------------------------------------- cli

def cmd_run(args, cfg: dict) -> int:
    ok, why = aiperf_ready(cfg)
    if not ok:
        print(f"AIPerf not available: {why}", file=sys.stderr)
        return 2

    profile = next((b for b in cfg.get("benchmarks", []) if b["id"] == args.profile), None)
    if not profile:
        print(f"unknown profile: {args.profile}", file=sys.stderr)
        return 2
    if args.dataset not in profile["datasets"] and not args.force_dataset:
        print(f"dataset '{args.dataset}' is not in profile '{args.profile}'.\n"
              f"  del perfil: {', '.join(profile['datasets'])}\n"
              f"  (--force_dataset to use it anyway; the run stays in its own profile)",
              file=sys.stderr)
        return 2

    spec = RunSpec(args.profile, args.dataset, args.concurrency, args.model,
                   requests=args.requests, output_tokens=args.output_tokens,
                   url=args.url, extra=args.flag)
    run_dir = Path(cfg["aiperf"]["runs_dir"]) / spec.run_id
    # The snapshot goes first: if the server is not up the run fails either way, but
    # the record keeps why.
    server = spec.capture_server(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "server.json").write_text(json.dumps(server, indent=2))
    runner = Runner(spec, cfg, run_dir)
    runner.start()
    print(f"run_id     {spec.run_id}")
    print(f"directory {run_dir}")
    print(f"server    {server.get('binary', '?')} · "
          f"spec {server.get('spec_backend', '?')}"
          f"{'/' + str(server['draft_depth']) if server.get('draft_depth') else ''} · "
          f"{server.get('slots', '?')} carriles · ctx {server.get('max_context', '?')}")

    last = 0.0
    while runner.proc.poll() is None:
        time.sleep(1.0)
        p = runner.progress()
        if p["records"] != last and p["records"]:
            last = p["records"]
            print(f"  {p['records']:>4} req  lat {p['latency_avg']:7.1f} ms  "
                  f"{p['out_tps_avg']:6.1f} tok/s", flush=True)
    code = runner.proc.returncode

    entry = ingest(run_dir, spec, cfg)
    if entry:
        append_registry(Path(cfg["aiperf"]["registry"]), entry)
        def num(v):
            return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"
        print(f"\nregistrado. TTFT {num(entry['ttft_avg'])} ms · "
              f"p95 {num(entry['latency_p95'])} ms · "
              f"out {num(entry['out_tps_avg'])} tok/s · err {num(entry['error_rate'])}%")
    else:
        print("\nno export: the run never got to writing results", file=sys.stderr)
        for line in runner.log()[-15:]:
            print(f"  {line}", file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(prog="bench.py", description=__doc__)
    ap.add_argument("--config", type=Path, default=CONFIG)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a benchmark")
    run.add_argument("--profile", required=True)
    run.add_argument("--dataset", default="")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--model", default="")
    run.add_argument("--requests", type=int, default=50)
    run.add_argument("--output-tokens", type=int, default=512)
    run.add_argument("--url", default="")
    run.add_argument("--flag", action="append", default=[])
    run.add_argument("--force_dataset", action="store_true")
    run.set_defaults(fn=lambda a, c: cmd_run(a, c))

    runs = sub.add_parser("runs", help="list the registry")
    runs.add_argument("--profile", default="")
    runs.add_argument("--json", action="store_true")
    runs.set_defaults(fn=lambda a, c: cmd_runs(a, c))

    sub.add_parser("reingest", help="rebuild the registry from the artifacts").set_defaults(
        fn=lambda a, c: cmd_reingest(a, c))

    sub.add_parser("profiles", help="listar perfiles y datasets").set_defaults(
        fn=lambda a, c: cmd_profiles(c))

    args = ap.parse_args(argv)
    return args.fn(args, cfg)


def cmd_reingest(args, cfg: dict) -> int:
    """
    Rebuild the registry from the artifacts already on disk.

    The registry schema changes (new metrics, renamed fields) and old runs stored
    the previous version. Reingesting fixes them without spending GPU re-running
    what was already measured.
    """
    registry = Path(cfg["aiperf"]["registry"])
    runs = list_runs(registry)
    rewritten = []
    for old in runs:
        run_dir = Path(old.get("artifacts_dir", ""))
        if not run_dir.is_dir():
            continue                      # deleted artifacts: left as they are
        spec = RunSpec(old["profile"], old["dataset"], old["concurrency"], old["model"],
                       requests=old.get("request_count_target") or 50,
                       output_tokens=old.get("output_tokens_mean") or 512,
                       url=old.get("url", ""))
        fresh = ingest(run_dir, spec, cfg)
        if not fresh:
            continue
        # run_id and ts are identity, not measurement: they come from the real run.
        fresh["run_id"], fresh["ts"] = old["run_id"], old["ts"]
        rewritten.append(fresh)
    if not rewritten:
        print("nada que reingerir (no hay artifacts en disco)")
        return 0
    with open(registry, "w") as fh:       # reconstruye el append-only
        for entry in rewritten:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    print(f"{len(rewritten)} runs reingested; current schema applied")
    return 0


def cmd_runs(args, cfg: dict) -> int:
    runs = list_runs(Path(cfg["aiperf"]["registry"]), args.profile or None)
    if args.json:
        print(json.dumps(runs, indent=2))
        return 0
    if not runs:
        print("empty registry")
        return 0
    head = f"{'run_id':<46} {'ds':<18} {'c':>2} {'p95 ms':>8} {'ttft ms':>8} {'tok/s':>7}"
    print(head)
    print("-" * len(head))
    for r in runs:
        def num(v):
            return f"{v:8.1f}" if isinstance(v, (int, float)) else " " * 8
        print(f"{r['run_id']:<46} {r['dataset']:<18} {r['concurrency']:>2} "
              f"{num(r['latency_p95'])} {num(r['ttft_avg'])} {num(r['out_tps_avg'])}")
    return 0


def cmd_profiles(cfg: dict) -> int:
    for b in cfg.get("benchmarks", []):
        print(f"{b['id']:<10} {b['label']}")
        print(f"{'':<10} datasets: {', '.join(b['datasets'])}")
        if b.get("concurrency"):
            print(f"{'':<10} concurrencia: {b['concurrency']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
