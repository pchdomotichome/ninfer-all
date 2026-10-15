#!/usr/bin/env python3
"""
The ninfer panel: HTTP server and API. No dependencies.

    python3 tools/panel/server.py            # reads config.toml

Serves the UI and exposes the benchmark registry. The AIPerf jobs live in this
process: they are started, their progress is followed over SSE, and they are
ingested when they finish.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import bench
import shutil

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.toml"

MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".json": "application/json"}


class Jobs:
    """
    Live runs. One lock because the handler is threaded and the SSE is several
    connections reading at once.
    """

    def __init__(self):
        self._runs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def add(self, run_id: str, state: dict):
        with self._lock:
            self._runs[run_id] = state

    def get(self, run_id: str) -> dict | None:
        with self._lock:
            return self._runs.get(run_id)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(v) for v in self._runs.values()]


class Baselines:
    """
    Normal decode speed per model, and the alerts derived from it.

    The baseline is learned from the cumulative counters of /v1/load -- the server's
    AGGREGATE -- and not from each request's wall time. With --max-concurrency N the
    requests share the decode batch, so one request's tok/s depends on how many run
    beside it: using it as a baseline would raise false positives as soon as the
    server is used in parallel. The aggregate does not depend on how the work is
    split.

    It is the failure the dashboard exists to catch: if the GPU drops out the engine
    still answers 200 and generates ten or twenty times slower.
    """

    WINDOW = 300          # samples retained per model (robust median)
    MIN_SAMPLES = 12      # antes de esto no se alerta contra la referencia
    RATIO = 0.25          # bajo el 25% de la mediana -> sospechoso
    STREAK = 3            # seguidos, porque uno lento no significa nada
    ABSOLUTE = 5.0        # tok/s: rule used when there is no baseline yet
    SAMPLE_EVERY = 2.0    # segundos entre muestras
    SAVE_EVERY = 30.0

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.rates: dict = {}          # model -> [tok/s]
        self.last: tuple | None = None  # (tokens, t) del muestreo anterior
        self.streak = 0
        self.slow_streak = 0
        self.alerts: list[dict] = []
        self.down_polls = 0
        self._last_save = 0.0
        self._stop = threading.Event()
        self._load()

    # ------------------------------------------------------------ persistence
    def _load(self):
        try:
            saved = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        with self.lock:
            for mid, entry in (saved.get("models") or {}).items():
                rates = [float(x) for x in (entry or []) if isinstance(x, (int, float))]
                if rates:
                    self.rates[mid] = rates[-self.WINDOW:]

    def _save(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_save < self.SAVE_EVERY:
            return
        self._last_save = now
        with self.lock:
            payload = {"version": 1,
                       "saved_unix_ms": int(now * 1000),
                       "models": {k: v[-self.WINDOW:] for k, v in self.rates.items()}}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload))
        except OSError:
            pass

    # ---------------------------------------------------------------- sampling
    @staticmethod
    def _median(xs: list) -> float:
        s = sorted(xs)
        n = len(s)
        if not n:
            return 0.0
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    def start(self, url: str):
        """
        Start sampling. It runs whether or not anyone has the panel open: an alert
        that only exists while I am looking at the page is not an alert.

        Reads only /v1/load, not the page's full reader: that is four endpoints plus
        the residency of every model, and this runs every 2 s forever.
        """
        def once() -> dict:
            try:
                with urllib.request.urlopen(url.rstrip("/") + "/v1/load", timeout=4) as fh:
                    return {"reachable": True, "load": json.loads(fh.read())}
            except urllib.error.HTTPError:
                # The router answered: no model is loaded, which is idle.
                return {"reachable": True, "load": {"error": "no model loaded"}}
            except (OSError, ValueError):
                return {"reachable": False, "load": None}

        def loop():
            while not self._stop.wait(self.SAMPLE_EVERY):
                try:
                    self.feed(once())
                except Exception:
                    continue
            self._save(force=True)

        threading.Thread(target=loop, daemon=True, name="baselines").start()

    def stop(self):
        self._stop.set()

    def feed(self, snap: dict, now: float | None = None):
        """Un muestreo. `snap` es {"reachable": bool, "load": dict|None}.

        `now` es inyectable para que las reglas se puedan ejercitar sin esperar
        el reloj: ver test_baselines.py."""
        now = time.time() if now is None else now
        # "Down" means it answers nothing at all. A 404 on /v1/load is the idle
        # router, not a fault: confusing the two would raise a false alert while the
        # server runs fine.
        reachable = bool(snap.get("reachable"))
        load = snap.get("load") or {}
        counters = load.get("counters") or {}
        model = load.get("model") or ""
        tokens = counters.get("committed_decode_tokens")

        self.down_polls = 0 if reachable else self.down_polls + 1
        self._raise("server_unreachable", "the server is not answering",
                    "no answer for %ds" % round(self.down_polls * self.SAMPLE_EVERY),
                    "ok" if reachable else ("critical" if self.down_polls >= 4 else "warning"))
        if not reachable or not model or tokens is None:
            with self.lock:
                self.last = None
            return

        prev = self.last
        self.last = (tokens, now)
        if not prev:
            return
        dtok, dt = tokens - prev[0], now - prev[1]
        # Only while requests are in flight: at rest the rate is zero by definition and
        # says nothing about the model's speed.
        if dt <= 0 or dtok <= 0 or not (load.get("requests") or {}).get("running"):
            return
        rate = dtok / dt
        if rate <= 0 or rate > 100000:
            return

        with self.lock:
            series = self.rates.setdefault(model, [])
            series.append(rate)
            if len(series) > self.WINDOW:
                del series[:-self.WINDOW]
            n, median = len(series), self._median(series)

        # Before a baseline exists the only possible rule is "this is not inference
        # speed": 5 tok/s is the floor of a healthy GPU.
        self.slow_streak = self.slow_streak + 1 if rate < self.ABSOLUTE else 0
        self._raise("decode_suspiciously_slow", "decode sospechosamente lento",
                    "%.1f tok/s" % rate, "serious" if self.slow_streak >= self.STREAK else "ok")

        if n < self.MIN_SAMPLES:
            return
        threshold = median * self.RATIO
        self.streak = self.streak + 1 if rate < threshold else 0
        self._raise("decode_below_baseline", "decode muy por debajo de su norma",
                    "%.1f tok/s · baseline %.1f (median of %d samples, %s)"
                    % (rate, median, n, model),
                    "critical" if self.streak >= self.STREAK else "ok")
        self._save()

    # ---------------------------------------------------------------- alerts
    def _raise(self, key: str, title: str, detail: str, level: str):
        with self.lock:
            prev = next((a for a in self.alerts if a["key"] == key), None)
            # "ok" is an internal state: when the rule does not fire, the alert closes.
            if level == "ok":
                if prev:
                    self.alerts.remove(prev)
                return
            if prev:
                prev.update(level=level, detail=detail)
                return
            self.alerts.append({"key": key, "level": level,
                                "title": title, "detail": detail,
                                "since_unix_ms": int(time.time() * 1000)})

    def view(self) -> dict:
        with self.lock:
            return {"ok": True,
                    "alerts": sorted(self.alerts, key=lambda a: a["level"]),
                    "models": {mid: {"median_tok_s": round(self._median(series), 1),
                                     "samples": len(series)}
                               for mid, series in self.rates.items()},
                    "learning": {"min_samples": self.MIN_SAMPLES,
                                 "streak": self.streak,
                                 "needed": self.STREAK},
                    "server_down_polls": self.down_polls}


class API:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.jobs = Jobs()
        self.hub: list[callable] = []          # suscriptores SSE
        self.hub_lock = threading.Lock()
        self._gpu_ts = 0.0
        self._gpu_cache: dict = {"online": False}
        self._build_ts = 0.0
        self._build_cache: tuple = ("", {})
        self.baselines = Baselines(Path(cfg["panel"].get("state_file") or
                                        (HERE / "panel-state.json")))
        self.baselines.start(cfg["ninfer"]["url"])

    # ---------------------------------------------------------------- notices
    def publish(self, event: dict):
        with self.hub_lock:
            for push in list(self.hub):
                try:
                    push(event)
                except Exception:
                    self.hub.remove(push)

    def subscribe(self, push: callable):
        with self.hub_lock:
            self.hub.append(push)

    def unsubscribe(self, push: callable):
        with self.hub_lock:
            if push in self.hub:
                self.hub.remove(push)

    # ---------------------------------------------------------------- routes
    def unquote_text(self, v): return v.strip().strip("'\"")
    def config(self) -> dict:
        ready, why = bench.aiperf_ready(self.cfg)
        ap = self.cfg.get("aiperf", {})
        nin = self.cfg.get("ninfer", {})
        url = nin.get("url", "").rstrip("/")
        # Endpoint for clients/agents: the router's OpenAI-compatible base. An agent
        # is configured with BASE_URL + MODEL from the catalog.
        return {
            "aiperf": {"enabled": bool(ap.get("enabled")), "ready": ready, "why": why,
                       "tokenizer": ap.get("tokenizer", ""),
                       "tokenizers": ap.get("tokenizers", {})},
            "endpoint": {
                "base_url": url + "/v1",
                "api_key": nin.get("api_key") or "",
                "model": "",         # the frontend fills it from the catalog
                "curl": f"curl {url}/v1/models",
            },
            "benchmarks": [
                {"id": b["id"], "label": b["label"], "datasets": b["datasets"],
                 "concurrency": b.get("concurrency", []),
                 "note": b.get("note", "")}
                for b in self.cfg.get("benchmarks", [])
            ],
            "defaults": {"model": self.cfg.get("ninfer", {}).get("model_alias", ""),
                         "url": self.cfg.get("ninfer", {}).get("url", "")},
        }

    def engine(self) -> dict:
        """
        Server state: what is loaded and what is available to load.

        The catalog comes from /models, so the list of models to benchmark is not
        hardcoded in the panel: it is the server's. The fine-grained state
        (loaded/sleeping/suspended) comes from /residency per model, the only source
        that tells sleeping from suspended.
        """
        url = self.cfg["ninfer"]["url"]
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as fh:
                data = json.loads(fh.read())
        except (OSError, ValueError, urllib.error.URLError):
            return {"url": url, "online": False, "loaded": [], "catalog": [],
                    "build": ""}
        models = [m for m in (data.get("data") or []) if m.get("id")]
        loaded = [m["id"] for m in models if (m.get("object") or "") != "model-unloaded"]
        catalog = []
        for m in models:
            mid = m["id"]
            state = (m.get("status") or {}).get("value", "")
            # residency.state is the truth for Sleep vs Suspend: a suspended model is still
            # `loaded` in /models but is `suspended` here.
            try:
                with urllib.request.urlopen(f"{url.rstrip('/')}/v1/models/{mid}/residency",
                                            timeout=5) as fh:
                    res = json.loads(fh.read())
                if res.get("state") in ("suspended",):
                    state = "suspended"
                elif res.get("state") == "resident" and state != "sleeping":
                    state = "loaded"
            except (OSError, ValueError, urllib.error.URLError):
                pass
            if state == "loaded" and mid not in loaded:
                loaded.append(mid)
            catalog.append({"id": mid, "status": state,
                            "context": m.get("max_model_len")})
        return {"url": url, "online": True, "loaded": loaded, "catalog": catalog,
                "build": self.engine_build(url)}

    # ---------------------------------------------------------------- build identity
    def engine_build(self, url: str) -> dict:
        """
        What the server says it is, from /props, cached for a minute.

        The badge used to be a fixed "v3 · sm_86", which described the artifact
        format rather than the binary actually serving: /props reports the real
        build (for example ninfer 0.12.0-rtx3090). Cached because /props costs a
        round trip and this only feeds a header chip.
        """
        url = url.rstrip("/")
        now = time.time()
        if self._build_ts and now - self._build_ts < 60.0 and self._build_cache[0] == url:
            return self._build_cache[1]
        out = {"version": "", "arch": "", "alias": "", "model": ""}
        try:
            with urllib.request.urlopen(url + "/props", timeout=5) as fh:
                props = json.loads(fh.read())
        except (OSError, ValueError, urllib.error.URLError):
            self._build_ts, self._build_cache = 0.0, (url, out)
            return out
        build = str(props.get("build_info") or "")
        # "ninfer 0.12.0-rtx3090+v0.6.0-...-dirty" -> version 0.12.0, arch RTX3090
        out = {"version": "", "arch": "", "alias": props.get("model_alias") or "",
               "model": Path(props.get("model_path", "")).name}
        head = build.split("+", 1)[0].split()          # "ninfer", "0.12.0-rtx3090"
        if len(head) > 1 and "-" in head[1]:
            version, arch = head[1].rsplit("-", 1)
            out["version"], out["arch"] = version, arch.upper()
        elif len(head) > 1:
            out["version"] = head[1]
        self._build_ts, self._build_cache = now, (url, out)
        return out

    def convert_recipes(self) -> dict:
        """
        The real recipes from tools/convert, grouped by module.

        They are not hardcoded: they are read from each module's RECIPES registry in
        the convert venv (torch), which is where they live. If the repo adds a new
        recipe it appears in the dropdown on its own.
        """
        venv = self.cfg.get("convert", {}).get("venv", "")
        python = str(Path(venv) / "bin" / "python") if venv else sys.executable
        if not Path(python).is_file():
            return {"ok": False, "error": f"no python at {python} ([convert] venv)"}
        probe = (
            "import sys, json\n"
            "sys.path.insert(0, '.')\n"
            "from tools.convert import (official_recipes, gguf_blocks,\n"
            "    gguf_blocks_dense, gguf_blocks_moe, qwen4_exp_gguf, gemma4, ternary)\n"
            "recs = {}\n"
            "for mod in (official_recipes, gguf_blocks, gguf_blocks_dense,\n"
            "            gguf_blocks_moe, qwen4_exp_gguf, gemma4, ternary):\n"
            "    for k in getattr(mod, 'RECIPES', {}):\n"
            "        recs[k] = mod.__name__.split('.')[-1]\n"
            "print(json.dumps(recs, sort_keys=True))\n"
        )
        try:
            out = subprocess.run([python, "-c", probe], capture_output=True,
                                 text=True, timeout=60, cwd=str(HERE.parent.parent))
        except (OSError, subprocess.SubprocessError) as err:
            return {"ok": False, "error": str(err)[:200]}
        if out.returncode != 0:
            return {"ok": False, "error": (out.stderr or out.stdout).strip()[:300]}
        try:
            return {"ok": True, "recipes": json.loads(out.stdout)}
        except json.JSONDecodeError:
            return {"ok": False, "error": "the recipes probe returned something odd"}

    def convert_inspect(self, query: dict) -> tuple[int, dict]:
        """
        Read a .ninfer and return its summary: components, tensors, formats.

        tools/artifact/inspect.py is pure stdlib, so it runs as a module in a
        subprocess: it uses relative imports and does not want to pollute the panel's
        path.
        """
        path = (query.get("path") or [""])[0]
        if not path:
            return 400, {"error": "path is required"}
        if not Path(path).is_file():
            return 404, {"error": f"{path} does not exist"}
        try:
            out = subprocess.run(
                [sys.executable, "-m", "tools.artifact.inspect", path, "--json"],
                capture_output=True, text=True, timeout=60,
                cwd=str(HERE.parent.parent))
        except (OSError, subprocess.SubprocessError) as err:
            return 500, {"error": str(err)[:200]}
        if out.returncode != 0:
            return 502, {"error": (out.stderr or out.stdout).strip()[:400]}
        try:
            return 200, json.loads(out.stdout)
        except json.JSONDecodeError:
            return 502, {"error": "the inspector returned something that is not JSON"}
    def convert_run(self, body: dict) -> tuple[int, dict]:
        """
        Start a GGUF/safetensors -> .ninfer conversion with a live log.

        tools/convert needs torch: it runs in its own venv, not the panel's.
        """
        venv = self.cfg.get("convert", {}).get("venv", "")
        python = str(Path(venv) / "bin" / "python") if venv else sys.executable
        if not Path(python).is_file():
            return 503, {"error": f"no python at {python} ([convert] venv)"}
        model = body.get("model")
        recipe = body.get("recipe")
        if not model and not body.get("base"):
            return 400, {"error": "model (or base) and recipe are required"}
        if not recipe:
            return 400, {"error": "model and recipe are required"}
        out_dir = Path(self.cfg["convert"]["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        # A GGUF source does not go in as --model: the base model is the checkpoint
        # (config+tokenizer) and the gguf comes in as --source gguf=. With the right
        # source the full command is assembled; without `base` the previous path stays
        # (checkpoint as --model).
        base = body.get("base")
        if base:
            argv = [python, "-m", "tools.convert", "--model", base,
                    "--recipe", recipe]
            for name in ("gguf", "dflash2", "vision", "imatrix", "mtp", "ngram"):
                if body.get("sources", {}).get(name):
                    argv += ["--source", f"{name}={body['sources'][name]}"]
            comps = body.get("components")
            if comps:
                argv += ["--components", ",".join(comps)]
            if body.get("template"):
                argv += ["--resource", f"chat_template.jinja={body['template']}"]
        else:
            argv = [python, "-m", "tools.convert", "--model", model,
                    "--recipe", recipe]
        # The artifact's name comes from the source: the gguf if there is one, else the
        # base.
        out_src = body.get("sources", {}).get("gguf") or base or model
        out_stem = Path(out_src).stem
        argv += ["--out", str(out_dir / (out_stem + ".ninfer")), "--proposal"]
        argv += list(body.get("flag", []))
        delete_src = bool(body.get("delete_source"))
        run_dir = out_dir / f"convert-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "cmd.txt").write_text(" ".join(argv) + "\n")
        try:
            proc = subprocess.Popen(argv, cwd=str(HERE.parent.parent),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, start_new_session=True)
        except (OSError, subprocess.SubprocessError) as err:
            return 500, {"error": f"could not start: {err}"}
        state = {"run_id": run_dir.name, "proc": proc, "log": [], "status": "running"}
        self.jobs.add(run_dir.name, state)

        def watch():
            for line in proc.stdout:
                state["log"].append(line.rstrip("\n"))
                del state["log"][:-400]
                self.publish({"type": "convert", "run_id": run_dir.name,
                              "log_tail": state["log"][-15:]})
            proc.wait()
            ok = proc.returncode == 0
            deleted = self._drop_inbox(model) if (ok and delete_src) else False
            state["status"] = "done" if ok else "failed"
            self.publish({"type": "convert_done", "run_id": run_dir.name,
                          "ok": ok, "deleted": deleted,
                          "log_tail": state["log"][-25:]})

        threading.Thread(target=watch, daemon=True).start()
        return 202, {"run_id": run_dir.name}

    def _drop_inbox(self, model_path: str) -> bool:
        """
        Delete a model's downloaded staging after a successful conversion.

        It only touches what we downloaded: a source inside ninfer/inbox/<slug>/. It
        never touches anything outside that root (the user could pass any path as the
        source).
        """
        inbox_root = Path(self.cfg.get("convert", {}).get(
            "out_dir", "/mnt/models_local/ninfer/artifacts")).parent / "inbox"
        p = Path(model_path)
        if not p.is_relative_to(inbox_root) or p.parent == inbox_root:
            return False
        slug_dir = p.parent          # ninfer/inbox/<slug>/
        try:
            shutil.rmtree(slug_dir)
            return not slug_dir.exists()
        except OSError:
            return False

    def gpu(self) -> dict:
        """
        GPU figures from nvidia-smi. Cached for 2s: spawning nvidia-smi on every 2s
        poll is throwaway work, but the process costs ~50ms.
        """
        now = time.time()
        if now - self._gpu_ts < 2.0:
            return self._gpu_cache
        out = {"online": False}
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                                "temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            r = None
        if r and r.returncode == 0:
            gpus = []
            for line in r.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append({"index": int(parts[0]), "util": float(parts[1]),
                                 "mem_used_mib": float(parts[2]), "mem_total_mib": float(parts[3]),
                                 "temp_c": float(parts[4]), "power_w": float(parts[5])})
            out = {"online": bool(gpus), "gpus": gpus}
        self._gpu_ts, self._gpu_cache = now, out
        return out

    def help_flags(self) -> tuple[int, dict]:
        """
        ninfer-serve flags parsed from --help, for the recipe form. Read live: if
        the binary changes, the form changes with it.
        """
        binary = self.cfg.get("ninfer", {}).get("binary", "")
        if not binary or not Path(binary).is_file():
            return 503, {"error": f"the binary {binary} does not exist ([ninfer] binary)"}
        try:
            out = subprocess.run([binary, "--help"], capture_output=True,
                                 text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as err:
            return 500, {"error": str(err)[:200]}
        text = out.stdout or out.stderr
        flags, current = [], None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("--"):
                parts = stripped[2:].split(None, 2)
                name = parts[0]
                if name in ("help", "version"):
                    continue
                current = {"flag": "--" + name,
                           "takes_value": len(parts) > 1 and not parts[1].startswith("--"),
                           "desc": (parts[-1] if len(parts) > 1 else "").strip()}
                flags.append(current)
            elif current:
                current["desc"] += ((" " if current["desc"] else "") + stripped)
        return 200, {"flags": flags, "binary": Path(binary).name}

    def convert_roots(self) -> list[str]:
        """
        Roots to search for artifacts and sources: the config's plus HF_HOME and the
        real model directories, if they exist. Nothing guessed: every path is checked
        against the disk.
        """
        roots = []
        cfg = self.cfg.get("convert", {})
        for r in cfg.get("roots", []):
            if Path(r).is_dir():
                roots.append(r)
        for env in ("HF_HOME", "HF_CACHE", "HUGGINGFACE_HUB_CACHE"):
            if os.environ.get(env) and Path(os.environ[env]).is_dir():
                roots.append(os.environ[env])
        for r in ("/mnt/models_local/huggingface", "/mnt/models_fast/huggingface",
                  "~/.cache/huggingface"):
            p = Path(r).expanduser()
            if p.is_dir():
                roots.append(str(p))
        # dedupe preserving order
        return list(dict.fromkeys(roots))

    def convert_files(self) -> dict:
        """
        The .ninfer artifacts and (GGUF/safetensors) sources available.

        The user does not type paths by hand: the panel lists what already exists in
        the configured roots (config.toml [convert] roots) and in HF_HOME.
        """
        roots = self.convert_roots()
        artifacts, sources, checkpoints = [], [], []
        for root in roots:
            root = Path(root)
            for f in root.rglob("*"):
                if f.name.startswith(".") or f.is_symlink() and not f.is_file():
                    continue
                try:
                    if f.suffix == ".ninfer":
                        artifacts.append({"path": str(f), "size": f.stat().st_size})
                    elif f.suffix == ".gguf":
                        sources.append({"path": str(f), "size": f.stat().st_size})
                except OSError:
                    continue
        # checkpoints: directories holding config.json (the model to convert)
        for root in roots:
            for cfg_dir in Path(root).rglob("config.json"):
                if "snapshots" in str(cfg_dir):
                    continue
                try:
                    checkpoints.append({"path": str(cfg_dir.parent)})
                except OSError:
                    continue
        def sort_key(item): return item["path"]
        artifacts.sort(key=sort_key); sources.sort(key=sort_key)
        checkpoints.sort(key=sort_key)
        return {"roots": roots,
                "artifacts": artifacts[-200:],   # demo: capped at what is manageable
                "sources": sources[-200:],
                "checkpoints": checkpoints[-100:]}

    # ---------------------------------------------- Hugging Face API (0 deps)
    # Quantisation prefix order, to group the .gguf files in the selector.
    GGUF_PREFIXES = ("IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ2_XXS", "IQ2_S", "IQ1_M",
                     "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S",
                     "Q4_0", "Q3_K_M", "Q3_K_S", "Q3_K_L", "Q2_K", "BF16", "F16")
    _HF_TIMEOUT = 20.0
    _HF_BASE = "https://huggingface.co/api/models"

    def hf_search(self, query: str) -> dict:
        """
        Search HF for models. query: substring of title/author.
        Returns {hits: [...], error: ...}.
        """
        if not query or len(query) < 3:
            return {"hits": [], "error": None}
        url = self._HF_BASE + "?search=" + urllib.parse.quote(query, safe="") + "&limit=50"
        try:
            with urllib.request.urlopen(url, timeout=self._HF_TIMEOUT) as resp:
                if resp.status != 200:
                    return {"hits": [], "error": "HF answered with something other than 200"}
                hits = json.loads(resp.read().decode())
        except OSError as e:
            return {"hits": [], "error": f"could not query HF: {e}"}
        out = []
        for h in hits:
            aid = h.get("id", "")
            tags = h.get("tags", [])
            gguf = 0 if any(t == "gguf" for t in tags) else None
            if aid.startswith("openai-"):
                gguf = None
            out.append({"id": aid, "title": h.get("title") or aid,
                        "desc": h.get("library_name", ""), "gguf": gguf,
                        "likes": int(h.get("likes", 0)), "downloads": int(h.get("downloads", 0)),
                        "private": bool(h.get("private"))})
        out.sort(key=lambda x: (x["gguf"] is None, -x["likes"]))
        return {"hits": out, "error": None}

    def hf_details(self, model: str) -> list | None:
        """
        A model's files (`.gguf` and the rest) with their sizes.

        The HF API does not put `size` in `siblings`, so the default branch's tree is
        queried instead: `.../tree/{branch}`. A subdirectory's tree (e.g.
        `ggml-model-...`) needs recursion, so `?recursive=true` is used. If the repo has
        no `main` branch it falls back to the model endpoint (without sizes).
        """
        q = urllib.parse.quote(model, safe="/")
        for branch in ("main", "master"):
            try:
                with urllib.request.urlopen(
                        f"{self._HF_BASE}/{q}/tree/{branch}?recursive=true&expand=false",
                        timeout=self._HF_TIMEOUT) as resp:
                    if resp.status != 200:
                        continue
                    data = json.loads(resp.read().decode())
                    if isinstance(data, list):
                        return data
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        try:
            with urllib.request.urlopen(f"{self._HF_BASE}/{q}",
                                       timeout=self._HF_TIMEOUT) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode()).get("siblings", [])
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return None

    def hf_ggufs(self, model: str) -> list:
        """A model's GGUFs, grouped by quantisation; mmproj/mtp separately."""
        tree = self.hf_details(model)
        if not tree:
            return []
        files = []
        for f in tree:
            path = f.get("path") or f.get("rfilename") or ""
            if path.endswith(".gguf"):
                files.append({"path": path, "size": f.get("size", 0)})
        byq: dict = {"__mmproj": [], "__mtp": [], "__otro": []}
        for f in files:
            mid = f["path"].lower()
            if "mmproj" in mid:
                byq["__mmproj"].append(f)
            elif mid.startswith("mtp") or "-mtp." in mid:
                byq["__mtp"].append(f)
            else:
                q = next((k for k in self.GGUF_PREFIXES if k.lower() in mid), None)
                byq.setdefault(q if q else "__otro", []).append(f)
        groups = [{"quant": k, **f} for k in self.GGUF_PREFIXES for f in byq.get(k, [])]
        groups += [{"quant": k, **f} for k in ("__mmproj", "__mtp", "__otro") for f in byq[k]]
        return groups

    def hf_inbox(self, model: str) -> Path:
        """Staging directory for the ephemeral GGUF: ninfer/inbox/<org>--<repo>."""
        out_dir = Path(self.cfg.get("convert", {}).get("out_dir",
                        "/mnt/models_local/ninfer/artifacts"))
        slug = re.sub(r"[^A-Za-z0-9_.-]", "-", model)
        return out_dir.parent / "inbox" / slug

    def hf_bin(self) -> str:
        """The HF CLI: `hf` from huggingface_hub is preferred (already installed)."""
        found = shutil.which("hf")
        if found:
            return found
        candidate = Path.home() / ".local" / "bin" / "hf"
        return str(candidate) if candidate.is_file() else "hf"

    def hf_download(self, body: dict) -> tuple[int, dict]:
        """
        Download specific GGUFs from an HF repo into ninfer/inbox/ with progress.

        Progress is read from the disk (bytes appearing in the folder), not from
        stdout: `hf download` paints a progress bar with \r that is useless for lines.
        The download goes to inbox/ -- ephemeral staging, not to the HF cache.
        """
        model, files = body.get("model", ""), body.get("files", [])
        total = int(body.get("total", 0) or 0)
        if not model or not isinstance(files, list) or not files:
            return 400, {"error": "model and files are required"}
        dest = self.hf_inbox(model)
        dest.mkdir(parents=True, exist_ok=True)
        # Disk guard: never let a conversion leave the NVMe without headroom.
        free = shutil.disk_usage(dest).free
        need = int(total * 1.2) + (5 << 30)
        if free < need:
            return 507, {"error": f"not enough disk: {free/2**30:.0f} GiB free, "
                                    f"{need/2**30:.0f} GiB estimated (1.2x the download). "
                                    "Free space or pick another quant."}
        hf = self.hf_bin()
        argv = [hf, "download", model, *files, "--local-dir", str(dest), "--quiet"]
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, start_new_session=True)
        except (OSError, subprocess.SubprocessError) as err:
            return 500, {"error": f"could not start hf: {err}"}
        run_id = f"dl-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        state = {"run_id": run_id, "proc": proc, "status": "running", "model": model,
                 "pct": 0, "done": 0, "total": total, "files": [], "error": ""}
        self.jobs.add(run_id, state)

        def watch():
            while proc.poll() is None:
                done = sum(f.stat().st_size for f in dest.rglob("*")
                           if f.is_file()) if dest.exists() else 0
                state["done"] = done
                state["pct"] = min(100, round(done / total * 100)) if total else 100
                threading.Event().wait(1.0)
            err = proc.stdout.read() if proc.stdout else ""
            ok = proc.returncode == 0
            state["status"] = "done" if ok else "failed"
            state["error"] = err.strip()[-300:]
            state["files"] = [str(p) for p in dest.rglob("*")
                              if p.is_file() and ".cache" not in str(p)] if ok else []
            self.publish({"type": "download_done", "run_id": run_id, "ok": ok,
                          "files": state["files"], "error": state["error"],
                          "dest": str(dest)})

        threading.Thread(target=watch, daemon=True).start()
        return 202, {"run_id": run_id, "dest": str(dest)}

    def hf_download_state(self, run_id: str) -> dict | None:
        st = self.jobs.get(run_id)
        if not st or "pct" not in st:
            return None
        return {"run_id": run_id, "status": st["status"], "pct": st["pct"],
                "done": st["done"], "total": st["total"], "files": st["files"],
                "error": st["error"], "dest": str(self.hf_inbox(st["model"]))}

    # ------------------------------------------- source probe + recipe
    # (arch, moe, layers, hidden) -> official recipe (checkpoint) / gguf / local base
    SHAPE_RECIPE = {
        (False, 64, 5120): ("qwen3_8_27b", "qwen3_8_27b_gguf", "qwen38-config"),
        (False, 32, 4096): ("qwen3_5_9b", "qwen3_5_dense_gguf", "qwen35-9b"),
        (True, 40, 2048):  ("qwen3_6_35b_a3b", "qwen3_6_35b_a3b_gguf", "qwen36-35b-config"),
        (False, 32, 2560): (None, None, "qwen35-4b"),
    }
    FLASH_RECIPE = "qwen3_8_flash_next_gguf"
    GEMMA4_RECIPE = "gemma4_gguf"
    _PARAM_KEYS = ("num_hidden_layers", "hidden_size", "num_attention_heads",
                   "num_key_value_heads", "num_experts")

    def _text_config(self, cfg: dict) -> dict:
        """The text tower of a multimodal config (nested text_config)."""
        t = cfg.get("text_config") or cfg
        return t if isinstance(t, dict) else {}

    def _read_checkpoint_params(self, path: Path) -> dict:
        try:
            cfg = json.loads((path / "config.json").read_text(errors="replace"))
        except (OSError, json.JSONDecodeError):
            return {}
        t = self._text_config(cfg)
        arch = (cfg.get("architectures") or [None])[0] or ""
        return {
            "architecture": arch,
            "layers": t.get("num_hidden_layers"),
            "hidden": t.get("hidden_size"),
            "heads": t.get("num_attention_heads"),
            "kv": t.get("num_key_value_heads"),
            "moe": bool(t.get("num_experts")),
        }

    def _read_gguf_params(self, path: Path) -> dict:
        """A .gguf's metadata via the convert venv (it has gguf-py)."""
        venv = self.cfg.get("convert", {}).get("venv", "")
        python = str(Path(venv) / "bin" / "python") if venv else sys.executable
        probe = (
            "import sys, json\n"
            "from gguf import GGUFReader\n"
            "r = GGUFReader(sys.argv[1], 'r')\n"
            "f = r.fields\n"
            "# layout gguf-py: parts = [dims, nombre, tipo, valor...] — el valor\n"
            "# escalar es parts[-1]; un string es bytes(parts[-1]).\n"
            "def val(k):\n"
            "    return f[k].parts[-1].tolist()[0] if k in f else None\n"
            "def bval(k):\n"
            "    return bytes(f[k].parts[-1].tolist()).decode('utf-8', 'replace') if k in f else None\n"
            "arch = bval('general.architecture') or ''\n"
            "p = arch + '.'\n"
            "print(json.dumps({\n"
            "  'architecture': arch,\n"
            "  'layers': val(p + 'block_count'),\n"
            "  'hidden': val(p + 'embedding_length'),\n"
            "  'heads': val(p + 'attention.head_count'),\n"
            "  'kv': val(p + 'attention.head_count_kv'),\n"
            "  'moe': 1 if val(p + 'expert_count') else (1 if 'moe' in arch else 0),\n"
            "}))\n"
        )
        try:
            out = subprocess.run([python, "-c", probe, str(path)],
                                 capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return {}
        try:
            return json.loads(out.stdout)
        except json.JSONDecodeError:
            return {}

    def _local_hf_base(self, patterns: tuple) -> str | None:
        """
        A local HF snapshot (config+tokenizer) to serve as a base, by name.

        It searches the HF cache of the roots for a snapshot that is not a GGUF repo
        and has tokenizer.json: that is the '--model' the conversion needs.
        """
        for root in self.cfg.get("convert", {}).get("roots", []):
            hub = Path(root) / "hub"
            if not hub.is_dir():
                continue
            for repo in hub.glob("models--*"):
                low = repo.name.lower()
                if "-gguf" in low or not any(p in low for p in patterns):
                    continue
                for snap in repo.glob("snapshots/*"):
                    if (snap / "tokenizer.json").is_file() and (snap / "config.json").is_file():
                        return str(snap)
        return None

    def _local_base(self, moe: bool, layers, hidden) -> str | None:
        for (is_moe, n_l, h), (_, _, name) in self.SHAPE_RECIPE.items():
            if is_moe == bool(moe) and n_l == layers and h == hidden:
                p = Path(self.cfg.get("convert", {}).get(
                    "out_dir", "/mnt/models_local/ninfer/artifacts")).parent \
                    / "checkpoints" / name
                if (p / "config.json").is_file():
                    return str(p)
        return None

    def convert_probe(self, path: str) -> dict:
        """
        Read the source (checkpoint or gguf), match the recipe and assemble the flags.

        It returns everything needed to convert without knowing anything by hand: the
        recipe (or 'unknown recipe'), the local base, the available sources
        (gguf/dflash2/vision/imatrix), components and template.
        """
        p = Path(path)
        if not p.exists():
            return {"known": False, "error": f"{path} does not exist"}
        if p.is_file() and p.suffix == ".gguf":
            params, kind = self._read_gguf_params(p), "gguf"
        elif p.is_dir() and (p / "config.json").is_file():
            params, kind = self._read_checkpoint_params(p), "checkpoint"
        else:
            return {"known": False, "error": "fuente no es checkpoint (config.json) ni .gguf"}

        out = {"kind": kind, "params": params, "known": False,
               "recipe": None, "recipe_kind": None, "base": None,
               "sources": {}, "components": ["text"], "template": None,
               "candidates": []}

        moe, layers, hidden = bool(params.get("moe")), params.get("layers"), params.get("hidden")
        arch_l = (params.get("architecture") or "").lower()
        flash = "qwen4exp" in arch_l
        gemma4 = "gemma4" in arch_l
        if flash:
            # Flash-Next: gguf recipe only; a local base does not cover it (other
            # architecture).
            out["recipe"], out["recipe_kind"] = self.FLASH_RECIPE, "gguf"
            out["known"] = True
        elif gemma4:
            # Gemma 4: gguf recipe (QAT Q4_0); the base is the HF repo (config+tokenizer).
            out["recipe"] = self.GEMMA4_RECIPE if kind == "gguf" else None
            out["recipe_kind"] = "gguf"
            out["known"] = bool(out["recipe"])
            out["base"] = self._local_hf_base(("gemma-4-26b-a4b", "gemma-4-12b", "gemma-4"))
        else:
            for (is_moe, n_l, h), (cp_recipe, gguf_recipe, base_name) in self.SHAPE_RECIPE.items():
                if is_moe == moe and n_l == layers and h == hidden:
                    out["recipe"] = gguf_recipe if kind == "gguf" else cp_recipe
                    out["recipe_kind"] = kind
                    out["known"] = bool(out["recipe"])
                    base_dir = Path(self.cfg.get("convert", {}).get(
                        "out_dir", "/mnt/models_local/ninfer/artifacts")).parent \
                        / "checkpoints" / base_name
                    if (base_dir / "config.json").is_file():
                        out["base"] = str(base_dir)
                    break

        # sources: whatever actually exists on disk
        if kind == "gguf":
            out["sources"]["gguf"] = str(p)
            # mmproj beside the gguf (as the lab's script does)
            mm = sorted(p.parent.glob("mmproj*.gguf")) or sorted(p.parent.glob("*mmproj*.gguf"))
            if mm:
                out["sources"]["vision"] = str(mm[0])
            im = next((x for x in p.parent.glob("*.gguf")
                       if "imatrix" in x.name.lower()), None)
            if im:
                out["sources"]["imatrix"] = str(im)
            if gemma4:
                # Gemma 4's drafter is the assistant GGUF, beside the source.
                asst = next((x for x in p.parent.glob("*.gguf")
                             if x != p and ("mtp" in x.name.lower()
                                            or "assistant" in x.name.lower())), None)
                if asst:
                    out["sources"]["mtp"] = str(asst)
        if out["base"]:
            out["sources"]["base"] = out["base"]
        # dflash2 is the draft head of the Qwen3.8-27B family; Gemma 4 does not use it.
        if not gemma4:
            dflash2 = Path("/mnt/models_local/strata/DFlash2")
            if dflash2.is_dir():
                out["sources"]["dflash2"] = str(dflash2)
            else:
                dg = Path("/mnt/models_local/strata/DFlash2-gguf/dflash2-q4km.gguf")
                if dg.is_file():
                    out["sources"]["dflash2"] = str(dg)

        # components: base text; +vision if there is an mmproj; mtp/dflash2 if there is a

        # draft
        comps = ["text"]
        if "vision" in out["sources"]:
            comps.append("vision")
        if "mtp" in out["sources"] or "dflash2" in out["sources"]:
            comps.append("mtp")
        if "dflash2" in out["sources"]:
            comps.append("dflash2")
        out["components"] = comps

        # template: the base's if it has one, otherwise the one proven in the lab.
        if out["base"] and (Path(out["base"]) / "chat_template.jinja").is_file():
            out["template"] = str(Path(out["base"]) / "chat_template.jinja")
        else:
            t = (Path(self.cfg.get("convert", {}).get(
                "roots", ["/mnt/models_local/ninfer"])[0]).parent
                 / "templates" / "froggeric-v22.5" / "chat_template.jinja")
            if t.is_file():
                out["template"] = str(t)
        return out

    # ---------------------------------------------------------------- router catalog
    def _router_paths(self) -> tuple[Path, Path]:
        rr = self.cfg.get("router", {})
        md = rr.get("models_dir", "/root/ninfer-lab/router-models")
        pr = rr.get("preset", "/root/ninfer-lab/router-preset.ini")
        return Path(md), Path(pr)

    def _parse_preset(self, path: Path) -> list[tuple[str, dict]]:
        """Read router-preset.ini into [(section_name, {key: value})].

        The order of sections is preserved so rewrites round-trip it. Missing
        files yield an empty list so a fresh deployment shows an empty catalog.
        """
        if not path.is_file():
            return []
        sections: list[tuple[str, dict]] = []
        cur: tuple[str, dict] | None = None
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                if cur is not None:
                    sections.append(cur)
                cur = (line[1:-1].strip(), {})
                continue
            if "=" not in line or cur is None:
                continue
            key, value = line.split("=", 1)
            value = self.unquote_text(value)
            cur[1][key.strip().lower()] = value
        if cur is not None:
            sections.append(cur)
        return sections

    def router_catalog(self) -> dict:
        models_dir, preset_path = self._router_paths()
        sections = self._parse_preset(preset_path)
        ids: dict[str, dict] = {}
        global_args: list[str] = []
        for name, entries in sections:
            if name == "*":
                continue
            ids[name] = dict(entries)
        models = []
        if models_dir.is_dir():
            for entry in sorted(models_dir.iterdir()):
                # Mirror discover_models() in model_catalog.cpp: a .ninfer file (or a
                # symlink pointing at one) is a model on its own; a directory holding
                # exactly one .ninfer is a model named after the directory.
                if entry.suffix == ".ninfer" and entry.is_file():
                    mid, artifact = entry.stem, entry
                elif entry.is_dir():
                    files = [f for f in entry.iterdir()
                             if f.suffix == ".ninfer" and f.is_file()]
                    if len(files) != 1:
                        continue
                    mid, artifact = entry.name, files[0]
                else:
                    continue
                pid = ids.get(mid, {})
                target = artifact.resolve()
                models.append({
                    "id": mid,
                    "artifact": pid.get("artifact") or pid.get("model") or str(artifact),
                    "artifact_real": str(target) if target != artifact else "",
                    "alias": pid.get("alias", ""),
                    "load_on_startup": pid.get("load-on-startup", ""),
                    "flags": {k: v for k, v in pid.items()
                              if k not in ("model", "artifact", "alias", "load-on-startup")},
                    "preset": mid in ids,
                    "on_disk": True,
                })
        # Add preset-only entries that have no directory on disk.
        for name, e in ids.items():
            if name == "*" or not name:
                continue
            if any(m["id"] == name for m in models):
                continue
            models.append({"id": name, "artifact": e.get("artifact") or e.get("model", ""),
                           "alias": e.get("alias", ""), "load_on_startup": e.get("load-on-startup", ""),
                           "flags": {k: v for k, v in e.items() if k not in ("model","artifact","alias","load-on-startup")},
                           "preset": True, "on_disk": False})
        return {"models_dir": str(models_dir), "preset": str(preset_path),
                "models": models,
                "global": (sections[0][1] if (sections and sections[0][0] == "*") else {})}

    @staticmethod
    def _split_preset(text: str) -> list:
        """[(section_name or None, [raw lines])], keeping the preamble as None.

        Line-oriented so comments and unknown keys survive a rewrite; only the
        one section being edited is rebuilt."""
        chunks: list = []
        for raw in text.splitlines():
            trimmed = raw.strip()
            if trimmed.startswith("[") and trimmed.endswith("]"):
                chunks.append((trimmed[1:-1].strip(), []))
            elif chunks:
                chunks[-1][1].append(raw)
            else:
                chunks.append((None, [raw]))
        return chunks

    @staticmethod
    def _join_preset(chunks: list) -> str:
        out: list[str] = []
        for name, lines in chunks:
            if name is not None:
                out.append(f"[{name}]")
            out.extend(lines)
        return "\n".join(out) + "\n"

    def _router_write_model(self, model_id: str, artifact: str | None,
                            alias: str | None, flags: dict | None,
                            load_on_startup: str | None = None) -> None:
        """Rewrite one [model] section, preserving every other line.

        section order, other sections and the [*] defaults are byte-preserved.
        """
        path = self._router_paths()[1]
        text = path.read_text() if path.is_file() else ""
        chunks = self._split_preset(text)
        existing: list[str] = []
        for name, lines in chunks:
            if name == model_id:
                existing = lines
                break
        block: list[str] = []
        entries: dict[str, str] = {}          # key -> value, in the section's own order
        for line in existing:
            if "=" in line:
                key, value = line.split("=", 1)
                entries[key.strip().lower()] = value.strip()
        if artifact:
            entries["artifact"] = artifact
        if alias:
            entries["alias"] = alias
        if load_on_startup not in (None, ""):
            entries["load-on-startup"] = load_on_startup
        for key, value in (flags or {}).items():
            entries[key] = value        # a new key lands at the end, as a human would write it
        for key, value in entries.items():
            block.append(f"{key} = {value}")
        index = next((i for i, (n, _) in enumerate(chunks) if n == model_id), None)
        # Blank line between sections: the file's own style. Without it a save
        # would reflow every section after the one being edited.
        if index is not None and index < len(chunks) - 1:
            block.append("")
        if index is not None:
            chunks[index] = (model_id, block)
        else:
            if chunks and any(ln.strip() for ln in chunks[-1][1]):
                chunks[-1][1].append("")
            chunks.append((model_id, block))
        path.write_text(self._join_preset(chunks))

    def router_set_alias(self, model_id: str, alias: str) -> tuple:
        if not model_id:
            return 400, {"error": "model is required"}
        sections = dict(self._parse_preset(self._router_paths()[1]))
        cur = sections.get(model_id, {})
        artifact = cur.get("artifact") or cur.get("model") or ""
        flags = {k: v for k, v in cur.items()
                 if k not in ("model", "artifact", "alias", "load-on-startup")}
        self._router_write_model(model_id, artifact, alias,
                                 flags, cur.get("load-on-startup"))
        return 200, {"ok": True, "restart_required": True,
                     "model": model_id, "alias": alias}

    def router_save_model(self, body: dict) -> tuple:
        """Save alias + flags of an existing model, or add one with an artifact."""
        model_id = (body.get("model") or "").strip()
        if not model_id:
            return 400, {"error": "model is required"}
        models_dir, preset_path = self._router_paths()
        sections = dict(self._parse_preset(preset_path))
        cur = sections.get(model_id)
        artifact = (body.get("artifact") or (cur or {}).get("artifact")
                    or (cur or {}).get("model") or "").strip()
        is_new = cur is None
        if is_new:
            if not artifact:
                return 400, {"error": "a new model needs an artifact (.ninfer path)"}
            src = Path(artifact)
            if not src.is_file():
                return 400, {"error": f"artifact does not exist: {artifact}"}
            link = models_dir / f"{model_id}.ninfer"
            if not link.exists():
                models_dir.mkdir(parents=True, exist_ok=True)
                link.symlink_to(src)
            artifact = str(link)
        flags = {k: v for k, v in (body.get("flags") or {}).items() if k}
        # A key absent from the body keeps its stored value: a form that only edits
        # flags must not silently erase the alias or load-on-startup.
        def keep(key: str) -> str:
            if key in body:
                return str(body.get(key) or "").strip()
            return str((cur or {}).get(key.replace("_", "-"), "") or "").strip()

        self._router_write_model(model_id, artifact, keep("alias"), flags,
                                 keep("load_on_startup"))
        return 200, {"ok": True, "restart_required": True, "model": model_id}

    def router_delete_model(self, body: dict) -> tuple:
        model_id = (body.get("model") or "").strip()
        if not model_id:
            return 400, {"error": "model is required"}
        models_dir, preset_path = self._router_paths()
        chunks = self._split_preset(preset_path.read_text() if preset_path.is_file() else "")
        kept = [c for c in chunks if c[0] != model_id]
        if len(kept) == len(chunks):
            return 404, {"error": f"{model_id} is not in the preset"}
        preset_path.write_text(self._join_preset(kept))
        removed_link = False
        if body.get("delete_link"):
            link = models_dir / f"{model_id}.ninfer"
            # Only the symlink this panel manages, never the artifact it points at.
            if link.is_symlink():
                link.unlink()
                removed_link = True
        return 200, {"ok": True, "restart_required": True,
                     "model": model_id, "removed_link": removed_link}

    def router_restart(self) -> dict:
        import subprocess
        try:
            out = subprocess.run(["systemctl", "restart", "ninfer-router"],
                                  capture_output=True, text=True, timeout=20)
            return {"ok": out.returncode == 0, "stderr": (out.stderr or "").strip()[:500],
                    "stdout": (out.stdout or "").strip()[:200]}
        except Exception as e:
            return {"ok": False, "error": str(e)[:300]}

    def history(self, query: dict) -> dict:
        """
        A throughput time series from the jsonl (`throughput` events).

        The charts start with history, not empty: opening the panel fills them with the
        last ~15 minutes the server already wrote.
        """
        path = Path(self.cfg["ninfer"].get("log", ""))
        if not path or not path.is_file():
            return {"ok": False}
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            return {"ok": False}
        points = []
        for line in lines:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event") != "throughput":
                continue
            tps = d.get("throughput_tokens_per_second") or {}
            cc = d.get("context_cache") or {}
            kv = (cc.get("device_main_kv_tokens") if isinstance(cc, dict)
                  else cc)
            points.append({
                "ts": d.get("timestamp_unix_ms"),
                "decode": tps.get("decode"),
                "prefill": tps.get("prefill"),
                "kv_tokens": kv,
            })
        points.sort(key=lambda p: p["ts"] or 0)
        return {"ok": True, "points": points[-360:]}

    def requests(self, query: dict) -> dict:
        """
        The most recent completed requests, from the server's jsonl.

        ninfer's jsonl already carries ttft/decode/prefill/cache and the per-request
        spec-decode: there is nothing else to parse. The whole file is read (including
        inode rotation: when the file changes it is read again).
        """
        path = Path(self.cfg["ninfer"].get("log", ""))
        if not path or not path.is_file():
            return {"ok": False, "error": "config.toml does not define [ninfer] log"}
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as err:
            return {"ok": False, "error": str(err)}
        n = int((query.get("n") or ["30"])[0])
        rows = []
        for line in lines:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event") != "request_done":
                continue
            req = d.get("request") or {}
            res = d.get("result") or {}
            t = d.get("timings_seconds") or {}
            spec = d.get("speculative") or {}
            hist = spec.get("accepted_per_position") or []
            total_acc = sum(hist)
            acc_avg = (sum(i * v for i, v in enumerate(hist)) / total_acc
                       if total_acc else None)
            drafted = spec.get("drafted_tokens", 0)
            rows.append({
                "ts_ms": d.get("timestamp_unix_ms"),
                "model": req.get("model"),
                "protocol": req.get("protocol", "?"),
                "ttft_ms": round(t.get("ttft", 0) * 1000, 1),
                "decode_s": round(t.get("decode", 0), 2),
                "prefill_s": round(t.get("prefill", 0), 3),
                "total_s": round(t.get("total", 0), 2),
                "completion_tokens": res.get("completion_tokens", res.get("generated_tokens")),
                "prefill_tokens": res.get("computed_prefill_tokens", res.get("prompt_tokens")),
                "cache_hit": res.get("prefix_cache_hit_tokens"),
                "finish": res.get("finish_reason"),
                "spec_backend": spec.get("backend"),
                "draft_window": spec.get("draft_window"),
                # Verification rounds: the real denominator of acceptance by position (how many
                # times a draft was proposed at each position).
                "spec_rounds": spec.get("rounds"),
                "accepted_tokens": spec.get("accepted_tokens"),
                "drafted_tokens": drafted,
                "acceptance_avg": acc_avg,
                "draft_rate": round(spec.get("accepted_tokens", 0) / drafted * 100, 1)
                               if drafted else None,
                "histogram": hist,
            })
        rows.sort(key=lambda r: r["ts_ms"] or 0, reverse=True)
        return {"ok": True, "path": str(path), "lines": len(lines),
                "requests": rows[:max(n, 1)]}

    def _live_once(self) -> dict:
        """
        One sample of the engine's state, with no GPU cache and no extra work.

        Both the /api/live endpoint and the baselines thread use it: sharing the reader
        keeps the baselines from being learned with a payload shape different from the
        one the page sees.
        """
        url = self.cfg["ninfer"]["url"].rstrip("/")
        out: dict = {"url": url, "online": False}
        for name, path in (("load", "/v1/load"), ("stats", "/stats"),
                           ("models", "/models"), ("slots", "/slots")):
            try:
                with urllib.request.urlopen(url + path, timeout=5) as fh:
                    out[name] = json.loads(fh.read())
                out["online"] = True
            except (OSError, ValueError, urllib.error.URLError) as err:
                out[name] = {"error": str(err)[:120]}
        # Per-model residency: what a suspend releases and how long the last one took.
        for model in (out.get("models") or {}).get("data") or []:
            mid = model.get("id")
            if not mid:
                continue
            try:
                with urllib.request.urlopen(f"{url}/v1/models/{mid}/residency", timeout=5) as fh:
                    model["residency"] = json.loads(fh.read())
            except (OSError, ValueError, urllib.error.URLError):
                pass
        out["engines"] = self._per_model(url, out)
        return out

    @staticmethod
    def _per_model(url: str, out: dict) -> dict:
        """
        State per model in memory: /v1/load accepts ?model= and answers for every
        Engine, not just the loaded one.

        Without this a router with two models in memory shows one row of occupancy and
        the rest look like they do not exist. The model the parameterless /v1/load
        already brought is not requested again.

        `releasable_device_bytes` (from residency) is weights + device state: for a
        loaded model it IS its VRAM; for a suspended one it is the footprint it would
        take when loaded, since what is already in host does not count. The page labels
        it by state instead of always calling it VRAM.
        """
        primary = (out.get("load") or {}).get("model")
        engines: dict = {}
        for model in (out.get("models") or {}).get("data") or []:
            mid = model.get("id")
            state = (model.get("status") or {}).get("value") or "unloaded"
            res = model.get("residency") or {}
            if state == "unloaded":
                continue
            # Same rule as /api/engine: residency.state says suspended even while /models
            # still says loaded. Its internal names are "resident" and "suspended", not the
            # UI's.
            if res.get("state") == "suspended":
                state = "suspended"
            engines[mid] = {"state": state,
                            "residency": res,
                            "load": (out.get("load") if mid == primary and "error" not in (out.get("load") or {})
                                     else None)}
            if engines[mid]["load"] is not None:
                continue
            try:
                with urllib.request.urlopen(f"{url}/v1/load?model={urllib.parse.quote(mid)}",
                                            timeout=5) as fh:
                    engines[mid]["load"] = json.loads(fh.read())
            except (OSError, ValueError, urllib.error.URLError):
                engines[mid]["load"] = {"error": "no data"}
        return engines

    def live(self) -> dict:
        """
        Server state for the page: /v1/load, /stats, /models, /slots and the GPUs,
        cached so a poll does not spawn nvidia-smi.
        """
        out = self._live_once()
        out["gpu"] = self.gpu()
        return out

    def control(self, body: dict) -> tuple[int, dict]:
        """
        Actions on the engine: sleep or wake a model.

        These are endpoints ninfer already has; the panel invents nothing.
        """
        url = self.cfg["ninfer"]["url"].rstrip("/")
        action = body.get("action")
        model = body.get("model")
        if action not in ("sleep", "load", "unload", "suspend", "resume"):
            return 400, {"error": f"unknown action: {action}"}
        if not model:
            return 400, {"error": "model is required"}
        path = {"sleep": "/models/sleep", "load": "/models/load",
                "unload": "/models/unload"}.get(action)
        if path:
            target = f"{url}{path}"
            payload = json.dumps({"model": model}).encode()
        else:
            target = f"{url}/v1/models/{model}/{action}"
            payload = json.dumps({}).encode()
        req = urllib.request.Request(target, data=payload, method="POST",
                                     headers={"Content-Type": "application/json"})
        # Loading and unloading swap weights through the host: a 15 GB model took
        # ~35 s to unload, and a client-side timeout here looks to the user like
        # the action never ran, while the engine keeps working on it.
        try:
            with urllib.request.urlopen(req, timeout=300) as fh:
                return 200, json.loads(fh.read())
        except urllib.error.HTTPError as err:
            # 409 model_busy is a valid answer: the model has requests in flight and cannot
            # sleep. It has to be reported, not hidden.
            detail = err.read().decode()[:300]
            return err.code, {"error": detail, "action": action, "model": model}
        except (OSError, ValueError) as err:
            return 502, {"error": str(err)[:200]}

    def runs(self, query: dict) -> dict:
        rows = bench.list_runs(Path(self.cfg["aiperf"]["registry"]))
        # engine and spec are not registry fields: they live nested under `server`.
        # They are derived before filtering and added to the row, so filtering and
        # sorting work on what the UI sees.
        for row in rows:
            server = row.get("server") or {}
            row["engine"] = server.get("binary") or ""
            row["spec"] = (server.get("spec_backend", "") +
                           (f"/{server['draft_depth']}" if server.get("draft_depth") else "")
                           if server.get("spec_backend") else
                           ("off" if server else ""))
        # Filters: empty means "all". Each one is a substring over a row field, to avoid
        # inventing a query language.
        for field in ("profile", "dataset", "model", "concurrency", "engine", "spec"):
            want = (query.get(field) or [""])[0]
            if want:
                rows = [r for r in rows if str(r.get(field, "")) == want]
        sort = (query.get("sort") or ["ts"])[0]
        desc = (query.get("dir") or ["1"])[0] == "1"
        # Nulls always sort last: a run without spec-decode is not "worse", it is
        # another measurement, and it must not sink the table when sorting by
        # acceptance.
        rows.sort(key=lambda r: (r.get(sort) is None, r.get(sort) or 0), reverse=desc)
        # Pagination after filtering and sorting: otherwise page 2 repeats runs when the
        # order changes between two requests.
        total = len(rows)
        size = int((query.get("size") or ["50"])[0])
        page = max(int((query.get("page") or ["1"])[0]), 1)
        if size > 0:
            start = (page - 1) * size
            rows = rows[start:start + size]
        return {"runs": rows, "total": total, "page": page, "size": size,
                "pages": max(1, (total + size - 1) // size) if size > 0 else 1}

    def run_detail(self, run_id: str) -> dict | None:
        for r in bench.list_runs(Path(self.cfg["aiperf"]["registry"])):
            if r.get("run_id") == run_id:
                detail = dict(r)
                cmd = Path(r.get("artifacts_dir", "")) / "cmd.txt"
                if cmd.is_file():
                    detail["aiperf_cmd"] = cmd.read_text().strip()
                return detail
        return None

    # ---------------------------------------------------------------- jobs
    def _resolve_tokenizer(self, model: str, explicit: str | None) -> str:
        """
        The tokenizer that belongs to the model, not the config's default.

        A wrong tokenizer does not abort AIPerf: it produces an empty
        profile_export_aiperf.json and the run is never registered (bug of 2026-10-11
        with qwen35-9b running on qwen35-4b's tokenizer). That is why it is validated
        here and fails fast with a 400 instead of launching a doomed run.
        """
        ap = self.cfg.get("aiperf", {})
        cand = (explicit or "").strip() or ap.get("tokenizers", {}).get(model, "") \
            or ap.get("tokenizer", "")
        if cand and Path(cand, "tokenizer_config.json").is_file():
            return cand
        raise ValueError(
            f"no valid tokenizer for {model!r} (tried {cand!r}). "
            "Add it in config.toml [aiperf.tokenizers].")

    def start(self, body: dict) -> tuple[int, dict]:
        if not self.config()["aiperf"]["ready"]:
            return 503, {"error": "AIPerf no disponible: " + self.config()["aiperf"]["why"]}
        profile = body.get("profile")
        dataset = body.get("dataset")
        if not profile or not dataset:
            return 400, {"error": "profile and dataset are required"}
        known = {b["id"] for b in self.cfg.get("benchmarks", [])}
        if profile not in known:
            return 400, {"error": f"unknown profile: {profile}"}
        probe = bench.RunSpec(profile, dataset, 1, "", url=body.get("url", ""))
        server = probe.capture_server(self.cfg)
        # The model name is the server's to decide, not the panel's: ask it with
        # /props. Hardcoding it here was the bug that left a run with an empty --model.
        model = body.get("model") or server.get("alias") or server.get("model")
        if not model:
            return 502, {"error": "the server did not answer /props; the model name is unknown. "
                                  "Is it up at "
                                  f"{body.get('url') or self.cfg['ninfer']['url']}?"}
        try:
            tokenizer = self._resolve_tokenizer(model, body.get("tokenizer"))
        except ValueError as err:
            return 400, {"error": str(err)}
        spec = bench.RunSpec(
            profile, dataset, int(body.get("concurrency", 1)), model,
            requests=int(body.get("requests", 50)),
            output_tokens=int(body.get("output_tokens", 512)),
            url=body.get("url", ""),
            extra=list(body.get("flag", [])),
            tokenizer=tokenizer,
        )
        run_dir = Path(self.cfg["aiperf"]["runs_dir"]) / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "server.json").write_text(json.dumps(server, indent=2))
        runner = bench.Runner(spec, self.cfg, run_dir)
        try:
            runner.start()
        except (OSError, subprocess.SubprocessError) as err:
            return 500, {"error": f"could not start: {err}"}
        state = {"run_id": spec.run_id, "spec": spec, "runner": runner,
                 "status": "running", "server": server}
        self.jobs.add(spec.run_id, state)

        def watch():
            while runner.proc.poll() is None:
                self.publish({"type": "progress", "run_id": spec.run_id,
                              "progress": runner.progress()})
                threading.Event().wait(1.0)
            entry = bench.ingest(run_dir, spec, self.cfg)
            if entry:
                bench.append_registry(Path(self.cfg["aiperf"]["registry"]), entry)
            state["status"] = "done" if entry else "failed"
            # The subprocess log travels with the event: a failure that only says "did not
            # register" is unreadable without it.
            self.publish({"type": "finished", "run_id": spec.run_id,
                          "ok": bool(entry), "run": entry,
                          "exit": runner.proc.returncode,
                          "log_tail": runner.log()[-12:]})
            self.jobs.add(spec.run_id, state)

        threading.Thread(target=watch, daemon=True).start()
        return 202, {"run_id": spec.run_id, "server": server}

    def stop(self, body: dict) -> tuple[int, dict]:
        run_id = body.get("run_id")
        state = self.jobs.get(run_id) if run_id else None
        if not state:
            return 404, {"error": "there is no such live run"}
        state["runner"].stop()
        state["status"] = "stopped"
        return 200, {"run_id": run_id, "status": "stopped"}


class Handler(BaseHTTPRequestHandler):
    api: API = None          # inyectado abajo
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):        # silencio: el log va al SSE
        pass

    def _json(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path, query = parsed.path, urllib.parse.parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path == "/vendor-chart.umd.js":
            return self._static("vendor-chart.umd.js")
        if path.startswith(("/js/", "/css/")):
            return self._static(path.lstrip("/"))
        if path == "/api/config":
            return self._json(self.api.config())
        if path == "/api/engine":
            return self._json(self.api.engine())
        if path == "/api/gpu":
            return self._json(self.api.gpu())
        if path == "/api/help":
            code, payload = self.api.help_flags()
            return self._json(payload, code)
        if path == "/api/convert/files":
            return self._json(self.api.convert_files())
        if path == "/api/convert/search":
            return self._json(self.api.hf_search(query.get("q", [""])[0]))
        if path == "/api/convert/details":
            return self._json(self.api.hf_details(query.get("model", [""])[0]))
        if path == "/api/convert/ggufs":
            return self._json(self.api.hf_ggufs(query.get("model", [""])[0]))
        if path == "/api/convert/download":
            st = self.api.hf_download_state(query.get("run_id", [""])[0])
            return self._json(st if st else {"error": "there is no such download"},
                             200 if st else 404)
        if path == "/api/convert/probe":
            return self._json(self.api.convert_probe(query.get("path", [""])[0]))
        if path == "/api/convert/recipes":
            return self._json(self.api.convert_recipes())
        if path == "/api/convert/inspect":
            code, payload = self.api.convert_inspect(query)
            return self._json(payload, code)
        if path == "/api/live":
            return self._json(self.api.live())
        if path == "/api/alerts":
            return self._json(self.api.baselines.view())
        if path == "/api/router/catalog":
            return self._json(self.api.router_catalog())
        if path == "/api/requests":
            return self._json(self.api.requests(query))
        if path == "/api/history":
            return self._json(self.api.history(query))
        if path == "/api/bench/runs":
            return self._json(self.api.runs(query))
        if path.startswith("/api/bench/run/"):
            detail = self.api.run_detail(path.rsplit("/", 1)[-1])
            return self._json(detail) if detail else self._json({"error": "no existe"}, 404)
        if path == "/api/bench/stream":
            return self._sse()
        self._json({"error": "unknown path"}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        body = self._body()
        if parsed.path == "/api/bench/run":
            code, payload = self.api.start(body)
            return self._json(payload, code)
        if parsed.path == "/api/bench/stop":
            code, payload = self.api.stop(body)
            return self._json(payload, code)
        if parsed.path == "/api/control":
            code, payload = self.api.control(body)
            return self._json(payload, code)
        if parsed.path == "/api/router/alias":
            code, payload = self.api.router_set_alias(body.get("model", ""),
                                                      body.get("alias", ""))
            return self._json(payload, code)
        if parsed.path == "/api/router/save":
            code, payload = self.api.router_save_model(body)
            return self._json(payload, code)
        if parsed.path == "/api/router/delete":
            code, payload = self.api.router_delete_model(body)
            return self._json(payload, code)
        if parsed.path == "/api/router/restart":
            payload = self.api.router_restart()
            return self._json(payload)
        if parsed.path == "/api/convert/run":
            code, payload = self.api.convert_run(body)
            return self._json(payload, code)
        if parsed.path == "/api/convert/download":
            code, payload = self.api.hf_download(body)
            return self._json(payload, code)
        self._json({"error": "unknown path"}, 404)

    def _static(self, name: str):
        # The name arrives from the URL now that /js and /css are served from a
        # directory, so it can carry ../ or an absolute path. Resolve and refuse
        # anything outside the panel's own directory.
        root = HERE.resolve()
        path = (root / name).resolve()
        if path != root and root not in path.parents:
            return self._json({"error": "path outside the panel"}, 404)
        if not path.is_file():
            return self._json({"error": "missing " + name}, 404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        # Without it the browser caches index.html and a panel change does not show.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _sse(self):
        """Progress of the live runs. One stream per client."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        queue: list = []
        wake = threading.Event()

        def push(event):
            queue.append(event)
            wake.set()

        self.api.subscribe(push)
        try:
            for state in self.api.jobs.snapshot():
                # Only Runner jobs emit progress here; the convert and download ones publish
                # their own events from their watch thread.
                if state["status"] == "running" and "runner" in state:
                    self._emit({"type": "progress", "run_id": state["run_id"],
                                "progress": state["runner"].progress()})
            while True:
                wake.wait(timeout=5.0)
                wake.clear()
                while queue:
                    self._emit(queue.pop(0))
                self._emit({"type": "ping"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.api.unsubscribe(push)

    def _emit(self, event: dict):
        chunk = f"data: {json.dumps(event)}\n\n".encode()
        self.wfile.write(chunk)
        self.wfile.flush()


def main() -> int:
    cfg = bench.load_config(CONFIG)
    api = API(cfg)
    Handler.api = api
    host, port = cfg["panel"]["host"], cfg["panel"]["port"]
    ready, why = bench.aiperf_ready(cfg)
    print(f"panel ninfer   http://{host}:{port}")
    print(f"AIPerf         {'listo' if ready else 'NO DISPONIBLE — ' + why}")
    print(f"registro       {cfg['aiperf']['registry']}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
