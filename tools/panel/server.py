#!/usr/bin/env python3
"""Panel ninfer — servidor HTTP y API. Sin dependencias.

    python3 tools/panel/server.py            # lee config.toml

Sirve la UI y expone el registro de benchmarks. Los jobs AIPerf viven en este
proceso: se lanzan, se sigue su progreso por SSE y se ingieren al terminar.
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
    """Corridas vivas. Un lock porque el handler es threaded y el SSE son varias
    conexiones leyendo a la vez."""

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


class API:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.jobs = Jobs()
        self.hub: list[callable] = []          # suscriptores SSE
        self.hub_lock = threading.Lock()
        self._gpu_ts = 0.0
        self._gpu_cache: dict = {"online": False}

    # ---------------------------------------------------------------- avisos
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

    # ---------------------------------------------------------------- rutas
    def config(self) -> dict:
        ready, why = bench.aiperf_ready(self.cfg)
        ap = self.cfg.get("aiperf", {})
        nin = self.cfg.get("ninfer", {})
        url = nin.get("url", "").rstrip("/")
        # Endpoint para clientes/agentes: la base OpenAI-compatible del router.
        # Un agente se configura con BASE_URL + MODEL del catálogo.
        return {
            "aiperf": {"enabled": bool(ap.get("enabled")), "ready": ready, "why": why,
                       "tokenizer": ap.get("tokenizer", ""),
                       "tokenizers": ap.get("tokenizers", {})},
            "endpoint": {
                "base_url": url + "/v1",
                "api_key": nin.get("api_key") or "",
                "model": "",         # lo llena el frontend con el catálogo
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
        """Estado del servidor: qué está cargado y qué hay para cargar.

        El catálogo sale de /models, así que la lista de modelos a benchmarkear
        no está hardcodeada en el panel: es la del servidor. El estado fino
        (loaded/sleeping/suspended) sale de /residency por modelo, que es el
        único que distingue dormir de suspend.
        """
        url = self.cfg["ninfer"]["url"]
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as fh:
                data = json.loads(fh.read())
        except (OSError, ValueError, urllib.error.URLError):
            return {"url": url, "online": False, "loaded": [], "catalog": []}
        models = [m for m in (data.get("data") or []) if m.get("id")]
        loaded = [m["id"] for m in models if (m.get("object") or "") != "model-unloaded"]
        catalog = []
        for m in models:
            mid = m["id"]
            state = (m.get("status") or {}).get("value", "")
            # residency.state es la verdad para Dormir vs Suspend: un modelo
            # suspendido sigue `loaded` en /models pero está `suspended` aquí.
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
        return {"url": url, "online": True, "loaded": loaded, "catalog": catalog}

    def convert_recipes(self) -> dict:
        """Recipes reales de tools/convert, agrupadas por módulo.

        No se hardcodean: se leen de los registros RECIPES de cada módulo en el
        venv de convert (torch), que es donde viven. Si el repo agrega una
        recipe nueva, aparece sola en el dropdown.
        """
        venv = self.cfg.get("convert", {}).get("venv", "")
        python = str(Path(venv) / "bin" / "python") if venv else sys.executable
        if not Path(python).is_file():
            return {"ok": False, "error": f"no hay python en {python} ([convert] venv)"}
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
            return {"ok": False, "error": "el probe de recipes devolvió algo raro"}

    def convert_inspect(self, query: dict) -> tuple[int, dict]:
        """Lee un .ninfer y devuelve su resumen: componentes, tensores, formatos.

        tools/artifact/inspect.py es stdlib puro, así que se corre como módulo
        en un subprocess: usa imports relativos y no quiere contaminar el path
        del panel.
        """
        path = (query.get("path") or [""])[0]
        if not path:
            return 400, {"error": "hace falta path"}
        if not Path(path).is_file():
            return 404, {"error": f"no existe {path}"}
        try:
            out = subprocess.run(
                [sys.executable, "-m", "tools.artifact.inspect", path, "--json"],
                capture_output=True, text=True, timeout=60,
                cwd=str(HERE.parent.parent))
        except (OSError, subprocess.SubprocessError) as err:
            return 500, {"error": str(err)[:200]}
        print(f"DEBUG inspect: rc={out.returncode} stdout_len={len(out.stdout)} stderr_len={len(out.stderr)}", file=sys.stderr)
        if out.returncode != 0:
            return 502, {"error": (out.stderr or out.stdout).strip()[:400]}
        try:
            return 200, json.loads(out.stdout)
        except json.JSONDecodeError:
            return 502, {"error": "el inspector devolvió algo que no es JSON"}
    def convert_run(self, body: dict) -> tuple[int, dict]:
        """Lanza una conversión GGUF/safetensors → .ninfer con log en vivo.

        tools/convert necesita torch: corre en su venv, no en el del panel.
        """
        venv = self.cfg.get("convert", {}).get("venv", "")
        python = str(Path(venv) / "bin" / "python") if venv else sys.executable
        if not Path(python).is_file():
            return 503, {"error": f"no hay python en {python} ([convert] venv)"}
        model = body.get("model")
        recipe = body.get("recipe")
        if not model and not body.get("base"):
            return 400, {"error": "hacen falta model (o base) y recipe"}
        if not recipe:
            return 400, {"error": "hacen falta model y recipe"}
        out_dir = Path(self.cfg["convert"]["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        # Una fuente GGUF no va como --model: el modelo base es el checkpoint
        # (config+tokenizer) y el gguf entra como --source gguf=. Con la fuente
        # correcta armamos el comando completo; sin `base`, queda el camino
        # anterior (checkpoint como --model).
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
        # El nombre del artifact sale de la fuente: el gguf (si hay) o el base.
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
            return 500, {"error": f"no se pudo lanzar: {err}"}
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
        """Borra el staging descargado de un modelo tras una conversión OK.

        Solo actúa sobre lo que nosotros bajamos: una fuente dentro de
        ninfer/inbox/<slug>/. Nunca toca nada fuera de esa raíz (el usuario
        podría pasar cualquier path como source).
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
        """Datos de la GPU por nvidia-smi. Se cachea 2s: lanzar nvidia-smi
        a cada poll de 2s es descartable, pero el proceso cuesta ~50ms."""
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
        """Flags de ninfer-serve parseadas de --help, para el formulario de
        recipes. Se leen en vivo: si el binario cambia, el formulario también."""
        binary = self.cfg.get("ninfer", {}).get("binary", "")
        if not binary or not Path(binary).is_file():
            return 503, {"error": f"no existe el binario {binary} ([ninfer] binary)"}
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
        """Raíces donde buscar artifacts y fuentes: las del config más HF_HOME
        y los directorios reales de modelos, si existen. Nada adivinado: cada
        ruta se verifica contra el disco."""
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
        # dedupe preservando orden
        return list(dict.fromkeys(roots))

    def convert_files(self) -> dict:
        """Artifacts .ninfer y fuentes (GGUF/safetensors) disponibles.

        El usuario no escribe rutas a mano: el panel lista lo que ya existe en
        las raíces configuradas (config.toml [convert] roots) y en HF_HOME."""
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
        # checkpoints: directorios con config.json (modelo a convertir)
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
                "artifacts": artifacts[-200:],   # demo: acotar a lo manejable
                "sources": sources[-200:],
                "checkpoints": checkpoints[-100:]}

    # ---------------------------------------------- Hugging Face API (0 deps)
    # Orden de prefijos de cuantización para agrupar los .gguf en el selector.
    GGUF_PREFIXES = ("IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ2_XXS", "IQ2_S", "IQ1_M",
                     "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S",
                     "Q4_0", "Q3_K_M", "Q3_K_S", "Q3_K_L", "Q2_K", "BF16", "F16")
    _HF_TIMEOUT = 20.0
    _HF_BASE = "https://huggingface.co/api/models"

    def hf_search(self, query: str) -> dict:
        """Buscar modelos en HF. query: substring de título/autor.
        Devuelve {hits: [...], error: ...}."""
        if not query or len(query) < 3:
            return {"hits": [], "error": None}
        url = self._HF_BASE + "?search=" + urllib.parse.quote(query, safe="") + "&limit=50"
        try:
            with urllib.request.urlopen(url, timeout=self._HF_TIMEOUT) as resp:
                if resp.status != 200:
                    return {"hits": [], "error": "respuesta de HF no es 200"}
                hits = json.loads(resp.read().decode())
        except OSError as e:
            return {"hits": [], "error": f"no se pudo consultar HF: {e}"}
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
        """Lista de archivos de un modelo (`.gguf` y demás) con sus tamaños.

        La API de HF no pone `size` en `siblings`, así que se consulta el
        árbol de la rama por defecto: `.../tree/{branch}`. El árbol de una
        subcarpeta (p.ej. `ggml-model-...`) exige recursión, así que se usa
        `?recursive=true`. Si el repo no tiene rama `main`, se cae al
        endpoint del modelo (sin tamaños).
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
        """Los GGUF de un modelo, agrupados por cuantización; mmproj/mtp aparte."""
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
        """Carpeta de staging para el GGUF efímero: ninfer/inbox/<org>--<repo>."""
        out_dir = Path(self.cfg.get("convert", {}).get("out_dir",
                        "/mnt/models_local/ninfer/artifacts"))
        slug = re.sub(r"[^A-Za-z0-9_.-]", "-", model)
        return out_dir.parent / "inbox" / slug

    def hf_bin(self) -> str:
        """El CLI de HF: se prefiere el `hf` de huggingface_hub (ya instalado)."""
        found = shutil.which("hf")
        if found:
            return found
        candidate = Path.home() / ".local" / "bin" / "hf"
        return str(candidate) if candidate.is_file() else "hf"

    def hf_download(self, body: dict) -> tuple[int, dict]:
        """Descarga GGUFs exactos de un repo HF a ninfer/inbox/ con progreso.

        El progreso se lee del disco (bytes aparecidos en la carpeta), no de
        stdout: `hf download` pinta un progress bar con \r que no sirve para
        líneas. La descarga va a inbox/ — staging efímero, no al cache de HF.
        """
        model, files = body.get("model", ""), body.get("files", [])
        total = int(body.get("total", 0) or 0)
        if not model or not isinstance(files, list) or not files:
            return 400, {"error": "hacen falta model y files"}
        dest = self.hf_inbox(model)
        dest.mkdir(parents=True, exist_ok=True)
        # Guard de disco: nunca dejar la NVMe sin aire por una conversión.
        free = shutil.disk_usage(dest).free
        need = int(total * 1.2) + (5 << 30)
        if free < need:
            return 507, {"error": f"disco insuficiente: {free/2**30:.0f} GiB libres, "
                                    f"se calculan {need/2**30:.0f} GiB (1,2× la descarga). "
                                    "Liberá espacio o elegí otra quant."}
        hf = self.hf_bin()
        argv = [hf, "download", model, *files, "--local-dir", str(dest), "--quiet"]
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, start_new_session=True)
        except (OSError, subprocess.SubprocessError) as err:
            return 500, {"error": f"no se pudo lanzar hf: {err}"}
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

    # ------------------------------------------- probe de fuentes + recipe
    # (arch, moe, layers, hidden) -> recipe oficial (checkpoint) / gguf / base local
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
        """La torre de texto de un config multimodal (text_config anidado)."""
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
        """Metadata de un .gguf via el venv de convert (tiene gguf-py)."""
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
        """Snapshot HF local (config+tokenizer) que sirva de base, por nombre.

        Busca en el cache de HF de las roots un snapshot que no sea un repo GGUF
        y tenga tokenizer.json: es el '--model' que la conversión necesita.
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
        """Lee la fuente (checkpoint o gguf), matchea recipe y arma los flags.

        Devuelve lo necesario para convertir sin saber nada a mano: recipe
        (o 'recipe desconocida'), base local, sources disponibles
        (gguf/dflash2/vision/imatrix), componentes y template.
        """
        p = Path(path)
        if not p.exists():
            return {"known": False, "error": f"no existe {path}"}
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
            # Flash-Next: solo recipe gguf; base local no lo cubre (otra arquitectura).
            out["recipe"], out["recipe_kind"] = self.FLASH_RECIPE, "gguf"
            out["known"] = True
        elif gemma4:
            # Gemma 4: recipe gguf (QAT Q4_0); el base es el repo HF (config+tokenizer).
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

        # sources: lo que exista realmente en el disco
        if kind == "gguf":
            out["sources"]["gguf"] = str(p)
            # mmproj al lado del gguf (como hace el script del lab)
            mm = sorted(p.parent.glob("mmproj*.gguf")) or sorted(p.parent.glob("*mmproj*.gguf"))
            if mm:
                out["sources"]["vision"] = str(mm[0])
            im = next((x for x in p.parent.glob("*.gguf")
                       if "imatrix" in x.name.lower()), None)
            if im:
                out["sources"]["imatrix"] = str(im)
            if gemma4:
                # El drafter de Gemma 4 es el assistant GGUF, al lado de la fuente.
                asst = next((x for x in p.parent.glob("*.gguf")
                             if x != p and ("mtp" in x.name.lower()
                                            or "assistant" in x.name.lower())), None)
                if asst:
                    out["sources"]["mtp"] = str(asst)
        if out["base"]:
            out["sources"]["base"] = out["base"]
        # dflash2 es el draft head de la familia Qwen3.8-27B; Gemma 4 no lo usa.
        if not gemma4:
            dflash2 = Path("/mnt/models_local/strata/DFlash2")
            if dflash2.is_dir():
                out["sources"]["dflash2"] = str(dflash2)
            else:
                dg = Path("/mnt/models_local/strata/DFlash2-gguf/dflash2-q4km.gguf")
                if dg.is_file():
                    out["sources"]["dflash2"] = str(dg)

        # componentes: texto base; +vision si hay mmproj; mtp/dflash2 si hay draft
        comps = ["text"]
        if "vision" in out["sources"]:
            comps.append("vision")
        if "mtp" in out["sources"] or "dflash2" in out["sources"]:
            comps.append("mtp")
        if "dflash2" in out["sources"]:
            comps.append("dflash2")
        out["components"] = comps

        # template: el del base si lo tiene, si no el probado del lab.
        if out["base"] and (Path(out["base"]) / "chat_template.jinja").is_file():
            out["template"] = str(Path(out["base"]) / "chat_template.jinja")
        else:
            t = (Path(self.cfg.get("convert", {}).get(
                "roots", ["/mnt/models_local/ninfer"])[0]).parent
                 / "templates" / "froggeric-v22.5" / "chat_template.jinja")
            if t.is_file():
                out["template"] = str(t)
        return out

    def history(self, query: dict) -> dict:
        """Serie temporal de throughput desde el jsonl (eventos `throughput`).

        Las gráficas arrancan con historia, no vacías: al abrir el panel se
        rellenan con los últimos ~15 minutos que ya escribió el servidor.
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
        """Últimas requests completadas, del jsonl del servidor.

        El jsonl de ninfer ya tiene ttft/decode/prefill/cache y el spec-decode
        por request: no hay que parsear otra cosa. Se lee el archivo completo
        (rotación por inode incluida: si el archivo cambia, se vuelve a leer).
        """
        path = Path(self.cfg["ninfer"].get("log", ""))
        if not path or not path.is_file():
            return {"ok": False, "error": "config.toml no define [ninfer] log"}
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
                # Rondas de verificación: el denominador real de la aceptación por
                # posición (cuántas veces se propuso un draft en cada posición).
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

    def live(self) -> dict:
        """Estado del servidor: /v1/load, /stats, /models, /slots.

        /v1/load y /stats son la verdad para el panel: no se parsea el log
        para saber qué está pasando.
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
        # Residency por modelo: qué libera un suspend y cuánto tardó el último.
        for model in (out.get("models") or {}).get("data") or []:
            mid = model.get("id")
            if not mid:
                continue
            try:
                with urllib.request.urlopen(f"{url}/v1/models/{mid}/residency", timeout=5) as fh:
                    model["residency"] = json.loads(fh.read())
            except (OSError, ValueError, urllib.error.URLError):
                pass
        return out

    def control(self, body: dict) -> tuple[int, dict]:
        """Acciones sobre el motor: dormir/despertar un modelo.

        Son los endpoints que ya trae ninfer; el panel no inventa nada.
        """
        url = self.cfg["ninfer"]["url"].rstrip("/")
        action = body.get("action")
        model = body.get("model")
        if action not in ("sleep", "load", "unload", "suspend", "resume"):
            return 400, {"error": f"acción desconocida: {action}"}
        if not model:
            return 400, {"error": "hace falta model"}
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
        try:
            with urllib.request.urlopen(req, timeout=30) as fh:
                return 200, json.loads(fh.read())
        except urllib.error.HTTPError as err:
            # 409 model_busy es una respuesta válida: el modelo tiene requests
            # en vuelo y no se puede dormir. Hay que decirla, no esconderla.
            detail = err.read().decode()[:300]
            return err.code, {"error": detail, "action": action, "model": model}
        except (OSError, ValueError) as err:
            return 502, {"error": str(err)[:200]}

    def runs(self, query: dict) -> dict:
        rows = bench.list_runs(Path(self.cfg["aiperf"]["registry"]))
        # engine y spec no son campos del registro: viven anidados en `server`.
        # Se derivan antes de filtrar y se agregan a la fila, para que el filtro
        # y la ordenación trabajen sobre lo mismo que ve la UI.
        for row in rows:
            server = row.get("server") or {}
            row["engine"] = server.get("binary") or ""
            row["spec"] = (server.get("spec_backend", "") +
                           (f"/{server['draft_depth']}" if server.get("draft_depth") else "")
                           if server.get("spec_backend") else
                           ("off" if server else ""))
        # Filtros: vacíos significan "todos". Cada uno es un substring sobre un
        # campo de la línea, para no inventar un lenguaje de consultas.
        for field in ("profile", "dataset", "model", "concurrency", "engine", "spec"):
            want = (query.get(field) or [""])[0]
            if want:
                rows = [r for r in rows if str(r.get(field, "")) == want]
        sort = (query.get("sort") or ["ts"])[0]
        desc = (query.get("dir") or ["1"])[0] == "1"
        # nulls al fondo siempre: una corrida sin spec-decode no es "peor", es
        # otra medición, y no debe hundirse la tabla al ordenar por acceptance.
        rows.sort(key=lambda r: (r.get(sort) is None, r.get(sort) or 0), reverse=desc)
        # Paginación después de filtrar y ordenar: si no, la página 2 repite
        # corridas cuando el orden cambia entre dos requests.
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
        """Tokenizer que corresponde al modelo, no al default del cfg.

        Un tokenizer equivocado no aborta AIPerf: produce profile_export_aiperf.json
        vacío y la corrida nunca se registra (bug del 2026-10-11 con qwen35-9b
        corriendo con el tokenizer de qwen35-4b). Por eso se valida acá y se
        falla rápido con 400 en vez de lanzar una corrida condenada.
        """
        ap = self.cfg.get("aiperf", {})
        cand = (explicit or "").strip() or ap.get("tokenizers", {}).get(model, "") \
            or ap.get("tokenizer", "")
        if cand and Path(cand, "tokenizer_config.json").is_file():
            return cand
        raise ValueError(
            f"sin tokenizer válido para {model!r} (probé {cand!r}). "
            "Agregalo en config.toml [aiperf.tokenizers].")

    def start(self, body: dict) -> tuple[int, dict]:
        if not self.config()["aiperf"]["ready"]:
            return 503, {"error": "AIPerf no disponible: " + self.config()["aiperf"]["why"]}
        profile = body.get("profile")
        dataset = body.get("dataset")
        if not profile or not dataset:
            return 400, {"error": "hacen falta profile y dataset"}
        known = {b["id"] for b in self.cfg.get("benchmarks", [])}
        if profile not in known:
            return 400, {"error": f"perfil desconocido: {profile}"}
        probe = bench.RunSpec(profile, dataset, 1, "", url=body.get("url", ""))
        server = probe.capture_server(self.cfg)
        # El nombre del modelo lo decide el servidor, no el panel: se lo preguntamos
        # con /props. Hardcodearlo acá fue el bug que dejó una corrida con --model vacío.
        model = body.get("model") or server.get("alias") or server.get("model")
        if not model:
            return 502, {"error": "el servidor no respondió /props; no se puede saber "
                                  "el nombre del modelo. ¿está arriba en "
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
            return 500, {"error": f"no se pudo lanzar: {err}"}
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
            # El log del subproceso viaja con el evento: un fallo que solo dice
            # "no registró" es indescifrable sin él.
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
            return 404, {"error": "no hay esa corrida viva"}
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
            return self._json(st if st else {"error": "no hay esa descarga"},
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
        self._json({"error": "ruta desconocida"}, 404)

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
        if parsed.path == "/api/convert/run":
            code, payload = self.api.convert_run(body)
            return self._json(payload, code)
        if parsed.path == "/api/convert/download":
            code, payload = self.api.hf_download(body)
            return self._json(payload, code)
        self._json({"error": "ruta desconocida"}, 404)

    def _static(self, name: str):
        path = HERE / name
        if not path.is_file():
            return self._json({"error": "falta " + name}, 404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        # Si no, el navegador cachea index.html y no se ve un cambio del panel.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _sse(self):
        """Progreso de las corridas vivas. Un stream por cliente."""
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
                # Solo los jobs con Runner emiten progress aquí; los de convert
                # y download publican sus propios eventos desde su watch thread.
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
