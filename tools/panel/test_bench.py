#!/usr/bin/env python3
"""Check del motor de benchmarks. Sin framework: `python3 test_bench.py`.

Cubre lo que se rompe en silencio: el perfil de comparabilidad, la ingesta de
una línea a medias, el pool de spec-decode ausente, y que argv no pierda el
--streaming (sin él desaparece TTFT y el ranking queda con un hueco).
"""
from __future__ import annotations

import json
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench
from bench import spec_config


def test_config_se_parsea():
    cfg = bench.load_config()
    assert cfg["aiperf"]["enabled"] is True
    assert any(b["id"] == "spec" for b in cfg["benchmarks"]), "falta el perfil spec"


def test_aiperf_detecta_el_venv():
    ok, why = bench.aiperf_ready(bench.load_config())
    assert ok, f"el venv movido a /root/AIPerf no responde: {why}"


def test_argv_conserva_streaming_y_tokenizer():
    cfg = bench.load_config()
    argv = bench.RunSpec("spec", "spec_al_gsm8k", 2, "m").argv(cfg)
    assert "--streaming" in argv, "sin --streaming no hay time_to_first_token"
    assert "--tokenizer" in argv
    assert argv[argv.index("--tokenizer") + 1] == cfg["aiperf"]["tokenizer"]
    assert argv[argv.index("--concurrency") + 1] == "2"


def test_scalar_desplana_las_tres_formas():
    """El export de resumen usa {"avg":..}, el registro por request {"value":..}."""
    assert bench.scalar({"value": 12.5, "unit": "ms"}) == 12.5
    assert bench.scalar({"avg": 7.0, "p50": 6.0}) == 7.0
    assert bench.scalar(3) == 3.0
    assert bench.scalar(None) is None
    assert bench.scalar({"unit": "ms"}) is None
    assert bench.scalar("n/a") is None


def test_ingesta_tolera_spec_ausente():
    """llama.cpp no emite metrics.speculative_decoding: debe quedar null, no reventar."""
    cfg = bench.load_config()
    spec = bench.RunSpec("spec", "spec_al_gsm8k", 1, "m")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "profile_export_aiperf.json").write_text(json.dumps({
            "aiperf_version": "0.13.0",
            "request_latency": {"avg": 500.0, "p95": 700.0},
            "time_to_first_token": {"avg": 120.0},
            "output_token_throughput": {"avg": 55.0},
            "request_count": {"avg": 4},
            "request_error_rate": {"avg": 0},
            "benchmark_duration": {"avg": 2.2},
            "input_sequence_length": {"avg": 550},
            "output_sequence_length": {"avg": 32},
            "request_throughput": {"avg": 1.8},
        }))
        (d / "profile_export.jsonl").write_text(
            # forma real de un registro por request: dict con value+unit
            json.dumps({"metrics": {"request_latency": {"value": 500.0, "unit": "ms"}}}) + "\n"
            + '{"metrics": {"request_lat'          # línea a medias, como al tail-ear
        )
        entry = bench.ingest(d, spec, cfg)
    assert entry is not None
    assert entry["profile"] == "spec"
    assert entry["ttft_avg"] == 120.0
    assert entry["spec_decode_acceptance_length"] is None, "spec ausente debe quedar null"
    assert entry["acceptance_histogram"] is None
    assert entry["isl_avg"] == 550


def test_ingesta_promedia_spec_presente():
    """Sin las métricas en el resumen, se caen a los registros por request."""
    cfg = bench.load_config()
    spec = bench.RunSpec("spec", "spec_al_gsm8k", 1, "m")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "profile_export_aiperf.json").write_text('{"aiperf_version": "0.13.0"}')
        (d / "profile_export.jsonl").write_text("".join(
            json.dumps({"metrics": {"spec_decode_acceptance_length": {"value": v}}}) + "\n"
            for v in (2.0, 4.0)))
        entry = bench.ingest(d, spec, cfg)
    assert entry["spec_decode_acceptance_length"] == 3.0


def test_ingesta_prefiere_el_pooled_del_resumen():
    """AIPerf ya poola con la ponderación correcta; no se re-promedia a mano."""
    cfg = bench.load_config()
    spec = bench.RunSpec("spec", "spec_al_gsm8k", 1, "m")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "profile_export_aiperf.json").write_text(json.dumps({
            "spec_decode_acceptance_length": {"avg": 2.79},
            "pooled_spec_decode_acceptance_histogram": {"0": 260, "1": 200, "2": 169, "3": 459},
        }))
        # Los per-request dirían otra cosa: el resumen gana.
        (d / "profile_export.jsonl").write_text(
            json.dumps({"metrics": {"spec_decode_acceptance_length": {"value": 9.9}}}) + "\n")
        entry = bench.ingest(d, spec, cfg)
    assert entry["spec_decode_acceptance_length"] == 2.79
    assert entry["acceptance_histogram"] == {"0": 260, "1": 200, "2": 169, "3": 459}


def test_ingesta_guarda_el_comando():
    """Sin el argv exacto, una corrida no es reproducible ni comparable."""
    cfg = bench.load_config()
    spec = bench.RunSpec("spec", "spec_al_gsm8k", 1, "m")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "profile_export_aiperf.json").write_text('{"aiperf_version": "0.13.0"}')
        (d / "cmd.txt").write_text("aiperf profile --model m --spec mtp\n")
        assert bench.ingest(d, spec, cfg)["aiperf_cmd"] == "aiperf profile --model m --spec mtp"
        (d / "cmd.txt").unlink()
        assert bench.ingest(d, spec, cfg)["aiperf_cmd"] == ""


def test_progress_muestra_fase_y_log():
    """Antes del primer request no hay jsonl: la UI debe decir qué está pasando,
    no mostrar '0 requests' en silencio (que es como se escondió un --model vacío)."""
    cfg = bench.load_config()
    spec = bench.RunSpec("spec", "spec_al_gsm8k", 1, "m")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        r = bench.Runner(spec, cfg, d)
        p = r.progress()
        assert p["phase"] == "starting" and p["records"] == 0

        r.log_lines.append("  Downloading dataset Math-500 …")
        p = r.progress()
        assert p["phase"] == "preparing", "log but no jsonl, it is preparing"
        assert "Downloading" in p["log_tail"]

        (d / "profile_export.jsonl").write_text(
            json.dumps({"metrics": {"request_latency": {"value": 500.0}}}) + "\n")
        assert r.progress()["phase"] == "measuring"


def test_spec_config_normaliza_los_dos_motores():
    """MTP depth es justo lo que hace distintas dos corridas; debe ser comparable."""
    ninfer = spec_config(["ninfer-serve", "m.ninfer", "--spec", "mtp", "--draft-tokens", "3",
                          "--lm-head-draft", "--max-concurrency", "4"])
    assert ninfer["spec_backend"] == "mtp" and ninfer["draft_depth"] == 3
    assert ninfer["lm_head_draft"] is True

    llama = spec_config(["llama-server", "-m", "x.gguf", "--spec-draft-n-max", "5",
                         "--spec-draft-p-min", "0.55"])
    assert llama["spec_backend"] == "llama.cpp-draft" and llama["draft_depth"] == 5
    assert llama["draft_p_min"] == "0.55"

    assert spec_config(["ninfer-serve", "m.ninfer"])["spec_backend"] == "none"
    assert spec_config(["ninfer-serve", "m.ninfer"])["draft_depth"] is None


def test_ingesta_trae_el_retrato_del_servidor():
    """Sin server.json la línea queda sin config: la UI no puede explicarla."""
    cfg = bench.load_config()
    spec = bench.RunSpec("spec", "spec_al_gsm8k", 1, "m")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "profile_export_aiperf.json").write_text('{"aiperf_version": "0.13.0"}')
        (d / "server.json").write_text(json.dumps(
            {"binary": "ninfer-serve", "spec_backend": "mtp", "draft_depth": 3, "slots": 4}))
        entry = bench.ingest(d, spec, cfg)
        assert entry["server"]["draft_depth"] == 3
        assert entry["server"]["slots"] == 4

        (d / "server.json").unlink()
        sin = bench.ingest(d, spec, cfg)
    assert sin["server"] == {}, "sin retrato la línea lo registra vacío, no lo inventa"


def test_reingest_conserva_identidad():
    """run_id y ts son identidad de la corrida, no medidas: no se recalculan.

    Trabaja sobre una copia del registro: reingerir de verdad es una operación
    destructiva (reescribe el append-only) y un check no puede tocar el real.
    """
    real = Path(bench.load_config()["aiperf"]["registry"])
    if not real.exists() or not real.read_text().strip():
        return  # sin corridas reales que reconstruir; nada que verificar aquí
    cfg = dict(bench.load_config())
    with tempfile.TemporaryDirectory() as tmp:
        cfg["aiperf"] = dict(cfg["aiperf"], registry=str(Path(tmp) / "registry.jsonl"))
        shutil.copy(real, cfg["aiperf"]["registry"])
        before = {r["run_id"]: r.get("spec_decode_acceptance_length")
                  for r in bench.list_runs(Path(cfg["aiperf"]["registry"]))}
        bench.cmd_reingest(type("A", (), {})(), cfg)
        after = {r["run_id"]: r.get("spec_decode_acceptance_length")
                 for r in bench.list_runs(Path(cfg["aiperf"]["registry"]))}
    assert before, "el registro real debe tener corridas para reconstruir"
    assert set(before) == set(after), "reingest no debe perder ni inventar corridas"
    assert any(v is not None for v in after.values()), \
        "tras reingerir, las corridas de ninfer deben traer spec-decode"


def test_registro_filtra_por_perfil():
    with tempfile.TemporaryDirectory() as tmp:
        reg = Path(tmp) / "registry.jsonl"
        for prof in ("spec", "chat"):
            bench.append_registry(reg, {"profile": prof, "latency_p95": 1.0})
        assert len(bench.list_runs(reg)) == 2
        assert len(bench.list_runs(reg, "spec")) == 1, "comparar exige el mismo perfil"


def demo():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("bench: todo ok")


if __name__ == "__main__":
    demo()
