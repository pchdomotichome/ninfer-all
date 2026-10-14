"""Self-check de Baselines: las tres reglas y el aprendizaje.

Sin esto las reglas solo se prueban provocando una falla real, que es caro y
no se puede repetir. Aca se alimentan muestreos sinteticos con un reloj
inyectado, asi que corren en milisegundos.

    python3 test_baselines.py
"""

import json
import sys
import tempfile
from pathlib import Path

from server import Baselines


def sample(tokens: int, running: int = 1, model: str = "m1"):
    """Un muestreo como el que produce el lector: /v1/load con contadores."""
    return {"reachable": True,
            "load": {"model": model,
                     "counters": {"committed_decode_tokens": tokens},
                     "requests": {"running": running}}}


def feed_rates(b: Baselines, rates: list, t0: float = 1000.0, **kw) -> float:
    """Muestrea una serie de tasas a paso fijo de 2 s.

    El primer muestreo solo deja fijado el delta anterior; desde el segundo se
    calcula la tasa real (delta de tokens / dt)."""
    b.feed(sample(0, **kw), now=t0)
    tokens, t = 0, t0
    for rate in rates:
        tokens += int(round(rate * 2.0))
        t += 2.0
        b.feed(sample(tokens, **kw), now=t)
    return t0 + 2.0 * len(rates)


def keys(b: Baselines) -> set:
    with b.lock:
        return {a["key"] for a in b.alerts}


def level_of(b: Baselines, key: str) -> str | None:
    with b.lock:
        found = next((a for a in b.alerts if a["key"] == key), None)
        return found["level"] if found else None


def main() -> int:
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    tmp = lambda: Path(tempfile.mkdtemp()) / "state.json"  # noqa: E731

    # --- 1. aprende y no alerta con velocidad normal ------------------------
    b = Baselines(tmp())
    feed_rates(b, [100.0] * 20)
    check(b._median(b.rates["m1"]) == 100.0,
          f"la mediana debe ser 100, es {b._median(b.rates.get('m1', []))}")
    check(not keys(b), f"no debe alertar en velocidad normal: {keys(b)}")

    # --- 2. sustained drop: 3 samples below 25% of the median ----------
    t = feed_rates(b, [10.0, 11.0, 9.0], t0=2000.0)
    check("decode_below_baseline" in keys(b),
          f"debio alertar decode_below_baseline: {keys(b)}")
    check(level_of(b, "decode_below_baseline") == "critical",
          "3 seguidas deben escalar a critical")

    # Dos seguidas no bastan: una sola lentitud no significa nada.
    b2 = Baselines(tmp())
    feed_rates(b2, [100.0] * 20)
    feed_rates(b2, [5.0, 5.0], t0=2000.0)
    check("decode_below_baseline" not in keys(b2),
          "con dos seguidas no debe alertar (hacen falta 3)")

    # --- 3. la alerta se cierra al volver a la norma -------------------------
    feed_rates(b, [100.0] * 4, t0=t)
    check("decode_below_baseline" not in keys(b),
          f"la alerta debe cerrarse al recuperarse: {keys(b)}")

    # --- 4. aprendizaje: sin 12 muestras no hay referencia, pero el piso
    #        absoluto sigue aplicando -----------------------------------------
    b3 = Baselines(tmp())
    feed_rates(b3, [100.0] * 5)
    feed_rates(b3, [2.0, 2.0, 2.0], t0=2000.0)
    check("decode_below_baseline" not in keys(b3),
          "con 5 muestras todavia no hay referencia: no debe alertar")
    check("decode_suspiciously_slow" in keys(b3),
          f"debe alertar el piso absoluto: {keys(b3)}")

    # --- 5. reposo no muestrea ----------------------------------------------
    b4 = Baselines(tmp())
    b4.last = (1000, 0.0)
    b4.feed(sample(1100, running=0), now=2.0)
    check("m1" not in b4.rates, "en reposo no debe tomar muestras")

    # --- 6. servidor caido: escala por muestras, no por una sola ------------
    b5 = Baselines(tmp())
    down = {"reachable": False, "load": None}
    b5.feed(down, now=1.0)
    check(level_of(b5, "server_unreachable") == "warning",
          "una muestra caida avisa, no critica")
    for i in range(3):
        b5.feed(down, now=2.0 + i)
    check(level_of(b5, "server_unreachable") == "critical",
          f"cuatro seguidas deben escalar a critical: {level_of(b5, 'server_unreachable')}")
    b5.feed(sample(2000), now=10.0)
    check("server_unreachable" not in keys(b5), "al volver, se cierra")

    # --- 7. el router en reposo NO es una caida -----------------------------
    b6 = Baselines(tmp())
    b6.feed({"reachable": True, "load": {"error": "no model loaded"}}, now=1.0)
    check(not keys(b6), "404 model_not_found es reposo, no una caida")

    # --- 8. persistencia entre reinicios -------------------------------------
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "state.json"
        a = Baselines(p)
        feed_rates(a, [100.0] * 20)
        a._save(force=True)
        saved = json.loads(p.read_text())
        check(len(saved["models"]["m1"]) == 20,
              f"debe guardar 20 muestras, guardo {len(saved['models'].get('m1', []))}")
        b7 = Baselines(p)
        check(len(b7.rates.get("m1", [])) == 20, "debe recuperar las muestras al reiniciar")

    for f in fails:
        print("FAIL:", f)
    print("OK" if not fails else f"{len(fails)} FALLAS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())