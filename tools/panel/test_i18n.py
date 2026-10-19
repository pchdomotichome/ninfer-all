"""Catch Spanish left in the panel's visible strings.

A heuristic, not a proof: it greps the sources for Spanish words and accents in
the places a label can live (text between tags, and quoted literals that are not
markup). It cannot see a label built at runtime from pieces.

Four rounds of this let strings through, because Spanish function words carry
no accent: `en cola`, `aceptados`, `contexto`, `residencia`, `motor`. The word
list is therefore broad and the accent check is the weaker of the two.

The rendered DOM is the real oracle; this keeps the obvious regressions from
surviving a commit. When a string slips through, add the word to SPANISH here
and fix the string -- the list is meant to grow.

    python3 test_i18n.py
"""
from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent

SPANISH = re.compile(
    r"\b(que|como|para|pero|esta|este|esto|hay|masi|menos|cada|debe|queda|"
    r"sirve|cambia|guarda|muestra|linea|tabla|modelo|modelos|motor|motores|"
    r"carga|descargar|descargando|cargando|dormir|dormido|despertar|activo|"
    r"reposo|caido|cola|ventana|ronda|rondas|hora|actualizado|ahora|nuevo|"
    r"vacio|vacia|usado|estado|cerrar|abrir|guardar|cancelar|volver|atras|"
    r"siguiente|anterior|buscar|elegir|seleccionar|ninguno|ninguna|todos|"
    r"todas|copiado|listo|preparando|iniciando|terminado|midiendo|corrida|"
    r"corridas|favorito|contexto|residencia|aceptados|arquitectura|capas|"
    r"concurrencia|correr|errores|seg|objetivo|promedio|protocolo|tensores|"
    r"verificado|promedio|descargados|encolado|admitidos|liberar|sobrevive|"
    r"ultima|ultimo|primera|queda|haber|puede|deben|vamos|estara|estar)\b", re.I)

ACCENTS = re.compile(r"[áéíóúñÁÉÍÓÚ¿¡]")

# Markup, code, paths, flags and css hooks are not labels.
NOT_LABEL = re.compile(r"[<>{};()=]|https?:|//|\{\{|\$\{|=>|\.\w|--\w|\bnone\b")

TAGS = re.compile(r">([^<>{}]{2,})<")
QUOTES = re.compile(r"'([^'\n]{3,})'|\"([^\"\n]{3,})\"")


def candidates(path: pathlib.Path):
    for n, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(("//", "#", "*", "/*")):
            continue
        for m in TAGS.finditer(line):
            text = m.group(1).strip()
            if len(text) >= 2 and not NOT_LABEL.search(text):
                yield n, text
        # A literal holding markup is a template: its quoted attributes were
        # already covered by the TAGS pass, and scanning them is pure noise.
        if path.suffix != ".js" or "<" in line:
            continue
        for m in QUOTES.finditer(line):
            text = (m.group(1) or m.group(2)).strip()
            if len(text) >= 3 and not NOT_LABEL.search(text):
                yield n, text


def main() -> int:
    hits = 0
    for pattern in ("index.html", "css/panel.css", "js/*.js"):
        for path in sorted(HERE.glob(pattern)):
            for n, text in candidates(path):
                bad = SPANISH.search(text)
                if bad or ACCENTS.search(text):
                    word = bad.group(0) if bad else "accent"
                    print(f"FAIL {path.name}:{n}: {text[:60]!r}  ({word})")
                    hits += 1
    if hits:
        print(f"i18n: {hits} visible strings are not English")
        return 1
    print("i18n: no Spanish found in the visible strings")
    return 0


if __name__ == "__main__":
    sys.exit(main())