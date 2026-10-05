# Compara dos campañas con la misma matriz y un solo factor distinto
# (p. ej. el cliente LLM tal como se midió vs con el razonamiento ajustado).
#
# Uso (desde la raíz del repo; no necesita VPN):
#   python -m medicion.comparar resultados/campana/campana_2026-10-02.jsonl \
#       resultados/campana/ajustada_2026-10-04.jsonl --nombres "tal como está" "razonamiento bajo"
#
# Tiempos y tokens: solo ejecuciones completas; Mann-Whitney a dos colas.
# Éxito: todas, con IC de Wilson (un timeout es fallo en las dos).

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from medicion.analizar import _completas, cargar
from medicion.casos import TAREAS, cargar_campana
from medicion.estadistica import mann_whitney, media_de, wilson


def _mediana(valores):
    return statistics.median(valores) if valores else None


def _p(r) -> str:
    return "—" if r["p"] is None else ("< 0.001" if r["p"] < 0.001 else f"{r['p']:.3f}")


def tabla_exito(a, b, nombres, tareas) -> list[str]:
    lineas = [f"| Tarea | Éxito {nombres[0]} | Éxito {nombres[1]} | Timeouts {nombres[0]} | Timeouts {nombres[1]} |",
              "|---|---|---|---|---|"]
    for t in tareas + ["Todas"]:
        fa = [f for f in a if t == "Todas" or f["tarea"] == t]
        fb = [f for f in b if t == "Todas" or f["tarea"] == t]
        celdas = []
        for g in (fa, fb):
            k, n = sum(f["exito"] for f in g), len(g)
            bajo, alto = wilson(k, n) if n else (0, 0)
            celdas.append(f"{k}/{n} ({100 * k / n:.1f}%, IC {100 * bajo:.0f}–{100 * alto:.0f})" if n else "—")
        lineas.append(f"| {t} | {celdas[0]} | {celdas[1]} | {sum(f['timeout'] for f in fa)} | {sum(f['timeout'] for f in fb)} |")
    return lineas


def tabla_metrica(a, b, nombres, tareas, titulo, extraer, decimales) -> list[str]:
    lineas = [f"| Tarea | {titulo} {nombres[0]} (media ± DE; mediana) | {titulo} {nombres[1]} | Cambio de la mediana | Mann-Whitney p |",
              "|---|---|---|---|---|"]
    for t in tareas + ["Todas"]:
        va = [extraer(f) for f in _completas(a) if t == "Todas" or f["tarea"] == t]
        vb = [extraer(f) for f in _completas(b) if t == "Todas" or f["tarea"] == t]
        if not va or not vb:
            continue
        celdas = []
        for v in (va, vb):
            m, de = media_de(v)
            celdas.append(f"{m:.{decimales}f} ± {de or 0:.{decimales}f}; {_mediana(v):.{decimales}f}")
        ma, mb = _mediana(va), _mediana(vb)
        cambio = f"{100 * (mb - ma) / ma:+.0f}%" if ma else "—"
        lineas.append(f"| {t} | {celdas[0]} | {celdas[1]} | {cambio} | {_p(mann_whitney(va, vb))} |")
    return lineas


def tabla_ruteo(a, b, nombres) -> list[str]:
    """Las frases que rutearon mal en alguna de las dos campañas."""
    def errores(filas):
        return Counter(f["frase"] for f in _completas(filas) if not f["chequeos"].get("intencion", True))
    ea, eb = errores(a), errores(b)
    na, nb = Counter(f["frase"] for f in _completas(a)), Counter(f["frase"] for f in _completas(b))
    frases = sorted(set(ea) | set(eb))
    if not frases:
        return ["Ningún error de ruteo en las dos campañas."]
    consulta = {f["frase"]: f["consulta"] for f in a + b}
    lineas = [f"| Frase | Errores de ruteo {nombres[0]} | {nombres[1]} |", "|---|---|---|"]
    for fr in frases:
        lineas.append(f"| {fr} {consulta[fr]!r} | {ea.get(fr, 0)}/{na.get(fr, 0)} | {eb.get(fr, 0)}/{nb.get(fr, 0)} |")
    return lineas


def tabla_t1_usuarios(a, b, nombres, usuarios) -> list[str]:
    lineas = [f"| Usuario | Tiempo de T1 {nombres[0]} (mediana, s) | {nombres[1]} | Tokens salida {nombres[0]} | {nombres[1]} |",
              "|---|---|---|---|---|"]
    for u in usuarios:
        ga = [f for f in _completas(a) if f["tarea"] == "T1" and f["usuario"] == u]
        gb = [f for f in _completas(b) if f["tarea"] == "T1" and f["usuario"] == u]
        if not ga or not gb:
            continue
        lineas.append(f"| {u} | {_mediana([f['segundos_sistema'] for f in ga]):.1f} | "
                      f"{_mediana([f['segundos_sistema'] for f in gb]):.1f} | "
                      f"{_mediana([f['tokens_sistema']['salida'] for f in ga]):.0f} | "
                      f"{_mediana([f['tokens_sistema']['salida'] for f in gb]):.0f} |")
    return lineas


def main():
    parser = argparse.ArgumentParser(description="Comparación entre dos campañas")
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--nombres", nargs=2, default=["A", "B"])
    parser.add_argument("--salida", type=Path)
    args = parser.parse_args()

    a, b = cargar(args.a), cargar(args.b)
    usuarios = cargar_campana()["usuarios"]
    tareas = [t for t in TAREAS if any(f["tarea"] == t for f in b)]
    n = args.nombres
    texto = [
        f"# Comparación: {n[0]} (`{args.a.stem}`) vs {n[1]} (`{args.b.stem}`)", "",
        f"- Ejecuciones: {len(a)} vs {len(b)}. Tiempos y tokens solo de ejecuciones completas.",
        "- Mann-Whitney a dos colas sobre todas las ejecuciones completas de cada tarea.", "",
        "## Éxito", "", *tabla_exito(a, b, n, tareas), "",
        "## Tiempo del sistema (s)", "",
        *tabla_metrica(a, b, n, tareas, "Tiempo", lambda f: f["segundos_sistema"], 1), "",
        "## Tokens de salida (incluye razonamiento)", "",
        *tabla_metrica(a, b, n, tareas, "Tokens", lambda f: f["tokens_sistema"]["salida"], 0), "",
        "## Tokens de razonamiento", "",
        *tabla_metrica(a, b, n, tareas, "Razonamiento", lambda f: f["tokens_sistema"]["razonamiento"], 0), "",
        "## Frases mal ruteadas", "", *tabla_ruteo(a, b, n), "",
        "## T1 por usuario", "", *tabla_t1_usuarios(a, b, n, usuarios), "",
    ]
    salida = args.salida or args.b.with_name(f"comparacion_{args.a.stem}_vs_{args.b.stem}.md")
    salida.write_text("\n".join(texto) + "\n", encoding="utf-8")
    print(f"Comparación en {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
