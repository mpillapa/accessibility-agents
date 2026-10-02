# Tablas y gráficos de la campaña a partir del JSONL (chuleta 8.1).
#
# Uso (desde la raíz del repo; no necesita VPN):
#   python -m medicion.analizar resultados/campana/campana_2026-10-01.jsonl
#
# Escribe junto al JSONL:
#   <experimento>_tablas.md        las 6 tablas, listas para leer o pasar a LaTeX
#   <experimento>_usuarios.png     tiempo por usuario en T1 y T6 (boxplot)
#   <experimento>_rag.png          dónde se va el tiempo dentro del RAG (T2 y T6)
#
# Reglas de cálculo (declararlas en el paper):
# - Media ± DE muestral sobre TODAS las ejecuciones completas de la tarea
#   (5 usuarios × 4 frases × 3 repeticiones = 60 por tarea si no hubo cortes).
# - Tiempos y tokens excluyen las ejecuciones con timeout o excepción: su
#   tiempo es el del corte, no el del sistema. Se informan aparte y SÍ cuentan
#   como fallo en la tasa de éxito.
# - "Por agente" usa las ejecuciones donde ese agente corrió. Si el Orchestrator
#   ruteó mal, el agente que corrió es el equivocado y aparece en la tabla de
#   esa tarea con su propio n; así se ve el costo real de un error de ruteo.

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from medicion.casos import TAREAS, cargar_campana
from medicion.estadistica import kruskal_wallis, media_de, wilson

ORDEN_AGENTES = ["orchestrator", "medicacion", "recetas", "emergencia", "familia", "small_talk",
                 "medicacion_cruce", "recetas_cruce", "integrador"]


def _f(valores, decimales=1) -> str:
    m, s = media_de(valores)
    if m is None:
        return "—"
    return f"{m:.{decimales}f} ± {s:.{decimales}f}" if s is not None else f"{m:.{decimales}f}"


def _p95(valores):
    return statistics.quantiles(valores, n=20)[18] if len(valores) >= 20 else max(valores)


def _completas(filas):
    return [f for f in filas if not f["timeout"] and not f["error"]]


def cargar(ruta: Path) -> list[dict]:
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def tabla_por_agente(filas, nombres) -> list[str]:
    lineas = ["| Tarea | Agente | n | Tiempo (s) | Llamadas LLM | Tokens entrada | Tokens salida | de ellos, razonamiento |",
              "|---|---|---|---|---|---|---|---|"]
    for tarea in TAREAS:
        completas = _completas([f for f in filas if f["tarea"] == tarea])
        agentes = {a for f in completas for a in f["agentes"]}
        for agente in sorted(agentes, key=lambda a: ORDEN_AGENTES.index(a) if a in ORDEN_AGENTES else 99):
            datos = [f["agentes"][agente] for f in completas if agente in f["agentes"]]
            lineas.append(
                f"| {tarea} {nombres[tarea]} | {agente} | {len(datos)} | {_f([d['segundos'] for d in datos], 2)} | "
                f"{_f([d['llamadas_llm'] for d in datos])} | {_f([d['tokens_entrada'] for d in datos], 0)} | "
                f"{_f([d['tokens_salida'] for d in datos], 0)} | {_f([d['tokens_razonamiento'] for d in datos], 0)} |")
    return lineas


def tabla_sistema(filas, nombres) -> list[str]:
    lineas = ["| Tarea | n | Tiempo (s) | Mediana | P95 | Llamadas LLM | Tokens entrada | Tokens salida | Razonamiento | % razonamiento |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    grupos = [(f"{t} {nombres[t]}", [f for f in filas if f["tarea"] == t]) for t in TAREAS] + [("Todas", filas)]
    for etiqueta, grupo in grupos:
        completas = _completas(grupo)
        if not completas:
            continue
        seg = [f["segundos_sistema"] for f in completas]
        tok = [f["tokens_sistema"] for f in completas]
        salida = sum(t["salida"] for t in tok)
        pct = 100 * sum(t["razonamiento"] for t in tok) / salida if salida else 0
        lineas.append(
            f"| {etiqueta} | {len(completas)} | {_f(seg, 2)} | {statistics.median(seg):.2f} | {_p95(seg):.2f} | "
            f"{_f([t['llamadas_llm'] for t in tok])} | {_f([t['entrada'] for t in tok], 0)} | "
            f"{_f([t['salida'] for t in tok], 0)} | {_f([t['razonamiento'] for t in tok], 0)} | {pct:.0f}% |")
    return lineas


def tabla_usuarios(filas, usuarios) -> list[str]:
    lineas = []
    for tarea in TAREAS:
        completas = _completas([f for f in filas if f["tarea"] == tarea])
        if not completas:
            continue
        por_usuario = {u: [f for f in completas if f["usuario"] == u] for u in usuarios}
        kw_t = kruskal_wallis([[f["segundos_sistema"] for f in g] for g in por_usuario.values()])
        kw_s = kruskal_wallis([[f["tokens_sistema"]["salida"] for f in g] for g in por_usuario.values()])
        celdas = " | ".join(f"{_f([f['segundos_sistema'] for f in por_usuario[u]], 1)}" for u in usuarios)
        lineas.append(f"| {tarea} | {celdas} | {_kw(kw_t)} | {_kw(kw_s)} |")
    encabezado = ["| Tarea | " + " | ".join(usuarios) + " | Kruskal-Wallis tiempo | Kruskal-Wallis tokens salida |",
                  "|---|" + "---|" * len(usuarios) + "---|---|"]
    return encabezado + lineas


def _kw(r) -> str:
    if r["H"] is None:
        return "—"
    p = f"p = {r['p']:.3f}" if r["p"] is not None else "p = ?"
    return f"H = {r['H']:.2f}, {p}"


def tabla_exito(filas, nombres) -> list[str]:
    lineas = ["| Tarea | Éxitos | % | IC 95% (Wilson) | Timeouts | Errores | Chequeos que fallaron |",
              "|---|---|---|---|---|---|---|"]
    grupos = [(f"{t} {nombres[t]}", [f for f in filas if f["tarea"] == t]) for t in TAREAS] + [("Todas", filas)]
    for etiqueta, grupo in grupos:
        if not grupo:
            continue
        k, n = sum(f["exito"] for f in grupo), len(grupo)
        bajo, alto = wilson(k, n)
        fallos = Counter(c for f in grupo for c, ok in f["chequeos"].items() if not ok)
        detalle = ", ".join(f"{c} {v}" for c, v in fallos.most_common()) or "—"
        lineas.append(f"| {etiqueta} | {k}/{n} | {100 * k / n:.1f}% | {100 * bajo:.1f}–{100 * alto:.1f}% | "
                      f"{sum(f['timeout'] for f in grupo)} | {sum(bool(f['error']) for f in grupo)} | {detalle} |")
    return lineas


def tabla_caminos_t6(filas) -> list[str]:
    t6 = [f for f in filas if f["tarea"] == "T6"]
    if not t6:
        return ["Sin ejecuciones de T6."]
    correctos = sum(f["chequeos"].get("camino", False) for f in t6)
    bajo, alto = wilson(correctos, len(t6))
    lineas = [f"Camino correcto (orchestrator → {{medicacion_cruce, recetas_cruce}} → integrador): "
              f"**{correctos}/{len(t6)}** ({100 * correctos / len(t6):.1f}%, IC 95% {100 * bajo:.1f}–{100 * alto:.1f}%).", "",
              "| Camino recorrido | Veces |", "|---|---|"]
    for camino, veces in Counter(" → ".join(f["camino"]) or "(sin camino: timeout o error)" for f in t6).most_common():
        lineas.append(f"| {camino} | {veces} |")
    lineas += ["", "Con camino correcto, qué más falló:", "", "| Chequeo | Fallos |", "|---|---|"]
    con_camino = [f for f in t6 if f["chequeos"].get("camino")]
    for chequeo in ("fuente", "interacciones", "sin_error"):
        lineas.append(f"| {chequeo} | {sum(not f['chequeos'].get(chequeo, True) for f in con_camino)} |")
    return lineas


def tabla_rag(filas) -> list[str]:
    lineas = ["| Tarea | Agente | Subnodo | Veces por ejecución | Tiempo por vez (s) | % del tiempo del agente |",
              "|---|---|---|---|---|---|"]
    for tarea, agente in (("T2", "recetas"), ("T6", "recetas_cruce")):
        completas = [f for f in _completas([f for f in filas if f["tarea"] == tarea]) if agente in f["agentes"]]
        if not completas:
            continue
        total_agente = sum(f["agentes"][agente]["segundos"] for f in completas)
        por_subnodo = defaultdict(list)
        for f in completas:
            for s in f["subnodos"]:
                if s["agente"] == agente:
                    por_subnodo[s["nodo"]].append(s["segundos"])
        for nodo, tiempos in sorted(por_subnodo.items(), key=lambda x: -sum(x[1])):
            lineas.append(f"| {tarea} | {agente} | {nodo} | {len(tiempos) / len(completas):.2f} | "
                          f"{_f(tiempos, 2)} | {100 * sum(tiempos) / total_agente:.0f}% |")
    return lineas


def figuras(filas, usuarios, base: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hechas = []
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
    for eje, tarea in zip(ejes, ("T1", "T6")):
        completas = _completas([f for f in filas if f["tarea"] == tarea])
        eje.boxplot([[f["segundos_sistema"] for f in completas if f["usuario"] == u] or [0] for u in usuarios],
                    tick_labels=usuarios)
        eje.set_title(f"{tarea}: tiempo del sistema por usuario")
        eje.set_ylabel("segundos")
    fig.tight_layout()
    ruta = base.with_name(base.name + "_usuarios.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    hechas.append(ruta.name)

    fig, eje = plt.subplots(figsize=(7, 4))
    for i, (tarea, agente) in enumerate((("T2", "recetas"), ("T6", "recetas_cruce"))):
        completas = [f for f in _completas([f for f in filas if f["tarea"] == tarea]) if agente in f["agentes"]]
        if not completas:
            continue
        suma = defaultdict(float)
        for f in completas:
            for s in f["subnodos"]:
                if s["agente"] == agente:
                    suma[s["nodo"]] += s["segundos"]
        abajo = 0.0
        for nodo, total in sorted(suma.items()):
            media = total / len(completas)
            eje.bar(i, media, bottom=abajo, label=nodo if i == 0 or nodo not in eje.get_legend_handles_labels()[1] else None)
            abajo += media
    eje.set_xticks([0, 1], ["T2 recetas", "T6 recetas_cruce"])
    eje.set_ylabel("segundos por ejecución (media)")
    eje.set_title("Dónde se va el tiempo dentro del RAG")
    eje.legend(fontsize=8)
    fig.tight_layout()
    ruta = base.with_name(base.name + "_rag.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    hechas.append(ruta.name)
    return hechas


def main():
    parser = argparse.ArgumentParser(description="Tablas de la campaña de medición")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--sin-figuras", action="store_true")
    args = parser.parse_args()

    filas = cargar(args.jsonl)
    campana = cargar_campana()
    nombres = {t: campana["tareas"][t]["nombre"] for t in TAREAS}
    usuarios = campana["usuarios"]
    base = args.jsonl.with_suffix("")
    meta_ruta = base.with_name(base.name + "_meta.json")
    meta = json.loads(meta_ruta.read_text(encoding="utf-8")) if meta_ruta.exists() else {}

    incompletas = len(filas) - len(_completas(filas))
    texto = [
        f"# Campaña `{base.name}`", "",
        f"- Ejecuciones: **{len(filas)}** de {meta.get('total_planificado', '?')} planificadas; "
        f"{incompletas} con timeout o error (excluidas de tiempos y tokens, cuentan como fallo).",
        f"- Commit: `{meta.get('commit')}` (cambios sin commit: {meta.get('cambios_sin_commit')}).",
        f"- Modelos: {sorted({m for f in filas for m in f['modelos']})}.",
        f"- `system_fingerprint` vistos: {sorted({x for f in filas for x in f['fingerprints']})}.",
        f"- T3 decididas por la red de emergencia (sin LLM): "
        f"{sum(f['decidio_la_red_de_emergencia'] for f in filas if f['tarea'] == 'T3')}"
        f"/{sum(1 for f in filas if f['tarea'] == 'T3')}.",
        "- Valores: media ± DE muestral. Los tokens de razonamiento son PARTE de los de salida.", "",
        "## 1. Tokens y tiempo por agente", "", *tabla_por_agente(filas, nombres), "",
        "## 2. Tokens y tiempo del sistema", "", *tabla_sistema(filas, nombres), "",
        "## 3. ¿Cambia con el usuario? Tiempo del sistema (s) por usuario", "", *tabla_usuarios(filas, usuarios), "",
        "## 4. Tasa de éxito", "", *tabla_exito(filas, nombres), "",
        "## 5. Caminos en T6", "", *tabla_caminos_t6(filas), "",
        "## 6. Desglose del tiempo dentro del RAG", "", *tabla_rag(filas), "",
    ]
    if not args.sin_figuras:
        texto += ["## Figuras", ""] + [f"![{n}]({n})" for n in figuras(filas, usuarios, base)]
    ruta = base.with_name(base.name + "_tablas.md")
    ruta.write_text("\n".join(texto) + "\n", encoding="utf-8")
    print(f"Tablas en {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
