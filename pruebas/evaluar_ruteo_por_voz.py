# Precisión de ruteo que se pierde al entrar por voz en vez de por texto (bitácora 10.3).
#
# Uso (desde la raíz del repo):
#   python -m pruebas.evaluar_ruteo_por_voz
#   python -m pruebas.evaluar_ruteo_por_voz --por-condicion 50
#
# Requiere VPN (usa el LLM) y haber corrido antes:
#   python -m pruebas.evaluar_asr_corpus
# que deja las transcripciones en corpus_audio/resultados_asr.json.
#
# Distingue ruteo incorrecto (eligió mal el agente) de no ruteable (el VAD
# descartó el audio y no hubo nada que rutear).

import argparse
import json
from collections import defaultdict
from pathlib import Path

from asr.config import CORPUS_DIR
from orquestacion_langgraph.grafo import clasificar_consulta

RESULTADOS_ASR = CORPUS_DIR / "resultados_asr.json"
SALIDA = CORPUS_DIR / "resultados_ruteo_voz.json"


def condicion_de(fila: dict) -> str:
    if fila["tipo_ruido"] == "ninguno":
        return "limpio"
    return f"{fila['tipo_ruido']} {fila['snr_db_objetivo']}dB"


def orden_condicion(clave: str):
    if clave == "limpio":
        return (0, 0, "")
    tipo, snr = clave.split()
    return (1, -int(snr.replace("dB", "")), tipo)


def clasificar(texto: str) -> str | None:
    """None cuando no hay texto que rutear."""
    if not texto.strip():
        return None
    try:
        return clasificar_consulta(texto)
    except Exception as e:
        return f"ERROR:{type(e).__name__}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--por-condicion", type=int, default=50,
                        help="frases por condición (default 50; 0 = todas)")
    args = parser.parse_args()

    if not RESULTADOS_ASR.exists():
        print(f"Falta {RESULTADOS_ASR}.")
        print("Correr primero: python -m pruebas.evaluar_asr_corpus")
        return 1

    datos = json.loads(RESULTADOS_ASR.read_text(encoding="utf-8"))
    por_condicion = defaultdict(list)
    for fila in datos:
        por_condicion[condicion_de(fila)].append(fila)

    # Muestra estratificada por intención para no sesgar la accuracy.
    if args.por_condicion:
        for clave, filas in por_condicion.items():
            agrupadas = defaultdict(list)
            for f in filas:
                agrupadas[f["intencion"]].append(f)
            cupo = max(1, args.por_condicion // len(agrupadas))
            por_condicion[clave] = [f for g in agrupadas.values() for f in g[:cupo]]

    total_llamadas = sum(len(v) for v in por_condicion.values())
    base = por_condicion.get("limpio", [])
    print(f"Clasificando {total_llamadas} transcripciones + {len(base)} textos "
          f"de referencia (requiere VPN)...\n")

    # Línea base: el texto de referencia, sin pasar por audio.
    aciertos_texto = 0
    for fila in base:
        aciertos_texto += clasificar(fila["referencia"]) == fila["intencion"]
    accuracy_texto = aciertos_texto / len(base) if base else 0.0
    print(f"Línea base con TEXTO de referencia: {accuracy_texto:.1%} "
          f"({aciertos_texto}/{len(base)})\n")

    filas_salida = []
    resumen = []
    for clave in sorted(por_condicion, key=orden_condicion):
        filas = por_condicion[clave]
        correctos = incorrectos = no_ruteables = errores = 0
        confusiones = defaultdict(int)

        for fila in filas:
            predicha = clasificar(fila["hipotesis"])
            if predicha is None:
                no_ruteables += 1
                estado = "no_ruteable"
            elif str(predicha).startswith("ERROR:"):
                errores += 1
                estado = "error"
            elif predicha == fila["intencion"]:
                correctos += 1
                estado = "correcto"
            else:
                incorrectos += 1
                estado = "incorrecto"
                confusiones[f"{fila['intencion']} -> {predicha}"] += 1

            filas_salida.append({
                "condicion": clave,
                "id_frase": fila["id_frase"],
                "intencion_real": fila["intencion"],
                "intencion_predicha": predicha,
                "estado": estado,
                "referencia": fila["referencia"],
                "hipotesis": fila["hipotesis"],
                "wer": fila["wer"],
            })

        evaluables = len(filas) - errores
        resumen.append({
            "condicion": clave,
            "n": len(filas),
            "correctos": correctos,
            "incorrectos": incorrectos,
            "no_ruteables": no_ruteables,
            "accuracy": correctos / evaluables if evaluables else 0.0,
            "confusiones": dict(sorted(confusiones.items(), key=lambda x: -x[1])[:3]),
        })
        print(f"  {clave:<18} accuracy={resumen[-1]['accuracy']:.1%}  "
              f"(bien {correctos}, mal {incorrectos}, sin transcripción {no_ruteables})")

    print("\n" + "=" * 78)
    print("DEGRADACIÓN DE RUTEO AL ENTRAR POR VOZ")
    print("=" * 78)
    print(f"{'condición':<18} {'n':>4} {'accuracy':>9} {'vs texto':>10} "
          f"{'mal ruteado':>12} {'sin transcr.':>13}")
    print("-" * 78)
    for r in resumen:
        delta = r["accuracy"] - accuracy_texto
        print(f"{r['condicion']:<18} {r['n']:>4} {r['accuracy']:>8.1%} "
              f"{delta:>+9.1%} {r['incorrectos']:>12} {r['no_ruteables']:>13}")

    print("\nConfusiones más frecuentes por condición:")
    for r in resumen:
        if r["confusiones"]:
            print(f"  {r['condicion']}:")
            for confusion, veces in r["confusiones"].items():
                print(f"     {confusion}  x{veces}")

    # Las emergencias mal ruteadas se destacan porque el costo no es simétrico.
    emergencias_perdidas = [
        f for f in filas_salida
        if f["intencion_real"] == "EMERGENCY" and f["estado"] in ("incorrecto", "no_ruteable")
    ]
    print("\n" + "=" * 78)
    print(f"EMERGENCIAS NO ATENDIDAS: {len(emergencias_perdidas)}")
    print("=" * 78)
    print("Una emergencia mal ruteada o sin transcribir es el peor fallo posible")
    print("de este sistema: la persona pidió ayuda y no pasó nada.\n")
    for f in emergencias_perdidas[:12]:
        destino = f["intencion_predicha"] or "SIN TRANSCRIPCIÓN"
        print(f"  [{f['condicion']:<14}] -> {destino}")
        print(f"     dijo    : {f['referencia'][:66]}")
        print(f"     entendió: {(f['hipotesis'] or '(nada)')[:66]}")
    if len(emergencias_perdidas) > 12:
        print(f"  ... y {len(emergencias_perdidas) - 12} más (ver el JSON)")

    SALIDA.write_text(
        json.dumps({"accuracy_texto": accuracy_texto, "resumen": resumen,
                    "detalle": filas_salida}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\nResultados crudos: {SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
