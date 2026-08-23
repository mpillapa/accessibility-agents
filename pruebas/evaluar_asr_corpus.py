# Evaluación de Whisper sobre el corpus grabado, por condición de ruido.
#
# Uso (desde la raíz del repo):
#   python -m pruebas.evaluar_asr_corpus                  # corpus completo
#   python -m pruebas.evaluar_asr_corpus --muestra 100    # solo N frases
#   python -m pruebas.evaluar_asr_corpus --solo-matriz    # generar audio y salir
#
# Requiere GPU y el corpus grabado (ver demo_voz/). No requiere VPN.
#
# Mide tres cosas por condición de ruido:
#   1. WER y CER — cuánto se degrada la transcripción.
#   2. Transcripciones vacías — cuándo el VAD se come voz real. Ésta es la
#      contracara del hallazgo de que el VAD elimina el 100% de las
#      alucinaciones: si además se come frases legítimas, el remedio tiene un
#      costo que hay que conocer.
#   3. Alucinaciones conocidas — si aparece la muletilla de YouTube sobre audio
#      que SÍ tiene voz.

import argparse
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from asr.config import CORPUS_DIR, CORPUS_GRABACIONES_DIR, CORPUS_INDICE
from asr.metricas import contiene_alucinacion_conocida, resumir, wer
from asr.ruido import NIVELES_SNR_DB, TIPOS_RUIDO_SINTETICO, generar_matriz
from asr.transcribir import transcribir

VARIANTES_DIR = CORPUS_DIR / "variantes"
RESULTADOS = CORPUS_DIR / "resultados_asr.json"


def cargar_corpus(limite: int | None = None) -> list[dict]:
    if not CORPUS_INDICE.exists():
        print(f"No hay corpus grabado en {CORPUS_INDICE}.")
        print("Grabar con: python -m demo_voz.servidor")
        return []
    with open(CORPUS_INDICE, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    if limite and limite < len(filas):
        # Muestra estratificada: misma cantidad por intención, para que la
        # accuracy de ruteo siga siendo medible sobre el subconjunto.
        por_intencion = defaultdict(list)
        for fila in filas:
            por_intencion[fila["intencion"]].append(fila)
        cupo = max(1, limite // len(por_intencion))
        filas = [f for grupo in por_intencion.values() for f in grupo[:cupo]]
    return filas


def construir_matriz(corpus: list[dict]) -> list[dict]:
    """Genera todas las variantes con ruido. Devuelve el índice de variantes."""
    if VARIANTES_DIR.exists():
        shutil.rmtree(VARIANTES_DIR)
    VARIANTES_DIR.mkdir(parents=True)

    condiciones = 1 + (len(NIVELES_SNR_DB) - 1) * len(TIPOS_RUIDO_SINTETICO)
    print(f"Generando {len(corpus)} audios x {condiciones} condiciones "
          f"= {len(corpus) * condiciones} archivos...")

    indice = []
    inicio = time.time()
    for i, fila in enumerate(corpus, 1):
        origen = CORPUS_GRABACIONES_DIR / fila["archivo"]
        if not origen.exists():
            continue
        for variante in generar_matriz(origen, VARIANTES_DIR):
            indice.append({**variante,
                           "id_frase": fila["id_frase"],
                           "intencion": fila["intencion"],
                           "referencia": fila["texto"]})
        if i % 50 == 0 or i == len(corpus):
            print(f"  {i}/{len(corpus)}  ({time.time() - inicio:.0f}s)")
    return indice


def transcribir_todo(indice: list[dict]) -> list[dict]:
    print(f"\nTranscribiendo {len(indice)} audios...")
    inicio = time.time()
    resultados = []
    for i, entrada in enumerate(indice, 1):
        r = transcribir(VARIANTES_DIR / entrada["archivo"])
        resultados.append({
            **entrada,
            "hipotesis": r.texto,
            "sin_voz": r.sin_voz,
            "wer": round(wer(entrada["referencia"], r.texto), 4),
            "wer_sin_tildes": round(
                wer(entrada["referencia"], r.texto, quitar_tildes=True), 4),
            "alucinacion": contiene_alucinacion_conocida(r.texto),
        })
        if i % 200 == 0 or i == len(indice):
            transcurrido = time.time() - inicio
            print(f"  {i}/{len(indice)}  ({transcurrido:.0f}s, "
                  f"faltan ~{transcurrido / i * (len(indice) - i):.0f}s)")
    return resultados


def condicion_de(fila: dict) -> str:
    if fila["tipo_ruido"] == "ninguno":
        return "limpio"
    return f"{fila['tipo_ruido']} {fila['snr_db_objetivo']}dB"


def reportar(resultados: list[dict]) -> None:
    por_condicion = defaultdict(list)
    for fila in resultados:
        por_condicion[condicion_de(fila)].append(fila)

    def orden(clave):
        if clave == "limpio":
            return (0, 0, "")
        tipo, snr = clave.split()
        return (1, -int(snr.replace("dB", "")), tipo)

    print("\n" + "=" * 78)
    print("WER POR CONDICIÓN DE RUIDO")
    print("=" * 78)
    print(f"{'condición':<18} {'n':>5} {'WER':>8} {'WER s/tild':>11} "
          f"{'CER':>8} {'vacías':>8} {'alucin.':>8}")
    print("-" * 78)

    for clave in sorted(por_condicion, key=orden):
        filas = por_condicion[clave]
        r = resumir([(f["referencia"], f["hipotesis"]) for f in filas])
        wer_st = sum(f["wer_sin_tildes"] for f in filas) / len(filas)
        print(f"{clave:<18} {r['n']:>5} {r['wer']:>8.3f} {wer_st:>11.3f} "
              f"{r['cer']:>8.3f} {r['transcripciones_vacias']:>8} "
              f"{r['con_alucinacion_conocida']:>8}")

    limpio = por_condicion.get("limpio", [])
    if limpio:
        r = resumir([(f["referencia"], f["hipotesis"]) for f in limpio])
        print("\n" + "-" * 78)
        print(f"Línea base (audio limpio): WER {r['wer']:.1%}")
        vacias = r["transcripciones_vacias"]
        if vacias:
            print(f"\nATENCIÓN: {vacias}/{r['n']} audios LIMPIOS quedaron sin "
                  f"transcripción.\n  El VAD descartó voz real. Es el costo del "
                  f"filtro que elimina las alucinaciones.")

    # Frases donde el VAD se comió voz real, por condición: la medición
    # pendiente más importante de esta fase.
    print("\n" + "=" * 78)
    print("FALSOS NEGATIVOS DEL VAD (audio con voz que quedó sin transcribir)")
    print("=" * 78)
    for clave in sorted(por_condicion, key=orden):
        filas = por_condicion[clave]
        vacias = sum(1 for f in filas if not f["hipotesis"].strip())
        if vacias:
            print(f"  {clave:<18} {vacias:>4}/{len(filas)}  ({vacias/len(filas):.0%})")


def reportar_por_intencion(resultados: list[dict]) -> None:
    """El WER por intención importa porque las consecuencias no son simétricas:
    una emergencia mal transcrita es más grave que un saludo mal transcrito."""
    limpio = [f for f in resultados if f["tipo_ruido"] == "ninguno"]
    if not limpio:
        return
    print("\n" + "=" * 78)
    print("WER POR INTENCIÓN (audio limpio)")
    print("=" * 78)
    por_intencion = defaultdict(list)
    for fila in limpio:
        por_intencion[fila["intencion"]].append(fila)
    for intencion in sorted(por_intencion):
        filas = por_intencion[intencion]
        r = resumir([(f["referencia"], f["hipotesis"]) for f in filas])
        print(f"  {intencion:<24} n={r['n']:>4}  WER {r['wer']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muestra", type=int, help="evaluar solo N frases")
    parser.add_argument("--solo-matriz", action="store_true",
                        help="generar los audios con ruido y salir")
    args = parser.parse_args()

    corpus = cargar_corpus(args.muestra)
    if not corpus:
        return 1
    print(f"Corpus: {len(corpus)} frases grabadas")

    indice = construir_matriz(corpus)
    if args.solo_matriz:
        print(f"\n{len(indice)} archivos en {VARIANTES_DIR}")
        return 0

    resultados = transcribir_todo(indice)
    RESULTADOS.write_text(json.dumps(resultados, ensure_ascii=False, indent=1),
                          encoding="utf-8")

    reportar(resultados)
    reportar_por_intencion(resultados)
    print(f"\nResultados crudos: {RESULTADOS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
