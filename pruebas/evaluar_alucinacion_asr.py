# ¿Cuánto texto inventa Whisper cuando no hay nadie hablando?
#
# Uso (desde la raíz del repo):
#   python -m pruebas.evaluar_alucinacion_asr
#   python -m pruebas.evaluar_alucinacion_asr --json resultados.json
#
# Requiere GPU y el modelo de Whisper. NO requiere VPN ni corpus grabado: el
# audio se genera acá, y la gracia del experimento es justamente que no
# contiene voz.
#
# Por qué importa: el sistema está pensado para un dispositivo que escucha en la
# casa de una persona mayor. La mayor parte del tiempo no va a haber nadie
# hablándole, pero sí va a haber televisión, cocina y ruido de calle. Si el ASR
# inventa texto sobre ese audio, ese texto entra al Orchestrator como si fuera
# una consulta real y puede activar un agente — en el peor caso, el de
# emergencias.
#
# Esto mide algo distinto del WER, que asume que hay voz que transcribir y
# compara palabra por palabra. Acá la referencia es la cadena vacía: cualquier
# texto es un error.

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

from asr.ruido import escribir_wav, generar_ruido
from asr.transcribir import transcribir

FRECUENCIA = 16000

# Cada condición es audio SIN VOZ. La amplitud está en unidades de 16 bits:
# 32768 es el máximo, así que 300 es un ambiente tranquilo y 6000 es ruido
# fuerte y constante.
CONDICIONES = [
    {"id": "silencio_digital",   "tipo": "silencio", "amplitud": 0,
     "nota": "silencio absoluto, el caso más fácil"},
    {"id": "ambiente_muy_bajo",  "tipo": "blanco",   "amplitud": 100,
     "nota": "habitación en calma"},
    {"id": "ambiente_bajo",      "tipo": "rosa",     "amplitud": 300,
     "nota": "murmullo lejano, aire acondicionado"},
    {"id": "ambiente_medio",     "tipo": "rosa",     "amplitud": 1500,
     "nota": "televisión en otra habitación"},
    {"id": "ruido_fuerte_rosa",  "tipo": "rosa",     "amplitud": 6000,
     "nota": "cocina en uso, calle con tráfico"},
    {"id": "ruido_fuerte_blanco","tipo": "blanco",   "amplitud": 6000,
     "nota": "ruido siseante fuerte (extractor, grifo abierto)"},
]

DURACIONES_S = [3, 10, 30]


def generar_audio(condicion: dict, segundos: int, ruta: Path) -> None:
    n = FRECUENCIA * segundos
    if condicion["tipo"] == "silencio":
        muestras = [0] * n
    else:
        # Semilla fija por condición y duración: el experimento tiene que dar el
        # mismo audio en cada corrida para que los resultados sean comparables.
        semilla = abs(hash((condicion["id"], segundos))) % (2**31)
        normalizado = generar_ruido(n, tipo=condicion["tipo"], semilla=semilla)
        muestras = [
            max(-32768, min(32767, int(condicion["amplitud"] * v))) for v in normalizado
        ]

    import array
    escribir_wav(ruta, array.array("h", muestras), FRECUENCIA)


def evaluar(usar_vad: bool, directorio: Path) -> list[dict]:
    filas = []
    for condicion in CONDICIONES:
        for segundos in DURACIONES_S:
            ruta = directorio / f"{condicion['id']}_{segundos}s.wav"
            generar_audio(condicion, segundos, ruta)

            r = transcribir(ruta, usar_vad=usar_vad)
            # Cualquier texto es una alucinación: no hay voz en la entrada.
            invento = bool(r.texto.strip())

            filas.append({
                "condicion": condicion["id"],
                "nota": condicion["nota"],
                "tipo_ruido": condicion["tipo"],
                "amplitud": condicion["amplitud"],
                "duracion_s": segundos,
                "vad": usar_vad,
                "invento_texto": invento,
                "texto": r.texto,
                "caracteres": len(r.texto),
                "segmentos": len(r.segmentos),
                "prob_idioma": r.probabilidad_idioma,
                "sin_voz_reportado": r.sin_voz,
            })
    return filas


def resumir(filas: list[dict], etiqueta: str) -> None:
    total = len(filas)
    inventos = sum(f["invento_texto"] for f in filas)
    print(f"\n{etiqueta}: {inventos}/{total} audios sin voz produjeron texto "
          f"({inventos/total:.0%})")

    if inventos:
        print("  Casos donde inventó:")
        for f in filas:
            if f["invento_texto"]:
                print(f"    {f['condicion']:20s} {f['duracion_s']:2d}s  "
                      f"prob_idioma={f['prob_idioma']:.2f}  {f['texto'][:70]!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="guarda los resultados crudos")
    args = parser.parse_args()

    print("Alucinación de Whisper sobre audio SIN VOZ")
    print(f"{len(CONDICIONES)} condiciones x {len(DURACIONES_S)} duraciones x 2 (con/sin VAD)")
    print("La referencia correcta es la cadena vacía: cualquier texto es un error.\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print("--- Sin filtro VAD (Whisper crudo) ---")
        sin_vad = evaluar(usar_vad=False, directorio=tmp)
        print("--- Con filtro VAD (la protección que trae faster-whisper) ---")
        con_vad = evaluar(usar_vad=True, directorio=tmp)

    resumir(sin_vad, "SIN VAD")
    resumir(con_vad, "CON VAD")

    todos = sin_vad + con_vad
    print("\n" + "=" * 74)
    a, b = sum(f["invento_texto"] for f in sin_vad), sum(f["invento_texto"] for f in con_vad)
    print(f"El filtro VAD redujo las alucinaciones de {a} a {b} "
          f"(de {len(sin_vad)} audios sin voz).")

    if args.json:
        Path(args.json).write_text(json.dumps(todos, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"Resultados crudos en {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
