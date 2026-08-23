# Transcripción de audio a texto con Whisper (faster-whisper).
#
# Este módulo es deliberadamente independiente del sistema multiagente: no
# importa nada de orquestacion_langgraph/ ni de rag/. Cristian pidió trabajar
# Whisper aislado antes de integrarlo al grafo, y mantener esa frontera permite
# medir el ASR por separado.
#
# Uso desde la línea de comandos (desde la raíz del repo):
#   python -m asr.transcribir ruta/al/audio.wav

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from asr.config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_IDIOMA,
    WHISPER_MODELO,
)

# El modelo se carga una sola vez y se reutiliza: cargarlo cuesta ~37 s con
# 'large-v3' en GPU, así que hacerlo por transcripción haría inservible la app
# de grabación. Se carga en la primera llamada y no al importar, para que
# importar este módulo (por ejemplo, en las pruebas) no cueste nada.
_modelo = None


def obtener_modelo():
    global _modelo
    if _modelo is None:
        from faster_whisper import WhisperModel

        print(
            f"Cargando Whisper '{WHISPER_MODELO}' en {WHISPER_DEVICE} "
            f"({WHISPER_COMPUTE_TYPE})... la primera vez tarda"
        )
        inicio = time.time()
        _modelo = WhisperModel(
            WHISPER_MODELO, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
        print(f"  listo en {time.time() - inicio:.0f} s")
    return _modelo


@dataclass
class Transcripcion:
    """Resultado de transcribir un audio.

    `texto` es la transcripción completa. Los demás campos son las señales que
    hacen falta para evaluar el ASR, en particular `sin_voz`: ver más abajo.
    """

    texto: str
    idioma: str
    probabilidad_idioma: float
    duracion_audio_s: float
    tiempo_transcripcion_s: float
    segmentos: list[dict] = field(default_factory=list)

    # True cuando el detector de actividad de voz (VAD) no encontró habla.
    #
    # Importa porque Whisper INVENTA texto sobre audio sin voz: con 3 segundos
    # de ruido gaussiano puro devolvió " Gracias." y reportó probabilidad de
    # idioma 1.00 (medido el 2026-08-19). Un dispositivo que escucha todo el día
    # en la casa de una persona mayor va a recibir mucho audio sin habla, y ese
    # texto inventado entraría al sistema como si fuera una consulta real.
    #
    # Quien consuma esta clase tiene que mirar este campo, no solo `texto`.
    sin_voz: bool = False

    def __str__(self):
        marca = " [SIN VOZ DETECTADA]" if self.sin_voz else ""
        return f"{self.texto!r}{marca}"


def transcribir(ruta_audio: Path | str, usar_vad: bool = True) -> Transcripcion:
    """Transcribe un archivo de audio.

    `usar_vad=True` activa el filtro de actividad de voz de faster-whisper, que
    descarta los tramos sin habla antes de pasarlos al modelo. Reduce las
    alucinaciones pero no las elimina: por eso el resultado trae `sin_voz`.

    Ponerlo en False sirve para medir cuánto texto inventa Whisper sin ninguna
    protección, que es una de las mediciones de esta fase.
    """
    ruta_audio = Path(ruta_audio)
    if not ruta_audio.exists():
        raise FileNotFoundError(f"No existe el audio: {ruta_audio}")

    modelo = obtener_modelo()

    inicio = time.time()
    segmentos_iter, info = modelo.transcribe(
        str(ruta_audio),
        language=WHISPER_IDIOMA,
        vad_filter=usar_vad,
    )
    # faster-whisper devuelve un generador perezoso: la transcripción real
    # ocurre al recorrerlo, así que el tiempo se mide después de materializarlo.
    segmentos = [
        {
            "inicio_s": round(s.start, 2),
            "fin_s": round(s.end, 2),
            "texto": s.text,
        }
        for s in segmentos_iter
    ]
    tiempo = time.time() - inicio

    texto = " ".join(s["texto"] for s in segmentos).strip()

    return Transcripcion(
        texto=texto,
        idioma=info.language,
        probabilidad_idioma=round(info.language_probability, 4),
        duracion_audio_s=round(info.duration, 2),
        tiempo_transcripcion_s=round(tiempo, 2),
        segmentos=segmentos,
        sin_voz=not segmentos,
    )


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m asr.transcribir ruta/al/audio.wav")
        return 1

    for ruta in sys.argv[1:]:
        r = transcribir(ruta)
        print()
        print(f"Archivo    : {ruta}")
        print(f"Texto      : {r.texto!r}")
        print(f"Idioma     : {r.idioma} (prob {r.probabilidad_idioma})")
        print(f"Duración   : {r.duracion_audio_s} s")
        print(f"Transcribió: {r.tiempo_transcripcion_s} s "
              f"({r.duracion_audio_s / max(r.tiempo_transcripcion_s, 0.01):.1f}x tiempo real)")
        print(f"Segmentos  : {len(r.segmentos)}")
        if r.sin_voz:
            print("AVISO: el VAD no detectó voz en este audio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
