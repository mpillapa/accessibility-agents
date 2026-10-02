# Transcripción de audio con Whisper (faster-whisper). No importa nada del grafo
# ni de rag/ (asr/README.md). Usa GPU por defecto.
#
# Uso (desde la raíz del repo):
#   python -m asr.transcribir ruta/al/audio.wav

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from asr.config import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_IDIOMA,
    WHISPER_MODELO,
)

# Carga perezosa y única: ~37 s con large-v3 en GPU.
_modelo = None

# La interfaz precarga en un hilo; el candado evita cargar dos modelos en la GPU.
_candado_carga = threading.Lock()


def obtener_modelo():
    global _modelo
    with _candado_carga:
        return _cargar_si_hace_falta()


def _cargar_si_hace_falta():
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
    """Resultado de transcribir un audio. Mirar `sin_voz`, no solo `texto`."""

    texto: str
    idioma: str
    probabilidad_idioma: float
    duracion_audio_s: float
    tiempo_transcripcion_s: float
    segmentos: list[dict] = field(default_factory=list)

    # El VAD no encontró habla. Whisper inventa texto sobre audio sin voz (asr/README.md).
    sin_voz: bool = False

    def __str__(self):
        marca = " [SIN VOZ DETECTADA]" if self.sin_voz else ""
        return f"{self.texto!r}{marca}"


def transcribir(ruta_audio: Path | str, usar_vad: bool = True) -> Transcripcion:
    """Transcribe un archivo de audio.

    `usar_vad=False` sirve para medir cuánto inventa Whisper sin filtro de voz.
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
    # El generador es perezoso: la transcripción ocurre al recorrerlo.
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
