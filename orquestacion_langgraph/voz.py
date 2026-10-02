# Entrada por voz del grafo: transcribe el audio y decide si se puede confiar
# en lo transcrito antes de dejarlo entrar al sistema. Whisper siempre devuelve
# texto, aun sin habla; ver README.md raíz, "Entrada por voz y su guardrail".

from typing import Optional

from orquestacion_langgraph.estado import EstadoConversacion

# Regla 1: si el VAD no detecta voz, se descarta aunque Whisper haya devuelto texto.
DESCARTAR_SI_VAD_NO_DETECTA_VOZ = True

# Regla 2: red secundaria con las muletillas que Whisper inventa
# (asr/metricas.MULETILLAS_ALUCINADAS). La lista es finita.
DESCARTAR_MULETILLAS_CONOCIDAS = True

# Texto descriptivo y no nombre de nodo: es la etiqueta de la flecha en el diagrama.
RAMA_CONTINUAR = "transcripcion confiable"
RAMA_DESCARTAR = "no se entendio"

# Ante duda, pedir que repita en vez de clasificar.
MENSAJE_NO_SE_ENTENDIO = (
    "Perdón, no le escuché bien. ¿Me lo puede repetir, por favor?"
)


def _motivo_para_descartar(texto: str, sin_voz: bool) -> Optional[str]:
    """Por qué no se puede confiar en la transcripción, o None."""
    if DESCARTAR_SI_VAD_NO_DETECTA_VOZ and sin_voz:
        return "el detector de voz no encontró habla en el audio"

    if not texto.strip():
        return "la transcripción quedó vacía"

    if DESCARTAR_MULETILLAS_CONOCIDAS:
        from asr.metricas import contiene_alucinacion_conocida

        muletilla = contiene_alucinacion_conocida(texto)
        if muletilla:
            return f"la transcripción es una muletilla típica de Whisper ({muletilla!r})"

    return None


def nodo_transcribir_voz(estado: EstadoConversacion) -> dict:
    """Transcribe `ruta_audio` y deja el texto en `consulta`, si es confiable."""
    # Import perezoso: faster-whisper tarda ~37 s en cargar y exige GPU.
    from asr.transcribir import transcribir

    resultado = transcribir(estado["ruta_audio"], usar_vad=True)
    motivo = _motivo_para_descartar(resultado.texto, resultado.sin_voz)

    return {
        # Vacía si se descarta: así no llega al Orchestrator.
        "consulta": "" if motivo else resultado.texto,
        "transcripcion": {
            "texto": resultado.texto,
            "sin_voz": resultado.sin_voz,
            "idioma": resultado.idioma,
            "probabilidad_idioma": resultado.probabilidad_idioma,
            "duracion_audio_s": resultado.duracion_audio_s,
            "tiempo_transcripcion_s": resultado.tiempo_transcripcion_s,
        },
        "entrada_descartada": motivo,
    }


def ruta_tras_transcribir(estado: EstadoConversacion) -> str:
    """Al Orchestrator solo si la transcripción es confiable.

    Es un edge y no un `if` para que el guardrail se vea en el diagrama y en LangSmith.
    """
    return RAMA_DESCARTAR if estado.get("entrada_descartada") else RAMA_CONTINUAR


def nodo_no_se_entendio(estado: EstadoConversacion) -> dict:
    """Responde pidiendo que repita, sin clasificar la entrada dudosa."""
    return {
        "respuesta": MENSAJE_NO_SE_ENTENDIO,
        "intencion": "NO_SE_ENTENDIO",
        "razonamiento": estado.get("entrada_descartada"),
    }
