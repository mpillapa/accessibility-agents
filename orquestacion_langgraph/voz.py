# Entrada por voz del grafo: transcribe el audio y decide si se puede confiar
# en lo transcrito antes de dejarlo entrar al sistema.
#
# POR QUÉ ESTE NODO NO ES SOLO "TRANSCRIBIR Y SEGUIR"
# ---------------------------------------------------
# Whisper SIEMPRE devuelve texto. No tiene forma de decir "no escuché nada":
# ante audio sin habla genera lo más frecuente de su entrenamiento, que son
# cierres de video de YouTube. Medido sobre 18 audios sin voz: 18/18 (100%)
# produjeron texto, y con probabilidad de idioma 1.00 — máxima confianza sobre
# algo inventado.
#
# El caso que resume el problema (2026-08-22):
#
#     La persona dijo : "Me caí en el baño y no me puedo levantar, ayúdame"
#     Whisper oyó     : "Gracias por ver el video."
#     El Orchestrator : SMALL_TALK (conversación trivial)
#     El sistema      : no hizo nada
#
# Ningún componente falló: cada uno procesó correctamente una entrada que
# parecía válida. Sobre el corpus completo, **31 de 180 emergencias (17%) no se
# atendieron**. Y los errores de ruteo no se reparten al azar: el 49% cae en
# SMALL_TALK y solo el 8% en EMERGENCY, o sea que el sistema falla hacia el lado
# inseguro — no genera falsas alarmas, pierde emergencias reales.
#
# Conectar el ASR al grafo sin verificar nada reproduciría exactamente ese
# fallo. Por eso la transcripción pasa por dos filtros antes de convertirse en
# una consulta, y si no los pasa el sistema PREGUNTA en vez de adivinar.
#
# LAS DOS CAPAS DEL GUARDRAIL
# ---------------------------
# 1. `sin_voz` — el detector de actividad de voz (VAD) no encontró habla.
#    Elimina el 100% de las alucinaciones sobre audio sin voz (18/18 -> 0/18),
#    a un costo del 2% de frases reales perdidas, medido sobre 415 frases.
#
# 2. `contiene_alucinacion_conocida()` — muletillas típicas de Whisper que el
#    VAD deja pasar cuando hay algo de ruido con estructura. Es una red de
#    seguridad, no la defensa principal: la lista es finita y el modelo puede
#    inventar frases que no estén en ella.
#
# Ninguna de las dos garantiza nada. Por eso el nodo de descarte pide repetir
# en vez de asumir: ante una entrada dudosa, preguntar es más seguro que
# clasificar.

from typing import Optional

from orquestacion_langgraph.estado import EstadoConversacion

# Un audio que el VAD marca sin voz no entra al sistema, aunque Whisper haya
# devuelto texto. Esta es la regla que evita el caso de la emergencia perdida.
DESCARTAR_SI_VAD_NO_DETECTA_VOZ = True

# Además del VAD, se descarta el texto que coincide con las muletillas que
# Whisper inventa. Ver asr/metricas.MULETILLAS_ALUCINADAS.
DESCARTAR_MULETILLAS_CONOCIDAS = True

# Las dos ramas del edge condicional que sale de `transcribir_voz`. Son texto
# descriptivo y no el nombre del nodo destino a propósito: LangGraph las usa
# como etiqueta de la flecha en el diagrama, así que el grafo dibujado explica
# POR QUÉ se toma cada rama sin tener que leer el código.
RAMA_CONTINUAR = "transcripcion confiable"
RAMA_DESCARTAR = "no se entendio"

# Qué se le responde a la persona cuando la entrada no se pudo verificar.
# Pedir que repita es deliberado: ante duda, preguntar en vez de clasificar.
MENSAJE_NO_SE_ENTENDIO = (
    "Perdón, no le escuché bien. ¿Me lo puede repetir, por favor?"
)


def _motivo_para_descartar(texto: str, sin_voz: bool) -> Optional[str]:
    """Devuelve por qué no se puede confiar en esta transcripción, o None.

    Separado del nodo para poder probar la regla sin cargar Whisper ni el
    grafo: es una función pura sobre el resultado del ASR.
    """
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
    """Transcribe `ruta_audio` y deja el texto en `consulta`, si es confiable.

    El import de `asr.transcribir` es perezoso a propósito: cargar
    faster-whisper cuesta ~37 s y exige GPU. Haciéndolo acá, el grafo se puede
    importar y las pruebas correr en una máquina sin ASR instalado; solo falla
    quien realmente use la entrada por voz.
    """
    from asr.transcribir import transcribir

    resultado = transcribir(estado["ruta_audio"], usar_vad=True)
    motivo = _motivo_para_descartar(resultado.texto, resultado.sin_voz)

    return {
        # La consulta solo se puebla si la transcripción es confiable. Si no,
        # queda vacía y el edge condicional desvía antes del Orchestrator.
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
    """Edge condicional: al Orchestrator solo si la transcripción es confiable.

    Este desvío es el guardrail hecho explícito en la topología del grafo. Que
    sea un edge y no un `if` dentro de un nodo es deliberado: así queda visible
    en el diagrama y en las trazas de LangSmith.
    """
    return RAMA_DESCARTAR if estado.get("entrada_descartada") else RAMA_CONTINUAR


def nodo_no_se_entendio(estado: EstadoConversacion) -> dict:
    """Responde pidiendo que repita, sin clasificar la entrada dudosa."""
    return {
        "respuesta": MENSAJE_NO_SE_ENTENDIO,
        "intencion": "NO_SE_ENTENDIO",
        "razonamiento": estado.get("entrada_descartada"),
    }
