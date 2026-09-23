import os
import re
from typing import Literal

from pydantic import BaseModel, Field

from orquestacion_langgraph.estado import EstadoConversacion
# La config del LLM vive en llm.py para que el subgrafo de RAG pueda usarla
# sin importar este módulo (que a su vez importa el subgrafo).
from orquestacion_langgraph.llm import (
    VLLM_API_KEY,
    VLLM_CHAT_BASE_URL,
    VLLM_CHAT_MODEL,
    llm,
)
from orquestacion_langgraph.rag_agentico.subgrafo import consultar_recetario

INTENCIONES = [
    "MEDICATION_HEALTH",
    "RECIPE_MULTIMEDIA",
    "FAMILY_COMMUNICATION",
    "EMERGENCY",
    "SMALL_TALK",
]



class DecisionIntencion(BaseModel):
    razonamiento: str = Field(description="Justificación en español de la intención detectada")
    intencion: Literal[
        "MEDICATION_HEALTH",
        "RECIPE_MULTIMEDIA",
        "FAMILY_COMMUNICATION",
        "EMERGENCY",
        "SMALL_TALK",
    ]


def _extraer_decision(texto: str) -> DecisionIntencion:
    match = re.search(r"CATEGORIA:\s*([A-Z_]+)", texto)
    candidata = match.group(1) if match else None
    intencion = candidata if candidata in INTENCIONES else next(
        (o for o in INTENCIONES if o in texto), "SMALL_TALK"
    )
    razonamiento = texto.split("CATEGORIA:")[0].strip() or texto
    return DecisionIntencion(intencion=intencion, razonamiento=razonamiento)


def nodo_orchestrator(estado: EstadoConversacion) -> dict:
    respuesta = llm.invoke(
        "Eres el primer punto de contacto de un asistente para adultos mayores. "
        "Hablan español coloquial ecuatoriano. Analiza brevemente (1-2 líneas) la "
        f"intención detrás de esta consulta: '{estado['consulta']}'. Las categorías "
        f"posibles son: {', '.join(INTENCIONES)}. Termina tu respuesta en una última "
        "línea con el formato exacto: CATEGORIA: <una de esas categorías>"
    ).content
    decision = _extraer_decision(respuesta)
    return {"intencion": decision.intencion, "razonamiento": decision.razonamiento}


# Edge condicional: devuelve la INTENCIÓN detectada, y el grafo la traduce a
# nodo (ver construir_grafo). Devolver la intención en vez del nombre del nodo
# mantiene la decisión en el vocabulario del dominio y deja que el diagrama
# etiquete cada flecha con la intención que la dispara.
def ruta_siguiente_nodo(estado: EstadoConversacion) -> str:
    return estado["intencion"]


# Especialista en consultas de medicación y salud básica (mismo rol que CrewAI).
def nodo_medicacion(estado: EstadoConversacion) -> dict:
    respuesta = llm.invoke(
        "Eres un asistente especializado en gestión de medicación para adultos "
        "mayores. Conoces el inventario de pastillas, horarios y posibles "
        "interacciones. Respondes en español, de forma clara y empática. "
        "En esta versión simulas el acceso a la base de datos de la familia.\n\n"
        f"Consulta del usuario: '{estado['consulta']}'"
    )
    return {"respuesta": respuesta.content}



# Especialista en recetas: delega en el subgrafo de RAG agéntico.
#
# Antes este nodo hacía la recuperación él mismo (una llamada a buscar_receta()
# con k fijo) y le pasaba al LLM lo que saliera, relevante o no. Ahora invoca
# un subgrafo que decide si buscar, evalúa lo recuperado, reformula la consulta
# si hace falta y admite cuando no encontró nada. Ver rag_agentico/README.md.
#
# El subgrafo tiene su propio estado (EstadoRAG): el grafo principal no necesita
# conocer los intentos ni los fragmentos descartados, solo la respuesta y la
# traza. Ese límite es lo que permite reemplazar el RAG sin tocar el grafo.
def nodo_recetas(estado: EstadoConversacion) -> dict:
    resultado = consultar_recetario(estado["consulta"])
    return {
        "respuesta": resultado["respuesta"],
        "traza_rag": resultado["traza"],
    }


# Stub de comunicación con familia: no ejecuta acción real, solo confirma.
def nodo_familia_stub(estado: EstadoConversacion) -> dict:
    return {
        "respuesta": (
            "Tu mensaje para la familia fue recibido y será procesado. "
            "(STUB — en la versión completa esto se sincroniza con la app familiar)"
        )
    }


# Número de emergencias. 911 es el del ECU 911 (Ecuador); se deja configurable
# para no fijar por código algo que cambia según el país donde se despliegue.
NUMERO_DE_EMERGENCIAS = os.getenv("NUMERO_DE_EMERGENCIAS", "911")

# Qué se responde ante una emergencia. Es texto FIJO, no generado por el LLM, y
# esa es una decisión deliberada por dos razones:
#
# 1. Un mensaje de emergencia no puede depender de que el modelo redacte bien
#    esta vez. Es el único camino del sistema donde equivocarse tiene
#    consecuencias físicas, así que no se delega en algo que puede alucinar.
#    Es la misma regla que aplica el guardrail del ASR (ver voz.py): lo crítico
#    va en código determinista.
#
# 2. Quitar la llamada al LLM quita también su latencia y su dependencia del
#    servidor. Si la VPN está caída o el endpoint rotando de modelo —lo que
#    pasó cuatro veces en trece días—, TODO el sistema falla menos esto.
#
# El aviso final es igual de deliberado: la persona tiene que saber que nadie
# fue avisado todavía. Dejar creer que ya viene ayuda, cuando no viene, es peor
# que no responder nada.
#
# NO da instrucciones sobre qué hacer físicamente, y eso también es a propósito.
# El sistema clasifica en una sola categoría EMERGENCY: no distingue una caída
# de un incendio o de un dolor de pecho. Un consejo único sería contraproducente
# en alguno de esos casos —"no se mueva" es correcto ante una posible fractura y
# peligroso ante humo en la cocina—, y un prototipo académico no validado
# clínicamente no está en posición de instruir sobre primeros auxilios.
# Lo único seguro que puede hacer es dirigir a quien sí sabe, rápido y claro.
MENSAJE_DE_EMERGENCIA = (
    "**Llame al {numero} ahora mismo.**\n\n"
    "Es el número de emergencias. Si hay alguien cerca, pídale ayuda.\n\n"
    "_Este asistente todavía no puede llamar por usted: la llamada la tiene que "
    "hacer usted o alguien que esté a su lado._"
)


# Especialista en emergencias.
#
# No llama al LLM a propósito (ver MENSAJE_DE_EMERGENCIA). Sigue siendo un
# prototipo: no marca el teléfono, no avisa a un contacto ni activa ningún
# protocolo. Lo único que hace es dar la instrucción correcta de inmediato y
# decir con claridad qué NO hizo.
def nodo_emergencia(estado: EstadoConversacion) -> dict:
    return {"respuesta": MENSAJE_DE_EMERGENCIA.format(numero=NUMERO_DE_EMERGENCIAS)}


# Small talk: no llamamos ningún especialista, igual que la versión A1 de CrewAI.
def nodo_small_talk(estado: EstadoConversacion) -> dict:
    return {"respuesta": "¡Hola! ¿En qué puedo ayudarte hoy?"}
