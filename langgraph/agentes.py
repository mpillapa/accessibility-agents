import os
import re
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from estado import EstadoConversacion

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.buscar import buscar_receta


load_dotenv(Path(__file__).parent.parent / ".env", override=True)

VLLM_CHAT_BASE_URL = os.getenv("VLLM_CHAT_BASE_URL", "http://172.28.230.10:12559/v1")
VLLM_CHAT_MODEL = os.getenv("VLLM_CHAT_MODEL", "google/gemma-4-12B-it")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "local")

llm = ChatOpenAI(
    model=VLLM_CHAT_MODEL,
    base_url=VLLM_CHAT_BASE_URL,
    api_key=VLLM_API_KEY,
    temperature=0.3,  # bajo para que el ruteo sea consistente
)

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


# Edge condicional: decide el próximo nodo según la intención detectada.
def ruta_siguiente_nodo(estado: EstadoConversacion) -> str:
    return {
        "MEDICATION_HEALTH": "medicacion",
        "RECIPE_MULTIMEDIA": "recetas",
        "FAMILY_COMMUNICATION": "familia",
        "EMERGENCY": "emergencia",
        "SMALL_TALK": "small_talk",
    }[estado["intencion"]]


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



# Especialista en recetas y multimedia culinaria (mismo rol que CrewAI), hace la recuperación de información antes de llamar al LLM.
def nodo_recetas(estado: EstadoConversacion) -> dict:
    fragmentos = buscar_receta(estado["consulta"], k=3)
    if fragmentos:
        contexto = "\n---\n".join(fragmentos)
    else:
        contexto = (
            "(El recetario no tiene resultados para esta consulta, o todavía "
            "no se ha ingerido — correr 'python rag/ingesta.py'.)"
        )

    respuesta = llm.invoke(
        "Eres un asistente culinario que ayuda a personas mayores a preparar "
        "comidas. Adaptas medidas técnicas a referencias cotidianas (ej: 300ml "
        "= un vaso grande) para que sean comprensibles sin instrumentos de "
        "medición. Guías paso a paso y respondes en español. No inventes "
        "ingredientes ni pasos que no estén en el recetario de abajo.\n\n"
        f"Recetario (resultado de la búsqueda):\n{contexto}\n\n"
        f"Consulta del usuario: '{estado['consulta']}'"
    )
    return {"respuesta": respuesta.content}


# Stub de comunicación con familia: no ejecuta acción real, solo confirma.
def nodo_familia_stub(estado: EstadoConversacion) -> dict:
    return {
        "respuesta": (
            "Tu mensaje para la familia fue recibido y será procesado. "
            "(STUB — en la versión completa esto se sincroniza con la app familiar)"
        )
    }


# Stub de emergencias: no ejecuta acción real, solo confirma.
def nodo_emergencia_stub(estado: EstadoConversacion) -> dict:
    return {
        "respuesta": (
            "Se registró tu aviso de emergencia. "
            "(STUB — en producción esto activaría protocolos locales de auxilio)"
        )
    }


# Small talk: no llamamos ningún especialista, igual que la versión A1 de CrewAI.
def nodo_small_talk(estado: EstadoConversacion) -> dict:
    return {"respuesta": "¡Hola! ¿En qué puedo ayudarte hoy?"}
