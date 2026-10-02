"""Nodos del grafo principal: Orchestrator y especialistas."""

import os
import re
from typing import Literal

from pydantic import BaseModel, Field

from orquestacion_langgraph.estado import EstadoConversacion
# Re-exportados por compatibilidad con la demo y el notebook.
from orquestacion_langgraph.llm import (
    VLLM_API_KEY,
    VLLM_CHAT_BASE_URL,
    VLLM_CHAT_MODEL,
    llm,
)
from orquestacion_langgraph.rag_agentico.subgrafo import consultar_recetario
from orquestacion_langgraph.red_emergencia import detectar_emergencia

INTENCIONES = [
    "MEDICATION_HEALTH",
    "RECIPE_MULTIMEDIA",
    "FAMILY_COMMUNICATION",
    "EMERGENCY",
    "SMALL_TALK",
    # Única intención que activa dos especialistas (ver interacciones/README.md).
    "MEDICATION_FOOD_CHECK",
]



class DecisionIntencion(BaseModel):
    razonamiento: str = Field(description="Justificación en español de la intención detectada")
    intencion: Literal[
        "MEDICATION_HEALTH",
        "RECIPE_MULTIMEDIA",
        "FAMILY_COMMUNICATION",
        "EMERGENCY",
        "SMALL_TALK",
        "MEDICATION_FOOD_CHECK",
    ]


def _extraer_decision(texto: str) -> DecisionIntencion:
    match = re.search(r"CATEGORIA:\s*([A-Z_]+)", texto)
    candidata = match.group(1) if match else None
    intencion = candidata if candidata in INTENCIONES else next(
        (o for o in INTENCIONES if o in texto), "SMALL_TALK"
    )
    razonamiento = texto.split("CATEGORIA:")[0].strip() or texto
    return DecisionIntencion(intencion=intencion, razonamiento=razonamiento)


def _como_se_decidio(texto: str) -> str:
    """Devuelve "formato", "nombre_suelto" o "por_defecto" (cayó a SMALL_TALK)."""
    match = re.search(r"CATEGORIA:\s*([A-Z_]+)", texto)
    if match and match.group(1) in INTENCIONES:
        return "formato"
    if any(o in texto for o in INTENCIONES):
        return "nombre_suelto"
    return "por_defecto"


def clasificar_detallado(consulta: str) -> dict:
    """Decisión del LLM con la respuesta cruda, cómo se extrajo y los tokens."""
    mensaje = llm.invoke(
        "Eres el primer punto de contacto de un asistente para adultos mayores. "
        "Hablan español coloquial ecuatoriano. Analiza brevemente (1-2 líneas) la "
        f"intención detrás de esta consulta: '{consulta}'. Las categorías "
        f"posibles son: {', '.join(INTENCIONES)}. "
        # Única categoría descrita: se confunde con MEDICATION_HEALTH y RECIPE_MULTIMEDIA.
        "Usa MEDICATION_FOOD_CHECK cuando pregunte si puede comer o preparar una "
        "comida o receta con los medicamentos que toma. "
        "Termina tu respuesta en una última "
        "línea con el formato exacto: CATEGORIA: <una de esas categorías>"
    )
    decision = _extraer_decision(mensaje.content)
    return {
        "intencion": decision.intencion,
        "razonamiento": decision.razonamiento,
        "como_se_decidio": _como_se_decidio(mensaje.content),
        "crudo": mensaje.content,
        "tokens": getattr(mensaje, "usage_metadata", None),
    }


def decidir_intencion(consulta: str) -> dict:
    """Decisión del Orchestrator: red de emergencias determinista y, si no coincide, el LLM."""
    patron = detectar_emergencia(consulta)
    if patron:
        return {
            "intencion": "EMERGENCY",
            "razonamiento": f"Red de seguridad: la consulta coincide con el patrón '{patron}'. No se consultó al LLM.",
            "como_se_decidio": "red_emergencia",
            "crudo": None,
            "tokens": None,
        }
    return clasificar_detallado(consulta)


def nodo_orchestrator(estado: EstadoConversacion) -> dict:
    decision = decidir_intencion(estado["consulta"])
    return {"intencion": decision["intencion"], "razonamiento": decision["razonamiento"]}


# El edge devuelve la intención, no el nodo, para que el diagrama la use de etiqueta.
# MEDICATION_FOOD_CHECK devuelve dos etiquetas: fan-out a dos ramas en paralelo.
RAMAS_CRUCE = ["MEDICATION_FOOD_CHECK: medicación", "MEDICATION_FOOD_CHECK: receta"]


def ruta_siguiente_nodo(estado: EstadoConversacion) -> str | list[str]:
    if estado["intencion"] == "MEDICATION_FOOD_CHECK":
        return RAMAS_CRUCE
    return estado["intencion"]


# Perfil si la consulta no dice de quién es. Solo para el prototipo: en
# producción sale de la autenticación.
PERFIL_POR_DEFECTO = os.getenv("PERFIL_ACTIVO", "rosa")


# Delega en medicacion/agente.py: lee la receta médica y el LLM solo redacta.
def nodo_medicacion(estado: EstadoConversacion) -> dict:
    from medicacion.agente import VARIANTE_REGLAS, responder

    resultado = responder(
        estado["consulta"],
        estado.get("id_perfil") or PERFIL_POR_DEFECTO,
        VARIANTE_REGLAS,
    )
    return {"respuesta": resultado["respuesta"]}



# Delega en el subgrafo de RAG agéntico (ver rag_agentico/README.md).
def nodo_recetas(estado: EstadoConversacion) -> dict:
    resultado = consultar_recetario(estado["consulta"])
    return {
        "respuesta": resultado["respuesta"],
        "traza_rag": resultado["traza"],
    }


# Tarea medicamento × comida: dos ramas en paralelo y un integrador.
# Ver interacciones/README.md.

# Rama de medicación: lee la prescripción vigente, sin LLM.
def nodo_medicacion_cruce(estado: EstadoConversacion) -> dict:
    from interacciones.reglas import medicamentos_vigentes

    return {"medicamentos_vigentes": medicamentos_vigentes(estado.get("id_perfil") or PERFIL_POR_DEFECTO)}


# Se pregunta por los ingredientes y no por las pastillas: si no, el evaluador
# de relevancia rechaza la receta porque no responde sobre medicamentos.
CONSULTA_RAG_CRUCE = "¿Qué ingredientes lleva el plato que menciona esta persona? Lo que dijo: '{consulta}'"


# Rama de recetas: subgrafo de RAG sin generación, solo para saber qué plato es.
def nodo_recetas_cruce(estado: EstadoConversacion) -> dict:
    resultado = consultar_recetario(CONSULTA_RAG_CRUCE.format(consulta=estado["consulta"]), con_generacion=False)
    return {
        "recetas_encontradas": {
            "fuentes": resultado["fuentes"],
            "hubo_resultado": resultado["hubo_resultado"],
            "intentos": resultado["intentos"],
        },
        "traza_rag": resultado["traza"],
    }


# Integrador: espera a las dos ramas, cruza en código y el LLM solo redacta.
def nodo_integrador(estado: EstadoConversacion) -> dict:
    from interacciones.agente import redactar
    from interacciones.reglas import cruzar

    recetas = estado.get("recetas_encontradas") or {}
    cruce = cruzar(
        estado.get("id_perfil") or PERFIL_POR_DEFECTO,
        recetas.get("fuentes") or [],
        medicamentos=estado.get("medicamentos_vigentes"),
    )
    salida = redactar(estado["consulta"], cruce, llm)
    return {"respuesta": salida["respuesta"], "cruce": cruce}


# Stub de comunicación con familia: no ejecuta acción real, solo confirma.
def nodo_familia_stub(estado: EstadoConversacion) -> dict:
    return {
        "respuesta": (
            "Tu mensaje para la familia fue recibido y será procesado. "
            "(STUB — en la versión completa esto se sincroniza con la app familiar)"
        )
    }


# 911 es el ECU 911; configurable porque depende del país.
NUMERO_DE_EMERGENCIAS = os.getenv("NUMERO_DE_EMERGENCIAS", "911")

# Texto fijo, sin LLM: no puede depender del modelo ni del servidor. Avisa que
# nadie fue llamado y no da instrucciones físicas, porque EMERGENCY no distingue
# una caída de un incendio (bitácora 15.2).
MENSAJE_DE_EMERGENCIA = (
    "**Llame al {numero} ahora mismo.**\n\n"
    "Es el número de emergencias. Si hay alguien cerca, pídale ayuda.\n\n"
    "_Este asistente todavía no puede llamar por usted: la llamada la tiene que "
    "hacer usted o alguien que esté a su lado._"
)


# No marca el teléfono ni avisa a nadie: solo responde MENSAJE_DE_EMERGENCIA.
def nodo_emergencia(estado: EstadoConversacion) -> dict:
    return {"respuesta": MENSAJE_DE_EMERGENCIA.format(numero=NUMERO_DE_EMERGENCIAS)}


# Respuesta fija, sin LLM.
def nodo_small_talk(estado: EstadoConversacion) -> dict:
    return {"respuesta": "¡Hola! ¿En qué puedo ayudarte hoy?"}
