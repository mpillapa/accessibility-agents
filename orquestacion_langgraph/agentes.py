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
from orquestacion_langgraph.red_emergencia import detectar_emergencia

INTENCIONES = [
    "MEDICATION_HEALTH",
    "RECIPE_MULTIMEDIA",
    "FAMILY_COMMUNICATION",
    "EMERGENCY",
    "SMALL_TALK",
    # Agregada el 2026-09-30 (pedido de Cristian): ¿este plato es compatible
    # con mis medicamentos? Es la única intención que activa dos especialistas.
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
    """Cómo salió la intención del texto del modelo. Existe para poder medir
    el fallo del valor por defecto: si el modelo no cierra con `CATEGORIA:`,
    _extraer_decision busca un nombre suelto y, si no hay, cae a SMALL_TALK.
    Una emergencia mal formateada termina así como conversación trivial."""
    match = re.search(r"CATEGORIA:\s*([A-Z_]+)", texto)
    if match and match.group(1) in INTENCIONES:
        return "formato"
    if any(o in texto for o in INTENCIONES):
        return "nombre_suelto"
    return "por_defecto"


def clasificar_detallado(consulta: str) -> dict:
    """La decisión del Orchestrator con todo lo necesario para medirla: la
    respuesta cruda del modelo, cómo se extrajo la intención y los tokens."""
    mensaje = llm.invoke(
        "Eres el primer punto de contacto de un asistente para adultos mayores. "
        "Hablan español coloquial ecuatoriano. Analiza brevemente (1-2 líneas) la "
        f"intención detrás de esta consulta: '{consulta}'. Las categorías "
        f"posibles son: {', '.join(INTENCIONES)}. "
        # Única categoría con descripción: es la nueva y la que más se puede
        # confundir con MEDICATION_HEALTH y RECIPE_MULTIMEDIA.
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
    """La decisión completa del Orchestrator: primero la red de seguridad
    determinista para emergencias y, si no coincide, el LLM.

    Con la red, una emergencia con palabras inequívocas ("me caí", "auxilio")
    no depende del LLM ni de que la transcripción de voz haya salido bien. Ver
    red_emergencia.py para el caso que la motiva y su límite."""
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


# Edge condicional: devuelve la INTENCIÓN detectada, y el grafo la traduce a
# nodo (ver construir_grafo). Devolver la intención en vez del nombre del nodo
# mantiene la decisión en el vocabulario del dominio y deja que el diagrama
# etiquete cada flecha con la intención que la dispara.
#
# MEDICATION_FOOD_CHECK devuelve DOS etiquetas: LangGraph ejecuta las dos ramas
# en paralelo (fan-out) y el integrador espera a ambas.
RAMAS_CRUCE = ["MEDICATION_FOOD_CHECK: medicación", "MEDICATION_FOOD_CHECK: receta"]


def ruta_siguiente_nodo(estado: EstadoConversacion) -> str | list[str]:
    if estado["intencion"] == "MEDICATION_FOOD_CHECK":
        return RAMAS_CRUCE
    return estado["intencion"]


# Especialista en consultas de medicación y salud básica (mismo rol que CrewAI).
# Perfil que se asume cuando la consulta no dice de quién es. En un sistema real
# esto saldría de la autenticación, nunca de una variable de entorno.
PERFIL_POR_DEFECTO = os.getenv("PERFIL_ACTIVO", "rosa")


# Especialista en medicación: delega en medicacion/agente.py.
#
# Antes este nodo tenía el prompt "simulas el acceso a la base de datos de la
# familia", o sea que el LLM inventaba la medicación entera. Ahora lee la RECETA
# MÉDICA de la persona (medicacion/datos/prescripciones.json), la consolida por
# horario y la verifica en código determinista; el modelo solo redacta.
#
# El cambio de fondo está documentado en medicacion/prescripciones.py: filtrar un
# vademécum por condición devuelve opciones elegibles, no un tratamiento, y
# presentarlas como tal producía respuestas clínicamente absurdas.
def nodo_medicacion(estado: EstadoConversacion) -> dict:
    from medicacion.agente import VARIANTE_REGLAS, responder

    resultado = responder(
        estado["consulta"],
        estado.get("id_perfil") or PERFIL_POR_DEFECTO,
        VARIANTE_REGLAS,
    )
    return {"respuesta": resultado["respuesta"]}



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


# --- Tarea medicamento × comida ---------------------------------------------
#
# Dos ramas en paralelo y un integrador. Ver interacciones/README.md.

# Rama de medicación: qué toma la persona. Determinista, sin LLM: es la misma
# lectura de la prescripción vigente que usa el agente de medicación.
def nodo_medicacion_cruce(estado: EstadoConversacion) -> dict:
    from interacciones.reglas import medicamentos_vigentes

    return {"medicamentos_vigentes": medicamentos_vigentes(estado.get("id_perfil") or PERFIL_POR_DEFECTO)}


# Cómo se le pregunta al RAG por el plato. La consulta original es sobre
# pastillas ("¿puedo comer hornado con mis pastillas?") y el evaluador de
# relevancia del RAG juzga si un fragmento RESPONDE la consulta: un fragmento
# de la receta del hornado no responde nada sobre pastillas. Se le pide lo que
# esta rama necesita, los ingredientes del plato, sin llamada extra al LLM.
CONSULTA_RAG_CRUCE = "¿Qué ingredientes lleva el plato que menciona esta persona? Lo que dijo: '{consulta}'"


# Rama de recetas: de qué plato se habla. Usa el subgrafo de RAG agéntico SIN
# generación: solo hace falta saber qué receta es, no redactarla.
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
