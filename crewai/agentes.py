import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.buscar import buscar_receta

# override=True: el .env del proyecto manda sobre variables ya presentes en el
# entorno (p. ej. las que VSCode inyecta desde un .env del workspace padre).
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

VLLM_CHAT_BASE_URL = os.getenv("VLLM_CHAT_BASE_URL", "http://172.28.230.10:12559/v1")
VLLM_CHAT_MODEL = os.getenv("VLLM_CHAT_MODEL", "google/gemma-4-12B-it")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "local")

# Las 5 intenciones que rutea el sistema. Idénticas a langgraph/agentes.py
INTENCIONES = [
    "MEDICATION_HEALTH",
    "RECIPE_MULTIMEDIA",
    "FAMILY_COMMUNICATION",
    "EMERGENCY",
    "SMALL_TALK",
]

llm = LLM(
    model=f"openai/{VLLM_CHAT_MODEL}",
    base_url=VLLM_CHAT_BASE_URL,
    api_key=VLLM_API_KEY,
    temperature=0.3,  # bajo para que el ruteo sea consistente
)


@tool("Buscar en recetario")
def buscar_en_recetario(consulta: str) -> str:
    """Busca fragmentos relevantes del recetario (PDFs/imágenes ya procesados
    con OCR y embeddings) para responder una consulta sobre una receta."""
    fragmentos = buscar_receta(consulta, k=3)
    if not fragmentos:
        return (
            "El recetario no tiene resultados para esta consulta (o todavía no "
            "se ha ingerido — correr 'python rag/ingesta.py')."
        )
    return "\n---\n".join(fragmentos)


# Agente coordinador: identifica el tipo de consulta y delega al especialista correcto.
def crear_orchestrator():
    return Agent(
        role="Coordinador de Asistencia para Adulto Mayor",
        goal=(
            "Entender qué necesita el usuario adulto mayor y delegar la consulta "
            "al agente especialista correcto. Las opciones son: medicación, "
            "recetas, comunicación con familia, emergencias, o conversación trivial."
        ),
        backstory=(
            "Eres el primer punto de contacto del sistema. Hablas español y "
            "entiendes el lenguaje coloquial ecuatoriano. Tu único trabajo es "
            "identificar la intención y delegar; no respondes tú directamente "
            "salvo en conversación trivial."
        ),
        llm=llm,
        allow_delegation=True,
        verbose=False,
    )


# Especialista en consultas de medicación y salud básica.
def crear_agente_medicacion():
    return Agent(
        role="Especialista en Medicación y Salud",
        goal=(
            "Responder consultas sobre medicación: horarios, dosis, síntomas leves. "
            "En esta versión simulas el acceso a la base de datos de la familia."
        ),
        backstory=(
            "Eres un asistente especializado en gestión de medicación para adultos "
            "mayores. Conoces el inventario de pastillas, horarios y posibles "
            "interacciones. Respondes en español, de forma clara y empática."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


# Especialista en lectura y adaptación de recetas.
def crear_agente_recetas():
    return Agent(
        role="Especialista en Recetas y Multimedia",
        goal=(
            "Buscar la receta en el recetario (tool 'Buscar en recetario') antes "
            "de responder, y adaptar medidas a objetos cotidianos (ej: 300ml = un "
            "vaso grande), guiando paso a paso al usuario."
        ),
        backstory=(
            "Eres un asistente culinario que ayuda a personas mayores a preparar "
            "comidas. Siempre consultas primero el recetario real (RAG sobre PDFs "
            "e imágenes de recetas, incluidas fotos de recetas manuscritas vía "
            "OCR) antes de responder; no inventas ingredientes ni pasos que no "
            "estén en lo que encontraste. Adaptas medidas técnicas a referencias "
            "cotidianas para que sean comprensibles sin instrumentos de medición."
        ),
        llm=llm,
        tools=[buscar_en_recetario],
        allow_delegation=False,
        verbose=False,
    )


# Stub de comunicación con familia: no ejecuta acción real, solo confirma la recepción.
def crear_agente_familia_stub():
    return Agent(
        role="Puente de Comunicación Familiar (STUB)",
        goal=(
            "Confirmar que la consulta sobre comunicación con familia fue recibida. "
            "Por ahora solo respondes que el mensaje será procesado, sin ejecutar "
            "ninguna acción real."
        ),
        backstory=(
            "Eres una versión simplificada del Family Bridge Agent. En la versión "
            "completa gestionarías la app de la familia; en este prototipo solo "
            "confirmas la recepción."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


# Stub de emergencias: no ejecuta acción real, solo confirma la recepción.
def crear_agente_emergencia_stub():
    return Agent(
        role="Centinela de Emergencias (STUB)",
        goal=(
            "Confirmar que la consulta de emergencia fue recibida. En la versión "
            "completa activaría protocolos locales de llamada de emergencia."
        ),
        backstory=(
            "Eres una versión simplificada del Emergency Sentinel. En producción "
            "correrías 100% local y activarías llamadas de auxilio."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


# Ensambla la crew: Orchestrator + los 4 especialistas.
def crear_crew():
    orchestrator = crear_orchestrator()
    medicacion   = crear_agente_medicacion()
    recetas      = crear_agente_recetas()
    familia      = crear_agente_familia_stub()
    emergencia   = crear_agente_emergencia_stub()

    tarea_principal = Task(
        description=(
            "El usuario adulto mayor dijo: '{consulta}'. "
            "Identifica de qué tipo de consulta se trata y delega al especialista "
            "correspondiente. Si es conversación trivial (saludo, comentario), "
            "responde tú directamente de forma amable y breve."
        ),
        expected_output=(
            "Exactamente dos partes separadas por una línea en blanco:\n"
            "1) Una línea con el formato: 'AGENTE: [rol del agente que responde]'\n"
            "2) La respuesta completa en español dirigida al usuario.\n"
            "Ejemplo:\n"
            "AGENTE: Especialista en Medicación y Salud\n\n"
            "Claro, tu pastilla del corazón..."
        ),
        agent=orchestrator,
    )

    crew = Crew(
        agents=[orchestrator, medicacion, recetas, familia, emergencia],
        tasks=[tarea_principal],
        process=Process.sequential,
        verbose=False,
    )

    return crew


# Versión async: CrewAI exige kickoff_async cuando ya hay un event loop
# corriendo (Jupyter). Devuelve el mismo dict que procesar_consulta(). En una
# celda de notebook: `res = await procesar_consulta_async("...")`.
async def procesar_consulta_async(consulta: str) -> dict:
    import time

    inicio = time.time()
    crew = crear_crew()
    resultado = await crew.kickoff_async(inputs={"consulta": consulta})
    latencia = time.time() - inicio
    return {
        "consulta": consulta,
        "intencion": None,  # el Orchestrator no expone la intención de forma estructurada
        "respuesta": str(resultado),
        "latencia_segundos": round(latencia, 2),
    }


# El Orchestrator decide a quién delegar. Para scripts/demos.
def procesar_consulta(consulta: str) -> dict:
    import asyncio
    import time

    inicio  = time.time()
    crew    = crear_crew()
    resultado = asyncio.run(crew.kickoff_async(inputs={"consulta": consulta}))
    latencia  = time.time() - inicio

    return {
        "consulta":           consulta,
        "intencion":          None,  # el Orchestrator no expone la intención de forma estructurada
        "respuesta":          str(resultado),
        "latencia_segundos":  round(latencia, 2),
    }


# --- Solo para la comparación de accuracy de ruteo (notebooks/comparativa.ipynb) ---
#
# clasificar_consulta() mide ÚNICAMENTE la decisión de clasificación del
# Orchestrator, no la delegación completa + respuesta del especialista. Se
# implementa con el mismo estilo de prompt que el nodo orchestrator de
# LangGraph (texto libre terminando en "CATEGORIA: <intención>"), para que la
# comparación entre frameworks mida la misma tarea con el mismo LLM.
#
# Nota metodológica: como ambos frameworks delegan la decisión al mismo modelo
# (google/gemma-4-12B-it), se espera una accuracy parecida; lo que difiere
# entre CrewAI y LangGraph es el MECANISMO de ruteo (delegación vs edge
# condicional), la latencia y qué tan explícito queda el flujo de datos — no
# la calidad de la clasificación en sí. Verificado el 2026-07-23 en el notebook
# (notebooks/comparativa.ipynb): ambos 100% en muestra de 50, latencias
# similares.
def _extraer_categoria(texto: str) -> str:
    match = re.search(r"CATEGORIA:\s*([A-Z_]+)", texto)
    candidata = match.group(1) if match else None
    if candidata in INTENCIONES:
        return candidata
    return next((o for o in INTENCIONES if o in texto), "SMALL_TALK")


def _crew_clasificacion(consulta: str) -> Crew:
    orchestrator = crear_orchestrator()
    tarea = Task(
        description=(
            f"El usuario adulto mayor dijo: '{consulta}'. Clasifica su intención "
            f"en UNA de estas categorías: {', '.join(INTENCIONES)}. Razona en "
            "1-2 líneas y termina con una última línea con el formato exacto: "
            "CATEGORIA: <una de esas categorías>"
        ),
        expected_output="Un razonamiento breve y una última línea 'CATEGORIA: <intención>'.",
        agent=orchestrator,
    )
    return Crew(agents=[orchestrator], tasks=[tarea], process=Process.sequential, verbose=False)


def clasificar_consulta(consulta: str) -> str:
    """Devuelve la intención (una de INTENCIONES) que el Orchestrator asigna a
    la consulta. Una sola llamada al LLM, comparable con
    langgraph.grafo.clasificar_consulta(). Versión síncrona (scripts/demos)."""
    resultado = str(_crew_clasificacion(consulta).kickoff(inputs={"consulta": consulta}))
    return _extraer_categoria(resultado)


async def clasificar_consulta_async(consulta: str) -> str:
    """Igual que clasificar_consulta pero con kickoff_async, para Jupyter
    (donde CrewAI exige kickoff_async por el event loop activo)."""
    resultado = str(await _crew_clasificacion(consulta).kickoff_async(inputs={"consulta": consulta}))
    return _extraer_categoria(resultado)
