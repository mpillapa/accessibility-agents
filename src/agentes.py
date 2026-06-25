"""
Sistema multi-agente de accesibilidad para adultos mayores.

Contiene:
- 1 Orchestrator: recibe la consulta y enruta al especialista correcto.
- 2 Agentes especialistas reales: Medication, Recipe.
- 2 Agentes stub: Family, Emergency.

Versiones de ruteo implementadas:
- B  (Fase 4): el LLM del Orchestrator decide a quién delegar.
- A1 (Fase 5): el clasificador decide la intención; Python despacha directo.
- A2 (Fase 5): el clasificador es una herramienta del Orchestrator-LLM.
"""

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# Nota de autoria: la estructura base de los agentes con CrewAI se desarrollo
# con ayuda de Claude. Prompt de referencia que se le dio:
#   "Como armo en CrewAI un Orchestrator que delegue la consulta a agentes
#    especialistas segun la intencion, y como expongo mi clasificador de
#    scikit-learn como una herramienta del agente?"

# ============================================================
# Configuración del LLM
# ============================================================
# Usamos Ollama local con qwen2.5:7b. El prefijo "ollama/" le indica a
# LiteLLM (usado por CrewAI) que enrute a Ollama en localhost.

llm = LLM(
    model="ollama/qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.3,  # bajo para que el ruteo sea consistente
)

# ============================================================
# Definición de los agentes
# ============================================================

def crear_orchestrator():
    """
    Agente coordinador. Recibe la consulta del usuario, identifica de qué tipo es,
    y delega al especialista correcto.
    """
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


def crear_agente_medicacion():
    """Agente especialista en consultas de medicación y salud básica."""
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


def crear_agente_recetas():
    """Agente especialista en lectura y adaptación de recetas."""
    return Agent(
        role="Especialista en Recetas y Multimedia",
        goal=(
            "Leer recetas, adaptar medidas a objetos cotidianos (ej: 300ml = un "
            "vaso grande), y guiar paso a paso al usuario."
        ),
        backstory=(
            "Eres un asistente culinario que ayuda a personas mayores a preparar "
            "comidas. Adaptas medidas técnicas a referencias cotidianas para que "
            "sean comprensibles sin instrumentos de medición."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )


def crear_agente_familia_stub():
    """
    Agente stub para comunicación con familia.
    No implementa lógica real, solo confirma que recibió la consulta.
    """
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


def crear_agente_emergencia_stub():
    """
    Agente stub para emergencias.
    No implementa lógica real, solo confirma que recibió la consulta.
    """
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


# ============================================================
# Ensamblado de la crew — versión B (Fase 4)
# ============================================================

def crear_crew():
    """
    Crea la crew completa con el Orchestrator y los 4 agentes especialistas.
    El Orchestrator puede delegar a cualquiera de los especialistas.
    """
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


# ============================================================
# Herramienta de clasificación — versión A2 (Fase 5)
# ============================================================

@tool("Clasificador de Intencion")
def herramienta_clasificador(consulta: str) -> str:
    """
    Clasifica la intención de una consulta del usuario. Devuelve la intención
    detectada y la confianza del modelo.

    Args:
        consulta: La frase del usuario a clasificar.

    Returns:
        Texto con la intención detectada y la confianza.
    """
    from clasificador import predecir
    resultado = predecir(consulta)
    return (
        f"Intención detectada: {resultado['intencion']} "
        f"(confianza: {resultado['confianza']:.2f})"
    )


def crear_orchestrator_con_herramienta():
    """Orchestrator que tiene acceso a la herramienta de clasificación (versión A2)."""
    return Agent(
        role="Coordinador de Asistencia para Adulto Mayor",
        goal=(
            "Usar la herramienta de clasificación para identificar la intención "
            "del usuario y delegar al especialista correcto."
        ),
        backstory=(
            "Eres el coordinador. SIEMPRE usa la herramienta de clasificación "
            "antes de delegar, para tomar decisiones basadas en el clasificador."
        ),
        llm=llm,
        tools=[herramienta_clasificador],
        allow_delegation=True,
        verbose=False,
    )


# ============================================================
# Helper interno — versión A1 (Fase 5)
# ============================================================

def _ejecutar_agente_individual(agente, consulta):
    """
    Ejecuta UN agente especialista con la consulta dada (sin Orchestrator).
    Helper interno para A1. Solo funciona desde scripts, no desde Jupyter
    (asyncio.run no puede anidarse dentro de un event loop ya activo).
    """
    import asyncio
    tarea = Task(
        description=f"Responde a esta consulta del usuario: '{consulta}'",
        expected_output="Respuesta en español, clara y empática.",
        agent=agente,
    )
    crew = Crew(
        agents=[agente],
        tasks=[tarea],
        process=Process.sequential,
        verbose=False,
    )
    return asyncio.run(crew.kickoff_async())


# ============================================================
# Funciones públicas
# ============================================================

async def procesar_consulta_async(consulta: str) -> str:
    """
    Versión async para usar desde Jupyter (Fase 4, versión B).
    CrewAI moderno requiere kickoff_async() cuando ya hay un event loop activo.
    """
    crew = crear_crew()
    resultado = await crew.kickoff_async(inputs={"consulta": consulta})
    return str(resultado)


def procesar_consulta_v_b(consulta: str) -> dict:
    """
    Versión B: el LLM del Orchestrator decide a quién delegar.
    Para usar desde scripts (evaluacion.py), no desde Jupyter.
    """
    import asyncio
    import time

    inicio  = time.time()
    crew    = crear_crew()
    resultado = asyncio.run(crew.kickoff_async(inputs={"consulta": consulta}))
    latencia  = time.time() - inicio

    return {
        "version":            "B",
        "consulta":           consulta,
        "intencion":          None,  # B no expone la intención de forma estructurada
        "respuesta":          str(resultado),
        "latencia_segundos":  round(latencia, 2),
    }


def procesar_consulta_v_a1(consulta: str) -> dict:
    """
    Versión A1: el clasificador decide la intención y un dispatcher Python
    plano llama directamente al agente especialista correspondiente.
    """
    import time
    from clasificador import predecir

    inicio    = time.time()
    prediccion = predecir(consulta)
    intencion  = prediccion["intencion"]

    if intencion == "MEDICATION_HEALTH":
        respuesta = _ejecutar_agente_individual(crear_agente_medicacion(), consulta)
    elif intencion == "RECIPE_MULTIMEDIA":
        respuesta = _ejecutar_agente_individual(crear_agente_recetas(), consulta)
    elif intencion == "FAMILY_COMMUNICATION":
        respuesta = _ejecutar_agente_individual(crear_agente_familia_stub(), consulta)
    elif intencion == "EMERGENCY":
        respuesta = _ejecutar_agente_individual(crear_agente_emergencia_stub(), consulta)
    elif intencion == "SMALL_TALK":
        # Para small talk no llamamos ningún agente; ahorra tokens y latencia
        respuesta = "¡Hola! ¿En qué puedo ayudarte hoy?"
    else:
        respuesta = "No entendí bien tu consulta, ¿puedes repetirla?"

    latencia = time.time() - inicio

    return {
        "version":            "A1",
        "consulta":           consulta,
        "intencion":          intencion,
        "respuesta":          str(respuesta),
        "latencia_segundos":  round(latencia, 2),
    }


def procesar_consulta_v_a2(consulta: str) -> dict:
    """
    Versión A2: el clasificador es una herramienta del Orchestrator-LLM.
    El Orchestrator llama a la herramienta y con ese resultado decide a quién delegar.
    """
    import asyncio
    import time

    inicio = time.time()

    orchestrator = crear_orchestrator_con_herramienta()
    medicacion   = crear_agente_medicacion()
    recetas      = crear_agente_recetas()
    familia      = crear_agente_familia_stub()
    emergencia   = crear_agente_emergencia_stub()

    tarea = Task(
        description=(
            f"El usuario dijo: '{consulta}'. "
            f"Usa la herramienta de clasificación para identificar la intención, "
            f"luego delega al especialista correcto."
        ),
        expected_output="Respuesta en español del especialista correspondiente.",
        agent=orchestrator,
    )

    crew = Crew(
        agents=[orchestrator, medicacion, recetas, familia, emergencia],
        tasks=[tarea],
        process=Process.sequential,
        verbose=False,
    )

    resultado = asyncio.run(crew.kickoff_async())
    latencia  = time.time() - inicio

    return {
        "version":            "A2",
        "consulta":           consulta,
        "intencion":          None,  # A2 no expone la intención de forma estructurada
        "respuesta":          str(resultado),
        "latencia_segundos":  round(latencia, 2),
    }
