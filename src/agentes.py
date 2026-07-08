# Sistema multi-agente de accesibilidad para adultos mayores.
# Orchestrator + 2 especialistas reales (medicación, recetas) + 2 stubs (familia, emergencia).
# Versiones de ruteo: B (el LLM decide), A1 (clasificador + dispatch en Python),
# A2 (clasificador como herramienta del Orchestrator-LLM).

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# Ollama local con qwen2.5:7b. El prefijo "ollama/" le dice a LiteLLM que enrute a Ollama.
llm = LLM(
    model="ollama/qwen2.5:7b",
    base_url="http://localhost:11434",
    temperature=0.3,  # bajo para que el ruteo sea consistente
)


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


# Ensambla la crew de la versión B (Fase 4): Orchestrator + los 4 especialistas.
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


# Herramienta de clasificación de la versión A2 (Fase 5).
@tool("Clasificador de Intencion")
def herramienta_clasificador(consulta: str) -> str:
    """Clasifica la intención de la consulta y devuelve la intención detectada y su confianza."""
    from clasificador import predecir
    resultado = predecir(consulta)
    return (
        f"Intención detectada: {resultado['intencion']} "
        f"(confianza: {resultado['confianza']:.2f})"
    )


# Orchestrator con acceso a la herramienta de clasificación (versión A2).
def crear_orchestrator_con_herramienta():
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


# Ejecuta un solo especialista sin Orchestrator (helper de A1).
# Solo funciona desde scripts, no desde Jupyter (asyncio.run no se anida en un event loop activo).
def _ejecutar_agente_individual(agente, consulta):
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


# Versión B async para usar desde Jupyter (requiere kickoff_async con event loop activo).
async def procesar_consulta_async(consulta: str) -> str:
    crew = crear_crew()
    resultado = await crew.kickoff_async(inputs={"consulta": consulta})
    return str(resultado)


# Versión B: el LLM del Orchestrator decide a quién delegar. Para scripts (evaluacion.py).
def procesar_consulta_v_b(consulta: str) -> dict:
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


# Versión A1: el clasificador decide la intención y Python despacha directo al especialista.
def procesar_consulta_v_a1(consulta: str) -> dict:
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


# Versión A2: el clasificador es una herramienta que el Orchestrator-LLM consulta antes de delegar.
def procesar_consulta_v_a2(consulta: str) -> dict:
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
