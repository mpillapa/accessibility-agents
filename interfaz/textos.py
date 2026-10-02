# Todo el texto que ve la persona. Trato de usted, como los agentes, sin emojis
# ni fórmulas de chatbot; los nombres técnicos solo en el panel de detalles.

TITULO_PESTANA = "Asistente en casa"

TITULO = "Su asistente en casa"
BAJADA = (
    "Le ayuda con sus medicinas, sus recetas de cocina y lo que necesite. "
    "Puede hablarle o escribirle."
)

# Voz
OTRAS_FORMAS_DE_AUDIO = "Enviar un audio ya grabado"
PESTANA_ARCHIVO = "Desde un archivo"
PESTANA_EJEMPLOS = "Audios de prueba"
ETIQUETA_ARCHIVO = "Elija un audio (wav, mp3, m4a u ogg)"
EXPLICACION_EJEMPLOS = (
    "Grabaciones del corpus de la tesis. La misma frase de emergencia sin ruido "
    "y con ruido: con ruido, el asistente pide que se la repitan en vez de "
    "adivinar."
)
BOTON_ENVIAR_EJEMPLO = "Enviar este audio"
SIN_EJEMPLOS = (
    "No están los audios de prueba en esta máquina (la carpeta corpus_audio/ no "
    "va en el repositorio)."
)
ETIQUETA_POR_VOZ = "Por voz"
NO_SE_ENTENDIO_EL_AUDIO = "(no se entendió el audio)"

# Escritura
PLACEHOLDER_CHAT = "Escriba su pregunta o toque el micrófono para hablar"
TITULO_SUGERENCIAS = "Puede empezar por aquí"
SUGERENCIAS = [
    "¿Qué pastillas me tocan hoy?",
    "¿Cómo se hace el llapingacho?",
    "Se me acabó el paracetamol, ¿qué hago?",
    "Quiero preparar sushi",
]

# Mientras trabaja
TRABAJANDO_TEXTO = "Un momento, por favor"
TRABAJANDO_VOZ = "Escuchando lo que dijo"
LISTO = "Respuesta lista"

# Clave: nombre del nodo en orquestacion_langgraph/grafo.py.
PASOS = {
    "transcribir_voz": "Escuchando lo que dijo",
    "no_se_entendio": "No se entendió bien el audio",
    "orchestrator": "Entendiendo qué necesita",
    "medicacion": "Revisando su receta médica",
    "recetas": "Buscando en el recetario",
    "familia": "Preparando el mensaje para su familia",
    "emergencia": "Atendiendo la emergencia",
    "small_talk": "Preparando la respuesta",
    "medicacion_cruce": "Revisando qué medicinas toma",
    "recetas_cruce": "Buscando el plato en el recetario",
    "integrador": "Comparando el plato con sus medicinas",
}

# Los pasos internos del RAG agéntico, para el panel de detalles.
PASOS_RECETARIO = {
    "decidir_busqueda": "Decide si hace falta el recetario",
    "recuperar": "Busca en el recetario",
    "evaluar_relevancia": "Revisa si lo encontrado sirve",
    "reformular": "No servía: busca con otras palabras",
    "expandir_contexto": "Trae la receta completa",
    "generar": "Redacta la respuesta",
    "sin_resultado": "No está en el recetario",
    "responder_sin_recetario": "Responde sin buscar",
}

TEMAS = {
    "MEDICATION_HEALTH": "Medicinas y salud",
    "RECIPE_MULTIMEDIA": "Recetas de cocina",
    "FAMILY_COMMUNICATION": "Mensajes a la familia",
    "EMERGENCY": "Emergencia",
    "SMALL_TALK": "Conversación",
    "MEDICATION_FOOD_CHECK": "Comida y medicinas",
    "NO_SE_ENTENDIO": "No se entendió el audio",
}

# Detalles (para quien evalúa)
TITULO_DETALLES = "Cómo se llegó a esta respuesta"
DETALLE_TEMA = "Tema"
DETALLE_TIEMPO = "Tiempo de respuesta"
DETALLE_QUE_OYO = "Qué se entendió del audio (Whisper)"
DETALLE_VOZ_DETECTADA = "Detector de voz"
DETALLE_DESCARTADO = "Se pidió repetir porque"
DETALLE_POR_QUE = "Por qué se eligió ese tema"
DETALLE_RECORRIDO = "Recorrido por el sistema"

# Barra lateral
TITULO_PERSONA = "¿Quién usa el asistente?"
AYUDA_PERSONA = "Cada persona tiene su propia receta médica. Son personas y recetas de prueba."
LEER_EN_VOZ_ALTA = "Leer las respuestas en voz alta"
BOTON_NUEVA = "Empezar una conversación nueva"
TITULO_TECNICO = "Información técnica"
TECNICO_MODELO = "Modelo de lenguaje en uso"
TECNICO_CAMBIO = (
    "El servidor cambió de modelo respecto a lo configurado. Estas respuestas no "
    "son comparables con mediciones anteriores."
)
TECNICO_TRAZAS_ACTIVAS = "Trazas en LangSmith activas, proyecto"
TECNICO_TRAZAS_INACTIVAS = "Trazas en LangSmith desactivadas"
TECNICO_VER_TRAZAS = "Abrir LangSmith"

# Errores
ERROR_SIN_SERVIDOR = (
    "No pude responder en este momento. El servidor que usa el asistente no está "
    "disponible; intente de nuevo en unos minutos."
)
ERROR_DETALLE = "Detalle del error"

# Pie
PIE = (
    "Prototipo de tesis de maestría · Universidad San Francisco de Quito USFQ.<br>"
    "Las personas y las recetas médicas son ficticias. El asistente no reemplaza "
    "a su médico."
)
