# Red de seguridad determinista para emergencias, ANTES del LLM.
#
# EL CASO QUE LA MOTIVA (demo del 2026-09-24, traza en LangSmith)
# ---------------------------------------------------------------
# Manuel dijo por voz "Acabé de caerme, ¿me puedes ayudar?". Whisper transcribió
# "A través de Caerme, ¿me puedes ayudar?", con probabilidad de idioma 1,0 y sin
# marcar falta de voz, así que el guardrail de voz lo dejó pasar. El Orchestrator
# razonó que "Caerme" era el nombre de un servicio y lo clasificó como SMALL_TALK
# (3 de 3 veces con esa frase). La emergencia se perdió sin que ningún
# componente fallara: cada uno procesó correctamente una entrada que parecía
# válida (bitácora 2 y 10).
#
# La palabra "caerme" SÍ estaba en la transcripción. Una regla que la busque no
# depende de que el LLM interprete bien una frase mal transcrita.
#
# QUÉ HACE
# --------
# Si el texto contiene un patrón inequívoco de emergencia, la intención es
# EMERGENCY sin preguntarle al modelo. Si no, decide el LLM como siempre. La red
# COMPLEMENTA al Orchestrator, no lo reemplaza: solo cubre lo que está en la
# lista, y la mayoría de las emergencias del dataset no usan estas palabras.
#
# CRITERIO PARA AGREGAR UN PATRÓN
# -------------------------------
# Tiene que ser casi imposible en una consulta que NO es una emergencia. Por eso
# quedan fuera, a propósito:
#   - "ayuda" o "ayúdame" sueltos: "ayúdame con la receta" no es una emergencia;
#   - "fuego": "a fuego lento" está en medio recetario;
#   - "caído/caída" sin "me": "se me ha caído el vaso" no es una caída.
# Una falsa alarma es molesta; una emergencia perdida es grave. Aun así, cada
# patrón se valida contra las 332 frases que NO son emergencias en dataset.csv
# (pruebas/prueba_emergencia.py): hoy no da ninguna falsa alarma.

import re
import unicodedata

# (nombre, expresión). Se aplican sobre el texto en minúsculas y sin tildes.
PATRONES: list[tuple[str, str]] = [
    # Caídas: la emergencia más frecuente en adultos mayores.
    ("caida", r"\bcai\b|\bcaerme\b|\bme caigo\b|\bme (he )?caid[oa]\b"),
    ("caida", r"\bresbale\b|\btropece\b"),
    ("no_puede_levantarse", r"\bno (me )?puedo (levantar|parar|mover)|\bno puedo levantarme\b"),
    # Síntomas cardiorrespiratorios.
    ("dolor_de_pecho", r"\b(me duele|dolor|dolores) (mucho |muy fuerte )?(de |en |en el |del |el )?pecho\b|\bpecho (muy )?apretado\b"),
    ("falta_de_aire", r"\bno puedo respirar\b|\bme falta el aire\b|\bme ahogo\b"),
    # Pedido explícito de auxilio.
    ("pedido_de_auxilio", r"\bauxilio\b|\bsocorro\b|\bemergencia\b|\bambulancia\b|\b911\b|\bparamedicos\b"),
    # Peligro en la casa.
    ("peligro_en_casa", r"\bolor a gas\b|\bhumo\b|\bincendio\b"),
]

_COMPILADOS = [(nombre, re.compile(expresion)) for nombre, expresion in PATRONES]


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def detectar_emergencia(texto: str) -> str | None:
    """El nombre del primer patrón que coincide, o None si ninguno."""
    normalizado = _normalizar(texto or "")
    for nombre, expresion in _COMPILADOS:
        if expresion.search(normalizado):
            return nombre
    return None
