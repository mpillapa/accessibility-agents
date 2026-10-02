# Red de seguridad determinista para emergencias, antes del LLM: si el texto
# contiene un patrón inequívoco, la intención es EMERGENCY sin consultar al
# modelo. Complementa al Orchestrator, no lo reemplaza (bitácora 20.1 y 20.2).
#
# Criterio para agregar un patrón: casi imposible en una consulta que no es
# emergencia. Por eso quedan fuera "ayuda"/"ayúdame" sueltos ("ayúdame con la
# receta"), "fuego" ("a fuego lento") y "caído/caída" sin "me" ("se me ha caído
# el vaso"). pruebas/prueba_emergencia.py valida contra las no-emergencias de
# dataset.csv.

import re
import unicodedata

# (nombre, expresión), sobre texto en minúsculas y sin tildes.
PATRONES: list[tuple[str, str]] = [
    ("caida", r"\bcai\b|\bcaerme\b|\bme caigo\b|\bme (he )?caid[oa]\b"),
    ("caida", r"\bresbale\b|\btropece\b"),
    ("no_puede_levantarse", r"\bno (me )?puedo (levantar|parar|mover)|\bno puedo levantarme\b"),
    ("dolor_de_pecho", r"\b(me duele|dolor|dolores) (mucho |muy fuerte )?(de |en |en el |del |el )?pecho\b|\bpecho (muy )?apretado\b"),
    ("falta_de_aire", r"\bno puedo respirar\b|\bme falta el aire\b|\bme ahogo\b"),
    ("pedido_de_auxilio", r"\bauxilio\b|\bsocorro\b|\bemergencia\b|\bambulancia\b|\b911\b|\bparamedicos\b"),
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
