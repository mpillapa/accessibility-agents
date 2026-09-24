# Gestión de medicación.
#
# La RECETA MÉDICA es la fuente de verdad: el sistema no elige medicamentos, lee
# lo que el médico indicó, lo organiza por horario, lo explica y lo verifica.
#
# Separado en capas, a propósito:
#   datos.py           — carga los JSON (acceso a datos, no decide nada)
#   prescripciones.py  — LAS REGLAS DE NEGOCIO: plan del día, verificación y
#                        alternativas del mismo grupo. Código determinista.
#   reglas.py          — criterios reutilizables (exclusión, tope ajustado) y el
#                        planteamiento original, conservado como registro.
#   agente.py          — las dos variantes que se comparan (reglas vs LLM)
#   (el nodo del grafo vive en orquestacion_langgraph/, y solo conversa)
#
# Las reglas NO están en un prompt. Ver prescripciones.py para el porqué.
#
# NOMBRES: "receta" en este repositorio significa receta DE COCINA (rag/, el
# intent RECIPE_MULTIMEDIA). La receta médica se llama PRESCRIPCIÓN en todo el
# código y en la memoria.
