# Tarea medicamento × comida: ¿esta receta de cocina es compatible con lo que
# me recetó el médico?
#
# Es la única tarea del sistema que necesita DOS especialistas a la vez: el de
# medicación (qué toma la persona) y el de recetas (qué lleva el plato). Un
# tercer nodo, el integrador, cruza las dos salidas. Lo pidió Cristian el
# 2026-09-30 para tener un flujo con más de un agente y poder medir si el camino
# recorrido fue el correcto.
#
# Separado en capas, igual que medicacion/:
#   datos/             — los JSON: catálogo de interacciones, ingredientes por
#                        receta y casos de prueba con la verdad de referencia
#   datos.py           — carga los JSON (acceso a datos, no decide nada)
#   (pendiente) reglas.py — el cruce determinista, que es la regla de negocio
#   (pendiente) el nodo integrador del grafo, que solo redacta
#
# Datos ficticios e ilustrativos: ver el campo "_aviso" de cada JSON.
