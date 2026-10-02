# Estado del subgrafo de RAG agéntico y las reglas de su ciclo. Separado de
# EstadoConversacion para que el RAG sea reemplazable.

import operator
from typing import Annotated, Optional, TypedDict

# Reglas de negocio del ciclo. Justificación de cada una en el README del paquete.

# Búsquedas máximas por consulta (original + reformulaciones). Más rondas suben
# el recall a costa de latencia.
MAX_INTENTOS_RECUPERACION = 2

# k de cada búsqueda; igual al del RAG anterior.
FRAGMENTOS_POR_BUSQUEDA = 3

# True: una llamada al LLM por fragmento (tipo CRAG). False: una sola para el
# conjunto, más rápido pero todo o nada.
EVALUAR_FRAGMENTO_POR_FRAGMENTO = True

# Si un fragmento pasa el filtro, se trae el resto de su archivo (documento
# padre); sin esto la respuesta sale con un paso suelto (README, "Hallazgos medidos").
EXPANDIR_A_RECETA_COMPLETA = True

# Tope del contexto tras expandir; un archivo puede traer varias recetas.
MAXIMO_CARACTERES_CONTEXTO = 6000


class EstadoRAG(TypedDict):
    """Estado interno del subgrafo de recetas."""

    # Entrada
    consulta: str                              # lo que dijo el usuario, sin tocar

    # Ciclo de recuperación
    necesita_recetario: Optional[bool]         # lo decide el primer nodo
    consulta_busqueda: Optional[str]           # lo que realmente se buscó (reformulada o no)
    intentos: int                              # búsquedas hechas hasta ahora
    fragmentos: Optional[list[dict]]           # lo último recuperado (texto, fuente, distancia)
    fragmentos_utiles: Optional[list[dict]]    # los que el evaluador aceptó
    fragmentos_contexto: Optional[list[dict]]  # los útiles + el resto de su receta

    # Salida
    respuesta: Optional[str]
    hubo_resultado: Optional[bool]             # False = se respondió "no lo encontré"

    # Reducer: cada nodo concatena a la traza en vez de reemplazarla.
    traza: Annotated[list[dict], operator.add]
