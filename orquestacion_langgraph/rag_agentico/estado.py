# Estado del subgrafo de RAG agéntico, y las reglas que gobiernan su ciclo.
#
# Es un estado SEPARADO del EstadoConversacion del grafo principal: el grafo
# principal no necesita saber que hubo reformulaciones o cuántos fragmentos se
# descartaron, solo recibe la respuesta final y la traza. Mantenerlos separados
# es lo que permite tratar el RAG como un componente reemplazable.

import operator
from typing import Annotated, Optional, TypedDict

# --- Reglas de negocio del ciclo de recuperación ---------------------------
# Explícitas y con nombre a propósito: son decisiones de diseño discutibles,
# no detalles de implementación, y tienen que poder revisarse sin leer los
# prompts. Ver el README de este paquete para la justificación de cada una.

# Cuántas búsquedas se permiten como máximo por consulta (la original más las
# reformulaciones). Con 2, el usuario espera a lo sumo dos rondas antes de
# recibir una respuesta o un "no lo encontré". Subirlo mejora el recall a
# costa de latencia, que en esta población importa: un adulto mayor esperando
# frente a un dispositivo asume que se dañó.
MAX_INTENTOS_RECUPERACION = 2

# Cuántos fragmentos pide a la base vectorial en cada búsqueda.
FRAGMENTOS_POR_BUSQUEDA = 3

# Si es True, el nodo evaluador juzga CADA fragmento por separado (patrón tipo
# CRAG: filtra los irrelevantes y conserva los buenos). Cuesta una llamada al
# LLM por fragmento. Si es False, juzga el conjunto en una sola llamada: más
# rápido, pero no puede descartar un fragmento malo y quedarse con los otros.
EVALUAR_FRAGMENTO_POR_FRAGMENTO = True

# Cuando un fragmento pasa el filtro de relevancia, se traen también los demás
# fragmentos de su mismo archivo, en orden, antes de redactar la respuesta
# (patrón conocido como recuperación del documento padre).
#
# Motivo medido el 2026-08-19: ante "como hago el llapingacho" el sistema
# recuperaba la receta correcta, el filtro aceptaba 1 de 3 fragmentos, y la
# respuesta salía con un paso suelto ("fríelos en la manteca") en lugar de la
# receta. El troceado por párrafos reparte una receta en varios fragmentos y el
# filtro, al ser estricto, descarta parte de ella.
EXPANDIR_A_RECETA_COMPLETA = True

# Tope de caracteres del contexto que se le pasa al generador tras expandir. Un
# archivo con muchas recetas (una página doble, por ejemplo) puede tener docenas
# de fragmentos, y traerlos todos llenaría el prompt de recetas que el usuario
# no pidió. Al recortar se conservan primero los fragmentos que pasaron el
# filtro.
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

    # Trazabilidad (para el notebook comparativo y la defensa de la tesis).
    #
    # El Annotated[..., operator.add] es un REDUCER: le dice a LangGraph que
    # cuando un nodo devuelve {"traza": [x]} debe CONCATENAR con lo que ya
    # había, en vez de reemplazarlo (que es el comportamiento por defecto de
    # los demás campos). Sin esto, cada nodo borraría la traza del anterior y
    # el ciclo quedaría invisible.
    #
    # Vale la pena anotarlo para el paper: es exactamente el tipo de manejo de
    # estado acumulado que los tutores señalaron como limitación de CrewAI.
    traza: Annotated[list[dict], operator.add]
