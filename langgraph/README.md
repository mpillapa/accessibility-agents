# Réplica en LangGraph

Misma arquitectura de agentes que `../crewai/`, reimplementada en LangGraph para
la comparación de frameworks que pidieron los tutores (correo del 2026-07-21).
La diferencia central: en CrewAI el Orchestrator **delega** a otro agente
(tool calling interno del framework); aquí el Orchestrator escribe la intención
en un **estado compartido** y un **edge condicional** rutea al nodo especialista.

## Alcance acordado

Replica el mismo recorte que ya existe en CrewAI, no el diseño completo de 4
agentes de la tesis:

- Orchestrator
- Especialista en Medicación y Salud (real)
- Especialista en Recetas y Multimedia (real, con RAG — ver `../rag/`)
- Family Bridge (stub)
- Emergency Sentinel (stub)

El ruteo lo decide el propio LLM dentro del grafo — no hay clasificador ML
(perteneció a otro proyecto y se descartó para esta comparación).

## Flujo y cruce de información entre agentes

```
START -> orchestrator -> (edge condicional según "intencion") -> especialista -> END
```

1. Entra la `consulta` del usuario al estado compartido.
2. El nodo `orchestrator` la analiza y escribe `intencion` + `razonamiento` en ese estado.
3. Un edge condicional lee `intencion` y decide a qué nodo especialista saltar.
4. El especialista lee `consulta` (y, en recetas, el resultado del RAG) y escribe `respuesta`.
5. El grafo termina (`END`) y devuelve el estado final completo.

**El punto clave para la tesis**: los agentes no se pasan mensajes directos.
Comparten un único objeto de estado (`EstadoConversacion`, en `estado.py`) que
cada nodo lee y sobre el que cada nodo escribe. `grafo.py` expone
`procesar_consulta_verbose()` / `traza_por_nodo()` usando `app.stream(...,
stream_mode="updates")`: cada elemento del stream es `{nombre_del_nodo:
cambios_que_escribió en el estado}`, así que deja ver ese intercambio en el
momento exacto en que ocurre. Es la forma de "ver la información que se cruza
entre los agentes" (ver `../notebooks/comparativa.ipynb`).

## Estructura

- `estado.py` — el `State` (`TypedDict`) que viaja entre nodos: `consulta` (entrada, no cambia), `intencion` y `razonamiento` (los escribe el orchestrator), `respuesta` (la escribe el especialista). Cada nodo devuelve un `dict` parcial y LangGraph lo mergea sobre el estado acumulado.
- `agentes.py` — LLM compartido y las funciones de cada nodo. Incluye `clasificar_consulta()` (solo la decisión del orchestrator, para medir accuracy de ruteo).
- `grafo.py` — construye el `StateGraph` (`add_node`, `add_edge`, `add_conditional_edges`, `compile`) y expone `procesar_consulta()`, `procesar_consulta_verbose()`, `traza_por_nodo()` y `clasificar_consulta()`.
- `demo.py` — demo en vivo (mismo formato que `../crewai/demo.py`): intención detectada, razonamiento y respuesta, con traza nodo por nodo.
- `visualizar.py` — dibuja el grafo (`draw_ascii` local vía `grandalf`, `draw_mermaid`, o PNG vía `mermaid.ink` en `grafo.png`).

Uso:
```bash
python langgraph/demo.py                 # corre las 3 frases de ejemplo
python langgraph/demo.py "tu frase aquí" # corre una frase propia
```

## Hallazgo: salida estructurada de Pydantic no fue confiable aquí

El plan inicial era que el Orchestrator devolviera la intención con
`llm.with_structured_output(DecisionIntencion)` (JSON forzado), tal como
sugiere el material que mandaron los tutores. En la práctica, contra el
servidor de la universidad, el modelo generaba un razonamiento correcto pero el
campo `intencion` quedaba desconectado de ese razonamiento (ej. razonaba "esto
es una emergencia" y aun así completaba `SMALL_TALK`). Se probó con
`llama3.1:8b` y `qwen3.6:latest`, reordenando el enum y con
`method="function_calling"` — falló de forma similar en los tres casos.

En texto libre (sin forzar JSON), el mismo modelo clasifica correctamente. La
solución actual: el LLM razona y responde en texto libre terminando con una
línea `CATEGORIA: <opción>`, se extrae esa etiqueta con una expresión regular,
y **Pydantic valida** el resultado extraído (`DecisionIntencion`) en vez de
forzar la generación. Sigue siendo Pydantic + LLM, pero validando después de
generar, no restringiendo la generación misma.

Esto es un hallazgo real, no solo un detalle de implementación — vale la pena
mencionarlo en la tesis como limitación de salida estructurada con modelos
pequeños servidos localmente.

## Decisiones y supuestos (para la tesis)

- **Mismo LLM que CrewAI**: `google/gemma-4-12B-it` vía vLLM (OpenAI-compatible) en el servidor de la universidad, para que la comparación no mezcle la variable "modelo". Antes del 2026-07-23 era `llama3.1:8b` vía Ollama (ver `../README.md`, "Reglas y supuestos de esta migración").
- **Mismos roles/prompts** que los agentes de `../crewai/agentes.py`, adaptados a un solo prompt por nodo, para que la comparación sea justa.
- **Family y Emergency son stubs**: respuestas fijas, sin LLM y sin integración real.
- **Small talk** responde fijo, sin llamar al LLM (mismo atajo que CrewAI).

## Pendiente

- Comparar formalmente contra `../crewai/` — ya arrancado en `../notebooks/comparativa.ipynb` (accuracy de ruteo con `../dataset.csv`); falta cerrar latencia, líneas de código y legibilidad del flujo de datos.
- **Repetir el hallazgo de salida estructurada con el modelo actual.** El hallazgo de arriba se probó con `llama3.1:8b`/`qwen3.6` en Ollama, no con `google/gemma-4-12B-it` en vLLM. La solución actual (texto libre + regex + validación posterior) se dejó igual como precaución y el ruteo funciona al 100% en la muestra evaluada (`../notebooks/comparativa.ipynb`), pero falta confirmar si `with_structured_output` (JSON forzado) seguiría fallando con este modelo/servidor.
