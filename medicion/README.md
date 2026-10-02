# medicion/ — campaña de tokens, tiempo y éxito

Mide lo que pidió Cristian el 2026-09-30:
- **tokens y tiempo**, por agente y del sistema, para cada tarea con 5 usuarios;
- **tasa de éxito**, donde un timeout cuenta como fallo;
- en la tarea medicamento × comida, si el **camino** recorrido fue el correcto.

Es un **prototipo exploratorio**: los datos son ficticios y el resultado es un
archivo local. No es un sistema de monitoreo productivo.

## Qué hay

| Archivo | Qué hace |
|---|---|
| `registro.py` | Callback de LangChain que registra en local cada llamada al LLM y cada nodo del grafo, con tokens de entrada, salida y razonamiento, segundos, modelo y `system_fingerprint`. Los agrupa por agente |
| `casos.py` | Carga los casos y arma el plan intercalado de ejecuciones |
| `datos/casos_campana.json` | Frases T1–T5 y su verdad de referencia, con el criterio de selección de cada tarea. T6 se lee de `interacciones/datos/casos_prueba.json` |
| `criterios.py` | Criterios de éxito por tarea. Son funciones puras y se prueban sin servidor |
| `campana.py` | Ejecuta la campaña y escribe `resultados/campana/<experimento>.jsonl`, con una línea por ejecución |
| `extraer_langsmith.py` | Saca el mismo experimento de LangSmith y lo compara con el JSONL |

## Matriz

6 tareas × 5 usuarios (rosa, manuel, carmen, jorge, elena) × 4 frases × 3
repeticiones = **360 ejecuciones**. Todas entran por **texto**; la voz se midió
aparte (bitácora 14-17).

| Tarea | Intención | Es éxito si… |
|---|---|---|
| T1 medicación | `MEDICATION_HEALTH` | ruteó a medicación y la respuesta nombra todos los medicamentos de la receta del usuario y ninguno más del catálogo |
| T2 recetas | `RECIPE_MULTIMEDIA` | ruteó a recetas y la fuente usada está entre las aceptadas, o no hay ninguna fuente si el plato no está en el recetario |
| T3 emergencia | `EMERGENCY` | ruteó a emergencia (por la red de seguridad o por el LLM) |
| T4 familia | `FAMILY_COMMUNICATION` | ruteó a familia |
| T5 small talk | `SMALL_TALK` | ruteó a small talk |
| T6 medicamento × comida | `MEDICATION_FOOD_CHECK` | el camino fue orchestrator → {medicacion_cruce, recetas_cruce} → integrador, la fuente es una de las aceptadas y las interacciones son exactamente las esperadas |
| Todas | — | terminó en menos de 120 s y sin excepción |

**De dónde salen las frases:**
- **T1:** del dataset; son preguntas de horario que no nombran un medicamento.
- **T2:** las de `evaluar_rag_real.py`.
- **T3 a T5:** muestra aleatoria del dataset con semilla 42, para no elegirlas a mano.
- **T6:** las de `casos_prueba.json`.

El detalle está en `datos/casos_campana.json`.

## Cómo se atribuyen los tokens a cada agente

LangGraph pone en la metadata de cada llamada el nodo que la hizo
(`langgraph_node`) y la ruta de nodos padre (`langgraph_checkpoint_ns`, por
ejemplo `recetas_cruce:…|evaluar_relevancia:…`):
- el **primer tramo** es el agente del grafo principal;
- el **último tramo** es el subnodo del RAG.

Así se mide sin tocar ningún agente. Los agentes que no llaman al LLM
(emergencia, familia, small talk, medicacion_cruce) aparecen con 0 tokens y su
tiempo real.

## Cómo ejecutar

Hace falta la VPN, el `.env` con los modelos y, para cruzar con LangSmith, la
clave de LangSmith. Todo se corre desde la raíz del repo.

```bash
python -m pruebas.prueba_medicion                                  # 9 pruebas, sin VPN
python -m medicion.campana --experimento piloto_AAAA-MM-DD --piloto  # 6 ejecuciones, ~1 min
python -m medicion.campana --experimento campana_AAAA-MM-DD          # 360 ejecuciones
python -m medicion.campana --experimento campana_AAAA-MM-DD --reanudar
python -m medicion.extraer_langsmith --experimento campana_AAAA-MM-DD \
    --comparar resultados/campana/campana_AAAA-MM-DD.jsonl
```

**Protocolo (chuleta 7.5):**
- Correr la campaña con todo commiteado. El `_meta.json` guarda el commit y si
  había cambios sin commitear.
- El orden está **intercalado** dentro de cada repetición y la repetición *r*
  termina antes de que empiece la *r+1*.
- Si cambia el `system_fingerprint`, el script avisa y **se repite ese bloque**.

## Validación (piloto del 2026-10-01)

Rosa × 6 tareas × 1 frase:
- **5/6 éxitos.** El fallo es T4 f1 ("Modifica lo de la cita médica…"): el
  dataset la etiqueta como familia y el modelo la manda a medicación. Es
  ambigüedad de la etiqueta y se deja así a propósito.
- **Tokens idénticos en 6/6** entre el registro local y LangSmith (entrada,
  salida y razonamiento).
- **Tiempos** con menos de 10 ms de diferencia.

## Límites que se declaran al reportar

- **Embeddings:** las llamadas a BGE-M3 no pasan por los callbacks de chat, así
  que no hay tokens de embeddings. Su costo queda dentro del **tiempo** del
  subnodo `recuperar`.
- **Ramas paralelas:** en T6 la suma de los tiempos por agente es mayor que el
  tiempo del sistema. El tiempo del sistema es el que vive la persona.
- **Timeout:** una ejecución que pasa de 120 s sigue corriendo en su hilo. Se la
  espera hasta 300 s, sin contarla, antes de lanzar la siguiente, para no cargar
  el servidor con dos consultas a la vez.
- **Razonamiento:** el servidor tiene `enable_thinking` activo y se deja así
  (chuleta 3.2). Los tokens de razonamiento son **parte** de los de salida y se
  reportan aparte. En el piloto son el 61-86% de la salida, según el nodo.
- **Servidor compartido:** la latencia depende de su carga (bitácora 11). Por eso
  se intercala y se repite.
