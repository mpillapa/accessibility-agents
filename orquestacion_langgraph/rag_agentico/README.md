# RAG agéntico del especialista en recetas

Subgrafo de LangGraph que reemplaza la recuperación fija que antes hacía
`nodo_recetas`. Responde al pedido de los tutores de la reunión del 2026-08-19:
que el RAG deje de ser un componente aislado y pase a ser parte del sistema
multiagente.

## Qué cambió, y por qué no era solo "migrar"

El RAG anterior ya se invocaba desde el grafo: `nodo_recetas` llamaba a
`buscar_receta(consulta, k=3)` y le pasaba al LLM lo que saliera. El problema
no era que estuviera aislado, era que la recuperación no era una decisión:

| | RAG anterior | RAG agéntico |
|---|---|---|
| ¿Busca? | Siempre | Solo si la consulta lo requiere |
| ¿Juzga lo recuperado? | No — se lo pasa al LLM tal cual | Sí, fragmento por fragmento |
| Si la búsqueda falla | Responde con contexto irrelevante | Reformula la consulta y reintenta |
| Si no existe la receta | El LLM improvisa con lo que haya | Lo admite explícitamente |
| Trazabilidad | Ninguna | Traza por paso, con distancias y veredictos |

El caso que lo motiva es concreto y propio de esta población: un adulto mayor
dice *"eso dulce del arrocito que hacía mi mamá"* y el recetario está indexado
como *"arroz con leche"*. La búsqueda falla por vocabulario, no porque falte la
receta. Un pipeline lineal responde mal; este vuelve sobre sus pasos.

## Flujo

```
START -> decidir_busqueda
           |-- (no hace falta) --> responder_sin_recetario --> END
           `-- (sí hace falta) --> recuperar
                                      |
                                      v
                                 evaluar_relevancia
                                      |
           .--------------------------+--------------------------.
           |                          |                          |
      (hay útiles)            (sin útiles,               (sin útiles,
           |                   quedan intentos)           sin intentos)
           v                          |                          |
        generar                   reformular                sin_resultado
           |                          |                          |
           v                          `--> recuperar (ciclo)     v
          END                                                   END
```

## Reglas de negocio

Están como constantes con nombre en `estado.py`, no dentro de los prompts, para
que se puedan revisar y discutir sin leer el código de los nodos.

| Regla | Valor | Justificación |
|---|---|---|
| `MAX_INTENTOS_RECUPERACION` | 2 | El usuario espera a lo sumo dos rondas. Subirlo mejora el recall a costa de latencia, que aquí no es cosmética: un adulto mayor esperando frente a un dispositivo asume que se dañó. |
| `FRAGMENTOS_POR_BUSQUEDA` | 3 | Se mantiene el `k=3` del RAG anterior para que la comparación no mezcle variables. |
| `EVALUAR_FRAGMENTO_POR_FRAGMENTO` | `True` | Permite descartar un fragmento malo y conservar los buenos (patrón tipo CRAG). Cuesta una llamada al LLM por fragmento; ponerlo en `False` juzga el conjunto en una sola llamada, más rápido pero todo-o-nada. |

Decisiones de diseño que no son constantes pero sí son reglas:

- **Ante duda, buscar.** Si el nodo de decisión devuelve algo ambiguo, se
  consulta el recetario. Es preferible una búsqueda de más que responder sobre
  una receta sin haberla mirado.
- **`sin_resultado` no usa el LLM.** La respuesta de "no encontré esa receta"
  es un texto fijo. Es el punto donde el sistema tiene más incentivo a
  inventar, y la forma más segura de no alucinar una receta es no darle al
  modelo la oportunidad de redactarla. Una receta inventada no es un error
  cosmético: puede terminar en alguien cocinando mal algo que después se come.
- **La distancia vectorial no se usa como umbral.** `buscar_receta_detallado()`
  la devuelve y se registra en la traza para poder analizarla, pero quien
  decide si un fragmento sirve es el evaluador. Un umbral numérico habría que
  calibrarlo contra consultas etiquetadas, y ese trabajo no está hecho —
  poner un número a dedo sería inventar una regla.

## Costo en llamadas al LLM

Es el trade-off principal y hay que tenerlo presente al comparar latencias con
el RAG anterior (que gastaba 1 llamada siempre):

| Escenario | Llamadas |
|---|---|
| No hace falta buscar | 2 (decidir + responder) |
| Acierta en el primer intento | 1 + 3 + 1 = 5 |
| Reformula y acierta | 1 + 3 + 1 + 3 + 1 = 9 |
| Se rinde | 1 + 3 + 1 + 3 = 8 (la respuesta final no usa LLM) |

Con `EVALUAR_FRAGMENTO_POR_FRAGMENTO = False` bajan a 3 / 5 / 4.

## Estado propio, y por qué importa para la tesis

El subgrafo tiene su propio `EstadoRAG`, separado del `EstadoConversacion` del
grafo principal. El grafo principal no necesita saber cuántos intentos hubo ni
qué fragmentos se descartaron: recibe la respuesta y la traza. Ese límite es lo
que permite reemplazar el RAG sin tocar el grafo.

Dos cosas de este subgrafo son material directo para el paper:

1. **El ciclo.** `reformular -> recuperar` es un arco que vuelve atrás. Los
   frameworks orientados a pipelines secuenciales no lo expresan.
2. **El reducer de estado.** El campo `traza` usa
   `Annotated[list[dict], operator.add]` para que cada nodo concatene en vez de
   sobreescribir. Es exactamente el manejo de estado acumulado que los tutores
   señalaron como limitación de CrewAI. Cuando se replique este subgrafo en
   CrewAI (Fase 8), la limitación queda demostrada y no citada.

## Pruebas

```bash
python -m pruebas.prueba_ciclo_rag
```

Cubren los cuatro caminos del grafo (acierto directo, reformulación exitosa,
rendirse en el tope de intentos, no consultar el recetario) con dobles de
prueba en lugar del LLM y la base vectorial. **No requieren VPN**: verifican la
lógica del grafo, no la calidad de los juicios del modelo. Eso último se mide
en `../../notebooks/comparativa.ipynb`, que sí requiere VPN.

## Pendientes

- Medir en el notebook, con el LLM real, cuántas veces el evaluador acierta al
  juzgar relevancia y cuántas la reformulación rescata una búsqueda fallida.
- El recetario tiene 2 recetas de prueba y 1 imagen sintética. El camino
  `sin_resultado` se dispara con casi cualquier consulta real; el corpus se
  amplía en la Fase 3 (fotos reales del recetario).
