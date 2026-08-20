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
   expandir_contexto              reformular                sin_resultado
           |                          |                          |
           v                          `--> recuperar (ciclo)     v
        generar                                                 END
           |
           v
          END
```

## Reglas de negocio

Están como constantes con nombre en `estado.py`, no dentro de los prompts, para
que se puedan revisar y discutir sin leer el código de los nodos.

| Regla | Valor | Justificación |
|---|---|---|
| `MAX_INTENTOS_RECUPERACION` | 2 | El usuario espera a lo sumo dos rondas. Subirlo mejora el recall a costa de latencia, que aquí no es cosmética: un adulto mayor esperando frente a un dispositivo asume que se dañó. |
| `FRAGMENTOS_POR_BUSQUEDA` | 3 | Se mantiene el `k=3` del RAG anterior para que la comparación no mezcle variables. |
| `EVALUAR_FRAGMENTO_POR_FRAGMENTO` | `True` | Permite descartar un fragmento malo y conservar los buenos (patrón tipo CRAG). Cuesta una llamada al LLM por fragmento; ponerlo en `False` juzga el conjunto en una sola llamada, más rápido pero todo-o-nada. |
| `EXPANDIR_A_RECETA_COMPLETA` | `True` | Cuando un fragmento pasa el filtro, se traen los demás fragmentos de su archivo, en orden. Sin esto la respuesta sale con un paso aislado de la receta (ver "Hallazgos medidos"). No cuesta llamadas al LLM, solo una consulta a ChromaDB por fuente. |
| `MAXIMO_CARACTERES_CONTEXTO` | 6000 | Tope del contexto tras expandir. Un archivo con varias recetas puede tener docenas de fragmentos; al recortar se conservan primero los que pasaron el filtro. |

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
- **La expansión no vuelve a evaluar relevancia.** Si un fragmento de la receta
  pasó el filtro, se asume que la receta entera es la que el usuario pidió. La
  contrapartida es que un falso positivo del evaluador arrastra una receta
  completa al contexto, no un fragmento: la expansión amplifica los errores del
  filtro en las dos direcciones.

## Hallazgos medidos (2026-08-19)

Medidos con `pruebas/evaluar_rag_real.py`, contra el LLM y el recetario reales.
Vale registrarlos porque ninguno se veía sin ejecutar el sistema completo.

**1. El filtrado estricto produce respuestas incompletas.** Ante *"como hago el
llapingacho"*, el sistema recuperaba la receta correcta, el filtro aprobaba 1 de
3 fragmentos y la respuesta era *"para preparar los llapingachos, fríalos en la
manteca de chancho"* — la receta correcta, un paso aislado de ella. El troceado
por párrafos reparte una receta en varios fragmentos y el filtro, al ser
estricto, descarta parte. Con `expandir_contexto`, la misma consulta pasó de 341
a 2.225 caracteres, y el generador incluso detectó que el recetario tiene dos
versiones del plato y explicó ambas.

Es el trade-off central del RAG agéntico: **el filtro de relevancia mejora la
precisión de lo que entra al contexto, pero fragmenta la respuesta.** Recuperar
el documento padre lo corrige sin renunciar al filtro.

**2. Los encabezados sueltos contaminan la búsqueda.** El troceado dejaba
`"PREPARACIÓN"`, `"Ingredientes:"` y los títulos como fragmentos propios. Al no
tener contenido, su embedding no representa ninguna receta y quedan cerca de
cualquier consulta: buscar *"sushi"* devolvía tres fragmentos `"PREPARACIÓN"` de
tres recetas distintas. Se corrigió fusionándolos con la sección que encabezan
(`MINIMO_CARACTERES_FRAGMENTO` en `rag/ingesta.py`), en lugar de descartarlos.

**3. Un ejemplo dentro del prompt puede contaminar el caso homólogo.** Al
endurecer el evaluador se le agregó como ejemplo *"que el sushi lleve arroz no
convierte una receta de arroz con leche en una receta de sushi"*. Con ese texto
en el prompt, evaluar una consulta real de sushi dio **peores** veredictos que
antes: el modelo mezclaba el ejemplo con el caso a juzgar. El ejemplo actual usa
la papa, que no aparece en los casos de prueba.

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

**Lógica del grafo, sin VPN** (dobles en lugar del LLM y de ChromaDB):

```bash
python -m pruebas.prueba_ciclo_rag
```

Cubre los caminos del grafo: acierto directo, expansión a la receta completa,
reformulación exitosa, rendirse en el tope de intentos y no consultar el
recetario. Verifica la lógica, no la calidad de los juicios del modelo.

**Comportamiento real, con VPN** (LLM y recetario de verdad):

```bash
python -m pruebas.evaluar_rag_real
python -m pruebas.evaluar_rag_real --json resultados.json
```

Mide lo que no se puede afirmar con aserciones fijas, porque la salida de un LLM
varía entre corridas: si el evaluador acierta, si la respuesta llega completa,
si el sistema admite lo que no sabe. Está pensado para comparar el antes y el
después de un cambio en los prompts o en la estrategia de recuperación —
guardar el `--json` antes de tocar algo y volver a correrlo después.

## Pendientes

- Medir en el notebook, con el LLM real, cuántas veces el evaluador acierta al
  juzgar relevancia y cuántas la reformulación rescata una búsqueda fallida.
- El recetario tiene 2 recetas de prueba y 1 imagen sintética. El camino
  `sin_resultado` se dispara con casi cualquier consulta real; el corpus se
  amplía en la Fase 3 (fotos reales del recetario).
