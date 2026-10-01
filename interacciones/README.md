# Tarea medicamento × comida

Responde a una pregunta como *"¿puedo comer hornado con las pastillas que tomo?"*
cruzando **lo que la persona tiene recetado** con **lo que lleva el plato**.

> **Datos ficticios e ilustrativos.** Las interacciones se inspiran en
> advertencias conocidas de estos principios activos, pero la selección, la
> severidad y las recomendaciones fueron redactadas con ayuda de IA y **no están
> validadas por un profesional**. Nada de este módulo sirve para decidir qué
> comer o qué tomar.

---

## Por qué existe

Es la única tarea del sistema que necesita **dos especialistas a la vez**: el de
medicación (qué toma la persona) y el de recetas (qué lleva el plato). Las otras
cinco siguen el camino orquestador → un agente → fin. Cristian la pidió el
2026-09-30 por dos motivos: tener un flujo con varios agentes y poder medir si
el camino recorrido fue el correcto.

## Diseño (implementado el 2026-09-30)

```
                        ┌─► medicacion_cruce ─┐
orchestrator ──(MEDICATION_FOOD_CHECK)──┤                     ├─► integrador ─► fin
                        └─► recetas_cruce ────┘
```

Diagrama generado: `../orquestacion_langgraph/grafo.png`.

| Nodo | Qué hace | ¿LLM? |
|---|---|---|
| `orchestrator` | Clasifica la intención. Para esta tarea devuelve **dos** ramas y LangGraph las corre en paralelo | Sí, 1 llamada |
| `medicacion_cruce` | Lee los medicamentos de la prescripción vigente (`reglas.medicamentos_vigentes`) | No |
| `recetas_cruce` | Subgrafo de RAG agéntico **sin generación**: solo averigua qué receta es. Se le pregunta por los ingredientes del plato, no por las pastillas, porque el evaluador de relevancia juzga si el fragmento responde la consulta | Sí, 2 o más llamadas |
| `integrador` | Espera a las dos ramas, cruza en código (`reglas.cruzar`) y el LLM redacta (`agente.redactar`). Si no es evaluable, responde con texto fijo y sin LLM | Sí, 1 llamada (0 si no es evaluable) |

Cada rama escribe su propio campo del estado (`medicamentos_vigentes`,
`recetas_encontradas`): si las dos escribieran el mismo, LangGraph no sabría cuál
conservar.

**El cruce es código determinista; el LLM solo redacta.** Es la regla que sigue
todo el proyecto: lo que puede hacer daño no lo decide el modelo (ver
`medicacion/README.md` y la sección 15 de la bitácora).

## Estado

| Pieza | Estado |
|---|---|
| `datos/interacciones_alimentarias.json` | Hecho (2026-09-30) |
| `datos/ingredientes_recetas.json` | Hecho (2026-09-30) |
| `datos/casos_prueba.json` | Hecho (2026-09-30) |
| `datos.py`: acceso a los datos | Hecho (2026-09-30) |
| `pruebas/prueba_datos_interacciones.py` | Hecho: 9/9 pasan |
| `reglas.py`: el cruce determinista, con veredicto (evitar / precaución / sin interacción / no evaluable) | Hecho (2026-09-30) |
| `agente.py`: la redacción; texto fijo si no es evaluable | Hecho (2026-09-30) |
| Intención nueva, fan-out y nodo integrador en el grafo | Hecho (2026-09-30) |
| `pruebas/prueba_interacciones.py`: cruce + camino del grafo con dobles | Hecho: 12/12 |
| `pruebas/evaluar_interacciones_real.py`: las 20 combinaciones con modelos reales | Hecho (resultados en la bitácora, sección 20) |

## Los datos

### `interacciones_alimentarias.json`: el catálogo

- **4 categorías de alimento** con lista cerrada: `alcohol`, `cafeina`,
  `alto_potasio` y `pomelo`. Un alimento que no está en la lista no dispara nada.
- **10 interacciones** que cubren los 11 medicamentos que aparecen en las
  prescripciones. Omeprazol y Amoxicilina están listados como *revisados sin
  interacción*, para que su ausencia sea una decisión y no un olvido.
- **Severidad:** `evitar` o `precaucion`.
- **`solo_si`:** una condición del perfil que tiene que cumplirse. Hoy la usa
  una sola regla (ver la decisión 1).

### `ingredientes_recetas.json`: qué lleva cada receta

Tiene una entrada por cada una de las **29 fuentes del índice** de ChromaDB (reingerido con qwen2.5vl:7b el 2026-09-30) y
usa como llave el mismo nombre de archivo que devuelve la traza del RAG. Cada
alimento marcado cita la **evidencia literal** del texto indexado. La prueba de
integridad verifica que las 46 citas estén de verdad en el índice.

Al reingerir con Qwen la prueba detectó dos fuentes que ya no estaban y 7 citas
con otra redacción. Se corrigieron sobre el texto nuevo y se agregaron dos
platos de `receta13.jpeg` que GLM-OCR no había leído (fritada y ají de
librillo), verificados contra la foto. La versión sobre GLM-OCR está en el
historial de git (commit `1ee8a80`).

Se construyó leyendo **lo que el OCR dejó en el índice**, no la foto ni lo que
el plato "suele llevar". Si el OCR no leyó un ingrediente, aquí tampoco está: el
cruce solo puede razonar sobre lo que el RAG puede recuperar.

### `casos_prueba.json`: la verdad de referencia

- **`desarrollo`:** 11 casos con la fuente dada, para probar el cruce sin el RAG
  (6 con interacción, 5 sin, 1 no evaluable).
- **`campana`:** las 4 frases de la tarea T6 para la medición × 5 usuarios =
  20 combinaciones (11 con interacción, 9 sin).

Se escribieron **a mano**, no con el código del cruce: si la misma lógica
calculara la respuesta correcta, la prueba sería circular. Un cálculo
independiente y descartable, hecho el 2026-09-30, coincidió en los 31 casos.

## Decisiones

1. **El potasio solo alerta con función renal reducida.** Losartán × alimentos
   altos en potasio aplica solo si el perfil tiene `funcion_renal: reducida`. El
   recetario tiene plátano, papa o aguacate en casi todos los platos, y alertar
   siempre daría un aviso en casi cada receta. Es el mismo tipo de sinsentido
   que la bitácora documenta en la sección 16. **Consecuencia que hay que
   declarar:** con los perfiles actuales esta regla nunca se activa, porque
   Carmen, la única con función renal reducida, no toma losartán. Se prueba con
   un perfil sintético en las pruebas del cruce.
2. **El alcohol cuenta en cualquier cantidad**, cocido o no. No se modela cuánto
   queda después de la cocción. El campo `parte` (principal, acompañamiento u
   opcional) permite matizar la respuesta, pero no cambia el veredicto.
3. **Solo interacciones medicamento–alimento.** Las de condición–alimento
   (diabetes con azúcar, hipertensión con sal) quedan fuera.
4. **El cacao no cuenta como cafeína**, para no alertar por una torta de
   chocolate.
5. **Una persona sin prescripción no es evaluable.** El sistema no puede decir
   "no hay problema" si no sabe qué toma (caso `d11`, Luis).
6. **Los datos van en archivos propios y `medicamentos.json` no se toca.** El
   plan inicial era agregarle un campo al vademécum; se separó para no alterar
   los datos que ya usa el agente de medicación y sus pruebas.

## Limitaciones

- **Una foto con varios platos es una sola fuente.** El cruce une las
  categorías de todos sus platos: preguntar por los calamares de `receta14.jpeg`
  alerta por la cerveza de los cangrejos de la misma página.
- **`berenjena.jpg`** (`calidad_ocr: baja`): con Qwen el texto es fluido pero
  no fiel. Omite los ingredientes e inventa frases, y pasa el control de
  calidad porque no repite nada (ver su `nota_ocr`).
- **Tres fotos no están en el índice** (`_fuentes_excluidas`): Qwen entró en
  bucle con ellas.
- **La variedad entre usuarios es baja:** todos toman algún medicamento que
  interactúa con alcohol, así que en las recetas con alcohol todos reciben
  aviso. Solo el café separa a un usuario (Elena) de los demás.

## Cómo probarlo

```bash
.venv/bin/python -m pruebas.prueba_datos_interacciones
```

No requiere VPN ni LLM. Las dos pruebas que leen el índice se omiten con aviso
si `rag/chroma_db/` no existe.
