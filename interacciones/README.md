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

## Diseño (la parte del grafo todavía no está implementada)

```
orchestrator ──(intención nueva)──► medicacion ─┐
                                               ├─► integrador ─► fin
                              └──► recetas ────┘
```

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
| `reglas.py`: el cruce | Pendiente |
| Intención nueva, fan-out y nodo integrador en el grafo | Pendiente |

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

Tiene una entrada por cada una de las **31 fuentes del índice** de ChromaDB y
usa como llave el mismo nombre de archivo que devuelve la traza del RAG. Cada
alimento marcado cita la **evidencia literal** del texto indexado. La prueba de
integridad verifica que las 42 citas estén de verdad en el índice.

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
- **Las recetas con OCR degradado** (`aguado de gallina.jpg` y `berenjena.jpg`,
  marcadas `calidad_ocr: baja`) tienen listas de ingredientes incompletas.
- **La variedad entre usuarios es baja:** todos toman algún medicamento que
  interactúa con alcohol, así que en las recetas con alcohol todos reciben
  aviso. Solo el café separa a un usuario (Elena) de los demás.

## Cómo probarlo

```bash
.venv/bin/python -m pruebas.prueba_datos_interacciones
```

No requiere VPN ni LLM. Las dos pruebas que leen el índice se omiten con aviso
si `rag/chroma_db/` no existe.
