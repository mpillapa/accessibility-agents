# Catálogo de prompts

Todos los prompts que el sistema le envía al LLM (GLM-5.3-Flash, temperatura
0,3), con:
- el nodo y el archivo de donde salen;
- las variables que reciben;
- la salida que se espera;
- **por qué están escritos así**.

Es la fuente para la sección de métodos del paper. Pedido de Cristian, reunión
del 2026-09-30.

**Transcrito del commit `250a11b` (2026-10-01).** Si se cambia un prompt en el
código, hay que actualizarlo aquí. Las variables van entre `{llaves}`.

## Resumen

| # | Prompt | Nodo del grafo | Archivo | En la campaña |
|---|---|---|---|---|
| 1 | Clasificación de intención | `orchestrator` | `orquestacion_langgraph/agentes.py` | Todas las tareas (salvo lo que atrapa la red de emergencia) |
| 2 | ¿Hace falta el recetario? | RAG `decidir_busqueda` | `orquestacion_langgraph/rag_agentico/nodos.py` | T2, T6 |
| 3 | ¿El fragmento es de la receta pedida? | RAG `evaluar_relevancia` | ídem | T2, T6 (uno por fragmento, hasta 3) |
| 4 | ¿Los fragmentos alcanzan? (alternativa) | RAG `evaluar_relevancia` | ídem | No: desactivado (`EVALUAR_FRAGMENTO_POR_FRAGMENTO = True`) |
| 5 | Reformular la búsqueda | RAG `reformular` | ídem | T2, T6, solo si el primer intento falla |
| 6 | Redactar la receta | RAG `generar` | ídem | T2 (en T6 no se genera) |
| 7 | Responder sin recetario | RAG `responder_sin_recetario` | ídem | Solo si (2) decide no buscar |
| 8 | Redactar el plan de medicación | `medicacion` | `medicacion/agente.py` | T1 |
| 9 | Alternativas a un medicamento faltante | `medicacion` | ídem | No: ninguna frase de T1 dice que le falta algo |
| 10 | Variante solo-LLM de medicación | — (experimento, fuera del grafo) | ídem | No: es la línea de comparación de la bitácora 16 |
| 11 | Integrador medicamento × comida | `integrador` | `interacciones/agente.py` | T6 |

**Patrón común de salida.** Los prompts 1 a 4 piden razonar en una línea y
cerrar con una línea fija (`CATEGORIA: X`, `DECISION: X`, `VEREDICTO: X`), que
se extrae con una expresión regular y se valida con Pydantic. No se usa
`with_structured_output()` porque no fue confiable con los modelos disponibles
(ver `orquestacion_langgraph/README.md`). Cuando la etiqueta no aparece, cada
nodo cae a la opción **segura**:
- el RAG busca igual;
- el evaluador rechaza el fragmento;
- el Orchestrator busca el nombre de una categoría en el texto y, si no
  encuentra ninguno, va a SMALL_TALK. Esa última salida nunca se activó en 415
  frases (bitácora 20.3).

**Lo que NO es prompt (texto fijo, sin LLM).** Emergencia, familia, small talk,
"no encontré la receta" (`sin_resultado`) y los casos no evaluables del
integrador responden con texto escrito en código. Es deliberado: son los puntos
donde equivocarse cuesta más, o donde el modelo tiene más incentivo a inventar.
El texto está en `agentes.py` (`MENSAJE_DE_EMERGENCIA`), `rag_agentico/nodos.py`
(`nodo_sin_resultado`) e `interacciones/agente.py` (`TEXTO_NO_EVALUABLE`).

---

## 1. Clasificación de intención (Orchestrator)

- **Variables:** `{consulta}`. `{intenciones}` es la lista de las 6 categorías
  separadas por coma.
- **Salida:** 1-2 líneas de análisis y `CATEGORIA: <una de las 6>`.
- **Antes del LLM** corre la red de seguridad de emergencias
  (`red_emergencia.py`): si la frase coincide con un patrón inequívoco, se rutea
  a EMERGENCY sin llamar al modelo.

```text
Eres el primer punto de contacto de un asistente para adultos mayores. Hablan español coloquial ecuatoriano. Analiza brevemente (1-2 líneas) la intención detrás de esta consulta: '{consulta}'. Las categorías posibles son: MEDICATION_HEALTH, RECIPE_MULTIMEDIA, FAMILY_COMMUNICATION, EMERGENCY, SMALL_TALK, MEDICATION_FOOD_CHECK. Usa MEDICATION_FOOD_CHECK cuando pregunte si puede comer o preparar una comida o receta con los medicamentos que toma. Termina tu respuesta en una última línea con el formato exacto: CATEGORIA: <una de esas categorías>
```

**Por qué así:**
- **"Español coloquial ecuatoriano".** Las frases del dataset usan "mijito",
  "verá", diminutivos.
- **Una sola categoría tiene descripción: MEDICATION_FOOD_CHECK.** Es la nueva y
  la que más se confunde con medicación y con recetas. Las otras cinco se
  entienden por su nombre. Agregarla bajó el acierto de 93,7% a 93,5%, y las 4
  frases que cambiaron son ambigüedad de etiqueta (bitácora 20.3).
- **Razonar antes de la etiqueta.** Deja escrito por qué se eligió, y eso se
  muestra en la interfaz y queda en LangSmith.

---

## 2. ¿Hace falta consultar el recetario? (RAG, `decidir_busqueda`)

- **Variables:** `{consulta}`. En T6 la consulta llega reescrita como
  `¿Qué ingredientes lleva el plato que menciona esta persona? Lo que dijo: '{consulta original}'`
  (`CONSULTA_RAG_CRUCE` en `agentes.py`).
- **Salida:** `DECISION: BUSCAR` o `DECISION: RESPONDER_DIRECTO`. Si es ambigua,
  se busca.

```text
Eres el componente de un asistente de cocina para adultos mayores que decide si hace falta consultar el recetario.

Consulta el recetario si el usuario pide una receta concreta, sus ingredientes, cantidades o pasos.
NO lo consultes si es un agradecimiento, un comentario sobre algo que ya cocinó, o una pregunta general que no requiere una receta puntual.

Consulta del usuario: '{consulta}'

Razona en una línea y termina con una última línea exactamente así:
DECISION: BUSCAR   (o)   DECISION: RESPONDER_DIRECTO
```

**Por qué así:**
- Es lo que hace **agéntica** la recuperación: el agente decide si usa la
  herramienta, en vez de usarla siempre.
- Cuesta una llamada extra en cada consulta de recetas. Se acepta ese costo.

**Por qué la consulta se reescribe en T6:** el evaluador (3) juzga si un
fragmento responde lo que se preguntó. Un fragmento de la receta del hornado no
responde "¿puedo comer hornado con mis pastillas?". Por eso se le pregunta lo que
esa rama necesita, los ingredientes, sin gastar otra llamada al LLM.

---

## 3. ¿El fragmento pertenece a la receta pedida? (RAG, `evaluar_relevancia`)

- **Variables:**
  - `{consulta}`;
  - `{receta}`: el nombre del archivo de origen, legible (`arroz_con_leche.txt` →
    `arroz con leche`);
  - `{fragmento}`: el texto del fragmento.
- **Salida:** `VEREDICTO: SI` o `VEREDICTO: NO`.
- **Cuántas llamadas:** una por cada fragmento recuperado (k = 3).

```text
Eres un evaluador estricto de un buscador de recetas.

El usuario pidió: '{consulta}'

Fragmento del recetario (receta: "{receta}"):
{fragmento}

Pregunta: ¿este fragmento pertenece a la receta que pidió el usuario?

SI = el fragmento es parte de ESA receta (su título, ingredientes, pasos o forma de servirla).
NO = el fragmento es de una receta distinta, aunque comparta ingredientes o técnica de cocción.

Compartir un ingrediente no basta: dos recetas que usan papa siguen siendo recetas distintas.

Razona en una línea y termina con una última línea exactamente así:
VEREDICTO: SI   (o)   VEREDICTO: NO
```

**Por qué así.** Cada decisión corrige un fallo medido (bitácora 3):
1. **El criterio es "pertenece a la receta pedida", no "es útil".** Con "¿ayuda
   a responder?", el modelo aceptaba fragmentos de otro plato que compartía un
   ingrediente.
2. **Se le dice de qué receta viene el fragmento.** Sin ese dato, ante "quiero
   preparar sushi", el fragmento "cocinar el arroz en 500 ml de agua" (de arroz
   con leche) salía SI en las tres corridas. Con la fuente en el prompt pasó a
   NO en las tres, sin volverse más estricto con las consultas legítimas. El
   problema era falta de información, no redacción.
3. **El ejemplo usa la papa a propósito.** Con el caso sushi / arroz con leche
   como ejemplo dentro del prompt, el modelo mezclaba el ejemplo con el caso a
   juzgar y daba veredictos peores.

---

## 4. ¿Los fragmentos alcanzan? (RAG, `evaluar_relevancia`, alternativa desactivada)

- **Variables:** `{consulta}`. `{fragmentos}` son los textos separados por `---`.
- **Salida:** `VEREDICTO: SI` o `VEREDICTO: NO`.

```text
Eres un evaluador estricto. Decide si los fragmentos de recetario de abajo, en conjunto, alcanzan para responder la consulta del usuario.

Consulta del usuario: '{consulta}'

Fragmentos:
{fragmentos}

Razona en una línea y termina con una última línea exactamente así:
VEREDICTO: SI   (o)   VEREDICTO: NO
```

**Por qué existe y no se usa:**
- Es una sola llamada en lugar de tres.
- Pero juzga todo o nada: un fragmento bueno deja pasar a dos de otra receta.

La evaluación fragmento por fragmento (3) es más cara y más precisa. La
constante `EVALUAR_FRAGMENTO_POR_FRAGMENTO` permite comparar las dos.

---

## 5. Reformular la búsqueda (RAG, `reformular`)

- **Variables:**
  - `{consulta}`: la original;
  - `{busqueda_anterior}`: la que falló.
- **Salida:** una sola línea con la nueva búsqueda. Si viene vacía o pasa de 200
  caracteres, se repite la original.
- **Cuándo corre:** solo si el evaluador rechazó todo. Hay un máximo de 2
  intentos de recuperación.

```text
La búsqueda en un recetario no dio resultados útiles. Reescribe la consulta del usuario para que funcione mejor en una búsqueda semántica sobre recetas de cocina.

Usa el nombre probable del plato y sus ingredientes principales. Quita muletillas, diminutivos y referencias personales ('el que hacía mi mamá'). No inventes un plato distinto del que pide el usuario.

Consulta original del usuario: '{consulta}'
Búsqueda que ya se intentó y falló: '{busqueda_anterior}'

Responde ÚNICAMENTE con la nueva búsqueda, sin comillas ni explicación, en una sola línea.
```

**Por qué así:**
- Un adulto mayor rara vez usa el nombre del plato: dice "eso dulce del
  arrocito que hacía mi mamá", y en el recetario figura "arroz con leche". La
  primera búsqueda falla por vocabulario, no porque falte la receta.
- **"No inventes un plato distinto"** evita que la reformulación convierta una
  consulta sin respuesta en otra que sí la tiene.

---

## 6. Redactar la receta (RAG, `generar`)

- **Variables:**
  - `{recetario}`: la receta completa de cada fuente aprobada, reconstruida por
    `expandir_contexto`, con un máximo de 6.000 caracteres;
  - `{consulta}`.
- **Salida:** texto libre para la persona.

```text
Eres un asistente culinario que ayuda a personas mayores a preparar comidas. Adaptas medidas técnicas a referencias cotidianas (ej: 300ml = un vaso grande) para que sean comprensibles sin instrumentos de medición. Guías paso a paso y respondes en español.

Usa ÚNICAMENTE la información del recetario de abajo. No agregues ingredientes ni pasos que no estén ahí. Si el recetario no cubre alguna parte de lo que se pregunta, dilo abiertamente en vez de completarlo por tu cuenta.

Recetario:
{recetario}

Consulta del usuario: '{consulta}'
```

**Por qué así:**
- **Medidas cotidianas.** La población no tiene instrumentos de medición en la
  cocina.
- **"ÚNICAMENTE" y "dilo abiertamente".** Es el anclaje que separa un RAG de un
  modelo que cocina de memoria.
- **Lo que el prompt no puede garantizar** (que el modelo no complete por su
  cuenta) lo cubren dos piezas en código: el evaluador (3) y la salida fija
  `sin_resultado`.

---

## 7. Responder sin recetario (RAG, `responder_sin_recetario`)

- **Variables:** `{consulta}`.
- **Salida:** 2-3 líneas.

```text
Eres un asistente culinario cálido y breve que conversa con una persona mayor. Responde en español, en dos o tres líneas como máximo.

No describas recetas concretas ni des ingredientes o cantidades: no has consultado el recetario. Si el usuario termina pidiendo una receta puntual, ofrécele buscarla.

Mensaje del usuario: '{consulta}'
```

**Por qué así:** cuando (2) decide no buscar, el modelo no tiene datos. La
prohibición explícita de dar ingredientes o cantidades evita que responda una
receta de memoria por esta salida lateral.

---

## 8. Redactar el plan de medicación (`medicacion`, variante de reglas)

- **Variables:**
  - `{nombre}` y `{edad}` del perfil;
  - `{plan}`: el JSON con las tomas agrupadas por hora, armado y verificado en
    código por `medicacion/prescripciones.plan_diario`;
  - `{avisos}`: el JSON con los avisos de la verificación (alergia,
    contraindicación, exceso sobre el tope ajustado, receta vencida);
  - `{consulta}`.
- **Salida:** texto libre agrupado por hora. El aviso de prototipo se agrega en
  código al final, siempre.

```text
Eres un asistente que le explica a una persona mayor la medicación que le recetó su médico, en español claro y tratándola de usted.

El plan de abajo SALE DE SU RECETA MÉDICA y ya fue verificado. Tu tarea es SOLO redactarlo de forma comprensible, agrupado por hora.

REGLAS ESTRICTAS:
- NO agregues ningún medicamento que no esté en el plan.
- NO quites ninguno, ni siquiera los que tienen un aviso.
- NO cambies dosis ni horarios.
- NO sugieras reemplazos ni cambios de tratamiento.
- Si un renglón trae 'aviso', dilo JUNTO A ESE MEDICAMENTO, antes de la indicación, y con claridad. No lo escondas al final.
- Lo que hay que hacer ante un aviso viene en el campo 'que_hacer'. Transmítelo tal cual: si dice NO tomarlo, díselo sin rodeos; si dice consultar, no le digas que lo suspenda.
- Repite la 'nota_medico' tal como está: la escribió el médico.

Persona: {nombre}, {edad} años.
PLAN DEL DÍA (de su receta):
{plan}

AVISOS DE LA VERIFICACIÓN:
{avisos}

Consulta: '{consulta}'
```

**Por qué así:**
- **El modelo solo redacta.** La receta médica es la fuente de verdad, y
  organizar y verificar se hace en código determinista (bitácora 16). Las
  reglas "NO agregues / NO quites" son las mismas que mide el criterio de éxito
  de T1.
- **"NO quites ninguno, ni siquiera los que tienen un aviso".** El sistema
  **marca** lo problemático en vez de esconderlo. Si un medicamento con
  contraindicación desaparece del plan, la persona no se entera del problema.
- **El aviso junto al medicamento, no al final.** Una advertencia al pie de un
  texto largo no se lee.
- **"Transmítelo tal cual".** El modelo tiende a suavizar ("quizás convendría…")
  o a endurecer ("suspéndalo") lo que la verificación dijo.

---

## 9. Alternativas a un medicamento faltante (`medicacion`, variante de reglas)

- **Variables:**
  - `{alternativas}`: el JSON de `prescripciones.alternativas_para`, con los
    medicamentos del mismo grupo, si está en su receta y la dosis de catálogo;
  - `{consulta}`.
- **Cuándo corre:** la consulta dice que le falta algo ("se me acabó…") **y**
  nombra un medicamento del catálogo. La detección es determinista
  (`clasificar_consulta`).

```text
Eres un asistente que le habla a una persona mayor en español claro, tratándola de usted. Se quedó sin uno de sus medicamentos.

REGLAS ESTRICTAS:
- NO le digas que tome otro en su lugar. El cambio lo autoriza SOLO su médico o su farmacéutico.
- Preséntalo así: existen estas otras del mismo grupo, llévele esta lista a su médico o al farmacéutico para que ellos decidan.
- NO indiques dosis para las alternativas: 'dosis_referencia_mg' es un valor de catálogo, NO una pauta para esta persona. No la menciones.
- Si 'en_su_receta' es false, dile que ese medicamento no figura en su receta y que lo consulte antes de tomar nada.
- Recuérdale que mientras tanto NO debe suspender el resto de su tratamiento.

DATOS:
{alternativas}

Consulta: '{consulta}'
```

**Por qué así:** cambiar un medicamento por otro es una decisión clínica. El
sistema puede informar qué existe en el mismo grupo, pero la decisión queda en
el médico o el farmacéutico. La dosis de catálogo no es una pauta para la
persona, y nombrarla invita a tomarla como tal.

---

## 10. Variante solo-LLM de medicación (experimento, fuera del grafo)

- **Variables:**
  - `{nombre}`, `{edad}`, `{condiciones}`, `{alergias}` y `{funcion_renal}` del
    perfil;
  - `{receta}`: el JSON de sus prescripciones;
  - `{base}`: el JSON de los 62 medicamentos del catálogo;
  - `{consulta}`.
- **Uso:** solo en `pruebas/evaluar_medicacion_comparativa.py`, como línea de
  comparación frente a (8).

```text
Eres un gestor de medicación que le explica a una persona mayor qué tiene que tomar, en español claro y tratándola de usted.

Organiza su día por horarios a partir de su RECETA MÉDICA. Verifica contra la BASE DE DATOS que nada esté contraindicado por sus condiciones, que no haya ningún medicamento al que sea alérgica y que no se exceda el máximo de miligramos diarios, teniendo en cuenta que con función renal reducida ese máximo baja. Si encuentras algún problema, avísaselo.

PERSONA: {nombre}, {edad} años. Condiciones: {condiciones}. Alergias: {alergias}. Función renal: {funcion_renal}.

SU RECETA MÉDICA:
{receta}

BASE DE DATOS DE MEDICAMENTOS:
{base}

Consulta: '{consulta}'
```

**Por qué así:** la comparación tiene que ser justa. El modelo recibe
exactamente la misma información que las reglas, incluidas alergias, función
renal y topes; lo único que no recibe es el resultado ya calculado. Resultado:
empate, 0 errores en 24 casos en las dos variantes (bitácora 16 y 17). La ventaja de las
reglas no está en el acierto sino en que son **auditables**.

---

## 11. Integrador medicamento × comida (`integrador`)

- **Variables:**
  - `{consulta}`;
  - `{cruce}`: el resultado del cruce determinista
    (`interacciones/reglas.cruzar`), convertido a texto con los medicamentos
    vigentes, las recetas revisadas y, para cada interacción, el medicamento, la
    categoría, la severidad, los alimentos (y si son parte del plato,
    acompañamiento u opcionales), el porqué y qué hacer.
- **Salida:** máximo 120 palabras. El aviso de prototipo se agrega en código.
- **Cuándo NO corre:** si el cruce no es evaluable (sin perfil, sin receta
  médica, receta vencida o plato no encontrado). Ahí responde texto fijo.

```text
Eres un asistente para adultos mayores. Hablas español sencillo, de usted, con frases cortas.

La persona preguntó: '{consulta}'

{cruce}

Responde a la persona:
1. Empieza con la respuesta directa en una frase (puede comerlo, con cuidado, o mejor evitarlo).
2. Explica cada interacción del cruce con palabras simples y di qué hacer. Si el alimento es acompañamiento u opcional, dilo: puede comer el plato sin ese ingrediente.
3. NO agregues interacciones, medicamentos ni alimentos que no estén en el cruce. NO des dosis ni cambies la medicación.
4. Si el cruce no encontró nada, dilo así, sin inventar precauciones.
Máximo 120 palabras.
```

Así llega `{cruce}` cuando hay interacciones (formato de
`_describir_cruce`):

```text
Medicamentos que toma (según su receta vigente): Metformina, Losartan, Aspirina.
Recetas de cocina revisadas: receta5.jpeg.
Resultado del cruce (estas y SOLO estas):
- Metformina con alcohol — severidad: evitar. Alimentos: ron (es parte del plato, en …). Por qué: … Qué hacer: …
```

**Por qué así:**
- **El veredicto sale del código.** El integrador no decide si hay interacción;
  la regla ya lo hizo, y el criterio de éxito de T6 compara las interacciones
  del **cruce**, no las del texto.
- **"Estas y SOLO estas"** y la regla 3 apuntan al fallo esperable: que el
  modelo agregue de memoria una interacción conocida que no está en el catálogo.
- **La regla 4** evita el fallo inverso: sobrealertar con precauciones genéricas
  cuando el cruce no encontró nada. El caso de prueba es Rosa con el bolón:
  toma losartán, pero su función renal es normal.
- **La respuesta directa primero** porque la persona preguntó "¿puedo?".
- **120 palabras**, para que se pueda leer en voz alta en la interfaz.
