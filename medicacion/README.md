# Agente de medicación

Le dice a una persona mayor qué medicación le tocó hoy, a qué hora y cómo
tomarla, a partir de **la receta que le dio su médico**.

> **Datos ficticios.** Ni las personas, ni las recetas, ni los valores del
> vademécum existen. Los principios activos son reales para que las categorías se
> entiendan, pero dosis, contraindicaciones y horarios son **ilustrativos** y no
> están validados clínicamente. Ninguna respuesta de este módulo sirve para tomar
> una decisión sobre medicación real.

---

## La idea en una línea

**El médico decide; el sistema lee, organiza, explica y verifica.**

El sistema no elige medicamentos. Esa es una decisión clínica y este prototipo no
está en posición de tomarla.

## Por qué, si antes hacía lo contrario

La primera versión filtraba el vademécum por las condiciones de la persona,
descartaba lo contraindicado y presentaba el resultado como su tratamiento. Para
Carmen —84 años, hipertensa— respondía:

```
Toma 7 pastillas, una de cada medicamento:
Enalapril · Losartán · Amlodipino · Valsartán · Furosemida · Bisoprolol · Carvedilol
```

Siete antihipertensivos juntos: tres del mismo eje, dos betabloqueantes. Nadie
toma eso.

**Ninguna regla se violó**: cada uno estaba indicado, ninguno contraindicado,
ninguno pasado de dosis. El error estaba en el planteamiento —un catálogo
filtrado por condición devuelve **opciones elegibles**, no un **régimen**— y por
eso **falló idéntico con las reglas en código y con el LLM decidiendo**.

Detalle completo: sección 16 de `BITACORA_HALLAZGOS.md`.

---

## Nombres

En este repositorio **`receta` significa receta de cocina** (`rag/recetas_data`,
el intent `RECIPE_MULTIMEDIA`). La receta médica se llama **prescripción** en
todo el código. No mezclar: son dos agentes distintos.

## Capas

| Archivo | Qué hace |
|---|---|
| `datos.py` | Carga los tres JSON y busca. **No decide nada.** |
| `prescripciones.py` | **Las reglas de negocio**: plan del día, verificación, alternativas. Código determinista. |
| `reglas.py` | Criterios reutilizables (exclusión, tope ajustado) + el planteamiento original, conservado como registro. |
| `agente.py` | Las dos variantes que se comparan y los prompts. |

El nodo del grafo (`orquestacion_langgraph/agentes.nodo_medicacion`) solo
conversa: delega en `agente.responder()`.

## Datos

| Archivo | Contenido |
|---|---|
| `datos/medicamentos.json` | 62 medicamentos, 22 categorías. **Material de consulta**, no un menú del que elegir. |
| `datos/perfiles.json` | 6 personas con condiciones, alergias y función renal. |
| `datos/prescripciones.json` | 6 recetas. **La fuente de verdad.** |

Cada receta es un caso de prueba deliberado:

| Perfil | Qué prueba |
|---|---|
| Rosa | Receta limpia: que no inventen un problema. |
| Manuel | Dos recetas a la vez **y** una alergia a penicilina en la aguda. |
| Carmen | 80 mg indicados contra un tope de 40 por función renal. |
| Jorge | Antiinflamatorio con úlcera: error de prescripción clásico. |
| Elena | Receta limpia, sin benzodiacepinas. |
| Luis | **Sin receta**: que no le inventen un tratamiento. |

---

## La regla que gobierna el módulo: marcar, nunca quitar

Si la verificación encuentra un problema, la indicación **sigue en el plan**,
marcada. El sistema no la borra.

Suspender un tratamiento es tan clínico como recetarlo, y una persona mayor que
deja su antihipertensivo porque una aplicación se lo ocultó corre más riesgo que
una advertida. Es el mismo criterio del guardrail de voz: ante duda, avisar en
vez de decidir.

Cada aviso trae **qué hacer**, y no todos piden lo mismo:

| Aviso | Acción |
|---|---|
| `alergia` | NO tomarlo, llamar hoy mismo al médico |
| `contraindicacion` | NO tomarlo sin hablar antes con el médico |
| `excede_tope` | Consultar antes de la próxima toma |
| `medicamento_desconocido` | Confirmar con médico o farmacéutico |
| `datos_inconsistentes` | Confirmar cuántas tomas son |
| `prescripcion_vencida` | Renovar la receta, **sin suspender** el tratamiento |

## "Se me acabó la pastilla"

El sistema **no sustituye**. Devuelve qué existe en la misma categoría, sin pauta
de toma —el campo se llama `dosis_referencia_mg` para que nadie lo lea como una
indicación— y deriva al médico o al farmacéutico.

La equivalencia por categoría es gruesa: mismo grupo no significa misma potencia
ni mismo perfil de efectos.

---

## Uso

```python
from medicacion.agente import responder, VARIANTE_REGLAS, VARIANTE_LLM

responder("¿Qué pastillas me toca hoy?", "carmen", VARIANTE_REGLAS)
responder("Se me acabó el paracetamol", "manuel", VARIANTE_REGLAS)
```

Sin LLM, solo los datos:

```python
from medicacion.prescripciones import plan_diario, alternativas_para

plan_diario("carmen")              # tomas por hora + avisos
alternativas_para("Naproxeno", "jorge")
```

## Pruebas

```bash
python -m pruebas.prueba_prescripciones      # 26, sin VPN ni LLM
python -m pruebas.prueba_reglas_medicacion   # 11, sin VPN ni LLM
```

## Experimento comparativo (requiere VPN)

```bash
python -m pruebas.evaluar_medicacion_comparativa --repeticiones 3
python -m pruebas.evaluar_medicacion_comparativa --ver carmen   # texto completo
```

Guarda todas las respuestas crudas y sus medidas en
`resultados/medicacion_comparativa.json`.

**Resultado actual: reglas 0/24 · LLM 0/24.** Las métricas automáticas **no**
distinguen las dos variantes, y la latencia es equivalente (11.2 s contra 12.0 s
de mediana). Lo que sí las separa es el tipo de garantía: las 26 pruebas de la
variante de reglas corren en milisegundos sin LLM y siguen valiendo mañana; los
24 aciertos del LLM son evidencia sobre un modelo que en este servidor **rotó
cuatro veces en trece días**.

Correr con `--repeticiones 1` no sirve: la salida del modelo no es determinista y
el primer experimento reportó un fallo que la siguiente llamada no reprodujo.

---

## Limitaciones declaradas

- **No modela interacciones entre fármacos** ni duplicidad terapéutica. Una
  receta que pase las cuatro validaciones **no** está clínicamente validada.
- El ajuste renal es un factor único (0.5). En la práctica depende del fármaco y
  del grado de insuficiencia.
- Sin perfil, el nodo del grafo cae al de `PERFIL_ACTIVO` para que la demo
  funcione sin login. En producción saldría de la autenticación, y sin sesión no
  habría respuesta.
- La detección de "se me acabó" es una lista de frases, no clasificación de
  intención: no cubre las formas que no estén en la lista.
- Si la persona dice que le falta algo **sin nombrar el medicamento**, la
  consulta cae en el plan del día en vez de preguntar cuál.
