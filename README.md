# Sistema Multiagente de Accesibilidad para Adultos Mayores

Clasificación de intención en español ecuatoriano coloquial, aplicada al ruteo de un sistema multiagente.

Mini tesis de maestría. Manuel Pillapa. 2026.

---

## 1. Contexto del problema

Los adultos mayores tienen dificultad para usar asistentes digitales. Hablan de forma coloquial con modismos ecuatorianos y rara vez usan las palabras clave que esperan los sistemas tradicionales. Una persona no dice "activar protocolo de emergencia" sino "ayúdame que me caí". No dice "consultar inventario de medicación" sino "¿ya me toca la pastillita del corazón?".

Un asistente útil para esta población tiene que entender la intención detrás de la frase y no las palabras exactas. Si confunde una emergencia con una consulta trivial las consecuencias para el usuario pueden ser graves.

## 2. Qué se va a hacer

Construí un clasificador de intención que recibe una frase hablada y la asigna a una de cinco categorías. Ese clasificador actúa como el router de un sistema multiagente: según la intención detectada la consulta se envía al agente especialista correcto.

Las cinco intenciones:

| Intención | Qué cubre |
|---|---|
| `MEDICATION_HEALTH` | Medicación, dosis, horarios, síntomas |
| `RECIPE_MULTIMEDIA` | Recetas y guías de cocina paso a paso |
| `FAMILY_COMMUNICATION` | Avisos y mensajes a familiares |
| `EMERGENCY` | Caídas, dolor intenso, peligro en el entorno |
| `SMALL_TALK` | Saludos y conversación trivial |

## 3. Resumen del trabajo

En este proyecto respondo dos preguntas de investigación:

1. **¿Qué representación de texto clasifica mejor el lenguaje coloquial?** Comparo TF-IDF y embeddings multilingües MiniLM manteniendo la misma cabeza Logistic Regression. Al variar solo el representador el resultado aísla el aporte de la representación.

2. **¿Cómo conviene rutear las consultas en el sistema multiagente?** Comparo tres estrategias: un clasificador dedicado, el razonamiento del modelo de lenguaje y un híbrido donde el modelo de lenguaje usa el clasificador como herramienta.

Todo el flujo de machine learning está en un solo notebook: carga de datos, análisis exploratorio, extracción de características, entrenamiento, optimización de hiperparámetros, evaluación estadística con ROC y AUC, y análisis de errores.

---

## Resultados principales

### Comparación de representaciones

| Modelo | Accuracy | F1 macro | AUC macro |
|---|---|---|---|
| TF-IDF + Logistic Regression | 85.5% | 0.86 | 0.97 |
| Embeddings MiniLM + Logistic Regression | **91.6%** | **0.92** | **0.99** |

Los embeddings ganan en todas las métricas. La mayor mejora está en emergencia y medicación que son las clases de mayor consecuencia, porque el modelo capta la urgencia implícita en frases que no tienen palabras clave explícitas.

### Comparación de estrategias de ruteo

| Estrategia | Cómo decide | Accuracy de ruteo | Latencia media |
|---|---|---|---|
| A1: clasificador dedicado | El clasificador decide y Python despacha | **91.6%** medido exacto | **3.39 s** |
| B: razonamiento del modelo de lenguaje | El modelo razona y delega solo | ~72% estimado | 4.48 s |
| A2: híbrido | El modelo usa el clasificador como herramienta | ~95% en muestra | 11.89 s |

La estrategia A1 es la recomendada para producción: es la más rápida, la única medible de forma exacta, determinista y auditable.

---

## Estructura del proyecto

```
accessibility-agents/
├── proyecto_final.ipynb          Notebook completo del proyecto, listo para Colab
├── data/
│   └── intents/
│       ├── dataset.csv           415 frases etiquetadas, 83 por intención
│       └── intents_schema.md     Definición y ejemplos de cada intención
├── src/
│   ├── clasificador.py           Carga del modelo entrenado y predicción
│   ├── agentes.py                Agentes especialistas y las tres estrategias de ruteo
│   └── evaluacion.py             Evaluación automática de las tres estrategias
├── notebooks/
│   ├── 01_clasificador.ipynb     Entrenamiento y comparación de modelos
│   ├── 02_sistema_agentes.ipynb  Demostración del sistema multiagente
│   └── 03_comparativa.ipynb      Análisis de resultados de la evaluación
├── modelo.joblib                 Mejor modelo entrenado, listo para usar
├── resultados_evaluacion.csv     Resultados crudos de la evaluación
├── requirements.txt              Dependencias
└── README.md
```

El entregable principal es `proyecto_final.ipynb`, que reúne todo en un solo documento. Los notebooks de la carpeta `notebooks/` son el desarrollo por fases.

---

## Cómo ejecutar el proyecto

### Opción 1: Google Colab

1. Abre `proyecto_final.ipynb` en [Google Colab](https://colab.research.google.com).
2. Ejecuta la primera celda de instalación de dependencias.
3. Ejecuta el resto de celdas en orden.

La carga de datos funciona automáticamente leyendo los archivos desde este repositorio. La sección del sistema multiagente no corre en Colab porque requiere un modelo de lenguaje local. El resto del proyecto sí corre completo.

### Opción 2: Entorno local

```bash
# 1. Clonar el repositorio
git clone https://github.com/mpillapa/accessibility-agents.git
cd accessibility-agents

# 2. Crear y activar un entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux o Mac

# 3. Instalar dependencias
pip install -r requirements.txt
```

Para correr el sistema multiagente en local hace falta [Ollama](https://ollama.com) con el modelo `qwen2.5:7b`:

```bash
ollama pull qwen2.5:7b
```

Luego se puede ejecutar la evaluación completa:

```bash
python src/evaluacion.py
```

---

## Metodología

El proyecto sigue el flujo estándar de un problema de machine learning supervisado de clasificación de texto:

1. Definición del problema y de las clases objetivo
2. Carga y exploración de datos, con nube de palabras y frecuencias
3. División estratificada en entrenamiento y prueba
4. Extracción de características con dos representaciones distintas
5. Entrenamiento de un modelo base y un modelo mejorado con la misma cabeza
6. Optimización de hiperparámetros con validación cruzada y curva de aprendizaje
7. Evaluación estadística con accuracy, F1 macro, curva ROC y AUC
8. Análisis de errores del mejor modelo
9. Persistencia del modelo entrenado
10. Aplicación en el sistema multiagente y comparación de estrategias de ruteo

---

## Stack técnico

- Clasificación: scikit-learn, sentence-transformers
- Embeddings: paraphrase-multilingual-MiniLM-L12-v2
- Sistema multiagente: CrewAI
- Modelo de lenguaje local: qwen2.5:7b servido por Ollama
- Visualización: matplotlib, seaborn, wordcloud
