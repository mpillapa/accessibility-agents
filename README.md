# Multi-agente de Accesibilidad para Adultos Mayores

Mini-tesis de maestría · Manuel Pillapa · 2026

## Objetivo académico

Demostrar el patrón **Supervisor/Router con clasificación de intención** aplicado
a un sistema multi-agente de accesibilidad en español ecuatoriano coloquial.

El núcleo del trabajo es comparar dos estrategias de enrutamiento:

| Estrategia | Mecanismo |
|---|---|
| A | Clasificador dedicado (embeddings multilingües + Logistic Regression) |
| B | Enrutamiento delegado al LLM del Orchestrator (CrewAI) |

Métricas de comparación: accuracy, latencia, consumo de tokens.

## Agentes del sistema

| Agente | Estado | Función |
|---|---|---|
| Orchestrator | Fase 3 | Clasifica la intención y delega al agente correcto |
| Medication & Health | Fase 3 — funcional | Recordatorios de medicación y consultas de salud |
| Recipe & Multimedia | Fase 3 — funcional | Lectura de recetas paso a paso |
| Family Bridge | Fase 3 — stub | Comunicación asíncrona con la familia |
| Emergency Sentinel | Fase 3 — stub | Detección y alerta de emergencias |

## Intenciones clasificadas

`MEDICATION_HEALTH` · `RECIPE_MULTIMEDIA` · `FAMILY_COMMUNICATION` · `EMERGENCY` · `SMALL_TALK`

Ver [data/intents/intents_schema.md](data/intents/intents_schema.md) para la definición
completa de cada intención con ejemplos en español ecuatoriano coloquial.

## Estructura del proyecto

```
accessibility-agents/
├── data/
│   └── intents/
│       ├── intents_schema.md   # Definición y ejemplos de las 5 intenciones
│       └── dataset.csv         # Dataset etiquetado (415 frases, 83 por intención)
├── src/
│   ├── classifier/             # Clasificador de intención (Fase 3)
│   ├── agents/                 # Agentes especialistas (Fase 3)
│   └── orchestrator/           # Orchestrator CrewAI (Fase 3)
├── notebooks/                  # Análisis y experimentos (Fase 3)
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

## Cómo correr el proyecto (Fase 1)

```bash
# 1. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Copiar variables de entorno
copy .env.example .env
```

## Fases del proyecto

- **Fase 1** ✅ Setup + schema + dataset semilla
- **Fase 2** ✅ Expansión del dataset (415 frases, 83 por intención)
- **Fase 3** — En curso
