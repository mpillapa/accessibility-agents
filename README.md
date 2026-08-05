# Sistema Multiagente de Accesibilidad para Adultos Mayores

Comparación de frameworks de orquestación multiagente (CrewAI vs LangGraph) aplicada a un asistente de accesibilidad para adultos mayores.

Mini tesis de maestría. Manuel Pillapa. 2026.

---

## 1. Contexto del problema

Los adultos mayores tienen dificultad para usar asistentes digitales. Hablan de forma coloquial con modismos ecuatorianos y rara vez usan las palabras clave que esperan los sistemas tradicionales. Una persona no dice "activar protocolo de emergencia" sino "ayúdame que me caí". No dice "consultar inventario de medicación" sino "¿ya me toca la pastillita del corazón?".

Un asistente útil para esta población tiene que entender la intención detrás de la frase y no las palabras exactas. Si confunde una emergencia con una consulta trivial las consecuencias para el usuario pueden ser graves.

## 2. Qué se va a hacer

Un Orchestrator identifica la intención de la consulta y la rutea al agente especialista correcto. Las cinco intenciones:

| Intención | Qué cubre |
|---|---|
| `MEDICATION_HEALTH` | Medicación, dosis, horarios, síntomas |
| `RECIPE_MULTIMEDIA` | Recetas y guías de cocina paso a paso |
| `FAMILY_COMMUNICATION` | Avisos y mensajes a familiares |
| `EMERGENCY` | Caídas, dolor intenso, peligro en el entorno |
| `SMALL_TALK` | Saludos y conversación trivial |

En esta fase el ruteo lo decide directamente el LLM (no hay clasificador ML dedicado — esa línea de trabajo perteneció a otro proyecto y se descartó aquí).

## 3. Resumen del trabajo

La pregunta de investigación de esta fase: **¿cómo se compara CrewAI contra LangGraph para orquestar el mismo sistema multiagente?**

Mismo alcance de agentes implementado en ambos frameworks, para que la comparación no mezcle variables:

- Orchestrator (decide la intención y rutea)
- Especialista en Medicación y Salud (real)
- Especialista en Recetas y Multimedia (real)
- Family Bridge (stub — sin integración real con ninguna app familiar)
- Emergency Sentinel (stub — sin lógica real de emergencia)

Mismo LLM en los dos: `google/gemma-4-12B-it`, servido por vLLM (OpenAI-compatible) en el servidor H200 de la universidad (requiere VPN institucional). Antes del 2026-07-23 se usaba `llama3.1:8b` vía Ollama; se migró porque el equipo de tutores pidió estandarizar todo el acceso a LLMs sobre el protocolo OpenAI-compatible (ver `Acceso a los Endpoints de LLMs.pdf`, correo del 2026-07-23). El cambio de proveedor implica también cambio de modelo — la comparación de frameworks previa a esa fecha corrió sobre `llama3.1:8b`, no sobre `gemma-4-12B-it`. El model id debe escribirse EXACTO al que devuelve `GET /v1/models` del servidor; el nombre del PDF (`gemma-4-31B`) no coincide con el modelo realmente desplegado.

- **CrewAI** (`crewai/`): el Orchestrator es un `Agent` con `allow_delegation=True` — decide y delega a otro agente mediante tool calling interno del framework.
- **LangGraph** (`langgraph/`): el Orchestrator es un nodo que escribe la intención en un estado compartido (`EstadoConversacion`), y un edge condicional rutea explícitamente al nodo especialista correspondiente.

Ver `langgraph/README.md` para el detalle de la implementación, el flujo de datos entre nodos y un hallazgo relevante sobre salida estructurada con Pydantic que no fue confiable con los modelos disponibles (documentado con `llama3.1:8b`/Ollama — pendiente repetir la prueba con `gemma-4-12B-it`/vLLM).

**Verificado en VPN institucional (2026-07-23)**: la migración completa (incluido el RAG de recetas) se ejecutó end-to-end contra los servidores de la universidad. Confirmado: `gemma-4-12B-it` soporta tool calling (por tanto `allow_delegation=True` de CrewAI funciona y el agente de recetas invoca la tool de RAG), los embeddings de `bge-m3` responden (1024-dim) y `GLM-OCR` transcribe imágenes con el formato `image_url` base64. El notebook `notebooks/comparativa.ipynb` está ejecutado con salidas reales.

## 3.1 RAG real para el agente de recetas

El especialista en recetas dejó de ser un rol sin datos: ahora consulta un recetario real vía RAG.

- **Embeddings**: `BGE-M3`, servido por vLLM (OpenAI-compatible, puerto 12556).
- **OCR/visión**: `glm-ocr` (OpenAI-compatible, puerto 12560), para fotos de recetas manuscritas.
- **Vector store**: ChromaDB local, persistida en `rag/chroma_db/` (no versionada — se regenera con `rag/ingesta.py`).
- **Datos**: `rag/recetas_data/` — 2 recetas en texto plano y 1 imagen **sintética** (texto tipeado renderizado como imagen con `rag/recetas_data/generar_imagen_mock.py`, NO una foto real de una receta manuscrita) para poder probar la ruta imagen → OCR → embeddings → ChromaDB de punta a punta. Cuando exista una foto real, reemplaza esa imagen y vuelve a correr la ingesta.

Uso:
```bash
python rag/ingesta.py   # (re)genera rag/chroma_db/ a partir de rag/recetas_data/
```

En CrewAI el RAG se expone como tool (`buscar_en_recetario`) que el propio agente decide invocar. En LangGraph el nodo `nodo_recetas` hace la recuperación explícitamente antes de llamar al LLM, siguiendo el mismo estilo de flujo de datos explícito del resto del grafo.

---

## Estructura del proyecto

```
accessibility-agents/
├── crewai/
│   ├── agentes.py                Agentes (Orchestrator + especialistas + stubs), procesar_consulta(), clasificar_consulta()
│   └── demo.py                   Demo en vivo (verbose=True: muestra razonamiento y delegación)
├── langgraph/
│   ├── estado.py                 State del grafo
│   ├── agentes.py                Nodos del grafo (mismos roles/prompts que crewai/agentes.py)
│   ├── grafo.py                  StateGraph + edges condicionales; traza_por_nodo() y clasificar_consulta()
│   ├── demo.py                   Demo en vivo (traza nodo por nodo vía stream())
│   ├── visualizar.py             Genera el diagrama del grafo (grafo.png)
│   └── README.md                 Alcance, flujo/cruce de información y hallazgos técnicos
├── rag/
│   ├── config.py                 Endpoints/modelos vLLM para embeddings y OCR (desde .env)
│   ├── embeddings.py              Llama a BGE-M3 (OpenAI-compatible)
│   ├── ocr.py                    Llama a glm-ocr para imágenes (OpenAI-compatible, formato "vision")
│   ├── ingesta.py                 Lee rag/recetas_data/, OCR+embeddings, guarda en ChromaDB
│   ├── buscar.py                  Búsqueda semántica sobre el recetario ya ingerido
│   └── recetas_data/              2 recetas en texto + 1 imagen sintética (ver generar_imagen_mock.py)
├── notebooks/
│   └── comparativa.ipynb         Cruce de información entre agentes (CrewAI vs LangGraph) + accuracy de ruteo
├── dataset.csv                    415 frases etiquetadas (83 × 5 intenciones), base simulada para evaluar ruteo
├── requirements.txt              Dependencias
├── .env.example                   Config de endpoints vLLM (chat, embeddings, OCR)
├── .gitignore
├── LICENSE
└── README.md
```

> **`notebooks/comparativa.ipynb`** es el artefacto central de esta entrega: para una frase, muestra lado a lado el cruce de información entre agentes en ambos frameworks (estado compartido inspeccionable en LangGraph vs delegación en CrewAI, con la decisión del Orchestrator como objeto Pydantic serializado a JSON), y calcula la accuracy de ruteo de cada framework contra `dataset.csv`. Reemplaza a los notebooks de la fase del clasificador ML (`01/02/03`, retirados; git conserva el historial). Requiere VPN institucional para ejecutarse — no se corrió en el entorno de desarrollo.

---

## Cómo ejecutar el proyecto

Requiere VPN institucional activa (el LLM y el RAG corren en servidores remotos de la universidad, no local).

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

# 4. Copiar la config de endpoints (los defaults ya apuntan a los servidores de la universidad)
cp .env.example .env

# 5. Ingerir el recetario (OCR + embeddings -> ChromaDB local), una sola vez
python rag/ingesta.py
```

Demo de CrewAI:
```bash
python crewai/demo.py                 # frases de ejemplo
python crewai/demo.py "tu frase aquí" # una frase propia
```

Demo de LangGraph (con traza nodo por nodo):
```bash
python langgraph/demo.py
python langgraph/demo.py "tu frase aquí"
```

Ver el diagrama del grafo de LangGraph:
```bash
python langgraph/visualizar.py
```

Comparativa (cruce de información entre agentes + accuracy de ruteo):
```bash
jupyter notebook notebooks/comparativa.ipynb
```

---

## Stack técnico

- Orquestación: CrewAI y LangGraph (comparación de frameworks)
- Salida estructurada: Pydantic
- Modelo de lenguaje: `google/gemma-4-12B-it`, servido por vLLM (OpenAI-compatible) en el servidor H200 de la universidad
- RAG de recetas: `BAAI/bge-m3` (embeddings) + `zai-org/GLM-OCR` (OCR/visión para recetas manuscritas) vía vLLM, ChromaDB local
- Visualización del grafo: Mermaid (vía LangGraph) y `grandalf` (ASCII local)

## Reglas y supuestos de esta migración (2026-07-23)

- **Regla de negocio**: los tutores pidieron estandarizar el acceso a LLMs sobre el protocolo OpenAI-compatible (vLLM), en vez de Ollama. Fuente: PDF "Acceso a los Endpoints de LLMs" + correo del 2026-07-23. No hay otra regla de negocio detrás del cambio de proveedor.
- **Model ids**: los nombres "amigables" del PDF (`gemma-4-31B`, `BGE-M3`, `glm-ocr`) NO coinciden con los model ids reales que exige la API. Los verificados contra `GET /v1/models` son `google/gemma-4-12B-it` (chat), `BAAI/bge-m3` (embeddings) y `zai-org/GLM-OCR` (OCR).
- **Verificado (2026-07-23)**: `gemma-4-12B-it` soporta tool calling vía vLLM → `allow_delegation=True` de CrewAI funciona; `bge-m3` devuelve embeddings de 1024-dim; `GLM-OCR` transcribe imágenes con el patrón "vision" de OpenAI (`image_url` + data URI base64). El notebook comparativo corre end-to-end.
- **Dato simulado**: `rag/recetas_data/` tiene 2 recetas en texto y 1 imagen sintética generada con Pillow (texto tipeado renderizado como imagen), no una foto real de una receta manuscrita. Sirve para probar el pipeline completo (imagen → OCR → embeddings → ChromaDB), no como contenido real del recetario. `dataset.csv` (415 frases) también es una base simulada.
- **Entorno**: el `.venv` estaba creado para Python 3.11 (intérprete ya inexistente en el host, ahora 3.12); se reconstruyó con 3.12 y se reinstaló `requirements.txt`. Si se clona en otra máquina, recrear el venv con la versión de Python disponible.
- **Seguridad**: el endpoint vLLM no requiere autenticación real (según el PDF); `VLLM_API_KEY` es un valor cualquiera, no un secreto. El acceso depende del aislamiento de la VPN institucional, no de esta clave.
- **Pendiente**: repetir con `gemma-4-12B-it` el hallazgo de salida estructurada con Pydantic documentado en `langgraph/README.md` (se probó con `llama3.1:8b`/`qwen3.6`, no con este modelo); ampliar la muestra de accuracy de 50 al dataset completo si se quiere el número definitivo.
