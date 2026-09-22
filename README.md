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

- **CrewAI** (`orquestacion_crewai/`): el Orchestrator es un `Agent` con `allow_delegation=True` — decide y delega a otro agente mediante tool calling interno del framework.
- **LangGraph** (`orquestacion_langgraph/`): el Orchestrator es un nodo que escribe la intención en un estado compartido (`EstadoConversacion`), y un edge condicional rutea explícitamente al nodo especialista correspondiente.

> Las carpetas se llaman `orquestacion_crewai/` y `orquestacion_langgraph/`, y no `crewai/` / `langgraph/`, a propósito: con el nombre del paquete pip Python resolvía un *namespace package* que mezclaba la carpeta del repo con la librería instalada. Funcionaba por casualidad y obligaba a manipular `sys.path` en cada módulo y en el notebook. Ver "Reglas y supuestos" al final.

Ver `orquestacion_langgraph/README.md` para el detalle de la implementación, el flujo de datos entre nodos y un hallazgo relevante sobre salida estructurada con Pydantic que no fue confiable con los modelos disponibles (documentado con `llama3.1:8b`/Ollama — pendiente repetir la prueba con `gemma-4-12B-it`/vLLM).

**Verificado en VPN institucional (2026-07-23)**: la migración completa (incluido el RAG de recetas) se ejecutó end-to-end contra los servidores de la universidad. Confirmado: `gemma-4-12B-it` soporta tool calling (por tanto `allow_delegation=True` de CrewAI funciona y el agente de recetas invoca la tool de RAG), los embeddings de `bge-m3` responden (1024-dim) y `GLM-OCR` transcribe imágenes con el formato `image_url` base64. El notebook `notebooks/comparativa.ipynb` está ejecutado con salidas reales.

## 3.1 RAG real para el agente de recetas

El especialista en recetas dejó de ser un rol sin datos: ahora consulta un recetario real vía RAG.

- **Embeddings**: `BGE-M3`, servido por vLLM (OpenAI-compatible, puerto 12556).
- **OCR/visión**: `glm-ocr` (OpenAI-compatible, puerto 12560), para fotos de recetas manuscritas.
- **Vector store**: ChromaDB local, persistida en `rag/chroma_db/` (no versionada — se regenera con `python -m rag.ingesta`).
- **Datos**: `rag/recetas_data/` — 2 recetas en texto plano y 1 imagen **sintética** (texto tipeado renderizado como imagen con `rag/recetas_data/generar_imagen_mock.py`, NO una foto real de una receta manuscrita) para poder probar la ruta imagen → OCR → embeddings → ChromaDB de punta a punta. Cuando exista una foto real, reemplaza esa imagen y vuelve a correr la ingesta.

Uso:
```bash
python -m rag.ingesta   # (re)genera rag/chroma_db/ a partir de rag/recetas_data/
```

La ingesta **recrea la colección** en cada corrida. Antes usaba `upsert`, lo que
dejaba fragmentos huérfanos: si un archivo se renombraba o se borraba, o si
cambiaba el troceado, los ids viejos seguían en el índice y el RAG los seguía
recuperando aunque ya no correspondieran a ningún archivo del disco.

### Control de calidad del OCR (2026-08-19)

`rag/calidad.py` valida el texto del OCR **antes** de indexarlo, y la ingesta
reporta al final qué archivos rechazó y por qué.

Existe por un caso concreto: al procesar `2 recetas mas.jpg` — una doble página
de un libro de cocina, foto nítida y bien iluminada — GLM-OCR leyó bien las dos
primeras líneas y después entró en un bucle, repitiendo *"Sive el mantequilla
que se dore por ambio."* 888 veces hasta degenerar en texto sin sentido con
caracteres chinos. Resultado: 60.648 caracteres inventados que llegaron al
índice vectorial y pasaron a ser el **66% de todo el recetario**.

Lo relevante para la tesis: **no falló por calidad de imagen sino por
complejidad de layout** (doble página, varias columnas, tipografía pequeña).
Otra doble página del mismo recetario, `2 recetas en 1.png`, se procesó sin
problema. Eso implica que evaluar OCR variando solo iluminación y ruido —el
enfoque intuitivo— deja fuera la variable que realmente lo rompió.

El detector usa dos señales, calibradas contra los 223 fragmentos legítimos que
había indexados:

| Señal | Legítimos | Caso degenerado | Umbral |
|---|---|---|---|
| Frecuencia del 5-grama más repetido | máx. 2 | 888 | 8 |
| Ratio de palabras únicas | mín. 0.32 | 0.04 | 0.20 |

Cero falsos positivos sobre esos 223 fragmentos. La frecuencia de n-grama es la
señal principal porque no depende de la longitud del texto; el ratio queda como
respaldo y solo se aplica a textos de 40 palabras o más.

Pruebas (no requieren VPN):
```bash
python -m pruebas.prueba_calidad_ingesta
```

En CrewAI el RAG se expone como tool (`buscar_en_recetario`) que el propio agente decide invocar. En LangGraph el nodo `nodo_recetas` delega en un subgrafo de RAG agéntico — ver abajo.

## 3.2 RAG agéntico (2026-08-19)

Pedido de los tutores en la reunión del 2026-08-19: que el RAG deje de ser un componente aislado y se integre al sistema multiagente. Al revisarlo, la integración ya existía (`nodo_recetas` llamaba a `buscar_receta()`); lo que faltaba era que la recuperación fuera una **decisión** y no un paso fijo.

`orquestacion_langgraph/rag_agentico/` es un subgrafo que:

1. decide si hace falta consultar el recetario (un agradecimiento no lo necesita),
2. recupera de ChromaDB,
3. juzga fragmento por fragmento si lo recuperado responde la consulta,
4. si no responde, **reformula la consulta y reintenta** (hasta `MAX_INTENTOS_RECUPERACION`),
5. si sigue sin encontrar, lo admite explícitamente en vez de improvisar una receta,
6. si sí encontró, **expande a la receta completa** antes de redactar: el filtro aprueba fragmentos sueltos, y una receta troceada por párrafos quedaría respondida con un paso aislado.

El caso que lo motiva es propio de esta población: un adulto mayor dice *"eso dulce del arrocito que hacía mi mamá"* y el recetario está indexado como *"arroz con leche"*. La búsqueda falla por vocabulario, no porque falte la receta — un pipeline lineal responde mal, este vuelve sobre sus pasos.

Detalle del flujo, reglas de negocio, costo en llamadas al LLM y qué de esto es material para el paper: `orquestacion_langgraph/rag_agentico/README.md`.

Pruebas del ciclo, sin VPN ni servidor (usan dobles en lugar del LLM):
```bash
python -m pruebas.prueba_ciclo_rag
```

---

## Estructura del proyecto

```
accessibility-agents/
├── infraestructura/               ACCESO A SERVICIOS EXTERNOS (aisla lo que no controlamos)
│   ├── modelos.py                Resuelve el id de modelo contra GET /v1/models; el .env es preferencia, no dato fijo
│   └── trazas.py                 Configura LangSmith; si no hay clave el sistema corre igual, sin trazas
├── interfaz/                      CAPA DE PRESENTACION (no decide nada)
│   ├── app.py                    Chat en Streamlit + recorrido por el grafo bajo cada respuesta
│   └── README.md                 Como ejecutarla y que muestra
├── orquestacion_crewai/
│   ├── agentes.py                Agentes (Orchestrator + especialistas + stubs), procesar_consulta(), clasificar_consulta()
│   └── demo.py                   Demo en vivo (verbose=True: muestra razonamiento y delegación)
├── orquestacion_langgraph/
│   ├── estado.py                 State del grafo principal
│   ├── llm.py                    Cliente LLM compartido (aparte, para evitar imports circulares)
│   ├── agentes.py                Nodos del grafo (mismos roles/prompts que orquestacion_crewai/agentes.py)
│   ├── grafo.py                  StateGraph + edges condicionales; traza_por_nodo(), procesar_consulta_en_vivo() y clasificar_consulta()
│   ├── rag_agentico/             SUBGRAFO de recuperación del especialista en recetas
│   │   ├── estado.py             EstadoRAG + reglas de negocio del ciclo (constantes con nombre)
│   │   ├── nodos.py              decidir / recuperar / evaluar / reformular / generar / sin_resultado
│   │   ├── subgrafo.py           StateGraph del ciclo + consultar_recetario()
│   │   └── README.md             Flujo, reglas, costo en llamadas al LLM y aportes al paper
│   ├── demo.py                   Demo en vivo (traza nodo por nodo vía stream())
│   ├── visualizar.py             Diagramas del grafo principal y del subgrafo de RAG
│   └── README.md                 Alcance, flujo/cruce de información y hallazgos técnicos
├── rag/                           CAPA DE ACCESO A DATOS (no decide nada, solo consulta)
│   ├── config.py                 Endpoints de embeddings y OCR; los ids se resuelven vía infraestructura/modelos.py
│   ├── embeddings.py              Llama a BGE-M3 en lotes que quepan en su ventana de contexto
│   ├── ocr.py                    Llama a glm-ocr para imágenes (OpenAI-compatible, formato "vision")
│   ├── calidad.py                 Detecta texto degenerado del OCR antes de indexarlo
│   ├── ingesta.py                 Lee rag/recetas_data/, OCR+calidad+troceado+embeddings, ChromaDB
│   ├── buscar.py                  Búsqueda semántica: buscar_receta() y buscar_receta_detallado()
│   └── recetas_data/              Recetario real: fotos de libros de cocina + 2 recetas en texto
├── pruebas/
│   ├── prueba_ciclo_rag.py       Los 4 caminos del subgrafo de RAG, con dobles (no requiere VPN)
│   └── prueba_calidad_ingesta.py Detección de OCR degenerado y troceado (no requiere VPN)
├── notebooks/
│   └── comparativa.ipynb         Cruce de información entre agentes + ciclo del RAG + accuracy de ruteo
├── dataset.csv                    415 frases etiquetadas (83 × 5 intenciones), base simulada para evaluar ruteo
├── requirements.txt              Dependencias
├── .env.example                   Config de endpoints (chat en vLLM; embeddings y OCR en Ollama)
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
#    No hace falta acertar el id del modelo: se resuelve contra el servidor al
#    arrancar (ver "Rotación de modelos" más abajo).

# 5. Ingerir el recetario (OCR + embeddings -> ChromaDB local), una sola vez
python -m rag.ingesta
```

> Todos los comandos se corren **desde la raíz del repo** y con `python -m`, porque
> el proyecto está organizado como paquetes de Python. Correr los archivos
> directamente (`python orquestacion_langgraph/demo.py`) falla con `ModuleNotFoundError`.

Interfaz web (chat con historial, accesible desde otra máquina de la red):
```bash
streamlit run interfaz/app.py                          # solo local
streamlit run interfaz/app.py --server.address 0.0.0.0 # accesible por IP, puerto 8501
```

Bajo cada respuesta muestra, plegado, el recorrido real por el grafo: a qué
agente ruteó el orquestador, por qué, y los ciclos del RAG agéntico. Ver
`interfaz/README.md`.

Demo de CrewAI:
```bash
python -m orquestacion_crewai.demo                 # frases de ejemplo
python -m orquestacion_crewai.demo "tu frase aquí" # una frase propia
```

Demo de LangGraph (con traza nodo por nodo):
```bash
python -m orquestacion_langgraph.demo
python -m orquestacion_langgraph.demo "tu frase aquí"
```

Ver el diagrama del grafo de LangGraph:
```bash
python -m orquestacion_langgraph.visualizar
```

Comparativa (cruce de información entre agentes + ciclo del RAG + accuracy de ruteo):
```bash
jupyter notebook notebooks/comparativa.ipynb
```

Pruebas (**no** requieren VPN: usan dobles en lugar del LLM, o son funciones puras):
```bash
python -m pruebas.prueba_ciclo_rag         # los caminos del subgrafo de RAG
python -m pruebas.prueba_calidad_ingesta   # deteccion de OCR degenerado y troceado
```

Evaluación del comportamiento real (**sí** requiere VPN y el recetario ingerido).
Guardar el `--json` antes de cambiar prompts o estrategia de recuperación, y
volver a correrlo después, permite comparar el antes y el después:
```bash
python -m pruebas.evaluar_rag_real --json antes.json
```

---

## Rotación de modelos en el servidor (2026-09-21)

Los endpoints de la Universidad cambian el modelo servido sin aviso y sin
cambiar el puerto. Registrado sobre el puerto 12559:

| Desde | Modelo | Cómo terminó |
|---|---|---|
| — | `google/gemma-4-12B-it` | detenido 2026-09-08, contenedor **eliminado** |
| 2026-09-09 | `zai-org/GLM-5.3-Flash` (FP16) | contenedor eliminado |
| 2026-09-21 17:57 | `canada-quant/GLM-5.3-Flash-W4A16-MTP` | SIGKILL a las 20:15 |
| 2026-09-21 20:24 | `zai-org/GLM-5.3-Flash` (FP16) | vigente |

Cuatro rotaciones en trece días, dos el mismo día; una de ellas ocurrió
**durante** una corrida de `pruebas.evaluar_rag_real`. Los endpoints de
embeddings (12556) y OCR (12560) no existen desde el 2026-09-09: se reemplazaron
por Ollama (`bge-m3:latest` y `qwen2.5vl:7b`).

La API OpenAI valida el campo `model` contra el id exacto del contenedor, así
que con el id fijado a mano en el `.env` **cada rotación devuelve 404 y tumba el
sistema**. `infraestructura/modelos.py` lo resuelve contra `GET /v1/models` al
arrancar; el `.env` queda como preferencia y respaldo sin VPN.

No sustituye a ciegas: si el endpoint sirve varios modelos y ninguno coincide
(caso Ollama, 14 modelos en un puerto), avisa y deja fallar, porque reemplazar
el modelo de embeddings por uno de chat cambiaría la dimensión del vector y
corrompería el índice de ChromaDB en silencio.

**Esto evita la caída, no restaura la reproducibilidad.** Un modelo distinto da
salidas distintas: `describir_resolucion()` deja registro de con qué modelo se
respondió, y toda medición debe anotarlo.

---

## Stack técnico

- Orquestación: CrewAI y LangGraph (comparación de frameworks)
- Salida estructurada: Pydantic
- Modelo de lenguaje: `google/gemma-4-12B-it`, servido por vLLM (OpenAI-compatible) en el servidor H200 de la universidad
- RAG de recetas: `BAAI/bge-m3` (embeddings) + `zai-org/GLM-OCR` (OCR/visión para recetas manuscritas) vía vLLM, ChromaDB local
- Visualización del grafo: Mermaid (vía LangGraph) y `grandalf` (ASCII local)

## Reestructuración del repo (2026-08-19)

Trabajo de base previo a incorporar los pendientes de la reunión con los tutores (RAG agéntico, Whisper, pruebas de OCR degradado). No cambia ningún comportamiento del sistema: mismos agentes, mismos prompts, mismo modelo.

- **Colisión de nombres resuelta**: las carpetas `crewai/` y `langgraph/` se llamaban igual que los paquetes de PyPI. Python resolvía un *namespace package* que fusionaba ambas rutas — funcionaba por casualidad y se habría roto al agregar un `__init__.py` o un módulo con nombre coincidente. Renombradas a `orquestacion_crewai/` y `orquestacion_langgraph/`.
- **Imports absolutos**: al ser paquetes reales, se eliminaron los `sys.path.insert(...)` que había en 8 módulos y el bloque de ~30 líneas de `importlib` en `notebooks/comparativa.ipynb` (que existía solo para sortear la colisión). El notebook ahora importa con `import orquestacion_langgraph.grafo as lg`.
- **Forma de ejecutar**: ahora es `python -m paquete.modulo` desde la raíz del repo, no `python carpeta/archivo.py`.
- **`.env.example` creado**: el README lo documentaba como paso obligatorio pero el archivo no existía ni estaba versionado, así que un clon del repo no era ejecutable. Contiene los mismos endpoints que ya estaban como valores por defecto en el código y una `VLLM_API_KEY` ficticia.

## Reglas y supuestos de esta migración (2026-07-23)

- **Regla de negocio**: los tutores pidieron estandarizar el acceso a LLMs sobre el protocolo OpenAI-compatible (vLLM), en vez de Ollama. Fuente: PDF "Acceso a los Endpoints de LLMs" + correo del 2026-07-23. No hay otra regla de negocio detrás del cambio de proveedor.
- **Model ids**: los nombres "amigables" del PDF (`gemma-4-31B`, `BGE-M3`, `glm-ocr`) NO coinciden con los model ids reales que exige la API. Los verificados contra `GET /v1/models` son `google/gemma-4-12B-it` (chat), `BAAI/bge-m3` (embeddings) y `zai-org/GLM-OCR` (OCR).
- **Verificado (2026-07-23)**: `gemma-4-12B-it` soporta tool calling vía vLLM → `allow_delegation=True` de CrewAI funciona; `bge-m3` devuelve embeddings de 1024-dim; `GLM-OCR` transcribe imágenes con el patrón "vision" de OpenAI (`image_url` + data URI base64). El notebook comparativo corre end-to-end.
- **Dato simulado**: `rag/recetas_data/` tiene 2 recetas en texto y 1 imagen sintética generada con Pillow (texto tipeado renderizado como imagen), no una foto real de una receta manuscrita. Sirve para probar el pipeline completo (imagen → OCR → embeddings → ChromaDB), no como contenido real del recetario. `dataset.csv` (415 frases) también es una base simulada.
- **Entorno**: el `.venv` estaba creado para Python 3.11 (intérprete ya inexistente en el host, ahora 3.12); se reconstruyó con 3.12 y se reinstaló `requirements.txt`. Si se clona en otra máquina, recrear el venv con la versión de Python disponible.
- **Seguridad**: el endpoint vLLM no requiere autenticación real (según el PDF); `VLLM_API_KEY` es un valor cualquiera, no un secreto. El acceso depende del aislamiento de la VPN institucional, no de esta clave.
- **Pendiente**: repetir con `gemma-4-12B-it` el hallazgo de salida estructurada con Pydantic documentado en `orquestacion_langgraph/README.md` (se probó con `llama3.1:8b`/`qwen3.6`, no con este modelo); ampliar la muestra de accuracy de 50 al dataset completo si se quiere el número definitivo.
