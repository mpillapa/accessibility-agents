# Funciones para usar el clasificador ya entrenado (el entrenamiento vive en
# notebooks/01_clasificador.ipynb). Aquí solo cargamos el modelo y predecimos.

import joblib
from sentence_transformers import SentenceTransformer

# Variables globales para no recargar en cada llamada
_modelo = None
_embedder = None


# Carga el modelo entrenado desde disco (y el embedder si el modelo lo usa).
def cargar_modelo(ruta="modelo.joblib"):
    global _modelo
    _modelo = joblib.load(ruta)
    # Si el modelo usa embeddings, cargamos también el embedder
    if _modelo["tipo"] == "embeddings":
        global _embedder
        if _embedder is None:
            _embedder = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
    return _modelo


# Recibe una frase y devuelve un dict con la intención predicha, la confianza
# y todas las probabilidades por etiqueta.
def predecir(texto):
    if _modelo is None:
        cargar_modelo()

    # Generar la representación según el tipo de modelo
    if _modelo["tipo"] == "tfidf":
        X = _modelo["vectorizador"].transform([texto])
    else:  # embeddings
        X = _embedder.encode([texto])

    clasificador = _modelo["clasificador"]
    probabilidades = clasificador.predict_proba(X)[0]
    indice_predicho = probabilidades.argmax()

    etiquetas = _modelo["etiquetas"]
    return {
        "intencion": etiquetas[indice_predicho],
        "confianza": float(probabilidades[indice_predicho]),
        "todas_las_probabilidades": {
            etiquetas[i]: float(probabilidades[i])
            for i in range(len(etiquetas))
        }
    }
