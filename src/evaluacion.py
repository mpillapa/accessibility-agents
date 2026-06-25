"""
Script de evaluación automática.

Toma las 80 frases del test set (mismas que usó el clasificador en 01_clasificador.ipynb),
las pasa por las 3 versiones del sistema, y guarda los resultados en CSV.

Es reanudable: si se interrumpe, al re-ejecutarlo retoma desde donde quedó.

Uso:
    cd accessibility-agents
    python src/evaluacion.py
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Permitir importar desde src/
sys.path.insert(0, str(Path(__file__).parent))

from agentes import procesar_consulta_v_b, procesar_consulta_v_a1, procesar_consulta_v_a2

RUTA_DATASET    = "data/intents/dataset.csv"
RUTA_RESULTADOS = "resultados_evaluacion.csv"


def obtener_test_set():
    """
    Reproduce el mismo split que usó notebooks/01_clasificador.ipynb.
    80/20 estratificado con random_state=42 → siempre las mismas 83 frases.
    """
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(RUTA_DATASET)
    _, df_test = train_test_split(
        df, test_size=0.2, stratify=df["intent"], random_state=42
    )
    return df_test.reset_index(drop=True)


def cargar_progreso_previo():
    """Si ya existe el CSV de resultados, lo carga para retomar donde quedó."""
    if os.path.exists(RUTA_RESULTADOS):
        print(f"Retomando desde {RUTA_RESULTADOS} ({os.path.getsize(RUTA_RESULTADOS)} bytes)")
        return pd.read_csv(RUTA_RESULTADOS)
    return pd.DataFrame(columns=[
        "version", "indice", "consulta", "intencion_real",
        "intencion_predicha", "respuesta", "latencia_segundos"
    ])


def guardar_resultado(df_resultados, nuevo_resultado):
    """Añade un resultado al CSV y guarda inmediatamente para no perder progreso."""
    df_resultados = pd.concat(
        [df_resultados, pd.DataFrame([nuevo_resultado])], ignore_index=True
    )
    df_resultados.to_csv(RUTA_RESULTADOS, index=False)
    return df_resultados


def ya_evaluado(df_resultados, version, indice):
    """Verifica si ya se evaluó esa frase con esa versión."""
    mask = (df_resultados["version"] == version) & (df_resultados["indice"] == indice)
    return mask.any()


def evaluar_version(version_nombre, funcion_procesar, df_test, df_resultados):
    """Evalúa una versión sobre todo el test set, guardando cada resultado al instante."""
    total = len(df_test)
    pendientes = sum(
        1 for i in range(total)
        if not ya_evaluado(df_resultados, version_nombre, i)
    )
    print(f"\n=== Versión {version_nombre} — {pendientes}/{total} pendientes ===")

    for i, fila in df_test.iterrows():
        if ya_evaluado(df_resultados, version_nombre, i):
            continue

        consulta       = fila["text"]
        intencion_real = fila["intent"]
        print(f"  [{i+1}/{total}] {consulta[:55]}...", end=" ", flush=True)

        try:
            resultado = funcion_procesar(consulta)
            nuevo = {
                "version":             version_nombre,
                "indice":              i,
                "consulta":            consulta,
                "intencion_real":      intencion_real,
                "intencion_predicha":  resultado.get("intencion"),
                "respuesta":           resultado["respuesta"][:200],
                "latencia_segundos":   resultado["latencia_segundos"],
            }
            df_resultados = guardar_resultado(df_resultados, nuevo)
            print(f"OK ({resultado['latencia_segundos']}s)")
        except Exception as e:
            print(f"ERROR: {e}")
            nuevo = {
                "version":             version_nombre,
                "indice":              i,
                "consulta":            consulta,
                "intencion_real":      intencion_real,
                "intencion_predicha":  "ERROR",
                "respuesta":           f"ERROR: {e}",
                "latencia_segundos":   -1,
            }
            df_resultados = guardar_resultado(df_resultados, nuevo)

    return df_resultados


def main():
    df_test = obtener_test_set()
    print(f"Test set: {len(df_test)} frases")

    df_resultados = cargar_progreso_previo()

    # A1 primero porque es la más rápida (no usa LLM para ruteo)
    df_resultados = evaluar_version("A1", procesar_consulta_v_a1, df_test, df_resultados)
    df_resultados = evaluar_version("B",  procesar_consulta_v_b,  df_test, df_resultados)
    df_resultados = evaluar_version("A2", procesar_consulta_v_a2, df_test, df_resultados)

    print(f"\nEvaluacion completa. Resultados en: {RUTA_RESULTADOS}")
    print(f"Total de filas: {len(df_resultados)}")


if __name__ == "__main__":
    main()
