# Evaluación del OCR sobre el recetario real, sin texto de referencia (bitácora 19).
#
# Uso (desde la raíz del repo):
#   python -m pruebas.evaluar_ocr                      # corre el OCR y guarda
#   python -m pruebas.evaluar_ocr --reanudar ARCHIVO   # completa una corrida cortada
#   python -m pruebas.evaluar_ocr --comparar A.json B.json
#
# Requiere acceso al endpoint de OCR (hoy qwen2.5vl:7b en Ollama).
#
# No mide exactitud (CER/WER): mide texto degenerado, volumen y tiempo, y con
# --comparar cuánto coinciden dos OCR. Guarda el texto crudo tras cada imagen,
# para reanudar y para reconstruir el índice con `rag.ingesta --desde-ocr`.

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from rag.calidad import evaluar_texto
from rag.config import RECETAS_DATA_DIR, VLLM_OCR_BASE_URL, VLLM_OCR_MODEL
from rag.ingesta import EXTENSIONES_IMAGEN
from rag.ocr import MAXIMO_TOKENS_SALIDA, _PROMPT_TRANSCRIPCION, extraer_texto_detallado
from infraestructura.modelos import describir_resolucion

DIRECTORIO_RESULTADOS = Path(__file__).parent.parent / "resultados"


def _slug(modelo: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.]+", "-", modelo).strip("-").lower()


def _sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _imagenes() -> list[Path]:
    return sorted(
        a for a in RECETAS_DATA_DIR.iterdir()
        if a.is_file() and a.suffix.lower() in EXTENSIONES_IMAGEN
    )


def _guardar(ruta: Path, datos: dict):
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")


def _resultado_de_imagen(imagen: Path) -> dict:
    inicio = time.time()
    try:
        salida = extraer_texto_detallado(imagen)
        texto, fin, tokens, error = salida["texto"], salida["fin"], salida["tokens_salida"], None
    except Exception as e:
        # Se registra y se sigue: una imagen que falla no debe tirar la corrida.
        texto, fin, tokens, error = "", None, None, f"{type(e).__name__}: {e}"
    segundos = round(time.time() - inicio, 2)
    diagnostico = evaluar_texto(texto)
    return {
        "sha256": _sha256(imagen),
        "segundos": segundos,
        "error": error,
        "fin": fin,
        "agoto_el_tope": fin == "length",
        "tokens_salida": tokens,
        "caracteres": len(texto),
        "palabras": diagnostico.palabras,
        "degenerado": diagnostico.es_degenerado,
        "motivo_rechazo": diagnostico.motivo,
        "ratio_palabras_unicas": diagnostico.ratio_palabras_unicas,
        "frecuencia_ngrama_maxima": diagnostico.frecuencia_ngrama_maxima,
        "texto": texto,
    }


def correr(ruta_salida: Path, datos: dict):
    imagenes = _imagenes()
    hechas = datos["imagenes"]
    pendientes = [i for i in imagenes if i.name not in hechas]
    print(f"OCR con {datos['modelo_ocr']} — {len(pendientes)} de {len(imagenes)} imágenes pendientes")
    print(f"Guardando en {ruta_salida}\n")

    for n, imagen in enumerate(pendientes, 1):
        print(f"  [{n}/{len(pendientes)}] {imagen.name} ...", end=" ", flush=True)
        resultado = _resultado_de_imagen(imagen)
        hechas[imagen.name] = resultado
        _guardar(ruta_salida, datos)
        if resultado["error"]:
            estado = f"ERROR {resultado['error'][:80]}"
        elif resultado["degenerado"]:
            estado = f"RECHAZADA ({resultado['motivo_rechazo']})"
        elif resultado["agoto_el_tope"]:
            estado = "AGOTÓ EL TOPE de tokens (revisar: posible bucle no detectado)"
        else:
            estado = "ok"
        print(f"{resultado['segundos']:.1f}s, {resultado['caracteres']} car. — {estado}")

    datos["terminado"] = datetime.now().isoformat(timespec="seconds")
    _guardar(ruta_salida, datos)
    resumir(datos)


def resumir(datos: dict):
    resultados = list(datos["imagenes"].values())
    if not resultados:
        return
    tiempos = sorted(r["segundos"] for r in resultados if not r["error"])
    errores = [n for n, r in datos["imagenes"].items() if r["error"]]
    rechazadas = [n for n, r in datos["imagenes"].items() if r["degenerado"] and not r["error"]]
    topadas = [n for n, r in datos["imagenes"].items() if r.get("agoto_el_tope")]
    print(f"\nResumen ({datos['modelo_ocr']}):")
    print(f"  imágenes: {len(resultados)} · errores: {len(errores)} · rechazadas por calidad: {len(rechazadas)}"
          f" · agotaron el tope de tokens: {len(topadas)}")
    if tiempos:
        mediana = tiempos[len(tiempos) // 2]
        print(f"  tiempo por imagen: mediana {mediana:.1f}s, mín {tiempos[0]:.1f}s, máx {tiempos[-1]:.1f}s, total {sum(tiempos):.0f}s")
    for nombre in rechazadas:
        print(f"  RECHAZADA {nombre}: {datos['imagenes'][nombre]['motivo_rechazo']}")
    for nombre in errores:
        print(f"  ERROR {nombre}: {datos['imagenes'][nombre]['error']}")


# Comparación entre dos OCR

def _palabras_normalizadas(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9ñ/]+", texto)


def _textos(datos: dict) -> dict[str, str]:
    """Acepta las dos formas de archivo: la salida de este script
    ('imagenes' con texto crudo) y la exportación del índice ('textos')."""
    if "imagenes" in datos:
        return {n: r["texto"] for n, r in datos["imagenes"].items() if not r["error"]}
    return datos["textos"]


def comparar(ruta_a: Path, ruta_b: Path):
    a, b = (json.loads(r.read_text(encoding="utf-8")) for r in (ruta_a, ruta_b))
    textos_a, textos_b = _textos(a), _textos(b)
    nombre_a, nombre_b = a.get("modelo_ocr", ruta_a.stem), b.get("modelo_ocr", ruta_b.stem)
    print(f"A = {nombre_a}\nB = {nombre_b}")
    print("Coincidencia = proporción de palabras en común en el mismo orden (difflib).")
    print("No mide exactitud: dos OCR pueden coincidir en el mismo error.\n")
    print(f"{'imagen':34} {'car. A':>7} {'car. B':>7} {'coinc.':>7}")

    filas = []
    for nombre in sorted(set(textos_a) | set(textos_b)):
        if Path(nombre).suffix.lower() not in EXTENSIONES_IMAGEN:
            continue  # los .txt no pasan por OCR
        ta, tb = textos_a.get(nombre), textos_b.get(nombre)
        if ta is None or tb is None:
            falta = "A" if ta is None else "B"
            print(f"{nombre[:34]:34} {len(ta or ''):>7} {len(tb or ''):>7}   (sin texto en {falta})")
            continue
        coincidencia = SequenceMatcher(None, _palabras_normalizadas(ta), _palabras_normalizadas(tb), autojunk=False).ratio()
        filas.append((nombre, len(ta), len(tb), coincidencia))
        print(f"{nombre[:34]:34} {len(ta):>7} {len(tb):>7} {coincidencia:>7.2f}")

    if filas:
        valores = sorted(f[3] for f in filas)
        print(f"\nCoincidencia en {len(filas)} imágenes: mediana {valores[len(valores) // 2]:.2f}, "
              f"mín {valores[0]:.2f}, máx {valores[-1]:.2f}")
        print("Las de menor coincidencia son las que hay que mirar a mano.")


def main():
    parser = argparse.ArgumentParser(description="Evalúa el OCR sobre el recetario, sin texto de referencia.")
    parser.add_argument("--reanudar", type=Path, help="completa una corrida guardada")
    parser.add_argument("--comparar", type=Path, nargs=2, metavar=("A", "B"), help="compara dos archivos de OCR")
    args = parser.parse_args()

    if args.comparar:
        comparar(*args.comparar)
        return 0

    if args.reanudar:
        ruta = args.reanudar
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        if datos["modelo_ocr"] != VLLM_OCR_MODEL:
            print(f"El archivo es de {datos['modelo_ocr']} y el endpoint sirve {VLLM_OCR_MODEL}: no se mezclan.")
            return 1
    else:
        DIRECTORIO_RESULTADOS.mkdir(exist_ok=True)
        ruta = DIRECTORIO_RESULTADOS / f"ocr_{_slug(VLLM_OCR_MODEL)}_{datetime.now():%Y-%m-%d}.json"
        if ruta.exists():
            print(f"Ya existe {ruta}. Usa --reanudar para completarlo o bórralo a propósito.")
            return 1
        datos = {
            "_que_es": "Salida CRUDA del OCR por imagen del recetario, antes de trocear. Generada por pruebas/evaluar_ocr.py. Sirve de evidencia y para reconstruir el índice con `python -m rag.ingesta --desde-ocr`.",
            "modelo_ocr": VLLM_OCR_MODEL,
            "base_url": VLLM_OCR_BASE_URL,
            "resolucion_de_modelos": describir_resolucion(),
            "prompt": _PROMPT_TRANSCRIPCION,
            "temperatura": 0.0,
            "maximo_tokens_salida": MAXIMO_TOKENS_SALIDA,
            "iniciado": datetime.now().isoformat(timespec="seconds"),
            "imagenes": {},
        }
    correr(ruta, datos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
