# WER y CER de transcripción. El WER puede pasar de 1.0 cuando el modelo inventa
# palabras, lo que Whisper hace con audio ruidoso.

import re
import unicodedata

# Texto que Whisper inventa cuando no entiende. Se cuenta aparte del WER para no
# esconder el fenómeno.
MULETILLAS_ALUCINADAS = [
    "gracias por ver el video",
    "gracias por ver",
    "subtítulos realizados por",
    "amara.org",
]


def normalizar(texto: str, quitar_tildes: bool = False) -> str:
    """Baja a minúsculas, quita puntuación y normaliza espacios.

    `quitar_tildes` no cuenta como error una tilde mal puesta; se reportan las dos.
    """
    texto = texto.lower().strip()
    if quitar_tildes:
        texto = "".join(
            c for c in unicodedata.normalize("NFD", texto)
            if unicodedata.category(c) != "Mn"
        )
    # \w con re.UNICODE incluye vocales acentuadas y ñ.
    texto = re.sub(r"[^\w\s]", " ", texto, flags=re.UNICODE)
    return re.sub(r"\s+", " ", texto).strip()


def _distancia_edicion(referencia: list, hipotesis: list) -> int:
    """Levenshtein sobre listas (de palabras o de caracteres)."""
    if not referencia:
        return len(hipotesis)
    if not hipotesis:
        return len(referencia)

    previa = list(range(len(hipotesis) + 1))
    for i, r in enumerate(referencia, 1):
        actual = [i]
        for j, h in enumerate(hipotesis, 1):
            actual.append(min(
                previa[j] + 1,          # borrado
                actual[j - 1] + 1,      # inserción
                previa[j - 1] + (r != h),  # sustitución
            ))
        previa = actual
    return previa[-1]


def wer(referencia: str, hipotesis: str, quitar_tildes: bool = False) -> float:
    """Word error rate. Devuelve 0.0 si ambos están vacíos, 1.0 si la
    referencia está vacía pero la hipótesis no (todo inventado)."""
    r = normalizar(referencia, quitar_tildes).split()
    h = normalizar(hipotesis, quitar_tildes).split()
    if not r:
        return 0.0 if not h else 1.0
    return _distancia_edicion(r, h) / len(r)


def cer(referencia: str, hipotesis: str, quitar_tildes: bool = False) -> float:
    """Character error rate."""
    r = list(normalizar(referencia, quitar_tildes).replace(" ", ""))
    h = list(normalizar(hipotesis, quitar_tildes).replace(" ", ""))
    if not r:
        return 0.0 if not h else 1.0
    return _distancia_edicion(r, h) / len(r)


def contiene_alucinacion_conocida(texto: str) -> str | None:
    """Devuelve la muletilla alucinada encontrada, o None."""
    normalizado = normalizar(texto, quitar_tildes=True)
    for muletilla in MULETILLAS_ALUCINADAS:
        if normalizar(muletilla, quitar_tildes=True) in normalizado:
            return muletilla
    return None


def resumir(pares: list[tuple[str, str]]) -> dict:
    """WER y CER agregados sobre una lista de (referencia, hipótesis).

    Total de errores sobre total de palabras, no promedio de WER: promediar
    sobrepondera las frases cortas.
    """
    errores_palabra = total_palabras = 0
    errores_caracter = total_caracteres = 0
    vacias = alucinadas = 0

    for referencia, hipotesis in pares:
        r_pal = normalizar(referencia).split()
        h_pal = normalizar(hipotesis).split()
        errores_palabra += _distancia_edicion(r_pal, h_pal)
        total_palabras += len(r_pal)

        r_car = list(normalizar(referencia).replace(" ", ""))
        h_car = list(normalizar(hipotesis).replace(" ", ""))
        errores_caracter += _distancia_edicion(r_car, h_car)
        total_caracteres += len(r_car)

        if not h_pal:
            vacias += 1
        if contiene_alucinacion_conocida(hipotesis):
            alucinadas += 1

    return {
        "n": len(pares),
        "wer": round(errores_palabra / total_palabras, 4) if total_palabras else 0.0,
        "cer": round(errores_caracter / total_caracteres, 4) if total_caracteres else 0.0,
        "transcripciones_vacias": vacias,
        "con_alucinacion_conocida": alucinadas,
    }
