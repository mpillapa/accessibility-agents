# Pruebas de las métricas de ASR (WER, CER).
#
#   python -m pruebas.prueba_metricas
#
# Sin GPU ni modelo: funciones puras sobre texto. Si el WER está mal calculado,
# toda la evaluación de Whisper reporta números equivocados.

import sys

from asr.metricas import cer, contiene_alucinacion_conocida, normalizar, resumir, wer


def caso_transcripcion_perfecta():
    t = "ya me toca la pastillita del corazón"
    assert wer(t, t) == 0.0 and cer(t, t) == 0.0
    # La puntuación y las mayúsculas no deben contar como error.
    assert wer(t, "¿Ya me toca la pastillita del corazón?") == 0.0
    return "transcripción idéntica da WER 0, ignorando puntuación y mayúsculas"


def caso_errores_conocidos():
    # 6 palabras de referencia, 1 sustitución -> 1/6
    assert abs(wer("uno dos tres cuatro cinco seis",
                   "uno dos TRESX cuatro cinco seis") - 1/6) < 1e-9
    # 1 borrado
    assert abs(wer("uno dos tres cuatro cinco seis",
                   "uno dos cuatro cinco seis") - 1/6) < 1e-9
    # 1 inserción
    assert abs(wer("uno dos tres", "uno dos extra tres") - 1/3) < 1e-9
    return "sustitución, borrado e inserción cuentan uno cada uno"


def caso_puede_pasar_de_uno():
    """Si el ASR inventa más palabras de las que había, el WER supera 1.0. No es
    un caso teórico: es lo que hace Whisper con audio ruidoso."""
    valor = wer("hola", "gracias por ver el video hola que tal amigos")
    assert valor > 1.0, f"esperaba WER > 1, dio {valor}"
    return f"el WER puede pasar de 1.0 cuando se inventa texto ({valor:.2f})"


def caso_referencia_vacia():
    assert wer("", "") == 0.0
    assert wer("", "gracias") == 1.0    # todo inventado
    assert cer("", "") == 0.0
    return "referencia vacía: 0.0 si no inventó nada, 1.0 si inventó"


def caso_tildes():
    """En español el ASR suele acertar la palabra y errar la tilde. Contar eso
    como error de palabra infla el WER sin que haya problema de comprensión."""
    r, h = "me caí en la cocina", "me cai en la cocina"
    assert wer(r, h) > 0, "sin quitar tildes debería contar como error"
    assert wer(r, h, quitar_tildes=True) == 0.0, "quitando tildes debería ser 0"
    return "el modo sin tildes no penaliza 'cai' por 'caí'"


def caso_detecta_alucinacion_conocida():
    assert contiene_alucinacion_conocida("Gracias por ver el video.") is not None
    assert contiene_alucinacion_conocida("gracias por ver el video") is not None
    assert contiene_alucinacion_conocida("Gracias mijito, ya me salió rico") is None, (
        "'gracias' sola es una palabra legítima del dominio, no una alucinación"
    )
    return "detecta la muletilla de YouTube sin marcar 'gracias' legítimo"


def caso_agregado_no_es_promedio():
    """El WER agregado se calcula sobre el total de errores y palabras. Promediar
    los WER individuales daría más peso a las frases cortas."""
    pares = [
        ("hola", "holo"),                                   # 1 error / 1 palabra  = 1.00
        ("uno dos tres cuatro cinco seis siete ocho nueve diez",
         "uno dos tres cuatro cinco seis siete ocho nueve diez"),  # 0 / 10 = 0.00
    ]
    r = resumir(pares)
    # Agregado correcto: 1 error / 11 palabras = 0.0909. Promedio ingenuo: 0.5
    assert abs(r["wer"] - 1/11) < 1e-4, f"esperaba 0.0909, dio {r['wer']}"
    return f"el WER agregado pesa por palabras, no por frase ({r['wer']:.4f}, no 0.5)"


def caso_resumen_cuenta_vacias_y_alucinadas():
    pares = [
        ("hola que tal", ""),                            # vacía
        ("hola que tal", "gracias por ver el video"),    # alucinada
        ("hola que tal", "hola que tal"),                # correcta
    ]
    r = resumir(pares)
    assert r["n"] == 3
    assert r["transcripciones_vacias"] == 1, r
    assert r["con_alucinacion_conocida"] == 1, r
    return "el resumen cuenta transcripciones vacías y alucinaciones conocidas"


CASOS = [
    caso_transcripcion_perfecta,
    caso_errores_conocidos,
    caso_puede_pasar_de_uno,
    caso_referencia_vacia,
    caso_tildes,
    caso_detecta_alucinacion_conocida,
    caso_agregado_no_es_promedio,
    caso_resumen_cuenta_vacias_y_alucinadas,
]


def main():
    print("Pruebas de métricas de ASR (sin GPU)\n")
    fallos = 0
    for caso in CASOS:
        try:
            print(f"  OK    {caso()}")
        except AssertionError as e:
            fallos += 1
            print(f"  FALLA {caso.__name__}: {e}")
        except Exception as e:
            fallos += 1
            print(f"  ERROR {caso.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(CASOS) - fallos}/{len(CASOS)} pruebas pasaron")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
