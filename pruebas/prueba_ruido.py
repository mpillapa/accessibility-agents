# Pruebas de la generación de la matriz de ruido.
#
# Uso (desde la raíz del repo):
#   python -m pruebas.prueba_ruido
#
# NO requieren GPU ni modelo: son funciones puras sobre señales de audio.
# Lo que verifican es que el SNR de las mezclas sea el que se pidió — si esa
# matemática está mal, toda la matriz de ruido de la evaluación de Whisper mide
# una condición distinta de la que dice medir, y los resultados no significan
# nada.

import array
import math
import sys
import tempfile
from pathlib import Path

from asr.ruido import (
    escribir_wav,
    generar_matriz,
    generar_ruido,
    leer_wav,
    mezclar_con_ruido,
    potencia_media,
    snr_medido_db,
)

FRECUENCIA = 16000


def señal_de_prueba(segundos: float = 1.0, amplitud: int = 8000) -> array.array:
    """Un tono de 220 Hz: no es voz, pero para verificar potencias y SNR se
    comporta igual y es determinista."""
    n = int(FRECUENCIA * segundos)
    return array.array(
        "h", [int(amplitud * math.sin(2 * math.pi * 220 * t / FRECUENCIA)) for t in range(n)]
    )


def caso_snr_es_el_pedido():
    """La propiedad central: si se pide 10 dB, la mezcla tiene 10 dB."""
    señal = señal_de_prueba(2.0)
    for snr in [20, 10, 5, 0, -5]:
        for tipo in ["blanco", "rosa"]:
            mezcla = mezclar_con_ruido(señal, snr, tipo=tipo, semilla=42)
            medido = snr_medido_db(señal, mezcla)
            assert abs(medido - snr) < 0.6, (
                f"ruido {tipo} a {snr} dB: se midió {medido:.2f} dB"
            )
    return "el SNR de la mezcla es el pedido (blanco y rosa, de -5 a 20 dB)"


def caso_ruido_normalizado():
    """El ruido sale con potencia media 1 para poder escalarlo por SNR; si no,
    el cálculo del SNR quedaría desplazado por un factor constante."""
    for tipo in ["blanco", "rosa"]:
        p = potencia_media(generar_ruido(20000, tipo=tipo, semilla=1))
        assert abs(p - 1.0) < 0.05, f"ruido {tipo}: potencia {p:.4f}, esperaba ~1.0"
    return "el ruido sale normalizado a potencia media 1"


def caso_reproducible():
    """Misma semilla, mismo archivo. Sin esto los resultados de una corrida no
    se pueden comparar con los de otra."""
    señal = señal_de_prueba()
    a = mezclar_con_ruido(señal, 10, tipo="blanco", semilla=7)
    b = mezclar_con_ruido(señal, 10, tipo="blanco", semilla=7)
    c = mezclar_con_ruido(señal, 10, tipo="blanco", semilla=8)
    assert list(a) == list(b), "la misma semilla dio ruido distinto"
    assert list(a) != list(c), "semillas distintas dieron el mismo ruido"
    return "la mezcla es reproducible (misma semilla, mismo resultado)"


def caso_no_desborda_16_bits():
    """A SNR bajo y con señal fuerte, la suma puede pasarse del rango de 16
    bits. Debe recortarse, no envolver: un desborde sonaría como un chasquido y
    Whisper lo interpretaría como habla."""
    señal = señal_de_prueba(amplitud=32000)  # casi el máximo
    mezcla = mezclar_con_ruido(señal, -10, tipo="blanco", semilla=3)
    assert all(-32768 <= m <= 32767 for m in mezcla), "hay muestras fuera de rango"
    return "recorta a 16 bits sin desbordar, incluso a -10 dB"


def caso_ida_y_vuelta_wav():
    """Escribir y leer no debe alterar las muestras."""
    señal = señal_de_prueba(0.5)
    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "prueba.wav"
        escribir_wav(ruta, señal, FRECUENCIA)
        leidas, frecuencia = leer_wav(ruta)
        assert frecuencia == FRECUENCIA, f"frecuencia cambió a {frecuencia}"
        assert list(leidas) == list(señal), "las muestras cambiaron al ir y volver"
    return "escribir y leer WAV conserva las muestras"


def caso_matriz_completa():
    """La matriz genera un archivo por condición, incluida la de control sin
    ruido, y reporta el SNR medido de cada una."""
    señal = señal_de_prueba(1.0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        origen = tmp / "frase001.wav"
        escribir_wav(origen, señal, FRECUENCIA)

        filas = generar_matriz(origen, tmp / "salida", niveles=[None, 10, 0],
                               tipos=["blanco", "rosa"])

        # 1 control + 2 niveles x 2 tipos = 5
        assert len(filas) == 5, f"esperaba 5 variantes, dio {len(filas)}"
        for fila in filas:
            assert (tmp / "salida" / fila["archivo"]).exists(), (
                f"no se escribió {fila['archivo']}"
            )
        controles = [f for f in filas if f["tipo_ruido"] == "ninguno"]
        assert len(controles) == 1, "debe haber exactamente una condición de control"

        # El SNR medido de cada variante con ruido debe estar cerca del pedido.
        for fila in filas:
            if fila["tipo_ruido"] == "ninguno":
                continue
            objetivo = float(fila["snr_db_objetivo"])
            medido = float(fila["snr_db_medido"])
            assert abs(medido - objetivo) < 0.6, (
                f"{fila['archivo']}: pedido {objetivo} dB, medido {medido} dB"
            )
    return "la matriz genera control + variantes, con el SNR verificado en cada una"


CASOS = [
    caso_snr_es_el_pedido,
    caso_ruido_normalizado,
    caso_reproducible,
    caso_no_desborda_16_bits,
    caso_ida_y_vuelta_wav,
    caso_matriz_completa,
]


def main():
    print("Pruebas de la matriz de ruido (sin GPU ni modelo)\n")
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
