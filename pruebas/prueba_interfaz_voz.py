# Pruebas del texto que la interfaz manda al lector de voz (interfaz/voz_salida.py).
# Uso: python -m pruebas.prueba_interfaz_voz
# No requiere VPN ni navegador; la síntesis del navegador no se prueba.

import sys

from interfaz.voz_salida import quitar_emojis, texto_para_leer

# Una respuesta real del agente de medicación para Carmen (2026-09-23), recortada.
RESPUESTA_MEDICACION = """## 🌅 A las 8:00 de la mañana

**1) Amlodipino** — 1 comprimido de 5 mg
- Es para la presión alta.
- ⚠️ **Aviso importante:** la receta indica 80 mg al día y el máximo es 40 mg.

---

_Prototipo académico con datos ficticios. No reemplaza a su médico: consulte siempre antes de cambiar su medicación._"""


def prueba_no_lee_marcas_de_formato():
    texto = texto_para_leer(RESPUESTA_MEDICACION)
    for marca in ("**", "##", "---", "- ", "_Prototipo"):
        assert marca not in texto, (marca, texto)
    return "no lee asteriscos, numerales, viñetas ni líneas"


def prueba_no_lee_emojis():
    """Un lector de voz dice "sol naciente" por 🌅: hay que sacarlos."""
    texto = texto_para_leer(RESPUESTA_MEDICACION)
    assert "🌅" not in texto and "⚠" not in texto, texto
    return "saca los emojis antes de leer"


def prueba_conserva_lo_importante():
    """Lo que se lee tiene que ser la respuesta: horas, dosis y el aviso."""
    texto = texto_para_leer(RESPUESTA_MEDICACION)
    for dato in ("8:00", "Amlodipino", "5 mg", "Aviso importante", "80 mg", "40 mg"):
        assert dato in texto, (dato, texto)
    return "conserva horas, dosis y el aviso de la receta"


def prueba_no_lee_el_aviso_de_datos_ficticios():
    """Sigue visible en pantalla; leído en cada respuesta se vuelve ruido."""
    assert "ficticios" not in texto_para_leer(RESPUESTA_MEDICACION)
    return "omite al leer el aviso de datos ficticios"


def prueba_cada_renglon_termina_en_pausa():
    """Sin punto al final de cada renglón, una lista se lee de corrido."""
    texto = texto_para_leer("Amlodipino\nFurosemida\n¿Le repito?")
    assert texto == "Amlodipino. Furosemida. ¿Le repito?", texto
    return "agrega la pausa al final de cada renglón"


def prueba_tablas_y_enlaces_se_leen_como_texto():
    texto = texto_para_leer("| Hora | Pastilla |\n|---|---|\n| 08:00 | Omeprazol |\n[su receta](http://x)")
    assert texto == "Hora, Pastilla. 08:00, Omeprazol. su receta.", texto
    return "lee tablas y enlaces sin símbolos"


def prueba_respuesta_de_emergencia_se_lee_completa():
    """Que el asistente no llama por la persona es parte de la respuesta, no formato."""
    emergencia = ("**Llame al 911 ahora mismo.**\n\nEs el número de emergencias.\n\n"
                  "_Este asistente todavía no puede llamar por usted._")
    texto = texto_para_leer(emergencia)
    assert "Llame al 911 ahora mismo." in texto and "no puede llamar por usted" in texto, texto
    return "la respuesta de emergencia se lee entera, incluida la advertencia"


def prueba_quitar_emojis_no_toca_el_texto():
    assert quitar_emojis("Hola, Carmen — a las 8:00 (ñ, á).") == "Hola, Carmen — a las 8:00 (ñ, á)."
    return "quitar emojis no altera tildes, eñes ni rayas"


CASOS = [
    prueba_no_lee_marcas_de_formato,
    prueba_no_lee_emojis,
    prueba_conserva_lo_importante,
    prueba_no_lee_el_aviso_de_datos_ficticios,
    prueba_cada_renglon_termina_en_pausa,
    prueba_tablas_y_enlaces_se_leen_como_texto,
    prueba_respuesta_de_emergencia_se_lee_completa,
    prueba_quitar_emojis_no_toca_el_texto,
]


def main():
    print("Pruebas de la lectura en voz alta (sin navegador, sin VPN)\n")
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
