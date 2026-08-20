# Pruebas del control de calidad del OCR y del troceado de la ingesta.
#
# Uso (desde la raíz del repo):
#   python -m pruebas.prueba_calidad_ingesta
#
# NO requieren VPN ni servidor: son funciones puras sobre texto.
#
# Motivo de estas pruebas: el 2026-08-19, GLM-OCR entró en un bucle al leer una
# doble página densa de un libro de cocina y produjo 60.648 caracteres de una
# frase repetida, que fueron a dar al índice vectorial y pasaron a ser el 66%
# del recetario. El detector tiene que atrapar eso sin descartar recetas
# legítimas.

import sys

from rag.calidad import (
    MAXIMO_FRECUENCIA_NGRAMA,
    evaluar_texto,
    ngrama_mas_frecuente,
    ratio_palabras_unicas,
)
from rag.ingesta import (
    MAXIMO_CARACTERES_FRAGMENTO,
    MINIMO_CARACTERES_FRAGMENTO,
    _trocear,
)

# Extracto real de lo que devolvió el OCR en el caso que motivó todo esto.
OCR_DEGENERADO = (
    "7 Con una cuchara, dibujue la carne entre las dos fujentes para el homo. "
    "Echá la mantequilla, hasta que se dore por ambio. "
    + "Sive el mantequilla que se dore por ambio. " * 200
)

RECETA_LEGITIMA = (
    "Bolón de verde\n\n"
    "Ingredientes:\n2 plátanos verdes\n100 g de queso manaba\n"
    "50 g de chicharrón\nmanteca\nsal\n\n"
    "Preparación:\nPela los plátanos y ponlos a hervir en agua con sal. "
    "Cuando estén cocidos, májalos mientras aún están calientes para que el "
    "bolón quede compacto y no se desarme. Añade la manteca y amasa. "
    "Rellena con el queso y el chicharrón, forma bolas y sirve caliente."
)


# --- Control de calidad ----------------------------------------------------

def caso_detecta_el_bucle_real():
    d = evaluar_texto(OCR_DEGENERADO)
    assert d.es_degenerado, "no detectó el bucle del OCR real"
    assert d.frecuencia_ngrama_maxima > MAXIMO_FRECUENCIA_NGRAMA
    assert d.motivo and "aparece" in d.motivo
    return f"detecta el bucle real del OCR ({d.frecuencia_ngrama_maxima} repeticiones)"


def caso_acepta_receta_legitima():
    d = evaluar_texto(RECETA_LEGITIMA)
    assert not d.es_degenerado, f"rechazó una receta válida: {d.motivo}"
    return f"acepta una receta legítima (ratio {d.ratio_palabras_unicas})"


def caso_detecta_bucles_de_cualquier_periodo():
    """Una primera versión del detector comparaba n-gramas separados por n
    posiciones, así que solo veía bucles cuyo período coincidía con el tamaño
    del n-grama. Una frase de 8 palabras repetida 30 veces pasaba sin ser
    detectada."""
    periodos = {
        3: "no se ve. " * 40,
        5: "el texto no se entiende " * 40,
        8: "Sive el mantequilla que se dore por ambio. " * 30,
        11: "una frase bastante mas larga que se repite sin parar en el texto " * 25,
    }
    for palabras_por_ciclo, texto in periodos.items():
        d = evaluar_texto(texto)
        assert d.es_degenerado, f"no detectó el bucle de período {palabras_por_ciclo}"
    return f"detecta bucles de período {sorted(periodos)}"


def caso_no_marca_listas_ni_textos_cortos():
    """Las listas de ingredientes repiten unidades ('g de', 'ml de') y los
    fragmentos cortos tienen pocas palabras: ninguno debe dar falso positivo."""
    textos = [
        "Ingredientes: 250 g de mantequilla, 120 ml de agua fría, 500 g de "
        "harina, 450 g de carne de oveja, 2 zanahorias picadas fino.",
        "Para 4 personas",
        "tiempo de cocción: 1 hora | preparación: 25 minutos",
        "Utensilios: un cuenco grande, un cuchillo de mesa, una tabla de "
        "cortar, una cuchara, un rodillo de cocina, 2 bandejas de horno.",
    ]
    for t in textos:
        d = evaluar_texto(t)
        assert not d.es_degenerado, f"falso positivo en {t[:40]!r}: {d.motivo}"
    return "no marca listas de ingredientes ni fragmentos cortos"


def caso_metricas_coherentes():
    # Texto más corto que dos n-gramas: no hay repetición que medir, así que la
    # función devuelve frecuencia 1 y ningún n-grama.
    frecuencia, ngrama = ngrama_mas_frecuente("uno dos tres cuatro cinco seis siete ocho")
    assert frecuencia == 1, f"frecuencia inesperada: {frecuencia}"
    assert ngrama is None, f"esperaba None para un texto corto, dio {ngrama!r}"

    # Con suficientes palabras sí devuelve el n-grama dominante.
    frecuencia, ngrama = ngrama_mas_frecuente("uno dos tres cuatro cinco " * 4)
    assert frecuencia == 4, f"esperaba 4 repeticiones, dio {frecuencia}"
    assert ngrama is not None

    assert ratio_palabras_unicas("a b c") == 1.0
    assert ratio_palabras_unicas("a a a a") == 0.25
    assert ratio_palabras_unicas("") == 1.0
    return "las métricas base son coherentes"


# --- Troceado --------------------------------------------------------------

def caso_trocea_por_parrafos():
    fragmentos = _trocear(RECETA_LEGITIMA)
    assert len(fragmentos) >= 2, f"trocear de más: dio {len(fragmentos)} fragmentos"
    assert all(f.strip() for f in fragmentos), "hay fragmentos vacíos"
    # Cada fragmento debe traer contenido, no solo el encabezado de una sección.
    for f in fragmentos:
        assert len(f) >= MINIMO_CARACTERES_FRAGMENTO, f"fragmento muy corto: {f!r}"
    return f"trocea una receta normal en {len(fragmentos)} fragmentos con contenido"


def caso_fusiona_encabezados_sueltos():
    """El troceado por párrafos dejaba encabezados como fragmentos propios
    ("PREPARACIÓN", "Ingredientes:", el título de la receta). Sin contenido, su
    embedding no representa ninguna receta y quedan cerca de cualquier consulta:
    buscar "sushi" devolvía tres fragmentos "PREPARACIÓN" de recetas distintas.

    Deben quedar pegados a la sección que encabezan.
    """
    texto = (
        "Llapingachos\n\nRecetas Ecuatorianas\n\nIngredientes:\n\n"
        "- 1 kg de papa chola\n- 500 g de queso fresco\n- 2 huevos\n\n"
        "PREPARACIÓN\n\n"
        "1. Pele las papas y cocine en agua hirviendo con sal.\n"
        "2. Cuando estén suaves escurra y reduzca a puré."
    )
    fragmentos = _trocear(texto)

    for f in fragmentos:
        assert len(f) >= MINIMO_CARACTERES_FRAGMENTO, f"encabezado suelto sin fusionar: {f!r}"
        assert f.strip() not in ("PREPARACIÓN", "Ingredientes:", "Llapingachos"), (
            f"quedó un encabezado como fragmento propio: {f!r}"
        )

    # "PREPARACIÓN" tiene que haber quedado junto a los pasos que encabeza.
    con_preparacion = [f for f in fragmentos if "PREPARACIÓN" in f]
    assert con_preparacion, "se perdió el encabezado PREPARACIÓN"
    assert "Pele las papas" in con_preparacion[0], (
        "PREPARACIÓN no quedó junto a los pasos que encabeza"
    )
    return f"fusiona encabezados sueltos ({len(fragmentos)} fragmentos, ninguno vacío de contenido)"


def caso_ningun_fragmento_excede_el_tope():
    """Es la propiedad que importa: si un fragmento excede la ventana del
    modelo de embeddings, rag/embeddings.py lo trunca para calcular el vector y
    el embedding deja de corresponder al texto que se guarda en ChromaDB."""
    entradas = [
        "palabra " * 3000,                                             # un bloque enorme
        "\n".join(f"Paso {i}: hacer algo con la receta." for i in range(200)),
        "corto",
        "a" * 5000,                                                    # una sola palabra larguísima
    ]
    for entrada in entradas:
        for fragmento in _trocear(entrada):
            assert len(fragmento) <= MAXIMO_CARACTERES_FRAGMENTO, (
                f"fragmento de {len(fragmento)} caracteres supera el tope "
                f"de {MAXIMO_CARACTERES_FRAGMENTO}"
            )
    return f"ningún fragmento supera {MAXIMO_CARACTERES_FRAGMENTO} caracteres"


def caso_no_produce_fragmentos_vacios():
    for entrada in ["", "\n\n\n", "   \n  \n  ", "texto\n\n\n\nmas texto", "\n"]:
        fragmentos = _trocear(entrada)
        assert all(f.strip() for f in fragmentos), f"fragmento vacío con {entrada!r}"
    return "no produce fragmentos vacíos"


# --- Runner ----------------------------------------------------------------

CASOS = [
    caso_detecta_el_bucle_real,
    caso_acepta_receta_legitima,
    caso_detecta_bucles_de_cualquier_periodo,
    caso_no_marca_listas_ni_textos_cortos,
    caso_metricas_coherentes,
    caso_trocea_por_parrafos,
    caso_fusiona_encabezados_sueltos,
    caso_ningun_fragmento_excede_el_tope,
    caso_no_produce_fragmentos_vacios,
]


def main():
    print("Pruebas de calidad de OCR y troceado (sin VPN)\n")
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
