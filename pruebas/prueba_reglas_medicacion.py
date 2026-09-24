# Pruebas de las reglas de gestión de medicación.
#
# Uso (desde la raíz del repo):
#   python -m pruebas.prueba_reglas_medicacion
#
# NO requieren VPN ni LLM: las reglas son funciones puras sobre los JSON. Que se
# puedan probar así es precisamente el argumento para que NO vivan en un prompt
# (ver medicacion/reglas.py).
#
# Escrito sin pytest, igual que el resto de pruebas/ (ver prueba_ciclo_rag.py).

import sys

from medicacion.datos import cargar_medicamentos, obtener_perfil
from medicacion.reglas import (
    FACTOR_AJUSTE_RENAL,
    dosis_maxima_para,
    horarios_de,
    medicamentos_excluidos_para,
    medicamentos_para,
    motivo_de_exclusion,
    plan_de,
)


def _medicamento(nombre):
    return next(m for m in cargar_medicamentos() if m["nombre"] == nombre)


# --- Exclusión --------------------------------------------------------------

def prueba_excluye_por_contraindicacion():
    """Jorge tiene úlcera y artrosis: lo que alivia la artrosis está
    contraindicado. Es el conflicto que el tutor pidió contemplar."""
    jorge = obtener_perfil("jorge")
    motivo = motivo_de_exclusion(_medicamento("Ibuprofeno"), jorge)
    assert motivo and "ulcera_gastrica" in motivo, motivo
    nombres = [m["nombre"] for m in medicamentos_para(jorge)]
    assert "Ibuprofeno" not in nombres, nombres
    return "excluye el antiinflamatorio contraindicado con úlcera"


def prueba_excluye_por_alergia():
    manuel = obtener_perfil("manuel")  # alérgico a penicilina
    motivo = motivo_de_exclusion(_medicamento("Amoxicilina"), manuel)
    assert motivo and "alérgic" in motivo, motivo
    return "excluye el antibiótico del grupo al que la persona es alérgica"


def prueba_el_motivo_es_explicable():
    """Devolver el motivo y no un booleano permite que el sistema explique por
    qué no recomienda algo, en vez de omitirlo en silencio."""
    excluidos = medicamentos_excluidos_para(obtener_perfil("jorge"))
    assert excluidos, "Jorge debería tener exclusiones"
    assert all(e["motivo"] for e in excluidos), excluidos
    return f"cada uno de los {len(excluidos)} excluidos trae su motivo"


def prueba_no_excluye_de_mas():
    """Contrapeso: un perfil sin contraindicaciones debe recibir opciones."""
    rosa = obtener_perfil("rosa")
    assert motivo_de_exclusion(_medicamento("Metformina"), rosa) is None
    assert len(medicamentos_para(rosa)) > 5, "Rosa debería tener varias opciones"
    return "no excluye medicamentos aptos (Rosa conserva sus opciones)"


# --- Dosis ------------------------------------------------------------------

def prueba_ajusta_la_dosis_por_funcion_renal():
    """El requisito textual del tutor: 'que no me devuelva lo que me tengo que
    tomar con un exceso de gramos'."""
    carmen = obtener_perfil("carmen")  # función renal reducida
    furosemida = _medicamento("Furosemida")
    ajustada = dosis_maxima_para(furosemida, carmen)
    esperada = furosemida["max_dosis_diaria_mg"] * FACTOR_AJUSTE_RENAL
    assert ajustada == esperada, f"{ajustada} != {esperada}"
    assert ajustada < furosemida["max_dosis_diaria_mg"]
    return f"ajusta el tope diario con función renal reducida ({furosemida['max_dosis_diaria_mg']} -> {ajustada} mg)"


def prueba_no_ajusta_a_quien_no_corresponde():
    rosa = obtener_perfil("rosa")  # función renal normal
    furosemida = _medicamento("Furosemida")
    assert dosis_maxima_para(furosemida, rosa) == furosemida["max_dosis_diaria_mg"]
    return "no ajusta la dosis de quien tiene función renal normal"


def prueba_recorta_las_tomas_si_el_tope_ajustado_no_alcanza():
    """Si el tope baja a la mitad, las tomas previstas pueden superarlo. Se
    recortan tomas antes que partir la dosis unitaria."""
    carmen = obtener_perfil("carmen")
    plan = plan_de(_medicamento("Furosemida"), carmen)
    assert plan["tomas_por_dia"] * plan["dosis_por_toma_mg"] <= plan["max_diario_mg"], plan
    assert plan["tomas_por_dia"] < _medicamento("Furosemida")["tomas_por_dia"], plan
    return "recorta el número de tomas para no superar el tope ajustado"


def prueba_ninguna_recomendacion_supera_su_tope():
    """Invariante sobre TODOS los perfiles: nunca se indica más de lo permitido.
    Es la garantía que un prompt no puede dar."""
    from medicacion.datos import cargar_perfiles

    for perfil in cargar_perfiles():
        for plan in medicamentos_para(perfil):
            total = plan["tomas_por_dia"] * plan["dosis_por_toma_mg"]
            assert total <= plan["max_diario_mg"], f"{perfil['nombre']}/{plan['nombre']}: {total} > {plan['max_diario_mg']}"
    return "ningún plan supera su tope diario, en ninguno de los 6 perfiles"


# --- Horarios ---------------------------------------------------------------

def prueba_los_horarios_siguen_el_numero_de_tomas():
    """Tres tomas dan 08:00, 15:00 y 20:00 — los que propuso el tutor."""
    tres = next(m for m in cargar_medicamentos() if m["tomas_por_dia"] == 3)
    assert horarios_de(tres) == ["08:00", "15:00", "20:00"], horarios_de(tres)
    return "tres tomas se reparten en 08:00, 15:00 y 20:00"


def prueba_el_plan_dice_como_tomarlo():
    """El tutor cerró con 'cómo se debería de tomar también'."""
    plan = plan_de(_medicamento("Metformina"), obtener_perfil("rosa"))
    assert plan["con_comida"] is True, plan
    assert plan["forma"] == "comprimido", plan
    return "el plan indica la forma y si va con comida"


# --- El caso que más importa ------------------------------------------------

def prueba_no_inventa_tratamiento_a_quien_no_declaro_condiciones():
    """Luis no tiene condiciones registradas. Un LLM tiende a responder algo
    igual; las reglas devuelven vacío y el sistema tiene que admitirlo."""
    luis = obtener_perfil("luis")
    assert medicamentos_para(luis) == [], medicamentos_para(luis)
    assert medicamentos_excluidos_para(luis) == []
    return "no le inventa tratamiento a quien no declaró condiciones"


CASOS = [
    prueba_excluye_por_contraindicacion,
    prueba_excluye_por_alergia,
    prueba_el_motivo_es_explicable,
    prueba_no_excluye_de_mas,
    prueba_ajusta_la_dosis_por_funcion_renal,
    prueba_no_ajusta_a_quien_no_corresponde,
    prueba_recorta_las_tomas_si_el_tope_ajustado_no_alcanza,
    prueba_ninguna_recomendacion_supera_su_tope,
    prueba_los_horarios_siguen_el_numero_de_tomas,
    prueba_el_plan_dice_como_tomarlo,
    prueba_no_inventa_tratamiento_a_quien_no_declaro_condiciones,
]


def main():
    print("Pruebas de las reglas de medicación (sin VPN, sin LLM)\n")
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
