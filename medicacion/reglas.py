# Criterios de medicación: exclusión (alergia, contraindicación), tope ajustado y horarios.
# Ilustrativos sobre datos ficticios, no clínicos; en código y no en un prompt (bitácora 16).
# medicamentos_para() y medicamentos_excluidos_para() son el planteamiento original, sin uso.

from typing import Optional

from medicacion.datos import cargar_medicamentos

# Tope diario con función renal reducida. Ilustrativo: en la práctica depende del fármaco.
FACTOR_AJUSTE_RENAL = 0.5

# Respaldo si la receta no fija horarios. Las 3 tomas (8, 15, 20 h) las propuso el tutor.
HORARIOS_POR_NUMERO_DE_TOMAS = {
    1: ["08:00"],
    2: ["08:00", "20:00"],
    3: ["08:00", "15:00", "20:00"],
    4: ["08:00", "13:00", "18:00", "23:00"],
}

# Cuando el JSON no trae `tomas_por_dia` o trae un valor fuera de la tabla.
HORARIOS_POR_DEFECTO = ["08:00"]


def _tiene_alergia(medicamento: dict, alergias: list[str]) -> bool:
    grupo = medicamento.get("grupo_alergia")
    return bool(grupo) and grupo in alergias


def motivo_de_exclusion(medicamento: dict, perfil: dict) -> Optional[str]:
    """Por qué no es apto para esta persona, o None. Devuelve el motivo para poder explicarlo."""
    condiciones = perfil.get("condiciones", [])

    if _tiene_alergia(medicamento, perfil.get("alergias", [])):
        return f"la persona es alérgica a {medicamento['grupo_alergia']}"

    contraindicado = set(medicamento.get("contraindicado_en", [])) & set(condiciones)
    if contraindicado:
        return f"está contraindicado con {', '.join(sorted(contraindicado))}"

    return None


def dosis_maxima_para(medicamento: dict, perfil: dict) -> float:
    """Tope diario ajustado por función renal (regla pedida por el tutor)."""
    maximo = medicamento["max_dosis_diaria_mg"]
    if perfil.get("funcion_renal") == "reducida" and "insuficiencia_renal" in medicamento.get("reducir_dosis_si", []):
        return round(maximo * FACTOR_AJUSTE_RENAL, 3)
    return maximo


def horarios_de(medicamento: dict) -> list[str]:
    return HORARIOS_POR_NUMERO_DE_TOMAS.get(
        medicamento.get("tomas_por_dia"), HORARIOS_POR_DEFECTO
    )


def plan_de(medicamento: dict, perfil: dict) -> dict:
    """Dosis por toma, tope ajustado, horarios y si va con comida."""
    maximo = dosis_maxima_para(medicamento, perfil)
    por_toma = medicamento["dosis_mg"]
    horarios = horarios_de(medicamento)

    # Si el tope ajustado no alcanza, se recortan tomas y no la dosis: no siempre se puede partir.
    tomas_que_caben = int(maximo // por_toma) if por_toma else 0
    if tomas_que_caben < len(horarios):
        horarios = horarios[:max(tomas_que_caben, 0)]

    return {
        "nombre": medicamento["nombre"],
        "categoria": medicamento["categoria"],
        "forma": medicamento["forma"],
        "dosis_por_toma_mg": por_toma,
        "max_diario_mg": maximo,
        "max_diario_sin_ajustar_mg": medicamento["max_dosis_diaria_mg"],
        "dosis_ajustada_por_rinon": maximo != medicamento["max_dosis_diaria_mg"],
        "horarios": horarios,
        "tomas_por_dia": len(horarios),
        "con_comida": medicamento.get("con_comida", False),
    }


def medicamentos_para(perfil: dict, condicion: Optional[str] = None) -> list[dict]:
    """Medicamentos aptos con su plan. `condicion` acota a una; sin condiciones, lista vacía."""
    condiciones = [condicion] if condicion else perfil.get("condiciones", [])
    if not condiciones:
        return []

    aptos = []
    for medicamento in cargar_medicamentos():
        if not set(medicamento.get("indicado_para", [])) & set(condiciones):
            continue
        if motivo_de_exclusion(medicamento, perfil):
            continue
        aptos.append(plan_de(medicamento, perfil))
    return aptos


def medicamentos_excluidos_para(perfil: dict, condicion: Optional[str] = None) -> list[dict]:
    """Los que tratan la condición pero se excluyen, con el motivo."""
    condiciones = [condicion] if condicion else perfil.get("condiciones", [])
    if not condiciones:
        return []

    excluidos = []
    for medicamento in cargar_medicamentos():
        if not set(medicamento.get("indicado_para", [])) & set(condiciones):
            continue
        motivo = motivo_de_exclusion(medicamento, perfil)
        if motivo:
            excluidos.append({"nombre": medicamento["nombre"], "motivo": motivo})
    return excluidos
