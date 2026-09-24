# Reglas de negocio de la gestión de medicación, en código determinista.
#
# POR QUÉ ESTO NO ESTÁ EN UN PROMPT
# ---------------------------------
# El tutor planteó el agente como "eres un gestor de medicamentos que lee de una
# base de datos". Implementado literalmente, eso deja al LLM decidiendo si una
# dosis es segura, y un LLM al que se le pregunta "¿cuántos miligramos puede
# tomar esta persona?" responde algo plausible aunque el dato no esté en la
# fuente. Es el mismo patrón que este proyecto ya midió dos veces:
#
#   - Whisper inventó "Gracias por ver el video" sobre ruido, con probabilidad
#     de idioma 1.00 (sección 10 de la bitácora).
#   - GLM-OCR inventó 60.648 caracteres ante una página densa (sección 3).
#
# En esos casos el costo fue una emergencia perdida y un recetario corrupto.
# Acá sería una dosis de medicamento. Por eso las reglas viven en funciones
# puras, verificables y probadas, y el LLM se limita a redactar lo que estas
# funciones ya decidieron.
#
# LIMITACIÓN QUE HAY QUE DECLARAR
# -------------------------------
# Estas reglas son ILUSTRATIVAS, no clínicas. Modelan cuatro criterios simples
# (indicación, contraindicación, alergia y ajuste renal) sobre datos ficticios.
# Un sistema real necesitaría interacciones entre fármacos, comorbilidades,
# función hepática, peso, edad y validación profesional. No sustituyen a nadie.
#
# CAMBIO DE ROL (2026-09-22)
# -------------------------
# Este módulo ya NO decide qué toma cada persona. Esa decisión ahora sale de la
# receta médica (medicacion/prescripciones.py), porque filtrar el vademécum por
# condición devuelve OPCIONES ELEGIBLES y presentarlas como un régimen producía
# disparates: a Carmen, hipertensa, le indicaba siete antihipertensivos a la vez.
#
# Lo que queda acá se sigue usando, con otro propósito:
#
#   motivo_de_exclusion()   verifica lo que la receta indica y filtra las
#   dosis_maxima_para()     alternativas del mismo grupo terapéutico.
#   HORARIOS_POR_NUMERO...  respaldo cuando la receta no fija horarios.
#
# medicamentos_para() y medicamentos_excluidos_para() se conservan porque
# documentan el planteamiento original y sus pruebas siguen siendo válidas como
# registro del experimento, pero NINGÚN agente las llama ya.

from typing import Optional

from medicacion.datos import cargar_medicamentos

# Factor por el que se multiplica el tope diario cuando la persona tiene función
# renal reducida. Valor ilustrativo: en la práctica el ajuste depende del
# fármaco y del grado de insuficiencia, no es un porcentaje único.
FACTOR_AJUSTE_RENAL = 0.5

# A qué hora se reparten las tomas del día. El tutor propuso explícitamente
# "8:00 de la mañana, 3 de la tarde y 8 de la noche" para tres tomas; el resto
# mantiene esa lógica de repartir en el horario de vigilia.
HORARIOS_POR_NUMERO_DE_TOMAS = {
    1: ["08:00"],
    2: ["08:00", "20:00"],
    3: ["08:00", "15:00", "20:00"],
    4: ["08:00", "13:00", "18:00", "23:00"],
}

# Cuando el JSON no trae `tomas_por_dia` o trae un valor fuera de la tabla.
HORARIOS_POR_DEFECTO = ["08:00"]


def _tiene_alergia(medicamento: dict, alergias: list[str]) -> bool:
    """El vademécum marca el grupo al que pertenece cada fármaco alergénico
    (`grupo_alergia`). Se compara contra las alergias declaradas del perfil."""
    grupo = medicamento.get("grupo_alergia")
    return bool(grupo) and grupo in alergias


def motivo_de_exclusion(medicamento: dict, perfil: dict) -> Optional[str]:
    """Por qué este medicamento NO es apto para esta persona, o None si lo es.

    Función pura: se puede probar sin LLM, sin red y sin el grafo. Devolver el
    motivo y no un booleano es deliberado — el sistema tiene que poder explicar
    la exclusión, no solo aplicarla.
    """
    condiciones = perfil.get("condiciones", [])

    if _tiene_alergia(medicamento, perfil.get("alergias", [])):
        return f"la persona es alérgica a {medicamento['grupo_alergia']}"

    contraindicado = set(medicamento.get("contraindicado_en", [])) & set(condiciones)
    if contraindicado:
        return f"está contraindicado con {', '.join(sorted(contraindicado))}"

    return None


def dosis_maxima_para(medicamento: dict, perfil: dict) -> float:
    """Tope diario ajustado al perfil.

    El ajuste por función renal es la regla que el tutor pidió explícitamente
    ("que no me devuelva lo que me tengo que tomar con un exceso de gramos").
    """
    maximo = medicamento["max_dosis_diaria_mg"]
    if perfil.get("funcion_renal") == "reducida" and "insuficiencia_renal" in medicamento.get("reducir_dosis_si", []):
        return round(maximo * FACTOR_AJUSTE_RENAL, 3)
    return maximo


def horarios_de(medicamento: dict) -> list[str]:
    """A qué horas tomarlo, según cuántas tomas diarias requiere."""
    return HORARIOS_POR_NUMERO_DE_TOMAS.get(
        medicamento.get("tomas_por_dia"), HORARIOS_POR_DEFECTO
    )


def plan_de(medicamento: dict, perfil: dict) -> dict:
    """Todo lo que hay que saber para tomar este medicamento: dosis por toma,
    tope diario ajustado, horarios y si va con comida."""
    maximo = dosis_maxima_para(medicamento, perfil)
    por_toma = medicamento["dosis_mg"]
    horarios = horarios_de(medicamento)

    # Si el tope bajó por ajuste renal, las tomas previstas pueden superarlo.
    # Se recorta el número de tomas antes que la dosis unitaria, porque partir
    # comprimidos no siempre es posible.
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
    """Los medicamentos aptos para esta persona, con su plan de toma.

    `condicion` acota a una sola de sus condiciones; sin ella, cubre todas las
    del perfil. Un perfil sin condiciones devuelve lista vacía: el sistema no
    le inventa tratamientos a quien no declaró ninguna.
    """
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
    """Los que SÍ tratan la condición pero no se pueden indicar, con el motivo.

    Existe para que el sistema pueda explicar por qué no recomienda algo, en vez
    de omitirlo en silencio. El caso de Jorge —úlcera y artrosis— es el ejemplo:
    casi todo lo que alivia la artrosis está contraindicado con úlcera.
    """
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
