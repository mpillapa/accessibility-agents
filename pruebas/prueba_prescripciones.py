# Pruebas de la receta médica como fuente de verdad (ver medicacion/prescripciones.py).
# Uso: python -m pruebas.prueba_prescripciones
# No requiere VPN.

import sys
from datetime import date

from medicacion.agente import (
    CONSULTA_ALTERNATIVAS,
    CONSULTA_PLAN,
    clasificar_consulta,
)
from medicacion.datos import obtener_perfil, prescripciones_de
from medicacion.prescripciones import (
    ACCION_POR_TIPO,
    AVISO_ALERGIA,
    AVISO_CONTRAINDICACION,
    AVISO_EXCEDE_TOPE,
    AVISO_MEDICAMENTO_DESCONOCIDO,
    AVISO_PRESCRIPCION_VENCIDA,
    GRAVEDAD_ALTA,
    alternativas_para,
    esta_vigente,
    horarios_de_indicacion,
    plan_diario,
    total_diario_mg,
    verificar_indicacion,
)

# Anterior a todo vencimiento: "vencida" depende de la fecha dada, no del día en que se corre.
HOY_FIJO = date(2026, 9, 22)
MUY_EN_EL_FUTURO = date(2030, 1, 1)


def _tipos(avisos):
    return {a["tipo"] for a in avisos}


def _medicamentos_del_plan(plan):
    return {i["medicamento"] for t in plan["tomas"] for i in t["items"]}


def prueba_consolida_varias_recetas_en_un_plan():
    """Manuel tiene una receta crónica y una aguda: se consolidan por horario."""
    assert len(prescripciones_de("manuel")) == 2
    plan = plan_diario("manuel", HOY_FIJO)
    medicamentos = _medicamentos_del_plan(plan)
    assert {"Paracetamol", "Omeprazol", "Amoxicilina"} == medicamentos, medicamentos
    return "consolida las 2 recetas de Manuel en un solo plan del día"


def prueba_agrupa_por_hora_y_en_orden():
    plan = plan_diario("manuel", HOY_FIJO)
    horas = [t["hora"] for t in plan["tomas"]]
    assert horas == sorted(horas), horas
    a_las_ocho = {i["medicamento"] for i in plan["tomas"][0]["items"]}
    assert plan["tomas"][0]["hora"] == "08:00"
    assert a_las_ocho == {"Paracetamol", "Omeprazol", "Amoxicilina"}, a_las_ocho
    return f"agrupa por hora y en orden: {horas}"


def prueba_los_horarios_de_la_receta_mandan():
    """Furosemida a las 08:00 y 16:00 por indicación médica, no 08:00 y 20:00 de la tabla."""
    receta = prescripciones_de("carmen")[0]
    furosemida = next(i for i in receta["indicaciones"] if i["medicamento"] == "Furosemida")
    assert horarios_de_indicacion(furosemida) == ["08:00", "16:00"]
    return "respeta los horarios que fijó el médico en vez de la tabla general"


def prueba_horarios_por_defecto_cuando_la_receta_no_los_trae():
    sin_horarios = {"medicamento": "Paracetamol", "dosis_mg": 500, "tomas_por_dia": 3}
    assert horarios_de_indicacion(sin_horarios) == ["08:00", "15:00", "20:00"]
    return "deriva los horarios de tomas_por_dia solo si la receta no los fija"


def prueba_total_diario_se_calcula_sobre_los_horarios_reales():
    receta = prescripciones_de("carmen")[0]
    furosemida = next(i for i in receta["indicaciones"] if i["medicamento"] == "Furosemida")
    assert total_diario_mg(furosemida) == 80, total_diario_mg(furosemida)
    return "suma el día sobre los horarios agendados (40 mg x 2 = 80 mg)"


def prueba_detecta_alergia_en_la_receta():
    """Manuel es alérgico a penicilina y su receta aguda trae Amoxicilina."""
    avisos = plan_diario("manuel", HOY_FIJO)["avisos"]
    alergia = [a for a in avisos if a["tipo"] == AVISO_ALERGIA]
    assert len(alergia) == 1, avisos
    assert alergia[0]["medicamento"] == "Amoxicilina"
    assert alergia[0]["gravedad"] == GRAVEDAD_ALTA
    return "detecta la alergia a penicilina en la receta de Manuel"


def prueba_detecta_contraindicacion_en_la_receta():
    """Naproxeno con úlcera gástrica: error de prescripción clásico."""
    avisos = plan_diario("jorge", HOY_FIJO)["avisos"]
    contra = [a for a in avisos if a["tipo"] == AVISO_CONTRAINDICACION]
    assert len(contra) == 1, avisos
    assert contra[0]["medicamento"] == "Naproxeno"
    return "detecta el antiinflamatorio contraindicado con la úlcera de Jorge"


def prueba_detecta_exceso_sobre_el_tope_ajustado_por_rinon():
    """No recorta la receta: avisa que no cuadra."""
    avisos = plan_diario("carmen", HOY_FIJO)["avisos"]
    exceso = [a for a in avisos if a["tipo"] == AVISO_EXCEDE_TOPE]
    assert len(exceso) == 1, avisos
    assert exceso[0]["medicamento"] == "Furosemida"
    assert "80" in exceso[0]["detalle"] and "40" in exceso[0]["detalle"], exceso[0]
    assert "renal" in exceso[0]["detalle"], exceso[0]
    return "detecta 80 mg indicados contra un tope de 40 mg por función renal"


def prueba_receta_limpia_no_genera_avisos():
    """Rosa y Elena tienen recetas coherentes con su perfil."""
    for id_perfil in ("rosa", "elena"):
        avisos = plan_diario(id_perfil, HOY_FIJO)["avisos"]
        assert avisos == [], (id_perfil, avisos)
    return "no avisa nada sobre las recetas limpias de Rosa y Elena"


def prueba_medicamento_desconocido_no_pasa_en_silencio():
    """Callarlo daría por buena una indicación que nunca se revisó."""
    indicacion = {"medicamento": "Pastilla del abuelo", "dosis_mg": 100, "tomas_por_dia": 1}
    avisos = verificar_indicacion(indicacion, obtener_perfil("rosa"), "rx-test")
    assert _tipos(avisos) == {AVISO_MEDICAMENTO_DESCONOCIDO}, avisos
    assert avisos[0]["gravedad"] == GRAVEDAD_ALTA
    return "avisa cuando no puede verificar un medicamento que no conoce"


def prueba_cada_aviso_dice_que_hacer():
    """Un aviso sin acción obliga a la persona a decidir sola qué hacer con él."""
    for id_perfil in ("manuel", "carmen", "jorge"):
        avisos = plan_diario(id_perfil, HOY_FIJO)["avisos"]
        assert avisos, id_perfil
        assert all(a["accion"] for a in avisos), (id_perfil, avisos)
    return "los avisos de los 3 perfiles traen qué hacer, no solo qué pasa"


def prueba_la_accion_depende_del_tipo_de_problema():
    """Alergia: no lo tome. Dosis alta: consulte antes de la próxima toma."""
    alergia = ACCION_POR_TIPO[AVISO_ALERGIA]
    exceso = ACCION_POR_TIPO[AVISO_EXCEDE_TOPE]
    assert "NO tomarlo" in alergia, alergia
    assert "NO tomarlo" not in exceso, exceso
    assert alergia != exceso

    manuel = plan_diario("manuel", HOY_FIJO)
    amoxicilina = [i for t in manuel["tomas"] for i in t["items"]
                   if i["medicamento"] == "Amoxicilina"]
    assert amoxicilina and all("NO tomarlo" in i["que_hacer"] for i in amoxicilina), amoxicilina

    carmen = plan_diario("carmen", HOY_FIJO)
    furosemida = [i for t in carmen["tomas"] for i in t["items"]
                  if i["medicamento"] == "Furosemida"]
    assert furosemida and all("consultar" in i["que_hacer"] for i in furosemida), furosemida
    return "la alergia pide no tomarlo; el exceso de dosis, consultar antes"


def prueba_la_receta_vencida_no_pide_suspender():
    """Dejar un antihipertensivo porque venció el papel es peor que el papel vencido."""
    accion = ACCION_POR_TIPO[AVISO_PRESCRIPCION_VENCIDA]
    assert "NO suspender" in accion, accion
    return "una receta vencida pide renovarla, no dejar el tratamiento"


# Regla del módulo: marcar, nunca quitar.

def prueba_no_quita_del_plan_lo_que_tiene_aviso():
    """Suspender es tan clínico como recetar: el sistema marca, decide el médico."""
    for id_perfil, medicamento in (("manuel", "Amoxicilina"),
                                   ("jorge", "Naproxeno"),
                                   ("carmen", "Furosemida")):
        plan = plan_diario(id_perfil, HOY_FIJO)
        assert plan["avisos"], f"{id_perfil} debería tener avisos"
        assert medicamento in _medicamentos_del_plan(plan), (id_perfil, medicamento)
    return "los 3 medicamentos con aviso siguen apareciendo en el plan"


def prueba_marca_el_renglon_y_no_solo_el_encabezado():
    """El aviso viaja con el medicamento: la persona lee la lista, no el bloque de arriba."""
    plan = plan_diario("jorge", HOY_FIJO)
    naproxeno = [i for t in plan["tomas"] for i in t["items"] if i["medicamento"] == "Naproxeno"]
    assert naproxeno, plan["tomas"]
    assert all(i["aviso"] and "ulcera" in i["aviso"] for i in naproxeno), naproxeno
    omeprazol = [i for t in plan["tomas"] for i in t["items"] if i["medicamento"] == "Omeprazol"]
    assert all(i["aviso"] is None for i in omeprazol), omeprazol
    return "marca el renglón del Naproxeno y deja limpio el del Omeprazol"


def prueba_receta_vencida_se_avisa_pero_sigue_en_el_plan():
    plan = plan_diario("rosa", MUY_EN_EL_FUTURO)
    assert AVISO_PRESCRIPCION_VENCIDA in _tipos(plan["avisos"]), plan["avisos"]
    assert "Metformina" in _medicamentos_del_plan(plan), plan["tomas"]
    assert not esta_vigente(prescripciones_de("rosa")[0], MUY_EN_EL_FUTURO)
    assert esta_vigente(prescripciones_de("rosa")[0], HOY_FIJO)
    return "avisa que la receta venció sin borrar el tratamiento crónico"


def prueba_los_avisos_graves_van_primero():
    plan = plan_diario("carmen", MUY_EN_EL_FUTURO)
    gravedades = [a["gravedad"] for a in plan["avisos"]]
    assert gravedades == sorted(gravedades, key=lambda g: 0 if g == GRAVEDAD_ALTA else 1)
    return "ordena los avisos por gravedad"


def prueba_sin_receta_no_inventa_plan():
    """Luis no tiene receta: es donde un LLM tiende a rellenar."""
    plan = plan_diario("luis", HOY_FIJO)
    assert plan["tiene_prescripcion"] is False
    assert plan["tomas"] == []
    assert plan["avisos"] == []
    return "no le arma un plan a quien no tiene receta"


def prueba_persona_desconocida_no_devuelve_plan():
    plan = plan_diario("no_existe", HOY_FIJO)
    assert plan["perfil"] is None and plan["tomas"] == []
    return "no devuelve plan para una persona que no está en el sistema"


# Alternativas: informar, no sustituir.

def prueba_alternativas_son_del_mismo_grupo():
    resultado = alternativas_para("Paracetamol", "manuel")
    assert resultado["encontrado"] and resultado["en_su_receta"]
    assert resultado["categoria"] == "analgesico"
    nombres = {a["nombre"] for a in resultado["del_mismo_grupo"]}
    assert "Paracetamol" not in nombres, nombres
    assert nombres == {"Metamizol", "Tramadol"}, nombres
    return f"para el Paracetamol ofrece {sorted(nombres)}, del mismo grupo"


def prueba_alternativas_descartan_lo_contraindicado():
    """A Jorge se le acaba el Naproxeno: los otros antiinflamatorios están
    contraindicados con su úlcera y no deben aparecer como opción."""
    resultado = alternativas_para("Naproxeno", "jorge")
    ofrecidas = {a["nombre"] for a in resultado["del_mismo_grupo"]}
    descartadas = {d["nombre"] for d in resultado["descartadas"]}
    assert {"Ibuprofeno", "Diclofenaco"} <= descartadas, descartadas
    assert not ({"Ibuprofeno", "Diclofenaco"} & ofrecidas), ofrecidas
    assert all(d["motivo"] for d in resultado["descartadas"])
    return f"descarta {sorted(descartadas)} por la úlcera de Jorge, con motivo"


def prueba_las_alternativas_no_son_una_sustitucion():
    """Autorizar el cambio es del médico; las dosis se llaman "de catálogo" para no leerse como pauta."""
    resultado = alternativas_para("Metformina", "rosa")
    assert resultado["requiere_autorizacion_medica"] is True
    for alternativa in resultado["del_mismo_grupo"]:
        assert "dosis_mg" not in alternativa, alternativa
        assert "dosis_referencia_mg" in alternativa, alternativa
        assert "horarios" not in alternativa, alternativa
    return "las alternativas no traen pauta de toma, solo dosis de catálogo"


def prueba_distingue_si_el_medicamento_esta_en_su_receta():
    """Si nunca se lo indicaron, el sistema no tiene por qué hablar de reemplazos."""
    assert alternativas_para("Paracetamol", "manuel")["en_su_receta"] is True
    assert alternativas_para("Paracetamol", "jorge")["en_su_receta"] is False
    return "distingue lo que está en su receta de lo que no"


def prueba_medicamento_inexistente_no_inventa_alternativas():
    resultado = alternativas_para("Pastilla del abuelo", "manuel")
    assert resultado["encontrado"] is False
    assert "del_mismo_grupo" not in resultado
    return "no inventa alternativas para un medicamento que no existe"


def prueba_clasificador_detecta_que_le_falta_una_pastilla():
    for consulta, esperado in (
        ("se me acabó el paracetamol", "Paracetamol"),
        ("Doctor, no tengo Metformina, ¿qué hago?", "Metformina"),
        ("no me queda naproxeno", "Naproxeno"),
    ):
        tipo, medicamento = clasificar_consulta(consulta)
        assert tipo == CONSULTA_ALTERNATIVAS, (consulta, tipo)
        assert medicamento == esperado, (consulta, medicamento)
    return "detecta el faltante y el medicamento en 3 formas de decirlo"


def prueba_clasificador_por_defecto_es_el_plan():
    """Sin faltante y medicamento nombrados, es el plan del día: lo más frecuente y menos riesgoso."""
    for consulta in ("¿qué pastillas me toca tomar hoy?",
                     "¿a qué hora tomo la furosemida?",
                     "se me acabaron las pastillas"):
        tipo, medicamento = clasificar_consulta(consulta)
        assert tipo == CONSULTA_PLAN and medicamento is None, (consulta, tipo)
    return "cae en el plan del día cuando no hay faltante con medicamento nombrado"


CASOS = [
    prueba_consolida_varias_recetas_en_un_plan,
    prueba_agrupa_por_hora_y_en_orden,
    prueba_los_horarios_de_la_receta_mandan,
    prueba_horarios_por_defecto_cuando_la_receta_no_los_trae,
    prueba_total_diario_se_calcula_sobre_los_horarios_reales,
    prueba_detecta_alergia_en_la_receta,
    prueba_detecta_contraindicacion_en_la_receta,
    prueba_detecta_exceso_sobre_el_tope_ajustado_por_rinon,
    prueba_receta_limpia_no_genera_avisos,
    prueba_medicamento_desconocido_no_pasa_en_silencio,
    prueba_cada_aviso_dice_que_hacer,
    prueba_la_accion_depende_del_tipo_de_problema,
    prueba_la_receta_vencida_no_pide_suspender,
    prueba_no_quita_del_plan_lo_que_tiene_aviso,
    prueba_marca_el_renglon_y_no_solo_el_encabezado,
    prueba_receta_vencida_se_avisa_pero_sigue_en_el_plan,
    prueba_los_avisos_graves_van_primero,
    prueba_sin_receta_no_inventa_plan,
    prueba_persona_desconocida_no_devuelve_plan,
    prueba_alternativas_son_del_mismo_grupo,
    prueba_alternativas_descartan_lo_contraindicado,
    prueba_las_alternativas_no_son_una_sustitucion,
    prueba_distingue_si_el_medicamento_esta_en_su_receta,
    prueba_medicamento_inexistente_no_inventa_alternativas,
    prueba_clasificador_detecta_que_le_falta_una_pastilla,
    prueba_clasificador_por_defecto_es_el_plan,
]


def main():
    print("Pruebas de la receta médica como fuente de verdad (sin VPN, sin LLM)\n")
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
