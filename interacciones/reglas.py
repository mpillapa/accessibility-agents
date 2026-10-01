# EL CRUCE medicamento × comida: la regla de negocio de esta tarea.
#
# Dada una persona y las recetas de cocina que recuperó el RAG, ¿qué
# medicamentos vigentes de la persona interactúan con qué alimentos del plato?
#
# Es código determinista a propósito, igual que la verificación de la
# prescripción (medicacion/prescripciones.py): lo que puede hacer daño no lo
# decide el modelo. El LLM del nodo integrador solo redacta lo que sale de acá.
#
# Reglas, en orden:
#   1. Sin prescripción no hay cruce posible: NO EVALUABLE. Decir "no hay
#      problema" sin saber qué toma la persona es inventar una respuesta.
#   2. Sin receta reconocida tampoco: NO EVALUABLE.
#   3. Solo cuentan las prescripciones VIGENTES a la fecha.
#   4. Una interacción aplica si el medicamento está en la prescripción
#      vigente, la categoría está entre los alimentos marcados de la receta y
#      se cumple su condición `solo_si` sobre el perfil (si la tiene).
#   5. Si el RAG devolvió varias recetas, se cruzan todas y cada interacción
#      dice de qué receta sale.

from datetime import date
from typing import Optional

from interacciones.datos import alimentos_marcados_de, interacciones_de, medicamentos_revisados
from medicacion.datos import obtener_perfil, prescripciones_de
from medicacion.prescripciones import esta_vigente

MOTIVO_SIN_PERFIL = "sin_perfil"
MOTIVO_SIN_PRESCRIPCION = "sin_prescripcion"
MOTIVO_SIN_PRESCRIPCION_VIGENTE = "sin_prescripcion_vigente"
MOTIVO_SIN_RECETA = "sin_receta"

VEREDICTO_EVITAR = "evitar"
VEREDICTO_PRECAUCION = "precaucion"
VEREDICTO_SIN_INTERACCION = "sin_interaccion"
VEREDICTO_NO_EVALUABLE = "no_evaluable"


def medicamentos_vigentes(id_perfil: str, hoy: Optional[date] = None) -> list[str]:
    """Los medicamentos de las prescripciones vigentes, sin repetir y en orden."""
    nombres = [
        indicacion["medicamento"]
        for prescripcion in prescripciones_de(id_perfil)
        if esta_vigente(prescripcion, hoy)
        for indicacion in prescripcion["indicaciones"]
    ]
    return list(dict.fromkeys(nombres))


def _cumple_condicion(solo_si: Optional[dict], perfil: dict) -> bool:
    return all(perfil.get(campo) == valor for campo, valor in (solo_si or {}).items())


def _veredicto(encontradas: list[dict]) -> str:
    """El más grave de los encontrados: basta una interacción 'evitar' para
    que el veredicto sea evitar."""
    severidades = {i["severidad"] for i in encontradas}
    if VEREDICTO_EVITAR in severidades:
        return VEREDICTO_EVITAR
    if VEREDICTO_PRECAUCION in severidades:
        return VEREDICTO_PRECAUCION
    return VEREDICTO_SIN_INTERACCION


def cruzar(id_perfil: str, fuentes: list[str], hoy: Optional[date] = None,
           medicamentos: Optional[list[str]] = None) -> dict:
    """El resultado del cruce, como datos.

    Devuelve:
    `medicamentos`: los vigentes, si ya los calculó la rama de medicación del
    grafo. Si no se pasan, se calculan aquí con la misma regla.

      evaluable          False si falta la persona, su prescripción o la receta
      veredicto          evitar | precaucion | sin_interaccion | no_evaluable
      motivo             por qué no es evaluable (None si lo es)
      hay_interaccion    True si al menos una interacción aplica
      interacciones      lista de {medicamento, categoria, severidad, motivo,
                         recomendacion, fuente, alimentos: [{alimento, parte, plato}]}
      medicamentos       los vigentes que se cruzaron
      sin_revisar        medicamentos vigentes que el catálogo no cubre (debería
                         ser vacío: lo garantiza prueba_datos_interacciones)
      fuentes_cruzadas   las recetas que se pudieron cruzar
      fuentes_desconocidas  las que no están en ingredientes_recetas.json
    """
    base = {
        "evaluable": False, "motivo": None, "veredicto": VEREDICTO_NO_EVALUABLE,
        "hay_interaccion": False, "interacciones": [],
        "medicamentos": [], "sin_revisar": [], "fuentes_cruzadas": [], "fuentes_desconocidas": [],
    }

    perfil = obtener_perfil(id_perfil) if id_perfil else None
    if perfil is None:
        return {**base, "motivo": MOTIVO_SIN_PERFIL}
    if not prescripciones_de(id_perfil):
        return {**base, "motivo": MOTIVO_SIN_PRESCRIPCION}

    if medicamentos is None:
        medicamentos = medicamentos_vigentes(id_perfil, hoy)
    if not medicamentos:
        return {**base, "motivo": MOTIVO_SIN_PRESCRIPCION_VIGENTE}

    alimentos_por_fuente = {f: alimentos_marcados_de(f) for f in dict.fromkeys(fuentes or [])}
    conocidas = [f for f, a in alimentos_por_fuente.items() if a is not None]
    desconocidas = [f for f, a in alimentos_por_fuente.items() if a is None]
    base.update(
        medicamentos=medicamentos,
        sin_revisar=[m for m in medicamentos if m not in medicamentos_revisados()],
        fuentes_cruzadas=conocidas,
        fuentes_desconocidas=desconocidas,
    )
    if not conocidas:
        return {**base, "motivo": MOTIVO_SIN_RECETA}

    encontradas = []
    for fuente in conocidas:
        alimentos = alimentos_por_fuente[fuente]
        for medicamento in medicamentos:
            for regla in interacciones_de(medicamento):
                if not _cumple_condicion(regla["solo_si"], perfil):
                    continue
                coinciden = [
                    {"alimento": a["alimento"], "parte": a["parte"], "plato": a["plato"]}
                    for a in alimentos if a["categoria"] == regla["categoria"]
                ]
                if coinciden:
                    encontradas.append({
                        "medicamento": medicamento,
                        "categoria": regla["categoria"],
                        "severidad": regla["severidad"],
                        "motivo": regla["motivo"],
                        "recomendacion": regla["recomendacion"],
                        "fuente": fuente,
                        "alimentos": coinciden,
                    })

    return {**base, "evaluable": True, "veredicto": _veredicto(encontradas),
            "hay_interaccion": bool(encontradas), "interacciones": encontradas}


def pares(resultado: dict) -> set[tuple[str, str]]:
    """Los pares (medicamento, categoría) de un resultado, para compararlo con
    la verdad de referencia sin importar de qué receta salió cada uno."""
    return {(i["medicamento"], i["categoria"]) for i in resultado["interacciones"]}
