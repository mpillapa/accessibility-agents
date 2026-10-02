# Cruce medicamento × comida, en código determinista: el LLM del integrador solo redacta.
# Reglas: sin perfil, prescripción vigente o receta reconocida, NO EVALUABLE; una interacción
# aplica si el medicamento es vigente, la categoría está en el plato y se cumple `solo_si`.

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
    """Medicamentos de las prescripciones vigentes, sin repetir y en orden."""
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
    """El más grave: basta una interacción 'evitar'."""
    severidades = {i["severidad"] for i in encontradas}
    if VEREDICTO_EVITAR in severidades:
        return VEREDICTO_EVITAR
    if VEREDICTO_PRECAUCION in severidades:
        return VEREDICTO_PRECAUCION
    return VEREDICTO_SIN_INTERACCION


def cruzar(id_perfil: str, fuentes: list[str], hoy: Optional[date] = None,
           medicamentos: Optional[list[str]] = None) -> dict:
    """El cruce como datos; los campos son los de `base`. Se cruzan todas las fuentes dadas.

    `medicamentos`: los vigentes si ya los calculó la rama de medicación; si no, se calculan aquí.
    `sin_revisar` debería quedar vacío (lo verifica prueba_datos_interacciones).
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
    """Pares (medicamento, categoría), para comparar con la referencia sin importar la fuente."""
    return {(i["medicamento"], i["categoria"]) for i in resultado["interacciones"]}
