# Plan del día y verificación a partir de la prescripción, en código determinista.
# Regla del módulo: marcar los problemas, nunca quitar una indicación (bitácora 16.3).
# Cuatro criterios sobre datos ficticios; no modela interacciones entre fármacos ni duplicidad.

from datetime import date
from typing import Optional

from medicacion.datos import (
    buscar_medicamento,
    cargar_medicamentos,
    obtener_perfil,
    prescripciones_de,
)
from medicacion.reglas import (
    HORARIOS_POR_DEFECTO,
    HORARIOS_POR_NUMERO_DE_TOMAS,
    dosis_maxima_para,
    motivo_de_exclusion,
)

# Constantes porque las pruebas y el evaluador cuentan avisos por tipo.
AVISO_ALERGIA = "alergia"
AVISO_CONTRAINDICACION = "contraindicacion"
AVISO_EXCEDE_TOPE = "excede_tope"
AVISO_MEDICAMENTO_DESCONOCIDO = "medicamento_desconocido"
AVISO_DATOS_INCONSISTENTES = "datos_inconsistentes"
AVISO_PRESCRIPCION_VENCIDA = "prescripcion_vencida"

GRAVEDAD_ALTA = "alta"
GRAVEDAD_MEDIA = "media"

# Regla de negocio: qué recomendar ante cada aviso. Sale de esta tabla, no del
# modelo (bitácora 16.7). Cambia la recomendación, nunca si la indicación se muestra.
ACCION_POR_TIPO = {
    AVISO_ALERGIA: "NO tomarlo y llamar hoy mismo al médico para que lo reemplace",
    AVISO_CONTRAINDICACION: "NO tomarlo sin hablar antes con el médico",
    AVISO_EXCEDE_TOPE: "consultar al médico antes de la próxima toma",
    AVISO_MEDICAMENTO_DESCONOCIDO: "confirmar esta indicación con el médico o el farmacéutico",
    AVISO_DATOS_INCONSISTENTES: "confirmar con el médico cuántas tomas son",
    AVISO_PRESCRIPCION_VENCIDA: "pedir al médico que renueve la receta; mientras tanto NO suspender el tratamiento",
}

_GRAVEDAD_POR_TIPO = {
    AVISO_ALERGIA: GRAVEDAD_ALTA,
    AVISO_CONTRAINDICACION: GRAVEDAD_ALTA,
    AVISO_EXCEDE_TOPE: GRAVEDAD_ALTA,
    AVISO_MEDICAMENTO_DESCONOCIDO: GRAVEDAD_ALTA,
    AVISO_DATOS_INCONSISTENTES: GRAVEDAD_MEDIA,
    AVISO_PRESCRIPCION_VENCIDA: GRAVEDAD_MEDIA,
}


def horarios_de_indicacion(indicacion: dict) -> list[str]:
    """Mandan los horarios de la receta; la tabla de reglas.py es solo respaldo."""
    horarios = indicacion.get("horarios")
    if horarios:
        return list(horarios)
    return HORARIOS_POR_NUMERO_DE_TOMAS.get(
        indicacion.get("tomas_por_dia"), HORARIOS_POR_DEFECTO
    )


def total_diario_mg(indicacion: dict) -> float:
    """Miligramos al día, sobre los horarios agendados y no sobre `tomas_por_dia`."""
    return indicacion["dosis_mg"] * len(horarios_de_indicacion(indicacion))


def _aviso(tipo: str, medicamento: str, detalle: str, id_prescripcion: str) -> dict:
    return {
        "tipo": tipo,
        "gravedad": _GRAVEDAD_POR_TIPO[tipo],
        "accion": ACCION_POR_TIPO[tipo],
        "medicamento": medicamento,
        "detalle": detalle,
        "prescripcion": id_prescripcion,
    }


def esta_vigente(prescripcion: dict, hoy: Optional[date] = None) -> bool:
    """Si la receta sigue vigente a la fecha dada (hoy por defecto)."""
    vigente_hasta = prescripcion.get("vigente_hasta")
    if not vigente_hasta:
        return True
    return (hoy or date.today()) <= date.fromisoformat(vigente_hasta)


def verificar_indicacion(indicacion: dict, perfil: dict, id_prescripcion: str) -> list[dict]:
    """Avisos de esta indicación para esta persona (puede haber varios)."""
    nombre = indicacion["medicamento"]
    medicamento = buscar_medicamento(nombre)

    # Fuera del vademécum no se puede verificar, y callarlo sería darla por buena.
    if medicamento is None:
        return [_aviso(
            AVISO_MEDICAMENTO_DESCONOCIDO,
            nombre,
            f"'{nombre}' no está en la base de datos del sistema, así que no se pudo verificar",
            id_prescripcion,
        )]

    avisos = []

    motivo = motivo_de_exclusion(medicamento, perfil)
    if motivo:
        tipo = AVISO_ALERGIA if "alérgic" in motivo else AVISO_CONTRAINDICACION
        avisos.append(_aviso(tipo, nombre, motivo, id_prescripcion))

    tope = dosis_maxima_para(medicamento, perfil)
    total = total_diario_mg(indicacion)
    if total > tope:
        sin_ajustar = medicamento["max_dosis_diaria_mg"]
        detalle = f"la receta indica {total:g} mg al día y el máximo es {tope:g} mg"
        if tope != sin_ajustar:
            detalle += f" (reducido desde {sin_ajustar:g} mg por la función renal)"
        avisos.append(_aviso(AVISO_EXCEDE_TOPE, nombre, detalle, id_prescripcion))

    horarios = indicacion.get("horarios")
    tomas = indicacion.get("tomas_por_dia")
    if horarios and tomas and len(horarios) != tomas:
        avisos.append(_aviso(
            AVISO_DATOS_INCONSISTENTES,
            nombre,
            f"la receta dice {tomas} tomas al día pero lista {len(horarios)} horarios",
            id_prescripcion,
        ))

    return avisos


def verificar_prescripcion(prescripcion: dict, perfil: dict, hoy: Optional[date] = None) -> list[dict]:
    """Todos los avisos de una receta completa, incluida su vigencia."""
    avisos = []

    if not esta_vigente(prescripcion, hoy):
        avisos.append(_aviso(
            AVISO_PRESCRIPCION_VENCIDA,
            "",
            f"la receta venció el {prescripcion['vigente_hasta']}",
            prescripcion["id"],
        ))

    for indicacion in prescripcion["indicaciones"]:
        avisos.extend(verificar_indicacion(indicacion, perfil, prescripcion["id"]))

    return avisos


def plan_diario(id_perfil: str, hoy: Optional[date] = None) -> dict:
    """Plan del día por hora, consolidando todas sus prescripciones.

    No omite ninguna indicación, ni las que tienen aviso: se marcan.
    """
    perfil = obtener_perfil(id_perfil)
    if perfil is None:
        return {"perfil": None, "tiene_prescripcion": False, "tomas": [], "avisos": []}

    recetas = prescripciones_de(id_perfil)
    if not recetas:
        return {
            "perfil": perfil,
            "tiene_prescripcion": False,
            "prescripciones": [],
            "tomas": [],
            "avisos": [],
        }

    avisos = []
    for receta in recetas:
        avisos.extend(verificar_prescripcion(receta, perfil, hoy))

    # Primer aviso grave por medicamento, para marcarlo en su propio renglón.
    aviso_por_medicamento: dict[str, str] = {}
    accion_por_medicamento: dict[str, str] = {}
    for aviso in avisos:
        if aviso["gravedad"] == GRAVEDAD_ALTA and aviso["medicamento"]:
            aviso_por_medicamento.setdefault(aviso["medicamento"], aviso["detalle"])
            accion_por_medicamento.setdefault(aviso["medicamento"], aviso["accion"])

    por_hora: dict[str, list[dict]] = {}
    for receta in recetas:
        for indicacion in receta["indicaciones"]:
            medicamento = buscar_medicamento(indicacion["medicamento"])
            for hora in horarios_de_indicacion(indicacion):
                por_hora.setdefault(hora, []).append({
                    "medicamento": indicacion["medicamento"],
                    "dosis_mg": indicacion["dosis_mg"],
                    "forma": medicamento["forma"] if medicamento else "desconocida",
                    "con_comida": indicacion.get("con_comida", False),
                    "motivo": indicacion.get("motivo"),
                    "nota_medico": indicacion.get("nota_medico"),
                    "prescripcion": receta["id"],
                    "aviso": aviso_por_medicamento.get(indicacion["medicamento"]),
                    "que_hacer": accion_por_medicamento.get(indicacion["medicamento"]),
                })

    tomas = [{"hora": hora, "items": por_hora[hora]} for hora in sorted(por_hora)]

    return {
        "perfil": perfil,
        "tiene_prescripcion": True,
        "prescripciones": [
            {
                "id": r["id"],
                "tipo": r["tipo"],
                "emitida": r["emitida"],
                "vigente_hasta": r.get("vigente_hasta"),
                "medico": r.get("medico"),
                "vigente": esta_vigente(r, hoy),
            }
            for r in recetas
        ],
        "tomas": tomas,
        "avisos": sorted(avisos, key=lambda a: 0 if a["gravedad"] == GRAVEDAD_ALTA else 1),
    }


def alternativas_para(nombre_medicamento: str, id_perfil: str) -> dict:
    """Opciones de la misma categoría para "se me acabó la pastilla". No sustituye.

    `dosis_referencia_mg` es el valor del vademécum, no una pauta para la persona.
    """
    perfil = obtener_perfil(id_perfil)
    if perfil is None:
        return {"medicamento": nombre_medicamento, "perfil_encontrado": False}

    medicamento = buscar_medicamento(nombre_medicamento)
    if medicamento is None:
        return {
            "medicamento": nombre_medicamento,
            "perfil_encontrado": True,
            "encontrado": False,
        }

    # Si nunca se lo indicaron, la respuesta no debe hablar de reemplazos.
    en_su_receta = any(
        i["medicamento"].lower() == medicamento["nombre"].lower()
        for r in prescripciones_de(id_perfil)
        for i in r["indicaciones"]
    )

    del_mismo_grupo, descartadas = [], []
    for otro in cargar_medicamentos():
        if otro["nombre"] == medicamento["nombre"]:
            continue
        if otro["categoria"] != medicamento["categoria"]:
            continue
        motivo = motivo_de_exclusion(otro, perfil)
        if motivo:
            descartadas.append({"nombre": otro["nombre"], "motivo": motivo})
        else:
            del_mismo_grupo.append({
                "nombre": otro["nombre"],
                "forma": otro["forma"],
                "dosis_referencia_mg": otro["dosis_mg"],
            })

    return {
        "medicamento": medicamento["nombre"],
        "perfil_encontrado": True,
        "encontrado": True,
        "categoria": medicamento["categoria"],
        "en_su_receta": en_su_receta,
        "del_mismo_grupo": del_mismo_grupo,
        "descartadas": descartadas,
        "requiere_autorizacion_medica": True,
    }
