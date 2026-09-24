# La receta médica como FUENTE DE VERDAD, en código determinista.
#
# POR QUÉ SE REESCRIBIÓ ESTE MÓDULO
# ---------------------------------
# La primera versión del agente filtraba el vademécum por las condiciones de la
# persona y presentaba el resultado como su tratamiento. Medido sobre los seis
# perfiles, eso produjo respuestas clínicamente absurdas en LAS DOS variantes
# (reglas y LLM): a Carmen, hipertensa, le indicaba SIETE antihipertensivos a la
# vez —tres del mismo eje y dos betabloqueantes— con el texto "tome 7 pastillas,
# una de cada medicamento".
#
# El error no era del modelo ni del filtro: era del planteamiento. Un catálogo
# filtrado por condición devuelve OPCIONES ELEGIBLES, no un RÉGIMEN. Convertir
# lo primero en lo segundo exige saber de interacciones, de clases terapéuticas
# y de titulación de dosis, que es exactamente el conocimiento clínico que este
# prototipo no tiene ni puede validar.
#
# La receta resuelve eso invirtiendo la responsabilidad: el médico decide, el
# sistema lee, organiza y explica. Lo que el sistema aporta es lo que sí puede
# hacer sin criterio clínico —consolidar varias recetas en un plan por horario,
# explicarlo en lenguaje llano y VERIFICAR la coherencia de lo indicado— y nada
# más.
#
# LA REGLA QUE GOBIERNA TODO ESTE MÓDULO
# --------------------------------------
# El sistema NUNCA quita nada del plan. Si detecta un problema —una alergia, una
# contraindicación, una dosis por encima del tope ajustado, una receta vencida—
# lo MARCA y lo avisa, pero la indicación sigue apareciendo.
#
# Suspender un tratamiento es una decisión clínica igual que recetarlo, y una
# persona mayor que deja de tomar su antihipertensivo porque una aplicación se
# lo ocultó corre más riesgo que una a la que se le advierte. Es el mismo
# criterio del guardrail de voz (orquestacion_langgraph/voz.py): ante duda,
# avisar en vez de decidir por cuenta propia.
#
# LIMITACIONES QUE HAY QUE DECLARAR
# ---------------------------------
# La validación cubre cuatro criterios sobre datos ficticios: alergia declarada,
# contraindicación por condición, tope diario con ajuste renal e integridad de
# los datos de la receta. NO modela interacciones entre fármacos, función
# hepática, peso, edad ni duplicidad terapéutica. Una receta que pase las cuatro
# validaciones NO está clínicamente validada.

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

# Tipos de aviso que puede emitir la verificación. Son constantes y no texto
# suelto porque las pruebas y el evaluador comparativo cuentan por tipo.
AVISO_ALERGIA = "alergia"
AVISO_CONTRAINDICACION = "contraindicacion"
AVISO_EXCEDE_TOPE = "excede_tope"
AVISO_MEDICAMENTO_DESCONOCIDO = "medicamento_desconocido"
AVISO_DATOS_INCONSISTENTES = "datos_inconsistentes"
AVISO_PRESCRIPCION_VENCIDA = "prescripcion_vencida"

GRAVEDAD_ALTA = "alta"
GRAVEDAD_MEDIA = "media"

# Qué hacer ante cada tipo de aviso. NO todos los problemas piden lo mismo, y
# tratarlos igual es un error que este proyecto cometió y midió: la primera
# versión respondía "acá está su medicación, consulte a su médico" tanto para una
# dosis un poco alta como para un antibiótico al que la persona es ALÉRGICA. La
# variante LLM del experimento, en cambio, dijo sin rodeos "no la tome, llame hoy
# mismo al médico", y para una alergia a penicilina eso es lo correcto.
#
# Que la acción salga de esta tabla y no del criterio del modelo es justamente el
# punto: es una decisión de diseño explícita, revisable y probada, no algo que
# cambie entre dos ejecuciones.
#
# Sigue valiendo la regla del módulo: el sistema NUNCA borra la indicación del
# plan. Cambia lo que recomienda hacer con ella, no si la muestra.
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
    """A qué horas se toma una indicación.

    Los horarios de la receta MANDAN: los fijó el médico y pueden no seguir
    ninguna regla general ("no la tome de noche porque le da ganas de orinar").
    La tabla de reglas.py es solo el respaldo para recetas que no los traigan.
    """
    horarios = indicacion.get("horarios")
    if horarios:
        return list(horarios)
    return HORARIOS_POR_NUMERO_DE_TOMAS.get(
        indicacion.get("tomas_por_dia"), HORARIOS_POR_DEFECTO
    )


def total_diario_mg(indicacion: dict) -> float:
    """Miligramos que suma la indicación en un día.

    Se calcula sobre los horarios efectivos, no sobre `tomas_por_dia`: lo que la
    persona va a tomar es lo que está agendado. Si ambos campos no coinciden, la
    verificación lo reporta como dato inconsistente.
    """
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
    """Qué problemas tiene esta indicación para esta persona.

    Función pura sobre datos: se prueba sin LLM, sin red y sin el grafo. Devuelve
    una lista porque una misma indicación puede tener más de un problema.
    """
    nombre = indicacion["medicamento"]
    medicamento = buscar_medicamento(nombre)

    # Un nombre que no está en el vademécum no se puede verificar. Decirlo es
    # obligatorio: callarlo equivaldría a dar por buena una indicación que el
    # sistema jamás revisó.
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
    """El día de esta persona, hora por hora, según sus recetas.

    Consolida TODAS sus prescripciones —una persona puede tener a la vez un
    tratamiento crónico y uno agudo— y agrupa por horario, que es como se vive:
    nadie toma "la receta A y la receta B", toma lo que le toca a las ocho.

    Ninguna indicación se omite, ni siquiera las que disparan un aviso. Ver la
    nota al inicio del módulo sobre por qué el sistema marca en vez de quitar.
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

    # Un índice medicamento -> aviso más grave, para poder marcar cada renglón
    # del plan y no solo encabezar la respuesta con una lista de advertencias.
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
    """Qué otras opciones hay del mismo grupo que un medicamento de su receta.

    Responde al caso "se me acabó la pastilla". El sistema NO sustituye: devuelve
    qué existe en la misma categoría terapéutica para que la persona lo consulte
    con su médico o su farmacéutico. Cambiar un fármaco por otro es una decisión
    clínica, y la equivalencia dentro de una categoría es aproximada: mismo grupo
    no significa misma potencia, misma dosis ni mismo perfil de efectos.

    Por eso las dosis que devuelve se llaman `dosis_referencia_mg` y no
    `dosis_mg`: son lo que figura en el vademécum, NO una pauta para esta
    persona. Solo el médico puede fijar la pauta de un cambio.
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

    # Que esté o no en su receta cambia la respuesta: si nunca se lo indicaron,
    # el sistema no tiene por qué ponerse a hablar de reemplazos.
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
