# Las dos variantes que se comparan: ¿quién transmite fielmente la receta? (bitácora 16.2, 16.8)
#   VARIANTE_REGLAS: prescripciones.py arma y verifica el plan; el LLM solo lo redacta.
#   VARIANTE_LLM:    receta, perfil y vademécum van al prompt y el modelo hace todo.

import json
import unicodedata
from typing import Literal, Optional

from medicacion.datos import cargar_medicamentos, obtener_perfil, prescripciones_de
from medicacion.prescripciones import alternativas_para, plan_diario

VARIANTE_REGLAS = "reglas"
VARIANTE_LLM = "llm"
Variante = Literal["reglas", "llm"]

CONSULTA_PLAN = "plan"
CONSULTA_ALTERNATIVAS = "alternativas"

# Lista cerrada: detección determinista, pero no cubre otras formas de decirlo.
FRASES_DE_FALTANTE = (
    "se me acabo", "se me acabaron", "se acabo", "se termino", "se me termino",
    "no tengo", "no me queda", "no me quedan", "no consegui", "no encontre",
    "me falta", "se me perdio",
)

# Obligatorio en toda respuesta; se agrega en código, no en el prompt.
AVISO = (
    "\n\n_Prototipo académico con datos ficticios. No reemplaza a su médico: "
    "consulte siempre antes de cambiar su medicación._"
)

SIN_PRESCRIPCION = (
    "No tengo ninguna receta registrada a su nombre, así que no puedo decirle qué "
    "tomar. Pídale a su médico la receta y la cargamos en el sistema."
)

PERSONA_DESCONOCIDA = "No encuentro esa persona en el sistema."


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in sin_tildes if not unicodedata.combining(c)).lower()


def clasificar_consulta(consulta: str) -> tuple[str, Optional[str]]:
    """(CONSULTA_ALTERNATIVAS, nombre) si dice que le falta un medicamento del vademécum.

    Si no, (CONSULTA_PLAN, None): el plan es la consulta más frecuente y la menos riesgosa.
    """
    normalizada = _normalizar(consulta)

    mencionado = next(
        (m["nombre"] for m in cargar_medicamentos()
         if _normalizar(m["nombre"]) in normalizada),
        None,
    )
    if mencionado and any(f in normalizada for f in FRASES_DE_FALTANTE):
        return CONSULTA_ALTERNATIVAS, mencionado

    return CONSULTA_PLAN, None


def _prompt_plan_con_reglas(plan: dict, consulta: str) -> str:
    """El LLM recibe el plan YA armado y verificado. No organiza ni calcula."""
    perfil = plan["perfil"]
    return (
        "Eres un asistente que le explica a una persona mayor la medicación que "
        "le recetó su médico, en español claro y tratándola de usted.\n\n"
        "El plan de abajo SALE DE SU RECETA MÉDICA y ya fue verificado. Tu tarea "
        "es SOLO redactarlo de forma comprensible, agrupado por hora.\n\n"
        "REGLAS ESTRICTAS:\n"
        "- NO agregues ningún medicamento que no esté en el plan.\n"
        "- NO quites ninguno, ni siquiera los que tienen un aviso.\n"
        "- NO cambies dosis ni horarios.\n"
        "- NO sugieras reemplazos ni cambios de tratamiento.\n"
        "- Si un renglón trae 'aviso', dilo JUNTO A ESE MEDICAMENTO, antes de la "
        "indicación, y con claridad. No lo escondas al final.\n"
        "- Lo que hay que hacer ante un aviso viene en el campo 'que_hacer'. "
        "Transmítelo tal cual: si dice NO tomarlo, díselo sin rodeos; si dice "
        "consultar, no le digas que lo suspenda.\n"
        "- Repite la 'nota_medico' tal como está: la escribió el médico.\n\n"
        f"Persona: {perfil['nombre']}, {perfil['edad']} años.\n"
        f"PLAN DEL DÍA (de su receta):\n{json.dumps(plan['tomas'], ensure_ascii=False, indent=2)}\n\n"
        f"AVISOS DE LA VERIFICACIÓN:\n{json.dumps(plan['avisos'], ensure_ascii=False, indent=2)}\n\n"
        f"Consulta: '{consulta}'"
    )


def _prompt_alternativas_con_reglas(alternativas: dict, consulta: str) -> str:
    """Las opciones del mismo grupo, con la autorización médica como condición."""
    return (
        "Eres un asistente que le habla a una persona mayor en español claro, "
        "tratándola de usted. Se quedó sin uno de sus medicamentos.\n\n"
        "REGLAS ESTRICTAS:\n"
        "- NO le digas que tome otro en su lugar. El cambio lo autoriza SOLO su "
        "médico o su farmacéutico.\n"
        "- Preséntalo así: existen estas otras del mismo grupo, llévele esta "
        "lista a su médico o al farmacéutico para que ellos decidan.\n"
        "- NO indiques dosis para las alternativas: 'dosis_referencia_mg' es un "
        "valor de catálogo, NO una pauta para esta persona. No la menciones.\n"
        "- Si 'en_su_receta' es false, dile que ese medicamento no figura en su "
        "receta y que lo consulte antes de tomar nada.\n"
        "- Recuérdale que mientras tanto NO debe suspender el resto de su "
        "tratamiento.\n\n"
        f"DATOS:\n{json.dumps(alternativas, ensure_ascii=False, indent=2)}\n\n"
        f"Consulta: '{consulta}'"
    )


def _prompt_solo_llm(perfil: dict, recetas: list[dict], consulta: str) -> str:
    """Recibe la misma información que las reglas, salvo el resultado ya calculado."""
    return (
        "Eres un gestor de medicación que le explica a una persona mayor qué "
        "tiene que tomar, en español claro y tratándola de usted.\n\n"
        "Organiza su día por horarios a partir de su RECETA MÉDICA. Verifica "
        "contra la BASE DE DATOS que nada esté contraindicado por sus "
        "condiciones, que no haya ningún medicamento al que sea alérgica y que "
        "no se exceda el máximo de miligramos diarios, teniendo en cuenta que "
        "con función renal reducida ese máximo baja. Si encuentras algún "
        "problema, avísaselo.\n\n"
        f"PERSONA: {perfil['nombre']}, {perfil['edad']} años. "
        f"Condiciones: {', '.join(perfil['condiciones']) or 'ninguna'}. "
        f"Alergias: {', '.join(perfil['alergias']) or 'ninguna'}. "
        f"Función renal: {perfil['funcion_renal']}.\n\n"
        f"SU RECETA MÉDICA:\n{json.dumps(recetas, ensure_ascii=False)}\n\n"
        f"BASE DE DATOS DE MEDICAMENTOS:\n{json.dumps(cargar_medicamentos(), ensure_ascii=False)}\n\n"
        f"Consulta: '{consulta}'"
    )


def responder(consulta: str, id_perfil: str, variante: Variante = VARIANTE_REGLAS) -> dict:
    """Responde con la variante indicada.

    En la de reglas devuelve también el plan y los avisos, para auditar el texto contra la receta.
    """
    from orquestacion_langgraph.llm import llm

    perfil = obtener_perfil(id_perfil)
    if perfil is None:
        return {"respuesta": PERSONA_DESCONOCIDA + AVISO, "variante": variante, "perfil": None}

    recetas = prescripciones_de(id_perfil)

    # Sin receta no se llama al LLM, en ninguna variante: es el piso de seguridad del sistema.
    if not recetas:
        return {
            "respuesta": SIN_PRESCRIPCION + AVISO,
            "variante": variante,
            "perfil": id_perfil,
            "tipo_consulta": CONSULTA_PLAN,
            "plan": None,
            "avisos": [],
        }

    tipo, medicamento = clasificar_consulta(consulta)

    if variante == VARIANTE_LLM:
        respuesta = llm.invoke(_prompt_solo_llm(perfil, recetas, consulta)).content
        return {
            "respuesta": respuesta + AVISO,
            "variante": variante,
            "perfil": id_perfil,
            "tipo_consulta": tipo,
        }

    if tipo == CONSULTA_ALTERNATIVAS:
        alternativas = alternativas_para(medicamento, id_perfil)
        respuesta = llm.invoke(_prompt_alternativas_con_reglas(alternativas, consulta)).content
        return {
            "respuesta": respuesta + AVISO,
            "variante": variante,
            "perfil": id_perfil,
            "tipo_consulta": tipo,
            "alternativas": alternativas,
        }

    plan = plan_diario(id_perfil)
    respuesta = llm.invoke(_prompt_plan_con_reglas(plan, consulta)).content
    return {
        "respuesta": respuesta + AVISO,
        "variante": variante,
        "perfil": id_perfil,
        "tipo_consulta": tipo,
        "plan": plan["tomas"],
        "avisos": plan["avisos"],
    }
