# Redacción de la respuesta de la tarea medicamento × comida.
#
# El cruce ya está hecho (reglas.py). Aquí solo se convierte en palabras:
#   - si NO es evaluable (falta la receta médica, la persona o el plato), el
#     texto es FIJO y no se llama al LLM: no hay nada que redactar, y lo que hay
#     que decir no puede depender de que el modelo lo diga bien;
#   - si es evaluable, el LLM redacta a partir del resultado del cruce, con la
#     instrucción explícita de no agregar interacciones que no estén en él.
# El aviso final se agrega en código, no en el prompt, por la misma razón que en
# medicacion/agente.py: tiene que estar siempre.

from interacciones.reglas import (
    MOTIVO_SIN_PERFIL,
    MOTIVO_SIN_PRESCRIPCION,
    MOTIVO_SIN_PRESCRIPCION_VIGENTE,
    MOTIVO_SIN_RECETA,
)
from medicacion.agente import AVISO, PERSONA_DESCONOCIDA

TEXTO_NO_EVALUABLE = {
    MOTIVO_SIN_PERFIL: PERSONA_DESCONOCIDA,
    MOTIVO_SIN_PRESCRIPCION: (
        "No tengo ninguna receta médica registrada a su nombre, así que no puedo "
        "saber si ese plato le cae bien con lo que toma. Pídale a su médico la "
        "receta y la cargamos en el sistema."
    ),
    MOTIVO_SIN_PRESCRIPCION_VIGENTE: (
        "Su receta médica registrada ya venció, así que no sé qué está tomando "
        "ahora. Pídale a su médico una receta actualizada."
    ),
    MOTIVO_SIN_RECETA: (
        "No encontré ese plato en su recetario, así que no sé qué ingredientes "
        "lleva y no puedo compararlo con sus medicamentos. ¿Me dice el nombre del "
        "plato de otra forma?"
    ),
}

PARTE_LEGIBLE = {
    "principal": "es parte del plato",
    "acompanamiento": "se sugiere como acompañamiento",
    "opcional": "es opcional en la receta",
}


def _describir_cruce(cruce: dict) -> str:
    lineas = [f"Medicamentos que toma (según su receta vigente): {', '.join(cruce['medicamentos'])}."]
    lineas.append(f"Recetas de cocina revisadas: {', '.join(cruce['fuentes_cruzadas'])}.")
    if not cruce["interacciones"]:
        lineas.append("Resultado del cruce: NINGUNA interacción registrada entre sus medicamentos y los ingredientes del plato.")
        return "\n".join(lineas)
    lineas.append("Resultado del cruce (estas y SOLO estas):")
    for i in cruce["interacciones"]:
        alimentos = "; ".join(
            f"{a['alimento']} ({PARTE_LEGIBLE.get(a['parte'], a['parte'])}, en {a['plato']})"
            for a in i["alimentos"]
        )
        lineas.append(
            f"- {i['medicamento']} con {i['categoria']} — severidad: {i['severidad']}. "
            f"Alimentos: {alimentos}. Por qué: {i['motivo']} Qué hacer: {i['recomendacion']}"
        )
    return "\n".join(lineas)


def prompt_integrador(consulta: str, cruce: dict) -> str:
    return (
        "Eres un asistente para adultos mayores. Hablas español sencillo, de usted, "
        "con frases cortas.\n\n"
        f"La persona preguntó: '{consulta}'\n\n"
        f"{_describir_cruce(cruce)}\n\n"
        "Responde a la persona:\n"
        "1. Empieza con la respuesta directa en una frase (puede comerlo, con "
        "cuidado, o mejor evitarlo).\n"
        "2. Explica cada interacción del cruce con palabras simples y di qué hacer. "
        "Si el alimento es acompañamiento u opcional, dilo: puede comer el plato sin "
        "ese ingrediente.\n"
        "3. NO agregues interacciones, medicamentos ni alimentos que no estén en el "
        "cruce. NO des dosis ni cambies la medicación.\n"
        "4. Si el cruce no encontró nada, dilo así, sin inventar precauciones.\n"
        "Máximo 120 palabras."
    )


def redactar(consulta: str, cruce: dict, llm) -> dict:
    """La respuesta final y si se usó el LLM para redactarla."""
    if not cruce["evaluable"]:
        return {"respuesta": TEXTO_NO_EVALUABLE[cruce["motivo"]] + AVISO, "uso_llm": False}
    texto = llm.invoke(prompt_integrador(consulta, cruce)).content
    return {"respuesta": texto.strip() + AVISO, "uso_llm": True}
