# Pruebas de integridad de los datos de la tarea medicamento × comida.
#
# Uso (desde la raíz del repo):
#   python -m pruebas.prueba_datos_interacciones
#
# NO requieren VPN ni LLM. Prueban que los tres JSON de interacciones/datos/
# sean coherentes entre sí y con el resto del sistema: que los medicamentos
# existan, que las recetas sean las del índice, que la evidencia de cada
# alimento esté de verdad en el texto indexado y que los casos de prueba solo
# esperen cosas posibles.
#
# NO prueban el cruce: eso es de pruebas/prueba_interacciones.py, cuando exista
# reglas.py. Acá se verifica que la verdad de referencia sea consistente, sin
# recalcularla (recalcularla con la misma lógica la volvería circular).
#
# Escrito sin pytest, igual que el resto de pruebas/ (ver prueba_ciclo_rag.py).

import re
import sys
import unicodedata
from pathlib import Path

from interacciones.datos import (
    PARTES,
    SEVERIDADES,
    alimentos_marcados_de,
    cargar_casos,
    cargar_catalogo,
    cargar_ingredientes,
    categorias_alimento,
    interacciones,
    medicamentos_revisados,
)
from medicacion.datos import cargar_medicamentos, cargar_perfiles, prescripciones_de

# Ruta del índice, escrita a mano en vez de importar rag.config: ese módulo
# consulta el servidor al importarse y esta prueba no debe depender de la red.
CHROMA_DIR = Path(__file__).parent.parent / "rag" / "chroma_db"
CHROMA_COLLECTION = "recetas"


def _normalizar(texto: str) -> str:
    """Minúsculas, sin tildes y con los espacios colapsados. El OCR mete saltos
    de línea y a veces letras de otro alfabeto que se ven iguales (una 'о'
    cirílica en 'ajо'), así que la evidencia se compara normalizada."""
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto).strip().lower()


def _texto_indexado_por_fuente() -> dict[str, str] | None:
    """El texto de ChromaDB agrupado por fuente, o None si el índice no está
    (no se versiona: se regenera con `python -m rag.ingesta`)."""
    if not CHROMA_DIR.exists():
        return None
    try:
        import chromadb
    except ImportError:
        return None
    coleccion = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(CHROMA_COLLECTION)
    datos = coleccion.get(include=["metadatas", "documents"])
    por_fuente: dict[str, list[str]] = {}
    for metadato, documento in zip(datos["metadatas"], datos["documents"]):
        por_fuente.setdefault(metadato["fuente"], []).append(documento)
    return {fuente: _normalizar(" ".join(docs)) for fuente, docs in por_fuente.items()}


def _nombres_vademecum() -> set[str]:
    return {m["nombre"] for m in cargar_medicamentos()}


def _medicamentos_recetados() -> set[str]:
    return {
        indicacion["medicamento"]
        for perfil in cargar_perfiles()
        for receta in prescripciones_de(perfil["id"])
        for indicacion in receta["indicaciones"]
    }


def _medicamentos_de(usuario: str) -> set[str]:
    return {i["medicamento"] for r in prescripciones_de(usuario) for i in r["indicaciones"]}


def _categorias_de(fuente: str) -> set[str]:
    return {a["categoria"] for a in alimentos_marcados_de(fuente) or []}


# --- Catálogo de interacciones ----------------------------------------------

def prueba_medicamentos_del_catalogo_existen_en_el_vademecum():
    vademecum = _nombres_vademecum()
    desconocidos = medicamentos_revisados() - vademecum
    assert not desconocidos, f"no están en medicamentos.json: {desconocidos}"
    return f"los {len(medicamentos_revisados())} medicamentos del catálogo existen en el vademécum"


def prueba_todo_medicamento_recetado_fue_revisado():
    """Si alguien recibe un medicamento que el catálogo no cubre, el cruce
    diría "sin interacciones" sin haberlo mirado. Eso tiene que fallar acá."""
    sin_revisar = _medicamentos_recetados() - medicamentos_revisados()
    assert not sin_revisar, f"recetados pero sin revisar en el catálogo: {sin_revisar}"
    return f"los {len(_medicamentos_recetados())} medicamentos recetados están cubiertos por el catálogo"


def prueba_interacciones_bien_formadas():
    categorias = set(categorias_alimento())
    campos_perfil = {clave for p in cargar_perfiles() for clave in p}
    for i in interacciones():
        assert i["categoria"] in categorias, i
        assert i["severidad"] in SEVERIDADES, i
        assert i["motivo"].strip() and i["recomendacion"].strip(), i
        for campo in (i["solo_si"] or {}):
            assert campo in campos_perfil, f"solo_si usa un campo que ningún perfil tiene: {i}"
    duplicadas = len(interacciones()) - len({(i["medicamento"], i["categoria"]) for i in interacciones()})
    assert duplicadas == 0, "hay pares (medicamento, categoría) repetidos"
    return f"{len(interacciones())} interacciones con categoría, severidad y condición válidas"


def prueba_ningun_medicamento_esta_en_las_dos_listas():
    catalogo = cargar_catalogo()
    ambas = {i["medicamento"] for i in catalogo["interacciones"]} & set(catalogo["sin_interacciones_registradas"])
    assert not ambas, f"con interacción y a la vez 'sin interacciones registradas': {ambas}"
    return "ningún medicamento figura a la vez con y sin interacciones"


# --- Ingredientes por receta ------------------------------------------------

def prueba_alimentos_marcados_bien_formados():
    categorias = categorias_alimento()
    total = 0
    for fuente, entrada in cargar_ingredientes().items():
        assert entrada["calidad_ocr"] in ("buena", "baja"), fuente
        assert entrada["platos"], f"{fuente} no tiene platos"
        for plato in entrada["platos"]:
            for a in plato["alimentos_marcados"]:
                total += 1
                assert a["categoria"] in categorias, (fuente, a)
                assert a["alimento"] in categorias[a["categoria"]]["alimentos"], (
                    f"{fuente}: '{a['alimento']}' no está en la lista cerrada de '{a['categoria']}'"
                )
                assert a["parte"] in PARTES, (fuente, a)
                assert a["evidencia"].strip(), (fuente, a)
    return f"{total} alimentos marcados con categoría, alimento y parte válidos"


def prueba_las_fuentes_son_exactamente_las_del_indice():
    indice = _texto_indexado_por_fuente()
    if indice is None:
        return "OMITIDA: no hay índice local (correr `python -m rag.ingesta`)"
    en_archivo = set(cargar_ingredientes())
    faltan = set(indice) - en_archivo
    sobran = en_archivo - set(indice)
    assert not faltan, f"recetas del índice sin ingredientes cargados: {faltan}"
    assert not sobran, f"fuentes cargadas que el índice no tiene: {sobran}"
    return f"las {len(indice)} fuentes del índice tienen sus ingredientes, y no sobra ninguna"


def prueba_la_evidencia_esta_en_el_texto_indexado():
    """Cada alimento marcado cita un fragmento del texto indexado. Si la cita no
    está, el dato salió de otro lado (de la foto, de lo que 'suele llevar' el
    plato) y no de lo que el RAG realmente puede recuperar."""
    indice = _texto_indexado_por_fuente()
    if indice is None:
        return "OMITIDA: no hay índice local (correr `python -m rag.ingesta`)"
    revisadas = 0
    for fuente, entrada in cargar_ingredientes().items():
        if fuente not in indice:
            continue  # lo reporta prueba_las_fuentes_son_exactamente_las_del_indice
        for plato in entrada["platos"]:
            for a in plato["alimentos_marcados"]:
                # "A ... B" cita dos trozos no contiguos del mismo texto.
                for trozo in a["evidencia"].split("..."):
                    trozo = _normalizar(trozo)
                    assert trozo in indice[fuente], f"{fuente}: la evidencia '{trozo}' no está en el texto indexado"
                revisadas += 1
    return f"las {revisadas} citas de evidencia aparecen literalmente en el texto indexado"


# --- Casos de prueba --------------------------------------------------------

def _revisar_esperado(usuario: str, fuente: str, esperado: dict, etiqueta: str):
    """Condiciones NECESARIAS, no el cálculo: un par esperado solo puede
    involucrar un medicamento que la persona tiene y una categoría que la
    fuente contiene."""
    pares = {tuple(p) for p in esperado["interacciones"]}
    assert len(pares) == len(esperado["interacciones"]), f"{etiqueta}: pares repetidos"
    assert esperado["hay_interaccion"] == bool(pares), f"{etiqueta}: hay_interaccion no coincide con la lista"
    registradas = {(i["medicamento"], i["categoria"]) for i in interacciones()}
    for medicamento, categoria in pares:
        assert (medicamento, categoria) in registradas, f"{etiqueta}: {medicamento}×{categoria} no está en el catálogo"
        assert medicamento in _medicamentos_de(usuario), f"{etiqueta}: {usuario} no toma {medicamento}"
        assert categoria in _categorias_de(fuente), f"{etiqueta}: {fuente} no contiene {categoria}"


def prueba_casos_de_desarrollo_consistentes():
    perfiles = {p["id"] for p in cargar_perfiles()}
    casos = cargar_casos()["desarrollo"]["casos"]
    for caso in casos:
        assert caso["usuario"] in perfiles, caso["id"]
        assert alimentos_marcados_de(caso["fuente"]) is not None, f"{caso['id']}: fuente desconocida"
        esperado = caso["esperado"]
        if not esperado["evaluable"]:
            assert not prescripciones_de(caso["usuario"]), f"{caso['id']}: no evaluable pero el usuario sí tiene receta"
            assert not esperado["interacciones"], caso["id"]
            continue
        assert prescripciones_de(caso["usuario"]), f"{caso['id']}: evaluable pero el usuario no tiene receta"
        _revisar_esperado(caso["usuario"], caso["fuente"], esperado, caso["id"])
    con = sum(c["esperado"]["hay_interaccion"] for c in casos)
    return f"{len(casos)} casos de desarrollo consistentes ({con} con interacción, {len(casos) - con} sin)"


def prueba_casos_de_campana_consistentes():
    campana = cargar_casos()["campana"]
    usuarios = campana["usuarios"]
    for usuario in usuarios:
        assert prescripciones_de(usuario), f"{usuario} está en la campaña pero no tiene receta"
    con = sin = 0
    for frase in campana["frases"]:
        assert set(frase["esperado_por_usuario"]) == set(usuarios), f"{frase['id']}: no cubre a todos los usuarios"
        # Si hay varias fuentes aceptadas, todas tienen que dar el mismo
        # resultado; si no, el éxito dependería de cuál recuperó el RAG.
        categorias = {frozenset(_categorias_de(f)) for f in frase["fuentes_aceptadas"]}
        assert len(categorias) == 1, f"{frase['id']}: las fuentes aceptadas no contienen las mismas categorías"
        for usuario, esperado in frase["esperado_por_usuario"].items():
            for fuente in frase["fuentes_aceptadas"]:
                _revisar_esperado(usuario, fuente, esperado, f"{frase['id']}/{usuario}")
            con += esperado["hay_interaccion"]
            sin += not esperado["hay_interaccion"]
    return f"{len(campana['frases'])} frases × {len(usuarios)} usuarios consistentes ({con} con interacción, {sin} sin)"


CASOS = [
    prueba_medicamentos_del_catalogo_existen_en_el_vademecum,
    prueba_todo_medicamento_recetado_fue_revisado,
    prueba_interacciones_bien_formadas,
    prueba_ningun_medicamento_esta_en_las_dos_listas,
    prueba_alimentos_marcados_bien_formados,
    prueba_las_fuentes_son_exactamente_las_del_indice,
    prueba_la_evidencia_esta_en_el_texto_indexado,
    prueba_casos_de_desarrollo_consistentes,
    prueba_casos_de_campana_consistentes,
]


def main():
    print("Integridad de los datos de medicamento × comida (sin VPN, sin LLM)\n")
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
