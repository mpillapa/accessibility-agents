# Ingesta del recetario: lee rag/recetas_data/ (texto e imágenes), pasa las
# imágenes por OCR (glm-ocr), trocea el texto resultante, descarta lo que no
# pasa el control de calidad, lo embebe (BGE-M3) y lo guarda en una colección
# local de ChromaDB persistida en rag/chroma_db/.
#
# Uso (desde la raíz del repo):
#   python -m rag.ingesta

from pathlib import Path

import chromadb

from rag.calidad import evaluar_texto
from rag.config import CHROMA_COLLECTION, CHROMA_DIR, RECETAS_DATA_DIR
from rag.embeddings import embed_textos
from rag.ocr import extraer_texto_de_imagen

EXTENSIONES_TEXTO = {".txt", ".md"}
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".webp"}

# Tope de caracteres por fragmento. Los fragmentos legítimos del recetario real
# llegan a ~900 caracteres, así que con 1200 casi nada se parte: el tope existe
# para las salidas de OCR que vienen como un bloque sin dobles saltos de línea
# (típico en fotos de páginas densas), no para trocear recetas normales.
#
# Importa que ningún fragmento exceda la ventana del modelo de embeddings: si la
# excede, rag/embeddings.py lo trunca para calcular el vector, y entonces el
# embedding representaría un texto distinto del que se guarda en ChromaDB.
MAXIMO_CARACTERES_FRAGMENTO = 1200

# Piso de caracteres por fragmento. Un fragmento más corto que esto se fusiona
# con el siguiente en vez de indexarse solo.
#
# Motivo medido el 2026-08-19: el troceado por párrafos dejaba encabezados
# sueltos como "PREPARACIÓN" (11 caracteres), "Ingredientes:" o "Recetas
# Ecuatorianas" como fragmentos propios. Al no tener contenido, su embedding no
# representa ninguna receta en particular y quedaban cerca de cualquier
# consulta: buscar "sushi" devolvía tres fragmentos "PREPARACIÓN" de tres
# recetas distintas. Con la expansión al documento padre eso es peor todavía,
# porque un solo falso positivo arrastra la receta completa al contexto.
#
# Fusionar en lugar de descartar conserva la información: el encabezado queda
# pegado a los pasos que encabeza, que es donde pertenece.
MINIMO_CARACTERES_FRAGMENTO = 40


def _partir_por_longitud(texto: str, maximo: int) -> list[str]:
    """Parte un bloque largo respetando límites de palabra.

    Una palabra que por sí sola excede el tope se corta a lo bruto: el OCR
    puede devolver cadenas sin espacios (basura de una imagen ilegible), y
    respetar límites de palabra en ese caso dejaría pasar un fragmento que no
    cabe en la ventana del modelo de embeddings.
    """
    partes, actual = [], ""
    for palabra in texto.split():
        while len(palabra) > maximo:
            if actual:
                partes.append(actual)
                actual = ""
            partes.append(palabra[:maximo])
            palabra = palabra[maximo:]
        if actual and len(actual) + 1 + len(palabra) > maximo:
            partes.append(actual)
            actual = palabra
        else:
            actual = f"{actual} {palabra}".strip()
    if actual:
        partes.append(actual)
    return partes


def _fusionar_cortos(fragmentos: list[str], minimo: int, maximo: int) -> list[str]:
    """Pega los fragmentos demasiado cortos al siguiente.

    Un encabezado ("PREPARACIÓN", "Ingredientes:") no dice nada por sí solo,
    pero sí encabeza lo que viene después: unirlos produce un fragmento con
    sentido en vez de dos, uno inútil y otro sin contexto.

    Si el corto es el último y no hay con qué fusionarlo, se pega al anterior.
    """
    resultado: list[str] = []
    pendiente = ""

    for fragmento in fragmentos:
        candidato = f"{pendiente}\n{fragmento}".strip() if pendiente else fragmento
        if len(candidato) < minimo:
            # Sigue siendo corto: se acumula esperando el próximo.
            pendiente = candidato
            continue
        if len(candidato) <= maximo:
            resultado.append(candidato)
            pendiente = ""
        else:
            # Unirlos excedería el tope: van separados.
            if pendiente:
                resultado.append(pendiente)
            resultado.append(fragmento)
            pendiente = ""

    if pendiente:
        if resultado:
            ultimo = f"{resultado[-1]}\n{pendiente}"
            if len(ultimo) <= maximo:
                resultado[-1] = ultimo
            else:
                resultado.append(pendiente)
        else:
            resultado.append(pendiente)

    return resultado


def _trocear(texto: str, maximo: int = MAXIMO_CARACTERES_FRAGMENTO) -> list[str]:
    """Trocea en fragmentos de a lo sumo `maximo` caracteres.

    Va de la separación más semántica a la más burda: párrafos, después líneas,
    y como último recurso corte por longitud. Así una receta normal queda con un
    fragmento por paso o sección, y solo el texto sin estructura termina cortado
    de forma arbitraria.
    """
    fragmentos = []
    for parrafo in texto.split("\n\n"):
        parrafo = parrafo.strip()
        if not parrafo:
            continue
        if len(parrafo) <= maximo:
            fragmentos.append(parrafo)
            continue

        # Demasiado largo: probar con líneas simples.
        acumulado = ""
        for linea in parrafo.split("\n"):
            linea = linea.strip()
            if not linea:
                continue
            if len(acumulado) + len(linea) + 1 <= maximo:
                acumulado = f"{acumulado}\n{linea}".strip()
            else:
                if acumulado:
                    fragmentos.append(acumulado)
                acumulado = linea if len(linea) <= maximo else ""
                if not acumulado:
                    fragmentos.extend(_partir_por_longitud(linea, maximo))
        if acumulado:
            fragmentos.append(acumulado)

    return _fusionar_cortos(fragmentos, MINIMO_CARACTERES_FRAGMENTO, maximo)


def _leer_archivo(archivo: Path) -> str | None:
    extension = archivo.suffix.lower()
    if extension in EXTENSIONES_TEXTO:
        return archivo.read_text(encoding="utf-8")
    if extension in EXTENSIONES_IMAGEN:
        print(f"  OCR: {archivo.name}...")
        return extraer_texto_de_imagen(archivo)
    return None


def cargar_fragmentos() -> tuple[list[tuple[str, str, str, int]], list[dict]]:
    """Devuelve (fragmentos, rechazos).

    Cada fragmento es una tupla (id, texto, archivo_fuente, orden). El `orden`
    es la posición del fragmento dentro de su archivo: permite reconstruir una
    receta completa a partir de uno de sus fragmentos, que es lo que hace el
    nodo de expansión de contexto del RAG agéntico.

    Cada rechazo es un dict con el archivo y el motivo, para reportarlos al
    final.
    """
    fragmentos: list[tuple[str, str, str, int]] = []
    rechazos: list[dict] = []

    for archivo in sorted(RECETAS_DATA_DIR.iterdir()):
        if not archivo.is_file():
            continue
        texto = _leer_archivo(archivo)
        if texto is None:
            continue

        # Primero se juzga el archivo completo. Cuando el OCR entra en un bucle,
        # falla la transcripción entera y no un trozo: descartar el archivo de
        # una vez da un mensaje claro ("revisá esta foto") en lugar de cuarenta
        # rechazos de fragmentos del mismo origen.
        diagnostico = evaluar_texto(texto)
        if diagnostico.es_degenerado:
            print(f"    RECHAZADO: {diagnostico.motivo}")
            rechazos.append({
                "archivo": archivo.name,
                "motivo": diagnostico.motivo,
                "caracteres": len(texto),
                "ratio_palabras_unicas": diagnostico.ratio_palabras_unicas,
            })
            continue

        for i, trozo in enumerate(_trocear(texto)):
            # Red de seguridad: un archivo puede estar bien en conjunto y traer
            # un tramo degenerado igual.
            diagnostico_trozo = evaluar_texto(trozo)
            if diagnostico_trozo.es_degenerado:
                rechazos.append({
                    "archivo": f"{archivo.name} (fragmento {i})",
                    "motivo": diagnostico_trozo.motivo,
                    "caracteres": len(trozo),
                    "ratio_palabras_unicas": diagnostico_trozo.ratio_palabras_unicas,
                })
                continue
            fragmentos.append((f"{archivo.stem}-{i}", trozo, archivo.name, i))

    return fragmentos, rechazos


def ingestar():
    fragmentos, rechazos = cargar_fragmentos()

    if not fragmentos:
        print(f"No se pudo indexar nada de {RECETAS_DATA_DIR}")
        if rechazos:
            print("Todo lo leído fue rechazado por el control de calidad.")
        return

    ids = [f[0] for f in fragmentos]
    textos = [f[1] for f in fragmentos]
    fuentes = [f[2] for f in fragmentos]
    ordenes = [f[3] for f in fragmentos]

    print(f"\nGenerando embeddings para {len(textos)} fragmento(s)...")
    embeddings = embed_textos(textos)

    # La colección se recrea en cada ingesta. Con upsert quedaban fragmentos
    # huérfanos de corridas anteriores: si un archivo se renombra o se borra, o
    # si cambia el troceado, los ids viejos siguen en el índice y el RAG los
    # sigue recuperando aunque ya no correspondan a ningún archivo del disco.
    cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        cliente.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass  # no existía todavía
    coleccion = cliente.create_collection(CHROMA_COLLECTION)
    coleccion.add(
        ids=ids,
        embeddings=embeddings,
        documents=textos,
        metadatas=[{"fuente": f, "orden": o} for f, o in zip(fuentes, ordenes)],
    )

    print(f"\nListo: {len(fragmentos)} fragmento(s) de {len(set(fuentes))} archivo(s) en {CHROMA_DIR}")

    if rechazos:
        print(f"\nRechazados por control de calidad ({len(rechazos)}):")
        for r in rechazos:
            print(f"  - {r['archivo']} ({r['caracteres']} caracteres)")
            print(f"      {r['motivo']}")
        print(
            "\n  Estos archivos NO están en el índice. Suele indicar que el OCR "
            "\n  no pudo leer la imagen y generó texto inventado — revisá la foto: "
            "\n  las páginas dobles muy densas y el texto en varias columnas son "
            "\n  las que más fallan."
        )


if __name__ == "__main__":
    ingestar()
