# Dibuja el grafo principal (grafo.png) y el subgrafo de RAG (grafo_rag.png).
# El PNG sale de mermaid.ink (requiere internet); el ASCII y el .mmd son locales.
#
# Uso (desde la raíz del repo):
#   python -m orquestacion_langgraph.visualizar

from pathlib import Path

from orquestacion_langgraph.grafo import construir_grafo
from orquestacion_langgraph.rag_agentico.subgrafo import construir_subgrafo_rag

AQUI = Path(__file__).parent
RUTA_PNG_PRINCIPAL = AQUI / "grafo.png"
RUTA_PNG_RAG = AQUI / "grafo_rag.png"


def dibujar(app, titulo: str, ruta_png: Path):
    print("=" * 70)
    print(titulo)
    print("=" * 70)

    g = app.get_graph()
    print("\n--- Vista ASCII (local) ---\n")
    print(g.draw_ascii())

    # Se guarda aunque falle el PNG: es la fuente editable del diagrama.
    ruta_png.with_suffix(".mmd").write_text(g.draw_mermaid(), encoding="utf-8")

    try:
        ruta_png.write_bytes(g.draw_mermaid_png())
        print(f"\nImagen guardada en: {ruta_png}")
    except Exception as e:
        print(f"\nNo se pudo generar el PNG (requiere internet): {e}")
        print("El código mermaid de abajo se puede pegar en https://mermaid.live:\n")
        print(g.draw_mermaid())
    print()


def main():
    dibujar(construir_grafo(), "GRAFO PRINCIPAL: ruteo por intención", RUTA_PNG_PRINCIPAL)
    dibujar(
        construir_subgrafo_rag(),
        "SUBGRAFO: RAG agéntico del especialista en recetas",
        RUTA_PNG_RAG,
    )


if __name__ == "__main__":
    main()
