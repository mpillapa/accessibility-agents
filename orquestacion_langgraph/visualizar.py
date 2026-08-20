# Genera imágenes de los grafos para ver visualmente el flujo de nodos y edges.
#
# Uso (desde la raíz del repo):
#   python -m orquestacion_langgraph.visualizar
#
# Dibuja dos grafos:
#   - el principal (Orchestrator -> especialista), en grafo.png
#   - el subgrafo de RAG agéntico (con su ciclo), en grafo_rag.png
#
# Los PNG salen vía la API pública de mermaid.ink (requiere internet). La
# versión ASCII se imprime 100% local.

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
