# Genera una imagen del grafo para ver visualmente el flujo de nodos y edges.
#
# Uso:
#   python langgraph/visualizar.py
#
# Genera langgraph/grafo.png (vía la API pública de mermaid.ink, requiere
# internet) y además imprime una versión ASCII 100% local en la terminal.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from grafo import construir_grafo

RUTA_PNG = Path(__file__).parent / "grafo.png"


def main():
    app = construir_grafo()
    g = app.get_graph()

    print("--- Vista ASCII (local) ---\n")
    print(g.draw_ascii())

    try:
        png = g.draw_mermaid_png()
        RUTA_PNG.write_bytes(png)
        print(f"\nImagen guardada en: {RUTA_PNG}")
    except Exception as e:
        print(f"\nNo se pudo generar el PNG (requiere internet): {e}")
        print("El código mermaid de abajo se puede pegar en https://mermaid.live para verlo:\n")
        print(g.draw_mermaid())


if __name__ == "__main__":
    main()
