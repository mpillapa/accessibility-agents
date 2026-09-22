# Interfaz web

Interfaz de chat del sistema multiagente, en Streamlit.

## Ejecutar

```bash
streamlit run interfaz/app.py                          # solo local
streamlit run interfaz/app.py --server.address 0.0.0.0 # accesible por IP
```

Queda en `http://localhost:8501` y, con `--server.address 0.0.0.0`, en la IP de
la máquina dentro de la red (por ejemplo `http://172.28.230.10:8501`).

Requiere VPN institucional activa: los modelos corren en los servidores de la
Universidad.

## Qué muestra

Un chat con historial. Bajo cada respuesta, plegado, el recorrido real por el
grafo: a qué agente ruteó el orquestador, por qué, y —si intervino el RAG
agéntico— los pasos del subgrafo, incluidos los ciclos de reformulación.

Mientras trabaja muestra el avance nodo por nodo. Una consulta con RAG tarda
entre 6 y 30 segundos y el 88% se va en redactar la respuesta final; sin
retroalimentación la pantalla parece colgada.

La barra lateral indica con qué modelo se está respondiendo y avisa si el
servidor cambió de modelo respecto a lo configurado en el `.env`. No es un
adorno: el puerto 12559 rotó de modelo cuatro veces en trece días (ver
`infraestructura/modelos.py` y la sección 11 de `BITACORA_HALLAZGOS.md`).

## Separación de responsabilidades

`app.py` solo presenta. No clasifica, no decide y no habla con los modelos: todo
eso vive en `orquestacion_langgraph/`. Si aparece lógica de negocio en este
paquete, está en el lugar equivocado.

El único punto de contacto con el sistema es `procesar_consulta_en_vivo()`, en
`orquestacion_langgraph/grafo.py`, que entrega cada paso del grafo apenas
ocurre.

## Estado

Prototipo académico. No tiene autenticación, control de acceso ni persistencia:
el historial vive en la sesión del navegador y se pierde al recargar.
