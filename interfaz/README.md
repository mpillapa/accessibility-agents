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

## Entrada por voz

Tres vías, en el desplegable "Hablar en vez de escribir":

| Vía | Funciona por IP | Para qué |
|---|---|---|
| **Grabar** | no | Hablar en vivo |
| **Subir un audio** | sí | Un `.wav`/`.mp3` cualquiera |
| **Ejemplos** | sí | El audio de emergencia, limpio y degradado |

**El micrófono necesita `localhost` o HTTPS.** La API del navegador que usa
`st.audio_input` (`getUserMedia`) solo está disponible en contextos seguros: si
abrís la app por `http://<IP>:8501`, el navegador bloquea la grabación y
Streamlit lo reporta con un error genérico que no explica la causa.

Para grabar desde otra máquina, hacé un túnel y entrá por localhost:

```bash
ssh -L 8501:localhost:8501 usuario@172.28.230.10
# después, en el navegador: http://localhost:8501
```

Las otras dos vías no tienen esa restricción, y para una demo son preferibles:
los ejemplos muestran el guardrail del ASR en dos clics, sin depender de que el
micrófono funcione ni de hablar en el momento.

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
