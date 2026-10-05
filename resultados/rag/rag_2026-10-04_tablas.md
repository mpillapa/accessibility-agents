# Evaluación ampliada del RAG `rag_2026-10-04`

- 30 consultas, 90 ejecuciones; fingerprints ['vllm-0.30.1rc1.dev630+g18f8f9602-tp4-ep-90ac72a8'].
- Acierto: la respuesta usó una fuente aceptada, o ninguna si el plato no está en el recetario.
- Sin mezclar: además no usó ninguna fuente ajena. Timeout (120 s) o excepción = fallo.

## Por tipo de consulta

| Tipo | Consultas | Ejecuciones | Acierto | IC 95% | Sin mezclar fuentes | Reformuló | Tiempo (s) | Tokens salida |
|---|---|---|---|---|---|---|---|---|
| exacto | 8 | 24 | 24/24 (100%) | 86–100% | 24/24 | 3/24 | 18.9 ± 14.5 | 4012 ± 2698 |
| descripcion | 8 | 24 | 24/24 (100%) | 86–100% | 24/24 | 0/24 | 18.5 ± 7.7 | 4429 ± 1828 |
| coloquial | 5 | 15 | 12/15 (80%) | 55–93% | 11/15 | 3/15 | 14.6 ± 4.3 | 3443 ± 1288 |
| ambigua | 5 | 15 | 15/15 (100%) | 80–100% | 15/15 | 3/15 | 16.3 ± 7.8 | 4293 ± 2216 |
| fuera | 4 | 12 | 12/12 (100%) | 76–100% | 12/12 | 12/12 | 8.0 ± 2.0 | 1790 ± 250 |
| Todas | 30 | 90 | 87/90 (97%) | 91–99% | 86/90 | 21/90 | 16.2 ± 9.8 | 3779 ± 2124 |

## Fallos (leer antes de contar)

| Consulta | Tipo | Aceptadas | Usó | Camino |
|---|---|---|---|---|
| r21-r1 'el verde majado con huevito y queso para el desayuno' | coloquial | ['tigrillo.png'] | —  | decidir_busqueda → recuperar → evaluar_relevancia → reformular → recuperar → evaluar_relevancia → sin_resultado |
| r21-r2 'el verde majado con huevito y queso para el desayuno' | coloquial | ['tigrillo.png'] | —  | decidir_busqueda → recuperar → evaluar_relevancia → reformular → recuperar → evaluar_relevancia → sin_resultado |
| r21-r3 'el verde majado con huevito y queso para el desayuno' | coloquial | ['tigrillo.png'] | —  | decidir_busqueda → recuperar → evaluar_relevancia → reformular → recuperar → evaluar_relevancia → sin_resultado |

