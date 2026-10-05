# Comparación: tal como está (`base_2026-10-04`) vs razonamiento bajo (`ajustada_2026-10-04`)

- Ejecuciones: 120 vs 120. Tiempos y tokens solo de ejecuciones completas.
- Mann-Whitney a dos colas sobre todas las ejecuciones completas de cada tarea.

## Éxito

| Tarea | Éxito tal como está | Éxito razonamiento bajo | Timeouts tal como está | Timeouts razonamiento bajo |
|---|---|---|---|---|
| T1 | 19/20 (95.0%, IC 76–99) | 20/20 (100.0%, IC 84–100) | 1 | 0 |
| T2 | 20/20 (100.0%, IC 84–100) | 20/20 (100.0%, IC 84–100) | 0 | 0 |
| T3 | 18/20 (90.0%, IC 70–97) | 20/20 (100.0%, IC 84–100) | 0 | 0 |
| T4 | 15/20 (75.0%, IC 53–89) | 15/20 (75.0%, IC 53–89) | 0 | 0 |
| T5 | 20/20 (100.0%, IC 84–100) | 20/20 (100.0%, IC 84–100) | 0 | 0 |
| T6 | 20/20 (100.0%, IC 84–100) | 20/20 (100.0%, IC 84–100) | 0 | 0 |
| Todas | 112/120 (93.3%, IC 87–97) | 115/120 (95.8%, IC 91–98) | 1 | 0 |

## Tiempo del sistema (s)

| Tarea | Tiempo tal como está (media ± DE; mediana) | Tiempo razonamiento bajo | Cambio de la mediana | Mann-Whitney p |
|---|---|---|---|---|
| T1 | 17.3 ± 11.5; 14.4 | 2.5 ± 1.0; 2.3 | -84% | < 0.001 |
| T2 | 13.6 ± 10.2; 10.8 | 3.7 ± 0.8; 3.7 | -66% | < 0.001 |
| T3 | 3.0 ± 5.2; 1.3 | 0.2 ± 0.1; 0.3 | -78% | 0.003 |
| T4 | 5.0 ± 9.6; 1.0 | 1.1 ± 1.4; 0.3 | -69% | 0.002 |
| T5 | 1.3 ± 0.6; 1.0 | 0.4 ± 0.1; 0.4 | -66% | < 0.001 |
| T6 | 11.2 ± 5.1; 9.8 | 3.2 ± 0.7; 3.2 | -67% | < 0.001 |
| Todas | 8.5 ± 9.7; 6.3 | 1.8 ± 1.6; 1.9 | -70% | < 0.001 |

## Tokens de salida (incluye razonamiento)

| Tarea | Tokens tal como está (media ± DE; mediana) | Tokens razonamiento bajo | Cambio de la mediana | Mann-Whitney p |
|---|---|---|---|---|
| T1 | 4440 ± 2954; 3872 | 415 ± 139; 412 | -89% | < 0.001 |
| T2 | 3446 ± 2139; 2715 | 565 ± 186; 582 | -79% | < 0.001 |
| T3 | 742 ± 1397; 352 | 31 ± 20; 38 | -89% | 0.002 |
| T4 | 1416 ± 2723; 278 | 143 ± 205; 36 | -87% | < 0.001 |
| T5 | 299 ± 99; 246 | 38 ± 8; 37 | -85% | < 0.001 |
| T6 | 2861 ± 1089; 2592 | 408 ± 85; 396 | -85% | < 0.001 |
| Todas | 2182 ± 2448; 1637 | 267 ± 243; 301 | -82% | < 0.001 |

## Tokens de razonamiento

| Tarea | Razonamiento tal como está (media ± DE; mediana) | Razonamiento razonamiento bajo | Cambio de la mediana | Mann-Whitney p |
|---|---|---|---|---|
| T1 | 3885 ± 2766; 3319 | 0 ± 0; 0 | -100% | < 0.001 |
| T2 | 2789 ± 2010; 2072 | 0 ± 0; 0 | -100% | < 0.001 |
| T3 | 656 ± 1258; 300 | 2 ± 4; 0 | -100% | < 0.001 |
| T4 | 1228 ± 2465; 224 | 0 ± 0; 0 | -100% | < 0.001 |
| T5 | 248 ± 94; 199 | 0 ± 0; 0 | -100% | < 0.001 |
| T6 | 2389 ± 1041; 2058 | 14 ± 28; 0 | -100% | < 0.001 |
| Todas | 1849 ± 2199; 1152 | 3 ± 12; 0 | -100% | < 0.001 |

## Frases mal ruteadas

| Frase | Errores de ruteo tal como está | razonamiento bajo |
|---|---|---|
| t3_f4 'Me desperté y no recuerdo nada de lo que pasó' | 2/5 | 0/5 |
| t4_f1 'Modifica lo de la cita médica que me cambiaron para la próxima semana' | 5/5 | 5/5 |

## T1 por usuario

| Usuario | Tiempo de T1 tal como está (mediana, s) | razonamiento bajo | Tokens salida tal como está | razonamiento bajo |
|---|---|---|---|---|
| rosa | 6.0 | 2.0 | 1740 | 308 |
| manuel | 37.6 | 3.4 | 9503 | 630 |
| carmen | 21.1 | 2.6 | 6087 | 496 |
| jorge | 16.1 | 2.1 | 3995 | 412 |
| elena | 5.9 | 1.5 | 1640 | 254 |

