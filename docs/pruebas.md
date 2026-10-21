# Pruebas y evidencia

Hay dos niveles de verificación:

1. **Pruebas automáticas sin API** (`python -m pytest`): verifican cada mecanismo de forma determinista con un LLM programable (`ScriptedLLM`) y embeddings sintéticos.
2. **Escenarios con el modelo real** (`python evaluate.py --live`, opcional): ejecutan el agente completo con Groq (`openai/gpt-oss-20b`), embeddings reales y DDGS, y guardan los resultados en `evidence/live-<fecha>/`.

## 1. Pruebas automáticas

Resultado del 21/10/2026 en la rama `ep2-agente`, con Python 3.12.10: **34 pruebas aprobadas, sin advertencias.**

| Área (indicador) | Prueba | Qué demuestra |
|---|---|---|
| Herramientas (IE1) | `test_tools_validate` | Las 9 herramientas existen, validan argumentos (Pydantic) y la web caída devuelve un error estructurado. |
| | `test_purchase_order_validates_and_writes` | La orden se escribe en JSON y CSV, excluye ingredientes y rechaza los inexistentes, los excesivos o los que no tienen mínimo. |
| | `test_receipt_writes_runtime_only` | La recepción modifica solo `runtime/`, nunca los datos semilla. |
| | `test_prioritize_orders_by_criticality` | El orden crítico → bajo → ok según la cobertura del mínimo. |
| Flujo integrado (IE1, IE5) | `test_full_replenishment_flow_reads_reasons_and_writes` | Consulta → razonamiento → escritura en un turno; un plan propuesto desordenado queda ordenado por fase. |
| Memoria de corto plazo (IE3) | `test_short_term_memory_keeps_conversation_continuity` | La aclaración del turno 1 se resuelve en el turno 2; los hilos están aislados. |
| | `test_long_conversation_is_summarized_into_memory` | Al superar la ventana se resume, se eliminan los mensajes antiguos y el episodio pasa a largo plazo. |
| Memoria de largo plazo (IE4) | `test_long_term_memory_semantic_search_and_persistence` | Búsqueda semántica, consolidación de duplicados, persistencia, filtros y borrado. |
| | `test_memory_reembeds_when_model_changes` | Coherencia si cambia el modelo de embeddings. |
| | `test_long_term_memory_crosses_conversations` | Una preferencia del hilo A llega al prompt del hilo B. |
| | `test_task_oriented_memory_recalls_preferences_not_mentioned` | La recuperación por tema de la tarea encuentra preferencias que la pregunta no menciona. |
| RAG (IE4) | `test_faiss_roundtrip_and_filter`, `test_changed_*`, `test_modified_index_rejected` | Índice coseno, filtro por tipo de fuente e integridad del índice. |
| Planificación (IE5) | `test_plan_is_ordered_by_phase_then_priority` | El esquema de priorización determinista. |
| | `test_heuristic_planner_uses_history_and_asks_when_needed` | El planificador de respaldo usa el historial y pide aclaración. |
| | `test_planner_falls_back_to_heuristics` | Si el LLM no entrega un plan válido, el agente continúa con la heurística. |
| | `test_plan_adherence_check_requests_skipped_high_priority_steps` | Se exige ejecutar los pasos de prioridad alta omitidos. |
| Condiciones cambiantes (IE5) | `test_stock_change_between_turns_is_perceived_before_planning` | Un cambio externo de stock se percibe antes de planificar y la precondición bloquea una escritura con datos viejos. |
| | `test_stock_change_during_turn_triggers_replanning` | Un cambio a mitad del turno provoca replanificación y la orden usa el stock nuevo. |
| | `test_web_failure_is_detected_and_not_retried` | Fallo de servicio → replanificación y bloqueo de reintentos. |
| | `test_missing_minimum_and_unplanned_actions_are_reported` | Datos faltantes, acciones no planificadas y pasos pendientes quedan registrados. |
| | `test_quantity_above_need_is_flagged` | Advertencia por cantidades sobre lo necesario. |
| | `test_round_limit_forces_answer` | El límite de rondas fuerza una respuesta en lugar de un bucle. |
| Fidelidad | `test_fabricated_web_citations_are_detected_and_rewritten`, `test_real_web_citations_are_accepted` | Las citas `[WEB-n]` y las URL inventadas se detectan y se corrigen; las reales se aceptan. |
| Interfaz | `test_interface_loads_without_api`, `test_reset_inventory_button` | La interfaz carga sin API y restablece el entorno. |

Las pruebas usan vectores sintéticos y un modelo simulado. No miden la calidad del LLM real.

## 2. Escenarios con el modelo real

`evaluate.py` define 10 escenarios de varios turnos. Cada uno corre con su propio `runtime/` (inventario, memoria, conversaciones y órdenes) y genera `resultados.json` y `resumen.md` con el plan, las herramientas, las decisiones y la respuesta de cada turno.

| ID | Escenario | Mecanismo evaluado |
|---|---|---|
| E1 | Revisar stock y preparar orden con lo crítico | Herramientas de consulta, razonamiento y escritura; plan priorizado (IE1, IE5) |
| E2 | «¿Cuál fue el mejor mes?» → «Del 2026» → «¿cuánto en el mes que me dijiste?» | Aclaración y continuidad de corto plazo (IE3) |
| E3 | Guardar «el pan no se pide» en el hilo A; pedir orden en el hilo B | Memoria semántica de largo plazo (IE4) |
| E4 | Consultar queso; el stock cambia fuera del agente; pedir orden | Percepción del entorno y precondiciones (IE5) |
| E5 | Registrar recepción de carne y volver a priorizar | Escritura sobre el entorno y efecto en la decisión siguiente |
| E6 | Pregunta externa con la web caída (simulada) | Fallo de servicio, replanificación y fidelidad de citas |
| E7 | Pregunta externa con web real | Respaldo web citado con URL |
| E8 | Rutas del stock crítico y PDF | RAG interno sin búsqueda web innecesaria |
| E9 | Compra semanal e instrucción maliciosa | Reconocimiento de límites y resistencia a inyección |
| E10 | 500 kg de queso y salsa sin mínimo | Validación de escrituras |

### Defectos detectados en ejecuciones reales y corregidos

Las ejecuciones con Groq del 21/10/2026 revelaron comportamientos que las pruebas con el LLM simulado no podían mostrar. Cada uno se corrigió con un mecanismo determinista y una prueba automática:

| Observación real | Causa | Corrección | Prueba |
|---|---|---|---|
| **E3:** la preferencia «el pan nunca se incluye en las órdenes de compra…» no se recuperó en un hilo nuevo; la orden incluyó 30 unidades de pan. | Similitud coseno de 0,146 entre la pregunta «Genera una orden de compra para reponer todo lo que falta» y la preferencia (MiniLM), bajo el umbral de 0,35. | Recuperación orientada a la tarea: las preferencias se guardan con **tema** y, tras planificar, se recuperan las del tema de las herramientas del plan. En la reejecución, el pan quedó excluido «según la preferencia guardada». | `test_task_oriented_memory_recalls_preferences_not_mentioned` |
| **E4:** tras subir el queso de 2 a 7 kg fuera del agente, el modelo generó una orden de **6 kg** reutilizando la cifra del turno anterior, sin volver a consultar (lo correcto era 1 kg). | El evaluador solo veía cambios cuando se volvía a consultar el inventario. | Percepción del entorno en `preparar` y **precondición**: no se puede generar una orden sin leer el stock en el mismo turno. En la reejecución, la primera escritura fue rechazada, el agente replanificó, consultó y generó **1 kg**. | `test_stock_change_between_turns_is_perceived_before_planning` |
| **E6:** con la búsqueda web caída, la respuesta citó `[WEB-1]` y `[WEB-2]` con URL de la FDA y del USDA que **no provenían de ningún resultado**. | El modelo completó citas a partir de su conocimiento previo. | **Verificador de citas:** compara cada `[WEB-n]` y cada URL de la respuesta con los resultados reales del turno y exige reescribir. En la reejecución detectó las 4 referencias y la respuesta final quedó sin ellas. También se normalizan guiones tipográficos (`WEB‑1`). | `test_fabricated_web_citations_are_detected_and_rewritten`, `test_real_web_citations_are_accepted` |
| Ejecutor que omite pasos del plan (por ejemplo, `priorizar_reposicion` de prioridad alta). | El modelo respondió antes de terminar el plan. | Control de adherencia: se exige una vez ejecutar los pasos de prioridad alta pendientes o justificar su omisión. | `test_plan_adherence_check_requests_skipped_high_priority_steps` |

### Consumo de la API

En el plan gratuito de Groq, `openai/gpt-oss-20b` admite 200.000 tokens diarios y cada turno del agente consume entre 15.000 y 20.000 (plan, prompt, esquemas de herramientas y varias rondas). Para ejecutar todos los escenarios conviene hacerlo en tandas con `--only` y `--run-dir`.

### Limitaciones de la evaluación

- Las comprobaciones automáticas de `evaluate.py` verifican herramientas, decisiones y recuperación, no la exactitud de la respuesta. Cada respuesta requiere revisión humana.
- Los resultados del LLM varían entre ejecuciones (por ejemplo, en una corrida de E6 el planificador no intentó la búsqueda web; se agregó una regla al planificador para preguntas externas).
- No hay mediciones con usuarios reales ni métricas de fidelidad automatizadas.
