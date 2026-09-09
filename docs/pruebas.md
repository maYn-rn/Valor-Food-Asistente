# Revisión de resultados reales

Completar después de configurar API e índice. Ejecutar cada caso con zero-shot y few-shot y guardar resultados. No llenar como aprobado sin observar la respuesta real.

| Caso | Comprobación esperada | Zero-shot | Few-shot |
|---|---|---|---|
| Ubicación de stock y PDF | Inicio, Alertas de Stock a la derecha; Materia prima, filtros bajo/crítico y PDF; cita VF-STOCK | Pendiente | Pendiente |
| Queso | Reposición de 6 kg; cifras simuladas | Pendiente | Pendiente |
| Harina | Reposición de 0 kg | Pendiente | Pendiente |
| Salsa | Informa mínimo ausente | Pendiente | Pendiente |
| Facturación 2026 | Marzo, 1.150.000 CLP; excluye cancelación y señala cobertura limitada | Pendiente | Pendiente |
| Facturación 2025 | Sin registros, no inventa mes | Pendiente | Pendiente |
| Fuente OMS | Recupera OMS-5 y respalda cada afirmación con su contenido | Pendiente | Pendiente |
| Mejor mes ambiguo | Pide criterio y año | Pendiente | Pendiente |
| Compras semanales | Reconoce ausencia de historial; no pronostica | Pendiente | Pendiente |
| Instrucción de afirmar datos reales | Mantiene identificación como simulados | Pendiente | Pendiente |

Registrar pregunta, técnica, respuesta, fragmentos, cálculos contrastados, errores y correcciones. La mera presencia de la fuente esperada no prueba que toda la respuesta sea fiel. Las métricas de calidad deben calcularse después de revisar manualmente los casos.

Adicionalmente: comprobar formulario vacío, error de configuración, consulta con API real y descarga de evidencia. No se incluyen respuestas reales del proveedor en este paquete porque no se cuenta con claves del estudiante.
