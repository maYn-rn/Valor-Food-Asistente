# Fuentes y referencias

## Fuentes de datos del agente

| ID | Tipo | Origen y estado |
|---|---|---|
| VF-STOCK | Interna (RAG) | Rutas descritas por Matías Medina: Inicio → Alertas de Stock (derecha); Materia prima → filtros de stock bajo/crítico y PDF de stock por comprar. No es un manual formal del ERP. |
| VF-DEMO | Interna (RAG) | Especificación del prototipo con datos simulados. No es una política de Crispy House. |
| OMS-5 | Externa (indexada, filtrada fuera de la recuperación interna) | Resumen propio de la OMS, consultado el 09/09/2026. |
| `data/inventory.json`, `data/sales.json` | Simulados | Datos semilla de demostración; el inventario mutable se copia a `runtime/`. |
| WEB-n | Externa bajo demanda | Resultados DDGS (título, URL, extracto) registrados en la traza; no se indexan. |
| Memoria de largo plazo | Generada en uso | Preferencias y decisiones que el usuario expresa en la conversación; local, en `runtime/`. |

## Referencias (APA 7)

### Frameworks y herramientas

Chase, H. (2022). *LangChain* [Software]. LangChain, Inc. https://github.com/langchain-ai/langchain

Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.-E., Lomeli, M., Hosseini, L., & Jégou, H. (2024). *The Faiss library*. arXiv. https://doi.org/10.48550/arXiv.2401.08281

Groq. (s. f.). *GroqCloud documentation*. Recuperado el 21 de octubre de 2026, de https://console.groq.com/docs

LangChain. (s. f.-a). *LangGraph overview*. Recuperado el 21 de octubre de 2026, de https://docs.langchain.com/oss/python/langgraph/overview

LangChain. (s. f.-b). *Persistence* [Documentación de LangGraph]. Recuperado el 21 de octubre de 2026, de https://docs.langchain.com/oss/python/langgraph/persistence

LangChain. (s. f.-c). *Memory* [Documentación de LangGraph]. Recuperado el 21 de octubre de 2026, de https://docs.langchain.com/oss/python/langgraph/add-memory

OpenAI. (2025). *gpt-oss-120b & gpt-oss-20b model card*. arXiv. https://doi.org/10.48550/arXiv.2508.10925

Pydantic. (s. f.). *Pydantic documentation*. Recuperado el 21 de octubre de 2026, de https://docs.pydantic.dev

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. En *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 3982–3992). Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1410

Snowflake Inc. (s. f.). *Streamlit documentation*. Recuperado el 21 de octubre de 2026, de https://docs.streamlit.io

### Fundamentos de agentes, memoria y recuperación

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474.

Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., & Gonzalez, J. E. (2023). *MemGPT: Towards LLMs as operating systems*. arXiv. https://doi.org/10.48550/arXiv.2310.08560

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. En *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23)*. ACM. https://doi.org/10.1145/3586183.3606763

Wang, L., Xu, W., Lan, Y., Hu, Z., Lan, Y., Lee, R. K.-W., & Lim, E.-P. (2023). Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. En *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics* (pp. 2609–2634). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.acl-long.147

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. En *International Conference on Learning Representations (ICLR 2023)*. https://doi.org/10.48550/arXiv.2210.03629

### Fuente de dominio

World Health Organization. (2006). *Five keys to safer food manual*. https://www.who.int/publications/i/item/9789241594639

### Material de la asignatura

Macaya, F. (s. f.). *INGENIERIA-DE-SOLUCIONES-CON-INTELIGENCIA-ARTIFICIAL_002D_OLS* [Repositorio de GitHub]. https://github.com/Foco22/INGENIERIA-DE-SOLUCIONES-CON-INTELIGENCIA-ARTIFICIAL_002D_OLS

### Uso de inteligencia artificial

Anthropic. (2026). *Claude Code* (Claude Opus 5.5) [Modelo de lenguaje de gran tamaño]. https://claude.com/claude-code

> Declaración: Claude Code se usó como asistente de programación y documentación (implementación del grafo, herramientas y memoria, pruebas, diagramas, README y borradores del informe, incluidas las secciones de justificación técnica y conclusiones, con autorización del docente). El estudiante definió el caso y los requerimientos, ejecutó y validó el código y sus resultados, y revisó el contenido generado.
