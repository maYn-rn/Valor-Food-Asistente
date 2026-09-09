# Fuentes del prototipo

| ID | Tipo | Origen y estado |
|---|---|---|
| VF-STOCK | Interna | Rutas descritas por Matías: Inicio, Alertas de Stock a la derecha; sección Materia prima, filtros de stock bajo/crítico y generación de PDF. Revisar redacción contra la app real. |
| VF-DEMO | Interna | Especificación del prototipo con datos simulados. No es una política de Crispy House. |
| OMS-5 | Externa | Resumen propio en español de la sección Overview de una publicación de la OMS, consultada el 09/09/2026. |

World Health Organization. (2006, May 15). *Five keys to safer food manual*. https://www.who.int/publications/i/item/9789241594639

La fuente externa se usa para una consulta general sobre manejo seguro de materias primas. No sustenta cantidades de reposición, precios ni normativa chilena. Se almacena un resumen breve con atribución, no el manual completo. Es una instantánea local, no una búsqueda web en tiempo real. Debe revisarse su pertinencia al defender el alcance del proyecto. Si se agrega un catálogo real de proveedor, registrar origen, fecha y condiciones de uso.

El directorio de proveedores mostrado en una captura del ERP es un registro interno; no se lo presenta como documentación externa. No se incorporan direcciones personales del cliente al dataset.

Referencia técnica consultada para la integración: OpenAI. (s. f.). *Text generation*. https://developers.openai.com/api/docs/guides/text

## Referencias de clase para esta adaptación

Repositorio del docente, revisión del commit `d9e1b78060417a17688ac92eb92b62e4bd7ecdf1`:
https://github.com/Foco22/INGENIERIA-DE-SOLUCIONES-CON-INTELIGENCIA-ARTIFICIAL_002D_OLS

Se revisaron los notebooks de clase 1.2 (ChatPromptTemplate, ChatOpenAI y ChatGroq), clase 1.3 (embeddings, RecursiveCharacterTextSplitter y FAISS) y app.py de clase 1.4 (Streamlit). La estructura se adapta al caso Valor Food; no se copia un sistema completo del docente ni se incorporan sus claves o datos.

LangChain. (s. f.). *ChatOpenAI integration*. https://docs.langchain.com/oss/python/integrations/chat/openai

Los paquetes del entorno verificado y sus versiones están en requirements.txt. Los imports se ajustan a esos paquetes. Antes de entregar, revisar las referencias en el formato APA solicitado y declarar la asistencia de IA.

## Búsqueda web dinámica

Desde v0.3, las consultas externas se realizan como respaldo mediante DDGS y se registran con URL en la traza. La recuperación FAISS inicial se restringe a fuentes internas para evitar que una fuente general, como la OMS, desplace documentación propia de Valor Food.
