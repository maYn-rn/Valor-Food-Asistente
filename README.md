# Valor Food Asistente IA

Prototipo académico de un asistente de inteligencia artificial para **Valor Food**, desarrollado como parte de la Evaluación Parcial 1 de la asignatura ISY0101.

El objetivo del proyecto es ayudar a consultar información de inventario, ventas y documentación de apoyo mediante un flujo RAG. El asistente trabaja con datos simulados y **no se conecta directamente al ERP real ni realiza compras**.

## ¿Cómo funciona?

El flujo general es:

1. El usuario escribe una pregunta en la interfaz de Streamlit.
2. El sistema intenta responder primero con información interna:
   - inventario simulado;
   - ventas simuladas;
   - documentos internos recuperados mediante FAISS.
3. Los documentos se representan con embeddings locales usando Sentence Transformers.
4. Si la información interna no es suficiente, el asistente puede realizar una búsqueda web de respaldo.
5. Groq genera la respuesta final usando el contexto recuperado.
6. La interfaz muestra la respuesta y la trazabilidad de las fuentes utilizadas.

### Arquitectura simplificada

```text
Usuario
  ↓
Streamlit
  ↓
Agente / Prompt
  ↓
├─ Inventario y ventas JSON
├─ FAISS + embeddings locales
└─ Búsqueda web de respaldo
  ↓
Groq
  ↓
Respuesta
```

## Requisitos

Se recomienda usar **Python 3.11 o 3.12**.

## Instalación

Desde la carpeta del proyecto:

```powershell
py -3.12 -m venv .venv
```

Instalar dependencias:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Crear el archivo de variables de entorno:

```powershell
Copy-Item .env.example .env
```

## Configuración del `.env`

```env
LLM_PROVIDER=groq
GROQ_API_KEY=TU_API_KEY_DE_GROQ
GROQ_MODEL=openai/gpt-oss-20b

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

WEB_SEARCH_ENABLED=true
```

## Crear el índice vectorial

Antes de realizar consultas con documentos internos, ejecutar:

```powershell
.\.venv\Scripts\python.exe create_vector_index.py
```

La primera ejecución puede tardar un poco porque el modelo de embeddings se descarga localmente.

## Iniciar la aplicación

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Luego abrir la dirección local que Streamlit muestre en el navegador.

## Ejemplos de preguntas

```text
¿Dónde veo el stock crítico?

¿Cuánto queso falta para llegar al stock mínimo?

¿Cuál fue el mes con mayor facturación en 2026?

¿Qué información existe sobre reposición de materias primas?
```

## Fuentes y prioridad de respuesta

El asistente sigue esta prioridad:

1. Datos internos de inventario o ventas.
2. Documentos internos recuperados con FAISS.
3. Búsqueda web externa solo cuando la información interna no es suficiente.

La búsqueda externa se usa como respaldo y no debe utilizarse para inventar información propia de Valor Food, como stock real, ventas reales, rutas internas o políticas no verificadas.

## Datos del prototipo

Los archivos de inventario y ventas son de demostración. Por esta razón, las respuestas relacionadas con cifras deben interpretarse como **datos simulados**.

El prototipo permite, entre otras cosas:

- calcular cuánto falta para llegar al stock mínimo;
- comparar facturación mensual;
- orientar sobre rutas documentadas del sistema;
- recuperar documentos mediante FAISS;
- usar búsqueda web como respaldo;
- mostrar trazabilidad de las fuentes utilizadas.

## Contexto académico

Valor Food es un ERP gastronómico utilizado en el caso analizado. El prototipo se desarrolló de manera independiente para demostrar el uso de:

- Prompt Engineering;
- LangChain;
- RAG;
- embeddings;
- FAISS;
- Groq como proveedor del LLM;
- Streamlit como interfaz;
- búsqueda web de respaldo.

El proyecto no reemplaza al ERP real y no está conectado a la base de datos productiva.

## Pruebas

Para ejecutar las pruebas automatizadas:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Las pruebas permiten revisar cálculos, recuperación, herramientas y carga básica de la aplicación, pero no sustituyen una validación manual de la calidad de las respuestas del modelo ni de las fuentes web.