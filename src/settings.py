import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
INDEX = ROOT / 'vector_index'


def load_settings():
    load_dotenv(ROOT / '.env', override=True)


def runtime_dir():
    """Carpeta de estado mutable del agente (inventario simulado, memoria, órdenes).

    Se separa de data/ para que los datos semilla del repositorio nunca se
    modifiquen. VALOR_RUNTIME_DIR permite aislar cada prueba en una carpeta temporal.
    """
    path = Path(os.getenv('VALOR_RUNTIME_DIR', str(ROOT / 'runtime')))
    path.mkdir(parents=True, exist_ok=True)
    return path


def env_int(name, default, low, high):
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f'{name} debe ser un número entero.') from exc
    if not low <= value <= high:
        raise ValueError(f'{name} debe estar entre {low} y {high}.')
    return value


def embedding_signature():
    """Identificador persistido junto a los índices para detectar cambios de configuración."""
    load_settings()
    provider = os.getenv('EMBEDDING_PROVIDER', 'local').strip().lower()
    default_model = (
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        if provider == 'local'
        else 'text-embedding-3-small'
    )
    model = os.getenv('EMBEDDING_MODEL', default_model).strip()
    return f'{provider}:{model}'


def get_embeddings():
    """Devuelve embeddings locales por defecto u OpenAI de forma opcional.

    El modo local evita cuotas/API keys. La primera ejecución puede descargar el
    modelo de Hugging Face; luego queda almacenado en caché en el equipo.
    """
    load_settings()
    provider = os.getenv('EMBEDDING_PROVIDER', 'local').strip().lower()

    if provider == 'local':
        model = os.getenv(
            'EMBEDDING_MODEL',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        ).strip()
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ValueError(
                'Falta langchain-huggingface. Ejecuta: '
                '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'
            ) from exc
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )

    if provider == 'openai':
        if not os.getenv('OPENAI_API_KEY', '').strip():
            raise ValueError(
                'EMBEDDING_PROVIDER=openai requiere OPENAI_API_KEY en .env. '
                'Para evitar cuotas usa EMBEDDING_PROVIDER=local.'
            )
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            request_timeout=45,
            max_retries=1,
        )

    raise ValueError('EMBEDDING_PROVIDER debe ser local u openai.')


def model_name():
    load_settings()
    if os.getenv('LLM_PROVIDER', 'groq').lower() == 'groq':
        return os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')
    return os.getenv('OPENAI_MODEL', 'gpt-4o-mini')


def get_llm():
    load_settings()
    provider = os.getenv('LLM_PROVIDER', 'groq').lower()
    if provider == 'openai':
        if not os.getenv('OPENAI_API_KEY', '').strip():
            raise ValueError('Configura OPENAI_API_KEY en .env para usar el modelo de chat.')
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            temperature=0,
            timeout=45,
            max_retries=1,
        )
    if provider == 'groq':
        if not os.getenv('GROQ_API_KEY', '').strip():
            raise ValueError('Configura GROQ_API_KEY en .env para usar Groq.')
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b'),
            temperature=0,
            timeout=45,
            max_retries=2,
        )
    raise ValueError('LLM_PROVIDER debe ser openai o groq.')
