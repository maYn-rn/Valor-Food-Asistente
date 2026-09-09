import os
from .settings import load_settings


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name, 'true' if default else 'false').strip().lower()
    return value in {'1', 'true', 'yes', 'si', 'sí', 'on'}


def search_web(query: str):
    """Realiza una búsqueda web de texto como respaldo de las fuentes internas.

    No usa una API key adicional. DDGS funciona como metabuscador y devuelve
    título, URL y fragmento de resultados. La búsqueda solo debe invocarse
    cuando las fuentes internas no permitan responder de forma suficiente.
    """
    if not isinstance(query, str) or not query.strip() or len(query) > 1000:
        raise ValueError('Consulta web inválida.')

    load_settings()
    if not _env_bool('WEB_SEARCH_ENABLED', True):
        return {
            'query': query,
            'source_type': 'web',
            'provider': 'DDGS metasearch',
            'results': [],
            'error': 'La búsqueda web está desactivada en .env.',
        }

    try:
        max_results = int(os.getenv('WEB_SEARCH_MAX_RESULTS', '5'))
    except ValueError as exc:
        raise ValueError('WEB_SEARCH_MAX_RESULTS debe ser un número entero.') from exc
    if not 1 <= max_results <= 10:
        raise ValueError('WEB_SEARCH_MAX_RESULTS debe estar entre 1 y 10.')

    region = os.getenv('WEB_SEARCH_REGION', 'cl-es').strip() or 'cl-es'
    backend = os.getenv('WEB_SEARCH_BACKEND', 'auto').strip() or 'auto'

    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise ValueError(
            'Falta la dependencia ddgs. Ejecuta: '
            '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'
        ) from exc

    raw = DDGS(timeout=10).text(
        query.strip(),
        region=region,
        safesearch='moderate',
        max_results=max_results,
        backend=backend,
    )

    results = []
    for row in raw or []:
        url = str(row.get('href') or row.get('url') or '').strip()
        if not url.startswith(('http://', 'https://')):
            continue
        title = str(row.get('title') or 'Resultado sin título').strip()[:300]
        snippet = str(row.get('body') or row.get('snippet') or row.get('description') or '').strip()[:1800]
        results.append({
            'web_id': f'WEB-{len(results) + 1}',
            'title': title,
            'url': url,
            'snippet': snippet,
        })
        if len(results) >= max_results:
            break

    return {
        'query': query.strip(),
        'source_type': 'web',
        'provider': 'DDGS metasearch',
        'region': region,
        'results': results,
    }
