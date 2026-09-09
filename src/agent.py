import json
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ConfigDict, StrictInt
from .operations import inventory, sales
from .prompts import make_prompt
from .web_search import search_web


class InventoryArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str | None = Field(default=None, max_length=100, description='Ingrediente; omitir para todos.')


class SalesArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    year: StrictInt = Field(ge=2000, le=2100, description='Año explícitamente solicitado.')


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1, max_length=1000)


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1, max_length=1000, description='Consulta concreta para buscar en Internet.')
    reason: str = Field(
        min_length=5,
        max_length=300,
        description='Explica brevemente qué información faltó en las fuentes internas.',
    )


def make_tools(retriever, web_searcher=search_web):
    @tool(args_schema=InventoryArgs)
    def consultar_inventario(name=None):
        """Obtiene stock SIMULADO y reposición hasta el mínimo. No predice compras futuras."""
        return inventory(name)

    @tool(args_schema=SalesArgs)
    def comparar_ventas(year):
        """Compara facturación bruta SIMULADA por mes de un año; no utilidad."""
        return sales(year)

    @tool(args_schema=SearchArgs)
    def buscar_documentos(query):
        """Busca SOLO documentación interna de Valor Food con FAISS. Debe usarse antes de recurrir a Internet."""
        return retriever.search(query, 'internal')

    @tool(args_schema=WebSearchArgs)
    def buscar_web(query, reason):
        """Búsqueda web GLOBAL de respaldo. Úsala SOLO si inventario, ventas y documentos internos no permiten responder."""
        try:
            result = web_searcher(query)
            if isinstance(result, dict):
                result = {**result, 'reason': reason}
            return result
        except Exception as exc:
            return {
                'query': query,
                'source_type': 'web',
                'results': [],
                'reason': reason,
                'error': f'Búsqueda web no disponible temporalmente ({type(exc).__name__}): {exc}',
            }

    return [consultar_inventario, comparar_ventas, buscar_documentos, buscar_web]


class Assistant:
    def __init__(self, llm, retriever, web_searcher=search_web):
        self.retriever = retriever
        self.tools = {t.name: t for t in make_tools(retriever, web_searcher)}
        self.model = llm.bind_tools(list(self.tools.values()))

    def ask(self, question, technique='zero_shot'):
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise ValueError('Escribe una pregunta de 1 a 1000 caracteres.')

        # La recuperación inicial es deliberadamente SOLO interna. De este modo
        # una fuente externa nunca desplaza a información propia de Valor Food.
        initial = self.retriever.search(question, 'internal')
        trace = [{'tool': 'buscar_documentos', 'arguments': {'query': question}, 'result': initial}]
        messages = make_prompt(technique).format_messages(
            context=json.dumps(initial, ensure_ascii=False),
            question=question,
        )
        tokens = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}

        for _ in range(4):
            reply = self.model.invoke(messages)
            usage = getattr(reply, 'usage_metadata', None) or {}
            for key in tokens:
                tokens[key] += usage.get(key, 0)
            messages.append(reply)
            calls = getattr(reply, 'tool_calls', [])

            if not calls:
                content = reply.content
                answer = content if isinstance(content, str) else '\n'.join(
                    b.get('text', '') for b in content if isinstance(b, dict)
                )
                if not answer.strip():
                    raise RuntimeError('El modelo no devolvió texto.')
                return {
                    'question': question,
                    'answer': answer,
                    'technique': technique,
                    'simulated_data': True,
                    'trace': trace,
                    'usage': tokens,
                    'validation': 'Respuesta pendiente de revisión humana.',
                }

            if len(calls) > 6:
                raise RuntimeError('Demasiadas herramientas solicitadas en una respuesta.')

            for call in calls:
                name, args = call['name'], call['args']
                if name not in self.tools:
                    result = {'error': 'Herramienta no permitida.'}
                else:
                    try:
                        result = self.tools[name].invoke(args)
                    except (ValueError, TypeError) as exc:
                        result = {'error': str(exc)}
                trace.append({'tool': name, 'arguments': args, 'result': result})
                messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call['id']))

        raise RuntimeError('Se alcanzó el máximo de cuatro rondas; reformula la pregunta.')
