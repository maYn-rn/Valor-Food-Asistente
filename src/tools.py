"""Herramientas del agente, agrupadas por categoría.

La categoría determina la fase del plan (ver planner.PHASES): primero se consulta,
luego se razona sobre los datos y al final se escribe.
"""
from langchain_core.tools import tool
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, StrictInt
from . import operations as ops
from .web_search import search_web

CATEGORIES = {
    'consultar_inventario': 'consulta',
    'comparar_ventas': 'consulta',
    'buscar_documentos': 'consulta',
    'recordar': 'consulta',
    'buscar_web': 'respaldo',
    'priorizar_reposicion': 'razonamiento',
    'generar_orden_compra': 'escritura',
    'guardar_preferencia': 'escritura',
    'registrar_recepcion': 'escritura',
}


class InventoryArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str | None = Field(default=None, max_length=100, description='Ingrediente; omitir para todos.')


class SalesArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    year: StrictInt = Field(ge=2000, le=2100, description='Año explícitamente solicitado o acordado en la conversación.')


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1, max_length=1000)


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1, max_length=1000, description='Consulta concreta para buscar en Internet.')
    reason: str = Field(min_length=5, max_length=300,
                        description='Explica brevemente qué información faltó en las fuentes internas.')


class PrioritizeArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    excluir: list[str] | None = Field(default=None, max_length=20,
                                      description='Ingredientes a excluir (por pedido del usuario o preferencia guardada).')


class OrderLine(BaseModel):
    model_config = ConfigDict(extra='forbid')
    nombre: str = Field(min_length=1, max_length=100)
    cantidad: float | None = Field(default=None, gt=0, le=10000,
                                   description='Omitir para reponer exactamente hasta el mínimo.')


class OrderArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    lineas: list[OrderLine] | None = Field(default=None, max_length=30,
                                           description='Líneas explícitas; omitir para incluir todo lo que falta para el mínimo.')
    excluir: list[str] | None = Field(default=None, max_length=20, description='Ingredientes a excluir.')
    nota: str | None = Field(default=None, max_length=300)


class PreferenceArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    texto: str = Field(min_length=3, max_length=500,
                       description='Preferencia o regla operativa expresada por el usuario, redactada en tercera persona.')
    tema: Literal['ordenes_compra', 'reposicion', 'ventas', 'navegacion', 'general'] = Field(
        default='general', description='Tema al que aplica la preferencia; permite recuperarla cuando se ejecute esa tarea.')


class ReceiptArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    nombre: str = Field(min_length=1, max_length=100)
    cantidad: float = Field(gt=0, le=1000, description='Cantidad recibida en la unidad del ingrediente.')


def make_tools(retriever, memory, web_searcher=search_web, context=None):
    """context es un dict compartido con el grafo (por ejemplo, thread_id del turno actual)."""
    context = context if context is not None else {}

    @tool(args_schema=InventoryArgs)
    def consultar_inventario(name=None):
        """CONSULTA. Stock SIMULADO actual y cantidad para reponer hasta el mínimo. No predice compras futuras."""
        return ops.inventory(name)

    @tool(args_schema=SalesArgs)
    def comparar_ventas(year):
        """CONSULTA. Facturación bruta SIMULADA por mes de un año; no utilidad ni pedidos."""
        return ops.sales(year)

    @tool(args_schema=SearchArgs)
    def buscar_documentos(query):
        """CONSULTA. Busca documentación interna de Valor Food (rutas, reglas) con FAISS. Usar antes que Internet."""
        return retriever.search(query, 'internal')

    @tool(args_schema=SearchArgs)
    def recordar(query):
        """CONSULTA. Busca en la memoria de largo plazo preferencias, decisiones y conversaciones pasadas."""
        return memory.search(query, k=5, min_similarity=0.3)

    @tool(args_schema=WebSearchArgs)
    def buscar_web(query, reason):
        """RESPALDO. Búsqueda web global. Úsala SOLO si inventario, ventas, documentos y memoria no permiten responder."""
        try:
            result = web_searcher(query)
            if isinstance(result, dict):
                result = {**result, 'reason': reason}
            return result
        except Exception as exc:
            return {'query': query, 'source_type': 'web', 'results': [], 'reason': reason,
                    'error': f'Búsqueda web no disponible temporalmente ({type(exc).__name__}): {exc}'}

    @tool(args_schema=PrioritizeArgs)
    def priorizar_reposicion(excluir=None):
        """RAZONAMIENTO. Clasifica cada ingrediente (crítico/bajo/ok) según cobertura del mínimo y ordena qué reponer primero."""
        return ops.prioritize(excluir)

    @tool(args_schema=OrderArgs)
    def generar_orden_compra(lineas=None, excluir=None, nota=None):
        """ESCRITURA. Crea un BORRADOR de orden de compra (JSON y CSV) validado contra el inventario. No compra ni envía nada."""
        lines = None if lineas is None else [{'name': l['nombre'] if isinstance(l, dict) else l.nombre,
                                              'quantity': l['cantidad'] if isinstance(l, dict) else l.cantidad}
                                             for l in lineas]
        result = ops.create_purchase_order(lines, excluir, nota)
        if result.get('created'):
            summary = ', '.join(f"{l['name']} {l['quantity']:g} {l['unit']}" for l in result['lines'])
            rejected = '; '.join(f"{r['name']}: {r['reason']}" for r in result['rejected'])
            text = f"Se generó el borrador {result['order_id']} con: {summary}." + (f' Rechazados: {rejected}' if rejected else '')
            result['memory'] = memory.add('decision', text[:500], context.get('thread_id'), 'ordenes_compra')
        return result

    @tool(args_schema=PreferenceArgs)
    def guardar_preferencia(texto, tema='general'):
        """ESCRITURA. Guarda en memoria de largo plazo una preferencia o regla operativa que el usuario pidió recordar."""
        return memory.add('preferencia', texto, context.get('thread_id'), tema)

    @tool(args_schema=ReceiptArgs)
    def registrar_recepcion(nombre, cantidad):
        """ESCRITURA. Suma al stock SIMULADO mercadería que el usuario informa como recibida. Solo si el usuario lo indica."""
        return ops.register_receipt(nombre, cantidad)

    tools = [consultar_inventario, comparar_ventas, buscar_documentos, recordar, buscar_web,
             priorizar_reposicion, generar_orden_compra, guardar_preferencia, registrar_recepcion]
    assert {t.name for t in tools} == set(CATEGORIES)
    return tools
