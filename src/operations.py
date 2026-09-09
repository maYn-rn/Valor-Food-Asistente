import json
import unicodedata
from decimal import Decimal
from .settings import DATA

def normalize(value):
    return ''.join(c for c in unicodedata.normalize('NFD', value.lower()) if unicodedata.category(c) != 'Mn')

def inventory(name=None):
    if name is not None and (not isinstance(name, str) or len(name) > 100):
        raise ValueError('Nombre de ingrediente inválido.')
    items = json.loads((DATA / 'inventory.json').read_text(encoding='utf-8'))
    selected = []
    for item in items:
        if name and normalize(name) not in normalize(item['name']):
            continue
        minimum, current = item['minimum'], item['stock']
        buy = None if minimum is None else float(max(Decimal('0'), Decimal(str(minimum)) - Decimal(str(current))))
        selected.append({**item, 'buy': buy})
    return {'simulated': True, 'scope': 'Reposición hasta el mínimo; no pronóstico semanal ni compra automática', 'items': selected}

def sales(year):
    if type(year) is not int or not 2000 <= year <= 2100:
        raise ValueError('Indica un año entero entre 2000 y 2100.')
    records = json.loads((DATA / 'sales.json').read_text(encoding='utf-8'))
    grouped = {}
    for record in records:
        if record['status'] == 'paid' and record['date'].startswith(str(year)):
            month = record['date'][:7]
            grouped[month] = grouped.get(month, 0) + record['total']
    months = [{'month': k, 'total': v} for k, v in sorted(grouped.items())]
    high = max(grouped.values(), default=None)
    return {'simulated': True, 'year': year, 'currency': 'CLP', 'criterion': 'Facturación bruta de registros paid; no utilidad',
            'coverage': 'Solo meses con registros. Dataset simulado enero-junio de 2026; no representa meses completos reales.',
            'months': months, 'best_months': [row for row in months if row['total'] == high]}
