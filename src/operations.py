"""Operaciones deterministas sobre los datos SIMULADOS (sin LLM).

Las cifras se calculan en Python para que el modelo nunca invente cantidades.
El inventario mutable vive en runtime/inventario.json, sembrado desde data/inventory.json.
"""
import csv
import json
import shutil
import unicodedata
import uuid
from datetime import datetime
from decimal import Decimal
from .settings import DATA, runtime_dir

MAX_ORDER_FACTOR = 5  # Una línea no puede superar 5 veces el mínimo del ingrediente.


def normalize(value):
    return ''.join(c for c in unicodedata.normalize('NFD', value.lower()) if unicodedata.category(c) != 'Mn').strip()


def inventory_file():
    path = runtime_dir() / 'inventario.json'
    if not path.exists():
        shutil.copyfile(DATA / 'inventory.json', path)
    return path


def load_inventory():
    return json.loads(inventory_file().read_text(encoding='utf-8'))


def save_inventory(items):
    for item in items:
        if not item.get('name') or not isinstance(item.get('stock'), (int, float)) or item['stock'] < 0:
            raise ValueError('Cada ítem necesita nombre y stock numérico no negativo.')
        if item.get('minimum') is not None and (not isinstance(item['minimum'], (int, float)) or item['minimum'] < 0):
            raise ValueError('El mínimo debe ser numérico no negativo o vacío.')
    path = inventory_file()
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def reset_inventory():
    shutil.copyfile(DATA / 'inventory.json', inventory_file())


def _buy(item):
    if item['minimum'] is None:
        return None
    return float(max(Decimal('0'), Decimal(str(item['minimum'])) - Decimal(str(item['stock']))))


def _find(items, name):
    wanted = normalize(name)
    exact = [i for i in items if normalize(i['name']) == wanted]
    return exact or [i for i in items if wanted in normalize(i['name'])]


def inventory(name=None):
    if name is not None and (not isinstance(name, str) or len(name) > 100):
        raise ValueError('Nombre de ingrediente inválido.')
    items = load_inventory()
    selected = _find(items, name) if name else items
    return {'simulated': True, 'scope': 'Reposición hasta el mínimo; no pronóstico semanal ni compra automática',
            'items': [{**item, 'buy': _buy(item)} for item in selected]}


def classify(item):
    """Nivel de criticidad según la cobertura del mínimo (stock / mínimo)."""
    if item['minimum'] is None:
        return 'sin_minimo', None
    if item['minimum'] == 0:
        return 'ok', None
    coverage = item['stock'] / item['minimum']
    if coverage < 0.5:
        return 'critico', coverage
    if coverage < 1:
        return 'bajo', coverage
    return 'ok', coverage


def prioritize(exclude=None):
    """Ordena la reposición: crítico (<50 % del mínimo) > bajo (<100 %) > ok; sin mínimo aparte."""
    excluded = {normalize(x) for x in (exclude or [])}
    rank = {'critico': 0, 'bajo': 1, 'ok': 2}
    lines, without_minimum, skipped = [], [], []
    for item in load_inventory():
        if normalize(item['name']) in excluded:
            skipped.append(item['name'])
            continue
        level, coverage = classify(item)
        if level == 'sin_minimo':
            without_minimum.append(item['name'])
            continue
        lines.append({'name': item['name'], 'unit': item['unit'], 'stock': item['stock'], 'minimum': item['minimum'],
                      'buy': _buy(item), 'level': level, 'coverage_pct': round(coverage * 100, 1) if coverage is not None else None})
    lines.sort(key=lambda x: (rank[x['level']], x['coverage_pct'] if x['coverage_pct'] is not None else 999))
    return {'simulated': True,
            'criterion': 'crítico: stock < 50 % del mínimo; bajo: < 100 %; ok: cubre el mínimo',
            'priorities': lines,
            'to_buy': [x['name'] for x in lines if x['buy']],
            'without_minimum': without_minimum,
            'excluded': skipped}


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


def create_purchase_order(lines=None, exclude=None, note=None):
    """Escribe un BORRADOR de orden de compra (JSON + CSV). Nunca ejecuta compras.

    Sin líneas explícitas, propone reponer hasta el mínimo todo lo que falte,
    respetando exclusiones. Cada línea se valida contra el inventario actual.
    """
    items = load_inventory()
    excluded = {normalize(x) for x in (exclude or [])}
    if lines is None:
        lines = [{'name': x['name'], 'quantity': None} for x in items]
    accepted, rejected = [], []
    for line in lines:
        name, quantity = line.get('name', ''), line.get('quantity')
        found = _find(items, name) if name else []
        if len(found) != 1:
            rejected.append({'name': name, 'reason': 'Ingrediente inexistente o ambiguo en el inventario.'})
            continue
        item = found[0]
        if normalize(item['name']) in excluded:
            rejected.append({'name': item['name'], 'reason': 'Excluido por indicación del usuario o preferencia guardada.'})
            continue
        if quantity is None:
            quantity = _buy(item)
            if quantity is None:
                rejected.append({'name': item['name'], 'reason': 'Sin stock mínimo definido: no se puede calcular la cantidad.'})
                continue
            if quantity == 0:
                continue  # Ya cubre el mínimo; no es un rechazo.
        if not isinstance(quantity, (int, float)) or quantity <= 0:
            rejected.append({'name': item['name'], 'reason': 'La cantidad debe ser mayor que cero.'})
            continue
        limit = (item['minimum'] or 0) * MAX_ORDER_FACTOR
        if limit and quantity > limit:
            rejected.append({'name': item['name'], 'reason': f'Cantidad {quantity} supera el límite de control ({limit} {item["unit"]}).'})
            continue
        line = {'name': item['name'], 'quantity': float(quantity), 'unit': item['unit'],
                'stock_at_order': item['stock'], 'minimum': item['minimum']}
        needed = _buy(item)
        if needed is not None and quantity > needed:
            line['warning'] = f'Supera lo necesario para el mínimo ({needed:g} {item["unit"]} con el stock actual).'
        accepted.append(line)
    if not accepted:
        return {'created': False, 'simulated': True, 'reason': 'No hay líneas válidas para la orden.', 'rejected': rejected}
    stamp = datetime.now()
    order_id = f"OC-{stamp:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4].upper()}"
    folder = runtime_dir() / 'ordenes'
    folder.mkdir(exist_ok=True)
    order = {'id': order_id, 'status': 'BORRADOR — requiere aprobación humana; no se envió a proveedores',
             'created_at': stamp.isoformat(timespec='seconds'), 'simulated': True, 'note': note or '',
             'lines': accepted, 'rejected': rejected}
    (folder / f'{order_id}.json').write_text(json.dumps(order, ensure_ascii=False, indent=2), encoding='utf-8')
    with open(folder / f'{order_id}.csv', 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle, delimiter=';')
        writer.writerow(['ingrediente', 'cantidad', 'unidad', 'stock_al_generar', 'minimo'])
        for line in accepted:
            writer.writerow([line['name'], line['quantity'], line['unit'], line['stock_at_order'], line['minimum']])
    return {'created': True, 'simulated': True, 'order_id': order_id, 'status': order['status'],
            'files': [f'runtime/ordenes/{order_id}.json', f'runtime/ordenes/{order_id}.csv'],
            'lines': accepted, 'rejected': rejected}


def list_orders():
    folder = runtime_dir() / 'ordenes'
    if not folder.exists():
        return []
    return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('OC-*.json'), reverse=True)]


def register_receipt(name, quantity):
    """Suma mercadería recibida al stock SIMULADO (escritura sobre runtime/inventario.json)."""
    if not isinstance(quantity, (int, float)) or not 0 < quantity <= 1000:
        raise ValueError('La cantidad recibida debe estar entre 0 y 1000.')
    items = load_inventory()
    found = _find(items, name)
    if len(found) != 1:
        return {'updated': False, 'error': f'Ingrediente "{name}" inexistente o ambiguo.'}
    item = found[0]
    before = item['stock']
    item['stock'] = float(Decimal(str(before)) + Decimal(str(quantity)))
    if item['stock'].is_integer():
        item['stock'] = int(item['stock'])
    save_inventory(items)
    return {'updated': True, 'simulated': True, 'name': item['name'], 'unit': item['unit'],
            'stock_before': before, 'stock_after': item['stock'], 'buy_now': _buy(item)}
