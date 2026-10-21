from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def test_interface_loads_without_api():
    app = AppTest.from_file(APP).run(timeout=30)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ['Agente', 'Entorno simulado', 'Memoria', 'Instrucciones']
    assert app.sidebar.selectbox[0].value.startswith('conv-')
    # Prioridad de reposición + facturación (el inventario editable es un data_editor aparte).
    assert len(app.dataframe) >= 2
    app.chat_input[0].set_value('   ').run()
    assert not app.exception
    assert any('Escribe una pregunta' in warning.value for warning in app.warning)


def test_reset_inventory_button(isolated_runtime):
    from src import operations as ops
    ops.register_receipt('Queso', 3)
    app = AppTest.from_file(APP).run(timeout=30)
    next(b for b in app.button if b.label == 'Restablecer datos de demostración').click().run()
    assert not app.exception
    assert ops.inventory('Queso')['items'][0]['stock'] == 2
