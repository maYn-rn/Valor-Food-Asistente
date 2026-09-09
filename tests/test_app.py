from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_interface_loads_without_api():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=20)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ['Consultar con IA', 'Revisar datos sin IA', 'Instrucciones']
    assert len(app.dataframe) == 2
    app.button[0].click().run()
    assert not app.exception
    assert any('Escribe una pregunta' in warning.value for warning in app.warning)
