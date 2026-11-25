"""Smoke test: the Streamlit app runs every tab without raising an exception.

Requires the processed data and the trained files in models/saved/.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.mark.skipif(not (ROOT / 'models' / 'saved' / 'artifacts.joblib').exists(),
                    reason='run python models/train.py first')
def test_app_runs_all_tabs():
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
    assert not at.exception
    # Older training season (in-sample warning branch), then submit a prediction.
    at.sidebar.selectbox[0].select('2015').run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any(m.label == 'Estimated salary' for m in at.metric)
