"""Exercise the real Streamlit form against each bundled classifier."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.utils import explainer

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.parametrize("choice", ["Logistic Regression (Recommended)", "Random Forest"])
def test_submit_applicant_from_another_working_directory(monkeypatch, tmp_path, choice):
    """Load resources and complete a submission independently of the launch directory."""
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert not app.error
    app.selectbox[0].select(choice).run(timeout=30)
    app.number_input[0].set_value(5000)
    app.number_input[1].set_value(1500)
    app.number_input[2].set_value(150)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert any(header.value == "📊 Model Score" for header in app.subheader)


def test_optional_explainer_has_clear_missing_dependency_error(monkeypatch):
    """The unused optional SHAP helper must not break ordinary application imports."""
    monkeypatch.setattr(explainer, "shap", None)
    with pytest.raises(ImportError, match="Install SHAP separately"):
        explainer.explain_prediction(None, None)
