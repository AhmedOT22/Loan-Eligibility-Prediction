"""Exercise the real Streamlit form against each bundled classifier."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.models.serving import load_bundle, predict_score
from src.utils import explainer

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.parametrize("choice", ["Logistic Regression", "Random Forest"])
def test_submit_applicant_from_another_working_directory(
    monkeypatch, tmp_path, choice, applicant
):
    """Load resources and complete a submission independently of the launch directory."""
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert not app.error
    app.selectbox[0].select(choice).run(timeout=30)
    next(widget for widget in app.selectbox if widget.label == "Self Employed").select("No")
    app.number_input[0].set_value(5000)
    app.number_input[1].set_value(1500)
    app.number_input[2].set_value(150)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert not (tmp_path / "loan_app.log").exists()
    assert not any("approval chances" in item.value for item in app.info)
    assert any("Release: train-only-v1" in item.value for item in app.caption)
    assert any(header.value == "📊 Model Score" for header in app.subheader)
    kind = "logistic_regression" if choice.startswith("Logistic") else "random_forest"
    model, manifest = load_bundle(kind)
    expected = predict_score(applicant, kind, model, manifest)
    figure = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert figure["data"][0]["value"] == pytest.approx(expected)
    assert any("Holdout accuracy:" in item.value for item in app.markdown)


def test_optional_explainer_has_clear_missing_dependency_error(monkeypatch):
    """The unused optional SHAP helper must not break ordinary application imports."""
    monkeypatch.setattr(explainer, "shap", None)
    with pytest.raises(ImportError, match="Install SHAP separately"):
        explainer.explain_prediction(None, None)


def test_unavailable_release_stops_without_legacy_fallback(monkeypatch):
    """A damaged release must never silently switch to a differently evaluated model."""

    def reject_release(*_args, **_kwargs):
        raise ValueError("corrupt release")

    monkeypatch.setattr("src.models.serving.load_bundle", reject_release)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert len(app.error) == 1
    assert "unavailable" in app.error[0].value
    assert not app.button
