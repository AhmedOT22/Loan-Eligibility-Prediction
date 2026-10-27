"""Exercise real release integrity, raw input contracts, and training/serving parity."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from build_serving import build_release
from src.models.pipeline import split_raw
from src.models.serving import MODEL_NAMES, SERVING_DIR, load_bundle, predict_score
from train_model import score_classifier

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", MODEL_NAMES)
def test_release_matches_published_metrics_and_does_not_fit(kind, monkeypatch, applicant):
    """Both exact serving artifacts reproduce the report and retain fitted statistics."""
    model, manifest = load_bundle(kind)
    _, holdout, _, target = split_raw(pd.read_csv(ROOT / "data/raw/credit.csv"))
    actual = score_classifier(model, holdout, target)
    published = json.loads((ROOT / "reports/baseline/metrics.json").read_text())[kind]
    assert actual == published == manifest["models"][kind]["holdout_metrics"]
    imputer = model["preprocess"].named_transformers_["numeric"]["impute"]
    before = imputer.statistics_.copy()

    def reject_fit(*_args, **_kwargs):
        raise AssertionError("Inference must not fit preprocessing or the classifier")

    monkeypatch.setattr(model, "fit", reject_fit)
    score = predict_score(applicant, kind, model, manifest)
    assert 0 <= score <= 100
    assert score == predict_score(applicant, kind, model, manifest)
    np.testing.assert_array_equal(imputer.statistics_, before)


@pytest.mark.parametrize(
    "change",
    [
        {"LoanAmount": np.inf},
        {"ApplicantIncome": -1},
        {"LoanAmount": 0},
        {"Loan_Amount_Term": 0},
        {"Credit_History": 2},
        {"Gender": "unknown"},
        {"LoanAmount": True},
        {"Dependents": None},
        {"LoanAmount": np.nan},
    ],
)
def test_reject_invalid_applicants(change, applicant):
    """Malformed complete inputs must not produce apparently usable model scores."""
    model, manifest = load_bundle("logistic_regression")
    with pytest.raises(ValueError):
        predict_score({**applicant, **change}, "logistic_regression", model, manifest)


def test_reject_missing_or_extra_fields(applicant):
    """Do not silently impute missing form fields or discard additional columns."""
    model, manifest = load_bundle("logistic_regression")
    for incomplete in (
        {**applicant, "Loan_ID": "example"},
        {k: v for k, v in applicant.items() if k != "Gender"},
    ):
        with pytest.raises(ValueError, match="eleven"):
            predict_score(incomplete, "logistic_regression", model, manifest)


def test_reject_corrupt_release_and_wrong_version(tmp_path):
    """Verify checks occur before unpickling damaged or incompatible files."""
    directory = tmp_path / "release"
    shutil.copytree(SERVING_DIR, directory)
    artifact = directory / "logistic_regression_pipeline.joblib"
    artifact.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle("logistic_regression", directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["packages"]["scikit-learn"] = "0.0.0"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="version"):
        load_bundle("logistic_regression", directory)


def test_builder_refuses_overwrite(tmp_path):
    """A rebuild cannot silently overwrite an existing serving release."""
    sentinel = tmp_path / "manifest.json"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        build_release(ROOT / "data/raw/credit.csv", tmp_path)
    assert sentinel.read_text() == "preserve"
