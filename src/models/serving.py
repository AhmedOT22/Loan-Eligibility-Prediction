"""Load versioned raw-input pipelines and validate a complete demonstration input."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import BASE_DIR
from src.models.pipeline import CATEGORICAL, NUMERIC

SERVING_DIR = BASE_DIR / "models/serving/v1"
MODEL_NAMES = ("logistic_regression", "random_forest")


def load_bundle(kind, directory=SERVING_DIR):
    """Check trusted release metadata, dependency version and checksum before loading."""
    if kind not in MODEL_NAMES:
        raise ValueError("Unknown serving model.")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1 or manifest["features"] != CATEGORICAL + NUMERIC:
        raise ValueError("Unsupported serving schema; rebuild the release with this code.")
    if manifest["packages"]["scikit-learn"] != version("scikit-learn"):
        raise ValueError(
            "Serving scikit-learn version differs from the release; install requirements."
        )
    artifact = directory / f"{kind}_pipeline.joblib"
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["models"][kind]["sha256"]:
        raise ValueError("Serving artifact checksum mismatch; restore or rebuild the release.")
    model = joblib.load(artifact)
    if not isinstance(model, Pipeline) or list(model.feature_names_in_) != manifest["features"]:
        raise ValueError("Serving pipeline feature schema does not match its manifest.")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Serving classifier must use classes 0 and 1.")
    encoder = model["preprocess"].named_transformers_["categorical"]["encode"]
    categories = {
        name: values.tolist() for name, values in zip(CATEGORICAL, encoder.categories_)
    }
    if categories != manifest["models"][kind]["categories"]:
        raise ValueError("Serving category schema does not match its manifest.")
    return model, manifest


def prepare_applicant(values, categories):
    """Reject missing, extra, nonfinite and unsupported inputs without fitting anything."""
    if set(values) != set(CATEGORICAL + NUMERIC):
        raise ValueError("Provide exactly the eleven applicant fields in the form.")
    frame = pd.DataFrame([values], columns=CATEGORICAL + NUMERIC)
    if frame.isna().any().any():
        raise ValueError("Complete all applicant fields before calculating a score.")
    for name in CATEGORICAL:
        if not isinstance(values[name], str) or values[name] not in categories[name]:
            raise ValueError(f"Unsupported category for {name}.")
    if any(isinstance(values[name], (bool, np.bool_)) for name in NUMERIC):
        raise ValueError("Numeric fields cannot be boolean values.")
    frame[NUMERIC] = frame[NUMERIC].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[NUMERIC].to_numpy(dtype=float)).all():
        raise ValueError("Numeric fields must contain finite values.")
    if (frame[NUMERIC] < 0).any().any():
        raise ValueError("Numeric fields must be non-negative.")
    if frame.at[0, "Credit_History"] not in (0, 1):
        raise ValueError("Credit history must be 0 or 1.")
    if frame.at[0, "LoanAmount"] <= 0 or frame.at[0, "Loan_Amount_Term"] <= 0:
        raise ValueError("Loan amount and term must be positive.")
    return frame


def predict_score(values, kind, model, manifest):
    """Return an uncalibrated positive-class score from one complete raw-input pipeline."""
    features = prepare_applicant(values, manifest["models"][kind]["categories"])
    score = float(model.predict_proba(features)[0, 1])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Model returned an invalid score.")
    return 100 * score
