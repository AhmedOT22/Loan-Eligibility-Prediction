"""Regression tests for training boundaries and legacy input encoding."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from src.models.pipeline import NUMERIC, make_pipeline, split_raw
from src.models.prediction import predict

DATA = Path(__file__).resolve().parents[1] / "data/raw/credit.csv"


def test_train_only_statistics_and_unknown_category():
    """Check fitting statistics stay fixed when transforming unseen observations."""
    x_train, x_test, y_train, _ = split_raw(pd.read_csv(DATA))
    model = make_pipeline("logistic_regression").fit(x_train, y_train)
    preprocessor = model["preprocess"]
    assert isinstance(preprocessor, ColumnTransformer)
    imputer = getattr(preprocessor, "named_transformers_")["numeric"]["impute"]
    np.testing.assert_allclose(imputer.statistics_, x_train[NUMERIC].median().to_numpy())
    before = imputer.statistics_.copy()
    changed = x_test.iloc[:2].copy()
    changed.loc[:, "LoanAmount"] = 1e12
    changed.loc[:, "Property_Area"] = "Unseen Area"
    assert np.isfinite(model.predict_proba(changed)).all()
    np.testing.assert_array_equal(before, imputer.statistics_)
    assert set(x_train.index).isdisjoint(x_test.index)


def test_single_applicant_encoding_does_not_refit_reference_data():
    """Verify legacy encoding ignores reference-table contents and rejects missing inputs."""
    raw = pd.read_csv(DATA)
    # Build a legacy-compatible complete numeric fixture, without loading pickles.
    fields = raw.drop(columns=["Loan_ID", "Loan_Approved"]).dropna().iloc[:100]
    encoded = pd.get_dummies(
        fields,
        columns=["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"],
        dtype=int,
    )
    scaler = MinMaxScaler().fit(encoded)
    model = LogisticRegression().fit(scaler.transform(encoded), np.arange(len(encoded)) % 2)
    applicant = fields.iloc[0].to_dict()
    first, row = predict(applicant, pd.DataFrame(), model, scaler, encoded.columns.tolist())
    second, _ = predict(
        applicant, pd.DataFrame({"irrelevant": [999999]}), model, scaler, encoded.columns.tolist()
    )
    assert first == second
    pd.testing.assert_frame_equal(
        row.reset_index(drop=True), encoded.iloc[:1].reset_index(drop=True), check_dtype=False
    )
    with pytest.raises(ValueError):
        predict(
            {**applicant, "LoanAmount": np.nan},
            pd.DataFrame(),
            model,
            scaler,
            encoded.columns.tolist(),
        )


def test_reject_invalid_targets():
    """Reject unrecognized approval labels before splitting."""
    raw = pd.read_csv(DATA)
    raw.loc[0, "Loan_Approved"] = "unknown"
    with pytest.raises(ValueError):
        split_raw(raw)
