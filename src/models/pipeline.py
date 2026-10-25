"""Raw-input pipelines keep fitted preprocessing inside the training boundary."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERIC = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]


def split_raw(frame):
    """Validate raw feature columns and reserve a stratified holdout."""
    missing = set(CATEGORICAL + NUMERIC + ["Loan_Approved"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    y = frame["Loan_Approved"].map({"Y": 1, "N": 0})
    if y.isna().any() or y.nunique() != 2:
        raise ValueError("Loan_Approved must contain both Y and N labels, without missing values.")
    x = frame[CATEGORICAL + NUMERIC].copy()
    x[NUMERIC] = x[NUMERIC].apply(pd.to_numeric, errors="raise")
    if np.isinf(x[NUMERIC].to_numpy(dtype=float)).any() or (x[NUMERIC] < 0).any().any():
        raise ValueError(
            "Numeric inputs must be non-negative and not infinite; "
            "missing values are imputed after splitting."
        )
    return train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)


def make_pipeline(kind):
    """Build a classifier with training-fitted numeric and categorical preparation."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", MinMaxScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)]
    )
    if kind == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif kind == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model: {kind}")
    return Pipeline([("preprocess", preprocessing), ("model", model)])
