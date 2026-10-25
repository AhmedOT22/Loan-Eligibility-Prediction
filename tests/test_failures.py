"""Legacy helpers must expose failures instead of reporting fake success values."""

import pandas as pd
import pytest

from src.data_processing.data_loader import load_data
from src.models.evaluation import evaluate_model
from src.models.training import train_logistic_regression


def test_missing_source_raises(tmp_path):
    """A missing CSV must not become an unexplained None downstream."""
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "absent.csv")


def test_invalid_model_raises():
    """A failed evaluator must not return an accuracy value of zero."""
    with pytest.raises(AttributeError):
        evaluate_model(None, pd.DataFrame({"a": [1]}), [1])


def test_invalid_training_labels_raise():
    """A classifier fit failure must not return a model-shaped None."""
    with pytest.raises(ValueError):
        train_logistic_regression(pd.DataFrame({"a": [1, 2]}), [1, 1])
