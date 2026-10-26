"""Check evaluation tradeoffs and the boundary around reserved holdout rows."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import evaluate_model
from src.models.pipeline import split_raw

DATA = Path(__file__).resolve().parents[1] / "data/raw/credit.csv"


def test_threshold_tradeoff_and_probability_error():
    """Hand-computed examples catch swapped labels and reversed score conventions."""
    target = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    result = evaluate_model.summarize_scores(target, scores)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["brier_score"] == pytest.approx(0.185)
    assert result["roc_auc"] == pytest.approx(0.75)
    higher = evaluate_model.summarize_scores(target, scores, threshold=0.7)
    assert higher["precision"] == 1
    assert higher["recall"] == 0.5
    assert higher["specificity"] == 1


def test_report_excludes_holdout_and_row_outputs(monkeypatch, tmp_path):
    """Assert the public-report path cannot pass holdout rows into validation."""
    raw = pd.read_csv(DATA)
    expected_features, holdout, expected_target, _ = split_raw(raw)
    calls = []

    def validate(features, target):
        pd.testing.assert_frame_equal(features, expected_features)
        pd.testing.assert_series_equal(target, expected_target)
        assert set(features.index).isdisjoint(holdout.index)
        calls.append(True)
        return {"fixture": {"summary": {"accuracy": {"mean": 0.5, "std": 0.1}}}}

    monkeypatch.setattr(evaluate_model, "validate_candidates", validate)
    monkeypatch.setattr(evaluate_model, "plot_validation", lambda *_: None)
    evaluate_model.evaluate(DATA, tmp_path)
    assert calls == [True]
    assert {path.name for path in tmp_path.iterdir()} == {"metrics.json", "run.json"}
    metadata = json.loads((tmp_path / "run.json").read_text())
    assert metadata["validation_rows"] == 491
    assert "indices" not in metadata


def test_repeated_folds_do_not_inflate_pooled_counts():
    """Pooled diagnostics count each input once even when validation repeats."""
    features, _, target, _ = split_raw(pd.read_csv(DATA))
    features, target = features.iloc[:60].copy(), target.iloc[:60].copy()
    original = features.copy(deep=True)
    report = evaluate_model.validate_candidates(features, target, splits=3, repeats=2)
    pd.testing.assert_frame_equal(features, original)
    for result in report.values():
        assert len(result["folds"]) == 6
        assert sum(map(sum, result["first_repeat_oof"]["confusion_matrix"])) == len(target)
        assert all(
            sum(map(sum, row["confusion_matrix"])) == len(target)
            for row in result["threshold_diagnostics"]
        )
        assert sum(sum(map(sum, row["confusion_matrix"])) for row in result["folds"]) == 120
