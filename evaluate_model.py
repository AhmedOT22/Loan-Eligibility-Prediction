"""Diagnose fixed candidates using only the training partition; export aggregates."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from src.models.pipeline import make_pipeline, split_raw

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parent
NAMES = ("logistic_regression", "random_forest", "majority_baseline")


def summarize_scores(actual, scores, threshold=0.5):
    """Compute aggregate discrimination, error, and threshold tradeoff metrics."""
    labels = scores >= threshold
    return {
        "accuracy": float(accuracy_score(actual, labels)),
        "precision": float(precision_score(actual, labels, zero_division=0)),
        "recall": float(recall_score(actual, labels, zero_division=0)),
        "specificity": float(recall_score(actual, labels, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, scores)),
        "average_precision": float(average_precision_score(actual, scores)),
        "brier_score": float(brier_score_loss(actual, scores)),
        "confusion_matrix": confusion_matrix(actual, labels, labels=[0, 1]).tolist(),
    }


def validate_candidates(features, target, splits=5, repeats=3):
    """Refit preprocessing in each fold; retain only aggregate output, never row scores."""
    splitter = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=42)
    report = {}
    for name in NAMES:
        candidate = (
            DummyClassifier(strategy="prior")
            if name == "majority_baseline"
            else make_pipeline(name)
        )
        folds = []
        # Only the first repeat is pooled: each training row contributes once.
        out_of_fold = np.full(len(target), np.nan)
        for fold, (train, validation) in enumerate(splitter.split(features, target)):
            model = clone(candidate).fit(features.iloc[train], target.iloc[train])
            scores = model.predict_proba(features.iloc[validation])[
                :, list(model.classes_).index(1)
            ]
            folds.append(summarize_scores(target.iloc[validation], scores))
            if fold < splits:
                out_of_fold[validation] = scores
        report[name] = {
            "summary": {
                metric: {
                    "mean": float(np.mean([result[metric] for result in folds])),
                    "std": float(np.std([result[metric] for result in folds], ddof=1)),
                }
                for metric in folds[0]
                if metric != "confusion_matrix"
            },
            "folds": folds,
            "first_repeat_oof": summarize_scores(target, out_of_fold),
            "threshold_diagnostics": [
                {"threshold": threshold, **summarize_scores(target, out_of_fold, threshold)}
                for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
            ],
        }
    return report


def plot_validation(report, output):
    """Save genuine aggregate validation charts with explicit sample boundaries."""
    labels = ["Logistic regression", "Random forest", "Majority baseline"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, metric, title in zip(
        axes, ("roc_auc", "accuracy"), ("ROC AUC", "Accuracy at threshold 0.5")
    ):
        means = [report[name]["summary"][metric]["mean"] for name in NAMES]
        errors = [report[name]["summary"][metric]["std"] for name in NAMES]
        axis.barh(
            labels, means, xerr=errors, capsize=4, color=["#236795", "#479c82", "#999999"]
        )
        axis.set(xlim=(0, 1), title=title, xlabel="Mean ± fold SD (not a confidence interval)")
        axis.invert_yaxis()
    figure.suptitle("Training partition only · repeated stratified 5-fold CV (3 repeats)")
    figure.savefig(output / "validation.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for axis, name, label in zip(axes, NAMES[:2], labels[:2]):
        values = report[name]["threshold_diagnostics"]
        thresholds = [row["threshold"] for row in values]
        for metric in ("precision", "recall", "specificity"):
            axis.plot(thresholds, [row[metric] for row in values], marker="o", label=metric)
        axis.set(ylim=(0, 1.05), xlabel="Score threshold", ylabel="Metric", title=label)
        axis.legend()
        axis.grid(alpha=0.2)
    figure.suptitle("First-repeat out-of-fold scores · no threshold selected or deployed")
    figure.savefig(output / "thresholds.png", dpi=160)
    plt.close(figure)


def evaluate(input_path, output):
    """Reserve the existing holdout and write only aggregate validation diagnostics."""
    input_path, output = Path(input_path), Path(output)
    features, _, target, _ = split_raw(pd.read_csv(input_path))
    report = validate_candidates(features, target)
    metadata = {
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "validation_rows": len(target),
        "positive_count": int(target.sum()),
        "splits": 5,
        "repeats": 3,
        "random_state": 42,
        "boundary": "Only the 80% training partition from split_raw; holdout excluded",
        "threshold": "0.5 fixed; diagnostics are not threshold optimization",
        "baseline": "Majority labels with training class-prior scores (DummyClassifier prior)",
        "uncertainty": "Fold SD is descriptive; overlapping folds are not independent samples",
        "calibration": "Brier scores diagnose uncalibrated outputs; no calibrator is fitted",
        "privacy": "Aggregate statistics only; no applicant indices, labels, or scores exported",
        "packages": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "joblib", "matplotlib")
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, content in (("metrics.json", report), ("run.json", metadata)):
        (output / filename).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    plot_validation(report, output)
    return {name: result["summary"] for name, result in report.items()}


def main():
    """Run validation without training or replacing serving artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/credit.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/latest/validation")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
