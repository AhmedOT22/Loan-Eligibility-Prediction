"""Evaluate raw-input pipelines without replacing legacy app artifacts."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.models.pipeline import split_raw, make_pipeline

ROOT = Path(__file__).resolve().parent


def train_pipeline(input_path=ROOT / 'data/raw/credit.csv', output_dir=ROOT / 'reports/latest', save_models=False):
    input_path, output = Path(input_path), Path(output_dir)
    raw = pd.read_csv(input_path)
    X_train, X_test, y_train, y_test = split_raw(raw)
    candidates = {name: make_pipeline(name).fit(X_train, y_train)
                  for name in ('logistic_regression', 'random_forest')}
    candidates['majority_baseline'] = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    metrics = {}
    predictions = pd.DataFrame({'row_index': X_test.index, 'actual': y_test.to_numpy()})
    output.mkdir(parents=True, exist_ok=True)
    for name, model in candidates.items():
        labels = model.predict(X_test)
        positive_index = list(model.classes_).index(1)
        scores = model.predict_proba(X_test)[:, positive_index]
        metrics[name] = {'accuracy': float(accuracy_score(y_test, labels)),
                         'precision': float(precision_score(y_test, labels, zero_division=0)),
                         'recall': float(recall_score(y_test, labels, zero_division=0)),
                         'f1': float(f1_score(y_test, labels, zero_division=0)),
                         'roc_auc': float(roc_auc_score(y_test, scores)),
                         'confusion_matrix_labels': [0, 1],
                         'confusion_matrix': confusion_matrix(y_test, labels, labels=[0, 1]).tolist()}
        predictions[name + '_score'] = scores
        predictions[name + '_class'] = labels
        if save_models:
            joblib.dump(model, output / (name + '_pipeline.joblib'))
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    predictions.to_csv(output / 'predictions.csv', index=False)
    metadata = {'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
                'rows': len(raw), 'train_rows': len(X_train), 'test_rows': len(X_test),
                'train_indices': X_train.index.tolist(), 'test_indices': X_test.index.tolist(),
                'train_positive_fraction': float(y_train.mean()), 'test_positive_fraction': float(y_test.mean()),
                'random_state': 42, 'test_size': 0.2,
                'evaluation': 'single stratified holdout; no cross-validation or probability calibration claimed',
                'packages': {p: version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'joblib')}}
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/raw/credit.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/latest')
    parser.add_argument('--save-models', action='store_true', help='Save new raw-input pipelines under OUTPUT, not legacy app model paths')
    args = parser.parse_args()
    print(json.dumps(train_pipeline(args.input, args.output, args.save_models), indent=2))


if __name__ == '__main__':
    main()
