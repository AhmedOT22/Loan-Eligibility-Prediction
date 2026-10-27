"""Build a new serving release without overwriting artifacts or exporting row results."""

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from src.models.pipeline import CATEGORICAL, NUMERIC, make_pipeline, split_raw
from src.models.serving import MODEL_NAMES, load_bundle
from train_model import ROOT, score_classifier


def build_release(input_path, output):
    """Fit only training rows, evaluate held-out rows, and atomically publish a new directory."""
    input_path, output = Path(input_path), Path(output)
    if output.exists():
        raise FileExistsError("Release directory exists; choose a new versioned directory.")
    features, holdout, target, holdout_target = split_raw(pd.read_csv(input_path))
    manifest = {
        "format_version": 1,
        "release": "train-only-v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "features": CATEGORICAL + NUMERIC,
        "target": {"column": "Loan_Approved", "positive": "Y", "classes": [0, 1]},
        "split": {
            "seed": 42,
            "test_size": 0.2,
            "train_rows": len(target),
            "holdout_rows": len(holdout_target),
        },
        "training": "Full preprocessing fitted on training rows only; no full-data refit",
        "threshold": 0.5,
        "calibrated": False,
        "packages": {
            p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "joblib")
        },
        "models": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "release"
        staging.mkdir()
        for name in MODEL_NAMES:
            model = make_pipeline(name).fit(features, target)
            artifact = staging / f"{name}_pipeline.joblib"
            joblib.dump(model, artifact, compress=3)
            encoder = getattr(model["preprocess"], "named_transformers_")["categorical"][
                "encode"
            ]
            manifest["models"][name] = {
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "holdout_metrics": score_classifier(model, holdout, holdout_target),
                "categories": {
                    field: values.tolist()
                    for field, values in zip(CATEGORICAL, encoder.categories_)
                },
            }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        for name in MODEL_NAMES:
            restored, _ = load_bundle(name, staging)
            if (
                score_classifier(restored, holdout, holdout_target)
                != manifest["models"][name]["holdout_metrics"]
            ):
                raise ValueError("Reloaded artifact evaluation differs from its manifest.")
        staging.rename(output)
    return manifest


def main():
    """Build an explicit new release; the default destination is an ignored local report path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/credit.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/latest/serving-v1")
    args = parser.parse_args()
    print(json.dumps(build_release(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
