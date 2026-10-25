"""Repository-anchored paths for the legacy serving artifacts."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models/loan_model.pkl"
SCALER_PATH = BASE_DIR / "models/scaler.pkl"
RAW_PATH = BASE_DIR / "data/raw/credit.csv"
PROCESSED_PATH = BASE_DIR / "data/processed/credit_processed.csv"
