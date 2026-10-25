"""Persist and load trusted legacy model artifacts."""

import logging
import os

import joblib

logging.basicConfig(
    filename="loan_app.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)


def save_models(models: dict, scaler, base_path):
    """
    Saves multiple trained models and a scaler to disk using joblib.

    Args:
        models (dict): Dictionary of model name to model object.
        scaler: Fitted scaler used for input features.
        base_path (str): Base directory where model files will be saved.
    """
    try:
        os.makedirs(base_path, exist_ok=True)
        for name, model in models.items():
            model_path = os.path.join(base_path, f"{name}_model.pkl")
            joblib.dump(model, model_path)
            logging.info("Model %s saved to %s", name, model_path)

        scaler_path = os.path.join(base_path, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        logging.info("Scaler saved to %s", scaler_path)

    except Exception as e:
        logging.error("Failed to save models or scaler: %s", e)
        raise


def load_model(model_path):
    """Load a trusted classifier artifact from disk."""
    try:
        model = joblib.load(model_path)
        logging.info("Model loaded from %s", model_path)
        return model
    except Exception as e:
        logging.error("Failed to load model from %s: %s", model_path, e)
        raise


def load_scaler(scaler_path):
    """Load the fitted legacy feature scaler."""
    try:
        scaler = joblib.load(scaler_path)
        logging.info("Scaler loaded from %s", scaler_path)
        return scaler
    except Exception as e:
        logging.error("Failed to load scaler from %s: %s", scaler_path, e)
        raise
