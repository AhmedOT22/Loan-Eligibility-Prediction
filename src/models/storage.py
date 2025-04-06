import os
import joblib
import logging

logging.basicConfig(filename='loan_app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

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
            logging.info(f"Model '{name}' saved to {model_path}")

        scaler_path = os.path.join(base_path, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        logging.info(f"Scaler saved to {scaler_path}")

    except Exception as e:
        logging.error(f"Failed to save models or scaler: {e}")
        raise

def load_model(model_path):
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise

def load_scaler(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        logging.info(f"Scaler loaded from {scaler_path}")
        return scaler
    except Exception as e:
        logging.error(f"Failed to load scaler from {scaler_path}: {e}")
        raise
