import logging
import json
from config import RAW_PATH, PROCESSED_PATH
from src.data_processing.data_loader import load_data
from src.data_processing.preprocessing import preprocess_data
from src.data_processing.features import split_and_scale_train_test
from src.models.training import train_random_forest, train_logistic_regression
from src.models.evaluation import evaluate_model
from src.models.storage import save_models

# Configure logging
logging.basicConfig(filename='loan_app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def train_pipeline():
    """
    Runs the full machine learning training pipeline:
    - Loads raw data
    - Preprocesses the data
    - Splits and scales features
    - Trains both Random Forest and Logistic Regression models
    - Evaluates models and saves their metrics to JSON files
    - Saves trained models and the scaler to disk
    """
    try:
        # Load raw dataset
        df_raw = load_data(RAW_PATH)

        # Preprocess and store the processed dataset
        df_processed = preprocess_data(df_raw, output_path=PROCESSED_PATH)

        # Split into train/test and scale the features
        X_train, X_test, y_train, y_test, scaler, feature_order = split_and_scale_train_test(df_processed)

        # Train both models
        models = {
            "random_forest": train_random_forest(X_train, y_train),
            "logistic_regression": train_logistic_regression(X_train, y_train)
        }

        # Evaluate models and save metrics
        metrics = {}
        for name, model in models.items():
            accuracy, _, _ = evaluate_model(model, X_test, y_test)
            print(f"{name.replace('_', ' ').title()} model trained with accuracy: {accuracy:.2%}")
            metrics[name] = {"accuracy": accuracy}
            metrics_path = f"models/{name}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics[name], f)

        # Save both trained models and the shared scaler
        save_models(models, scaler, base_path="models/")

        print(f"Feature order: {feature_order}")

    except Exception as e:
        logging.error("Training pipeline failed: %s", e)
        print(f"Training pipeline failed: {e}")

if __name__ == "__main__":
    train_pipeline()
