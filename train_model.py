import pandas as pd
import os
import joblib
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.data_processing.preprocessing import preprocess_data
from src.models.training import train_random_forest

logging.basicConfig(filename='loan_app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def load_and_preprocess_data(raw_path, processed_path):
    """
    Loads the raw dataset, preprocesses it, and saves the processed data.
    """
    try:
        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path)
            logging.info("Loaded processed data from %s", processed_path)
        else:
            df_raw = pd.read_csv(raw_path).drop(['Loan_ID'], axis=1, errors='ignore')
            df = preprocess_data(df_raw)
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            df.to_csv(processed_path, index=False)
            logging.info("Processed raw data and saved to %s", processed_path)
        return df
    except Exception as e:
        logging.error("Error loading or preprocessing data: %s", e)
        raise

def train_and_save_model(df_processed, model_path, scaler_path):
    """
    Trains the model and scaler, saves both to disk, and returns the accuracy and feature names.
    """
    try:
        X = df_processed.drop('Loan_Approved', axis=1)
        y = df_processed['Loan_Approved']
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        model = train_random_forest(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        logging.info("Model trained with accuracy: %.2f and saved to %s", accuracy, model_path)
        return accuracy, X.columns.tolist()
    except Exception as e:
        logging.error("Error training or saving model: %s", e)
        raise

if __name__ == "__main__":
    RAW_PATH = 'data/raw/credit.csv'
    PROCESSED_PATH = 'data/processed/credit_processed.csv'
    MODEL_PATH = 'models/loan_model.pkl'
    SCALER_PATH = 'models/scaler.pkl'

    df_processed = load_and_preprocess_data(RAW_PATH, PROCESSED_PATH)
    accuracy, feature_order = train_and_save_model(df_processed, MODEL_PATH, SCALER_PATH)
    print(f"Model trained with accuracy: {accuracy:.2%}")
