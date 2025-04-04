import pandas as pd
from src.data_processing.preprocessing import preprocess_data


def predict(user_input_dict, df_processed, model, scaler, feature_order):
    """
    Generate a probability prediction using a trained model and user input.

    Parameters:
    - user_input_dict (dict): Dictionary of user inputs.
    - df_processed (pd.DataFrame): Processed dataset (includes all training data).
    - model: Trained machine learning model.
    - scaler: Pre-fitted scaler for feature scaling.
    - feature_order (list): Ordered list of feature names used during training.

    Returns:
    - float: Probability of loan approval (0-100 scale).
    """
    try:
        input_df = pd.DataFrame([user_input_dict])

        # Add input to dataset for consistent one-hot encoding
        reference_df = df_processed.drop('Loan_Approved', axis=1).copy()
        combined_df = pd.concat([input_df, reference_df], axis=0, ignore_index=True)

        # Preprocess
        combined_processed = preprocess_data(combined_df)
        user_processed = combined_processed.iloc[0:1].copy()

        # Ensure feature_order has no duplicates and matches user_processed
        user_processed = user_processed.loc[:, ~user_processed.columns.duplicated()].copy()
        feature_order = pd.Index(feature_order).drop_duplicates().tolist()

        # Align with feature order
        user_processed = user_processed.reindex(columns=feature_order, fill_value=0)

        # Scale
        user_scaled = scaler.transform(user_processed)

        # Predict
        probability = model.predict_proba(user_scaled)[0][1] * 100
        return probability

    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")
