import pandas as pd
from src.data_processing.preprocessing import preprocess_data


def predict(user_input_dict, df_processed, model, scaler, feature_order):
    """
    Predicts the probability of loan approval based on user input and a trained model.
    
    Converts user input into a format consistent with the training data, applies preprocessing and scaling, and returns the predicted probability of loan approval as a percentage (0-100). Raises a ValueError if prediction fails.
    	
    Args:
    	user_input_dict: Dictionary containing user input features.
    	df_processed: Processed DataFrame of training data.
    	feature_order: List specifying the order of features used during model training.
    
    Returns:
    	Probability of loan approval as a float between 0 and 100.
    
    Raises:
    	ValueError: If prediction cannot be completed due to an error.
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
