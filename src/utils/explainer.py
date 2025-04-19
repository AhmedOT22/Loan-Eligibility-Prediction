import shap
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Not used in the current code but kept for potential future use
def explain_prediction(model, user_processed, background_data=None):
    """
    Explains the individual prediction using SHAP values.

    Args:
        model: Trained model.
        user_processed: Preprocessed (but not scaled) user input DataFrame.
        background_data: Background data for SHAP TreeExplainer (for Random Forest only).

    Returns:
        pd.DataFrame: Top 3 feature contributions sorted by absolute impact.
    """
    try:
        feature_names = user_processed.columns.tolist()
        
        if isinstance(model, LogisticRegression):
            explainer = shap.Explainer(model.predict, masker=user_processed)
            shap_values = explainer(user_processed)
            shap_contributions = shap_values.values[0]

        elif isinstance(model, RandomForestClassifier):
            if background_data is None:
                raise ValueError("Background data is required for explaining Random Forest!")

            explainer = shap.TreeExplainer(model, background_data)

            shap_values = explainer.shap_values(user_processed)

            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_contributions = shap_values[1][0]
            else:
                shap_contributions = shap_values[0][0]

        else:
            raise ValueError("Unsupported model type for SHAP explanations.")

        shap_df = pd.DataFrame({
            'feature': feature_names,
            'contribution': shap_contributions
        }).sort_values(by='contribution', key=abs, ascending=False)

        top_features = shap_df.head(3)
        return top_features

    except Exception as e:
        raise ValueError(f"SHAP explanation failed: {e}")
