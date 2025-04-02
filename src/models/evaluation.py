from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import logging

logging.basicConfig(filename='loan_app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate the given model's accuracy, confusion matrix, and classification report.

    Returns:
    tuple: accuracy, confusion matrix, classification report.
    """
    try:
        prob_pred = model.predict_proba(X_test)[:, 1]
        y_pred = (prob_pred >= threshold).astype(int)
        acc = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logging.info("Model evaluation completed successfully with accuracy: %.2f", acc)
        return acc, conf_mat, report
    except Exception as e:
        logging.error("Error evaluating model: %s", e)
        return 0, None, None

def feature_importance(model, feature_names):
    """
    Returns the importance of each feature.

    Returns:
    pd.DataFrame: DataFrame sorted by feature importance.
    """
    try:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        logging.info("Feature importance calculated successfully.")
        return importance_df
    except Exception as e:
        logging.error("Error calculating feature importance: %s", e)
        return None

def cross_validate_model(model, X_train, y_train, n_splits=5):
    """
    Perform cross-validation and return accuracy scores.

    Returns:
    dict: Cross-validation results containing mean and standard deviation of accuracy.
    """
    try:
        kfold = KFold(n_splits=n_splits)
        scores = cross_val_score(model, X_train, y_train, cv=kfold)
        logging.info("Cross-validation completed successfully. Mean accuracy: %.2f", scores.mean())
        return {'scores': scores, 'mean_accuracy': scores.mean(), 'std_accuracy': scores.std()}
    except Exception as e:
        logging.error("Error during cross-validation: %s", e)
        return {'scores': [], 'mean_accuracy': 0, 'std_accuracy': 0}
