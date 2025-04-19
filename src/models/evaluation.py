from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import logging

logging.basicConfig(filename='loan_app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluates a classification model on test data and returns accuracy, confusion matrix, and classification report.
    
    Args:
        threshold: Probability threshold for converting predicted probabilities to binary class labels. Default is 0.5.
    
    Returns:
        A tuple containing the accuracy (float), confusion matrix (numpy.ndarray), and classification report (str). If evaluation fails, returns (0, None, None).
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
    Returns a DataFrame of feature importances for the given model.
    
    The DataFrame pairs each feature name with its corresponding importance score and is sorted in descending order of importance. Returns None if the model does not provide feature importances or an error occurs.
    
    Returns:
        pd.DataFrame or None: DataFrame of features and their importances, or None on error.
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
    Performs k-fold cross-validation on the training data and returns accuracy metrics.
    
    Args:
        n_splits: Number of folds for cross-validation.
    
    Returns:
        A dictionary containing the array of accuracy scores, their mean, and standard deviation.
    """
    try:
        kfold = KFold(n_splits=n_splits)
        scores = cross_val_score(model, X_train, y_train, cv=kfold)
        logging.info("Cross-validation completed successfully. Mean accuracy: %.2f", scores.mean())
        return {'scores': scores, 'mean_accuracy': scores.mean(), 'std_accuracy': scores.std()}
    except Exception as e:
        logging.error("Error during cross-validation: %s", e)
        return {'scores': [], 'mean_accuracy': 0, 'std_accuracy': 0}
