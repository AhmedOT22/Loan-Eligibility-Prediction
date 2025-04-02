from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(filename='loan_app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model.

    Returns:
    model: Trained logistic regression model, or None if training fails.
    """
    try:
        model = LogisticRegression().fit(X_train, y_train)
        logging.info("Logistic Regression trained successfully.")
        return model
    except Exception as e:
        logging.error("Error training Logistic Regression: %s", e)
        return None

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Train Random Forest model.

    Returns:
    model: Trained random forest model, or None if training fails.
    """
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        logging.info("Random Forest trained successfully.")
        return model
    except Exception as e:
        logging.error("Error training Random Forest: %s", e)
        return None
