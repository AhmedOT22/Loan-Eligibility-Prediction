"""Train legacy classifiers while propagating fitting failures."""

import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    filename="loan_app.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)


def train_logistic_regression(x_train, y_train):
    """
    Train Logistic Regression model.

    Returns:
    model: Trained logistic regression model. Fitting failures propagate.
    """
    try:
        model = LogisticRegression().fit(x_train, y_train)
        logging.info("Logistic Regression trained successfully.")
        return model
    except Exception as e:
        logging.error("Error training Logistic Regression: %s", e)
        raise


def train_random_forest(x_train, y_train, n_estimators=100, max_depth=None):
    """
    Train Random Forest model.

    Returns:
    model: Trained random forest model. Fitting failures propagate.
    """
    try:
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(x_train, y_train)
        logging.info("Random Forest trained successfully.")
        return model
    except Exception as e:
        logging.error("Error training Random Forest: %s", e)
        raise
