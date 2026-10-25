"""Legacy feature splitting and train-only scaling helpers."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def split_and_scale_train_test(df, target_col="Loan_Approved", test_size=0.2, random_state=42):
    """
    Splits the data into train and test sets and scales using MinMaxScaler.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    x = df.drop(target_col, axis=1)
    y = df[target_col]
    feature_names = x.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Convert back to DataFrame to retain feature names
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=feature_names)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=feature_names)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler, feature_names
