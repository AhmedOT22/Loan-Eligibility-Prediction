from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def split_and_scale_train_test(df, target_col='Loan_Approved', test_size=0.2, random_state=42):
    """
    Splits the data into train and test sets and scales using MinMaxScaler.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to retain feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names