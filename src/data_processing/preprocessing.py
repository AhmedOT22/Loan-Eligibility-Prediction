import pandas as pd
import logging

logging.basicConfig(filename='loan_app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def preprocess_data(df):
    """
    Preprocess data by handling missing values and encoding categorical features.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: Preprocessed DataFrame, or None if error occurs.
    """
    try:
        df = df.copy()
        df.drop('Loan_ID', axis=1, inplace=True, errors='ignore')

        for column in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']:
            df[column].fillna(df[column].mode()[0], inplace=True)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

        df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'], dtype=int)

        if 'Loan_Approved' in df:
            df['Loan_Approved'] = df['Loan_Approved'].replace({'Y': 1, 'N': 0})

        logging.info("Data preprocessing completed successfully.")
        return df
    except Exception as e:
        logging.error("Error in preprocessing data: %s", e)
        return None
