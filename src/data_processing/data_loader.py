import pandas as pd
import logging

logging.basicConfig(filename='loan_app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def load_data(filepath):
    """
    Load dataset from a CSV file.

    Parameters:
    filepath (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_csv(filepath)
        df['Credit_History'] = df['Credit_History'].astype('object')
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('object')
        logging.info("Data loaded successfully from %s", filepath)
        return df
    except Exception as e:
        logging.error("Failed to load data from %s: %s", filepath, e)
        return None
