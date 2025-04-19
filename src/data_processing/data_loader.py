import pandas as pd
import logging

logging.basicConfig(filename='loan_app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def load_data(filepath):
    """
    Loads a dataset from a CSV file and converts specific columns to object type.
    
    Attempts to read the CSV file at the given filepath into a pandas DataFrame. If successful, converts the 'Credit_History' and 'Loan_Amount_Term' columns to the object data type. Returns the DataFrame, or None if loading fails.
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
