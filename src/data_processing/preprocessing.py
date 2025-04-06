import pandas as pd
import logging
import os

pd.set_option('future.no_silent_downcasting', True)

logging.basicConfig(filename='loan_app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def preprocess_data(df, output_path=None):
    """
    Preprocess data by handling missing values and encoding categorical features.
    
    Parameters:
        df (pd.DataFrame): Raw input DataFrame.
        output_path (str, optional): Path to save the processed DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame, or None if error occurs.
    """
    try:
        df = df.copy()
        df.drop('Loan_ID', axis=1, inplace=True, errors='ignore')

        for column in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']:
            df[column] = df[column].fillna(df[column].mode()[0])
        df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())


        df = pd.get_dummies(df, columns=[
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'
        ], dtype=int)

        if 'Loan_Approved' in df:
            df['Loan_Approved'] = df['Loan_Approved'].replace({'Y': 1, 'N': 0}).infer_objects(copy=False)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logging.info("Processed data saved to %s", output_path)

        logging.info("Data preprocessing completed successfully.")
        return df

    except Exception as e:
        logging.error("Error in preprocessing data: %s", e)
        return None
