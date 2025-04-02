import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging
import joblib
import os

from config import MODEL_PATH, SCALER_PATH, PROCESSED_PATH
from src.models.prediction import predict
from src.utils.form import fetch_input
from src.utils.gauge import generate_gauge_chart

# Configure logging
logging.basicConfig(filename='loan_app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

def interpret_probability(probability):
    if probability >= 80:
        return "Very Likely", "green"
    elif probability >= 60:
        return "Likely", "limegreen"
    elif probability >= 40:
        return "Somewhat Likely", "yellow"
    elif probability >= 20:
        return "Unlikely", "orange"
    else:
        return "Very Unlikely", "red"

# Main app logic
st.set_page_config(page_title='Loan Eligibility Prediction', layout='centered')
st.title('Loan Eligibility Prediction App')


try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df_processed = pd.read_csv(PROCESSED_PATH)
    feature_order = df_processed.drop('Loan_Approved', axis=1).columns.tolist()
    logging.info("Model, scaler, and processed data loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    logging.error("Error loading model or data: %s", e)
    st.stop()

st.markdown("## Current Model Accuracy: **Pretrained**")
st.subheader('Please enter applicant details below')

user_input = fetch_input()
if user_input:
    try:
        prediction_proba = predict(user_input, df_processed, model, scaler, feature_order)
        interpretation, color = interpret_probability(prediction_proba)
        st.subheader('Loan Approval Probability')
        fig = generate_gauge_chart(prediction_proba, interpretation, color)
        st.plotly_chart(fig)
        logging.info("Prediction successful with probability: %.2f", prediction_proba)
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        st.error(f"Error during prediction: {e}")
