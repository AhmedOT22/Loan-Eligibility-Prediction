import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging
import joblib
import os
import json

from src.config import PROCESSED_PATH
from src.models.prediction import predict
from src.utils.form import fetch_input
from src.utils.gauge import generate_gauge_chart
from src.utils.advice import generate_advice

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

# Model selection
model_choice = st.selectbox(
    "Select Prediction Model",
    ("Logistic Regression (Recommended)", "Random Forest")
)

# Correct mapping after selection
if model_choice.startswith("Logistic"):
    model_key = "logistic_regression"
else:
    model_key = "random_forest"

model_path = f"models/{model_key}_model.pkl"
metrics_path = f"models/{model_key}_metrics.json"
scaler_path = "models/scaler.pkl"

# Load model, scaler, processed data
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df_processed = pd.read_csv(PROCESSED_PATH)
    feature_order = df_processed.drop('Loan_Approved', axis=1).columns.tolist()

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        accuracy_display = f"{metrics['accuracy']:.2%}"
    else:
        accuracy_display = "Pretrained"

    logging.info("Model, scaler, and processed data loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    logging.error("Error loading model or data: %s", e)
    st.stop()

st.markdown(f"### 🎯 Current Model Accuracy: **{accuracy_display}**")
st.subheader('Applicant Details')

user_input = fetch_input()

if user_input:
    with st.spinner('Making prediction...'):
        try:
            prediction_proba, user_processed = predict(user_input, df_processed, model, scaler, feature_order)
            interpretation, color = interpret_probability(prediction_proba)

            # Center the gauge
            st.markdown("<h2 style='text-align: center;'>Loan Approval Probability</h2>", unsafe_allow_html=True)
            gauge = generate_gauge_chart(prediction_proba, interpretation, color)
            st.plotly_chart(gauge, use_container_width=True)

            logging.info("Prediction successful with probability: %.2f", prediction_proba)

            # Loan Advice inside expander
            loan_advice = generate_advice(user_input)

            with st.expander('📋 Loan Officer Advice'):
                if loan_advice:
                    for tip in loan_advice:
                        st.info(tip)
                else:
                    st.info("No specific advice for this applicant.")

        except Exception as e:
            logging.error("Error during prediction: %s", e)
            st.error(f"Error during prediction: {e}")
