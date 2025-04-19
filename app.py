import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging
import joblib

from src.config import PROCESSED_PATH
from src.models.prediction import predict
from src.utils.form import fetch_input, interpret_probability, display_model_info, load_custom_styles, model_selector
from src.utils.gauge import generate_gauge_chart
from src.utils.advice import generate_advice

# --- Streamlit Page Config ---
st.set_page_config(page_title='Loan Eligibility Prediction', layout='centered')

# --- Load Custom Styles ---
load_custom_styles()

# --- Title ---
st.title('🏦 Loan Eligibility Prediction App')

# --- Model Selection ---
model_key = model_selector()
model_path = f"models/{model_key}_model.pkl"
metrics_path = f"models/{model_key}_metrics.json"
scaler_path = "models/scaler.pkl"

# --- Configure Logging ---
logging.basicConfig(filename='loan_app.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- Load Model, Scaler, Processed Data ---
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df_processed = pd.read_csv(PROCESSED_PATH)
    feature_order = df_processed.drop('Loan_Approved', axis=1).columns.tolist()

    logging.info("Model, scaler, and processed data loaded successfully.")

except Exception as e:
    st.error(f"Error loading model or data: {e}")
    logging.error("Error loading model or data: %s", e)
    st.stop()

# --- Display Model Info ---
display_model_info(metrics_path)

# --- Applicant Form ---
user_input = fetch_input()

# --- Prediction ---
if user_input:
    with st.spinner('Making prediction...'):
        try:
            prediction_proba, user_processed = predict(user_input, df_processed, model, scaler, feature_order)
            interpretation, color = interpret_probability(prediction_proba)

            # Display only Gauge
            st.subheader("📊 Loan Approval Probability")
            gauge = generate_gauge_chart(prediction_proba, interpretation, color)
            st.plotly_chart(gauge, use_container_width=True)

            logging.info("Prediction successful with probability: %.2f", prediction_proba)

            # --- Loan Officer Advice ---
            loan_advice = generate_advice(user_input)

            with st.expander('📋 Loan Officer Advice'):
                if loan_advice:
                    for tip in loan_advice:
                        st.info(f"✅ {tip}")
                else:
                    st.info("No specific advice for this applicant.")

        except Exception as e:
            logging.error("Error during prediction: %s", e)
            st.error(f"Error during prediction: {e}")
