"""Streamlit demonstration of scores from the bundled loan classifiers."""

import logging

import joblib
import streamlit as st

from src.config import BASE_DIR
from src.models.prediction import predict
from src.utils.form import (
    display_model_info,
    fetch_input,
    interpret_probability,
    load_custom_styles,
    model_selector,
)
from src.utils.gauge import generate_gauge_chart


def main():
    """Render the applicant form and score using the existing serving artifacts."""
    # --- Streamlit Page Config ---
    st.set_page_config(page_title="Loan Eligibility Prediction", layout="centered")

    # --- Load Custom Styles ---
    load_custom_styles()

    # --- Title ---
    st.title("🏦 Loan Eligibility Prediction App")
    st.caption(
        "Educational demonstration. Scores are uncalibrated model outputs, "
        "not lender approval probabilities or decisions."
    )

    # --- Model Selection ---
    model_key = model_selector()
    metrics_path = BASE_DIR / "models" / f"{model_key}_metrics.json"

    # --- Load Model and Scaler ---
    try:
        model = joblib.load(BASE_DIR / "models" / f"{model_key}_model.pkl")
        scaler = joblib.load(BASE_DIR / "models/scaler.pkl")
        feature_order = scaler.feature_names_in_.tolist()

        logging.info("Bundled model and scaler loaded successfully.")

    except (OSError, ValueError, TypeError, KeyError, ImportError, EOFError) as e:
        st.error(f"Error loading model or data: {e}")
        logging.error("Error loading model or data: %s", e)
        st.stop()

    # --- Display Model Info ---
    display_model_info(metrics_path)

    # --- Applicant Form ---
    user_input = fetch_input()

    # --- Prediction ---
    if user_input:
        with st.spinner("Making prediction..."):
            try:
                prediction_proba, _ = predict(user_input, None, model, scaler, feature_order)
                interpretation, color = interpret_probability(prediction_proba)

                # Display only Gauge
                st.subheader("📊 Model Score")
                gauge = generate_gauge_chart(prediction_proba, interpretation, color)
                st.plotly_chart(gauge, use_container_width=True)

                st.caption(
                    "This score comes from the older bundled model. The README's evaluation "
                    "covers separate pipelines with train-only preprocessing."
                )

            except (OSError, ValueError, TypeError, KeyError, ImportError, EOFError) as e:
                logging.error("Error during prediction: %s", e)
                st.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
