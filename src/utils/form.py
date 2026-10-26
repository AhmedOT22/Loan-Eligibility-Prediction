"""Applicant form widgets and model score display helpers."""

import json
from pathlib import Path

import streamlit as st

from src.config import BASE_DIR


def fetch_input():
    """Render raw applicant fields and return a dictionary only after submission."""
    with st.form("loan_form"):
        st.subheader("📋 Applicant Details")
        values = {}
        columns = st.columns(3)
        fields = [
            ("Gender", ("Male", "Female")),
            ("Married", ("Yes", "No")),
            ("Dependents", ("0", "1", "2", "3+")),
            ("Education", ("Graduate", "Not Graduate")),
            ("Self_Employed", ("Yes", "No")),
            ("Credit_History", ("1.0", "0.0")),
            ("Property_Area", ("Urban", "Semiurban", "Rural")),
            ("Loan_Amount_Term", ("360", "180", "120", "240")),
        ]
        for index, (name, choices) in enumerate(fields):
            with columns[min(index // 3, 2)]:
                values[name] = st.selectbox(name.replace("_", " "), choices)
        st.subheader("💰 Financial Details")
        columns = st.columns(2)
        for column, name in zip(columns, ["ApplicantIncome", "CoapplicantIncome"]):
            with column:
                values[name] = st.number_input(name, min_value=0, step=100)
        values["LoanAmount"] = st.number_input("Loan Amount", min_value=1, step=1)
        submitted = st.form_submit_button("Calculate Model Score", use_container_width=True)
    return values if submitted else None


def interpret_probability(probability):
    """Label uncalibrated classifier-score bands without implying lender odds."""
    if probability >= 80:
        return "Very high model score", "green"
    if probability >= 60:
        return "High model score", "limegreen"
    if probability >= 40:
        return "Middle model score", "yellow"
    if probability >= 20:
        return "Low model score", "orange"
    return "Very low model score", "red"


def model_selector():
    """Return the selected bundled classifier's filename key."""
    choice = st.selectbox(
        "Choose your prediction model:", ("Logistic Regression", "Random Forest")
    )
    return "logistic_regression" if choice.startswith("Logistic") else "random_forest"


def display_model_info(metrics_path):
    """Display stored legacy accuracy, or an explicit unavailable status."""
    try:
        with Path(metrics_path).open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        accuracy_display = f"{metrics['accuracy']:.2%}"
    except (OSError, ValueError, KeyError, TypeError):
        accuracy_display = "Unavailable"
    with st.expander("About this bundled model"):
        st.write(f"Historical stored accuracy: {accuracy_display}")
        st.caption(
            "This figure belongs to the older encoded model and its original evaluation. "
            "It is not the train-only pipeline result reported in the README. The legacy "
            "preparation used whole-dataset statistics, so this is not a clean holdout estimate."
        )
    st.caption(
        "Amounts use the CSV's original numeric scale; currency and amount units are "
        "unverified. Gender and other demographic fields reflect the legacy feature schema."
    )


def load_custom_styles():
    """Read app styles from a repository-relative resource."""
    try:
        css = (BASE_DIR / "src/assets/style.css").read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except OSError as error:
        st.error(f"Failed to load custom styles: {error}")
