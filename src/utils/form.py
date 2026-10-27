"""Applicant form widgets and model score display helpers."""

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


def display_model_info(manifest, model_key):
    """Show holdout metrics belonging to the exact loaded serving pipeline."""
    metrics = manifest["models"][model_key]["holdout_metrics"]
    with st.expander("About this model and its evaluation"):
        st.write(f"Release: {manifest['release']}")
        st.write(f"Holdout accuracy: {metrics['accuracy']:.2%}")
        st.write(f"Holdout ROC AUC: {metrics['roc_auc']:.3f}")
        st.caption(
            f"Evaluated on {manifest['split']['holdout_rows']} reserved rows. "
            "The held-out rows were excluded from all fitting. Cross-validation in the "
            "README shows variation across folds; a single holdout is not a performance guarantee."
        )
    st.caption(
        "Amounts use the CSV's original numeric scale; currency and amount units are "
        "unverified. Demographic fields reflect the supplied dataset, not a real lending policy."
    )


def load_custom_styles():
    """Read app styles from a repository-relative resource."""
    try:
        css = (BASE_DIR / "src/assets/style.css").read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except OSError as error:
        st.error(f"Failed to load custom styles: {error}")
