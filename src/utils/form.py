import streamlit as st
import os
import json

def fetch_input():
    """
    Renders a loan application form in the Streamlit UI and collects user input.

    Returns:
        dict or None: A dictionary of user inputs if submitted, else None.
    """
    with st.form("loan_form"):
        st.subheader("📋 Applicant Details") 
        col1, col2, col3 = st.columns(3)

        with col1:
            Gender = st.selectbox('Gender', ('Male', 'Female'))
            Married = st.selectbox('Married', ('Yes', 'No'))
            Dependents = st.selectbox('Dependents', ('0', '1', '2', '3+'))

        with col2:
            Education = st.selectbox('Education', ('Graduate', 'Not Graduate'))
            Self_Employed = st.selectbox('Self Employed', ('Yes', 'No'))
            Credit_History = st.selectbox('Credit History', ('1.0', '0.0'))

        with col3:
            Property_Area = st.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))
            Loan_Amount_Term = st.selectbox('Loan Amount Term', ('360', '180', '120', '240'))

        st.subheader("💰 Financial Details")

        col4, col5 = st.columns(2)

        with col4:
            ApplicantIncome = st.number_input('Applicant Income', min_value=0, step=100)

        with col5:
            CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0, step=100)

        LoanAmount = st.number_input('Loan Amount', min_value=1, step=1)

        submitted = st.form_submit_button("Predict Loan Eligibility", use_container_width=True)

    if submitted:
        return {
            'Gender': Gender,
            'Married': Married,
            'Dependents': Dependents,
            'Education': Education,
            'Self_Employed': Self_Employed,
            'ApplicantIncome': ApplicantIncome,
            'CoapplicantIncome': CoapplicantIncome,
            'LoanAmount': LoanAmount,
            'Loan_Amount_Term': Loan_Amount_Term,
            'Credit_History': Credit_History,
            'Property_Area': Property_Area
        }
    return None

def interpret_probability(probability):
    """
    Returns interpretation text and associated color based on the loan approval probability.

    Args:
        probability (float): Loan approval probability in percentage.

    Returns:
        tuple: (interpretation_text, color_code)
    """
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

def model_selector():
    """
    Renders a model selection dropdown and returns the selected model key.

    Returns:
        str: Model key ('logistic_regression' or 'random_forest')
    """
    model_choice = st.selectbox(
    "Choose your prediction model:",
    ("Logistic Regression (Recommended)", "Random Forest")
    )

    if model_choice.startswith("Logistic"):
        return "logistic_regression"
    else:
        return "random_forest"

def display_model_info(metrics_path):
    """
    Displays the model accuracy or pretrained status.

    Args:
        metrics_path (str): Path to the metrics JSON file.
    """
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            accuracy_display = f"{metrics['accuracy']:.2%}"
        except Exception as e:
            accuracy_display = "Pretrained"
    else:
        accuracy_display = "Pretrained"

    st.subheader(f"🎯 Current Model Accuracy: **{accuracy_display}**")

def load_custom_styles():
    """
    Loads external CSS styling for the app from src/assets/style.css.
    """
    try:
        with open('src/assets/style.css') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load custom styles: {e}")
