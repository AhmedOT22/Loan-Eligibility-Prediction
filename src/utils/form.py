import streamlit as st

def fetch_input():
    """
    Renders a loan application form in the Streamlit UI and collects user input.

    Returns:
        dict or None: A dictionary of user inputs if submitted, else None.
    """
    with st.form("loan_form"):
        Gender = st.selectbox('Gender', ('Male', 'Female'))
        Married = st.selectbox('Married', ('Yes', 'No'))
        Dependents = st.selectbox('Dependents', ('0', '1', '2', '3+'))
        Education = st.selectbox('Education', ('Graduate', 'Not Graduate'))
        Self_Employed = st.selectbox('Self Employed', ('Yes', 'No'))
        ApplicantIncome = st.number_input('Applicant Income', min_value=0, step=100)
        CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0, step=100)
        LoanAmount = st.number_input('Loan Amount', min_value=1, step=1)
        Loan_Amount_Term = st.selectbox('Loan Amount Term', ('360', '180', '120', '240'))
        Credit_History = st.selectbox('Credit History', ('1.0', '0.0'))
        Property_Area = st.selectbox('Property Area', ('Urban', 'Semiurban', 'Rural'))

        submitted = st.form_submit_button("Predict")

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
