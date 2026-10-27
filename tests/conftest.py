"""Synthetic form inputs shared by serving and application integration tests."""

import pytest


@pytest.fixture
def applicant():
    """Return a hypothetical complete form, independent of any dataset applicant."""
    return {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Urban",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 150,
        "Loan_Amount_Term": "360",
        "Credit_History": "1.0",
    }
