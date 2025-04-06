# Loan Eligibility Prediction App

Streamlit application that predicts loan approval probability based on applicant data. It uses a cleaned and preprocessed version of the UCI Loan Prediction dataset and supports both Logistic Regression and Random Forest models.

![Scikit-learn](https://img.shields.io/badge/framework-scikit--learn-blue)
![Streamlit](https://img.shields.io/badge/ui-streamlit-orange)
![Model Accuracy](https://img.shields.io/badge/logistic%20regression-85%25-brightgreen)


## Streamlit Demo

[Launch App](https://loan-eligibility-prediction-djdstcprojdapfxugpdqwi.streamlit.app/)


## Features

- Loads and preprocesses raw loan application data
- Trains Logistic Regression and Random Forest classifiers (or loads pre-trained models)
- Saves preprocessed data, model, and scaler for reuse
- Displays model accuracy at the top of the interface
- Renders a user-friendly form for applicant input
- Predicts loan approval in real time based on user input
- Visualizes results using a gauge with arrow pointer and color-coded interpretation
- Logs application behavior and handles errors gracefully
- Supports dynamic model selection in the UI


## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/AhmedOT22/loan-eligibility-app.git
   cd loan-eligibility-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Launch the app**
   ```bash
   streamlit run app.py
   ```

## Model

- **Algorithm**: Logistic Regression, Random Forest Classifier
- **Preprocessing**: Handling missing values, one-hot encoding, scaling
- **Cross-validation**: 5-fold
- **Metric**: Accuracy Score


## Requirements

- Python ≥ 3.8  
- Streamlit  
- Scikit-learn  
- Pandas  
- Plotly  


## Dataset
- Path: `data/raw/credit.csv`
- Target Column: `Loan_Approved`
- Features Used:
  - Categorical: `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`
  - Numerical: `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`
- Preprocessing:
  - Fill missing values (mode for categorical, median for numerical)
  - One-hot encoding for categorical variables
  - Min-Max scaling for numerical features


## Model Architecture
- Models: Logistic Regression, Random Forest
- Evaluation Metric: Accuracy
- Cross-validation: 5-fold optional for deeper evaluation
- Logging: Activity stored in `loan_app.log`
- Metrics: Saved in JSON format (`logistic_regression_metrics.json`, `random_forest_metrics.json`)


## Results
- Logistic Regression Accuracy: **85.37%** (Best Performer)
- Random Forest Accuracy: **79.67%**
- Evaluation performed on a validation split
- Visual prediction results with interpretive gauge (very unlikely → very likely)


## Requirements
- Python ≥ 3.8
- streamlit
- scikit-learn
- pandas
- numpy
- plotly


## Future Enhancements
- Fine-tuning with grid/random search
- Probability calibration for better probability interpretation
- Exportable prediction reports (PDF/CSV)
- Secure access & role-based dashboard


## Author
Developed by [Ahmed Ouazzani](https://github.com/AhmedOT22)


## License
MIT License © 2025
