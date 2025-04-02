# Loan Eligibility Prediction App

This is a modular, production-ready Streamlit application that predicts loan approval likelihood based on applicant data. The model uses a cleaned and preprocessed version of the UCI Loan Prediction dataset and is backed by a trained Random Forest classifier.

---

## Features

- Loads and preprocesses raw loan application data
- Trains a Random Forest classifier on cleaned data (or loads a pre-trained model)
- Saves preprocessed data, model, and scaler for reuse
- Displays model accuracy at the top of the interface
- Renders a user-friendly form for applicant input
- Predicts loan approval in real time based on user input
- Visualizes results using a gauge with arrow pointer and color-coded interpretation
- Logs application behavior and handles errors gracefully


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

---

## Model

- **Algorithm**: Logistic Regression, Random Forest Classifier
- **Preprocessing**: Handling missing values, one-hot encoding, scaling
- **Cross-validation**: 5-fold
- **Metric**: Accuracy Score

---

## Requirements

- Python ≥ 3.8  
- Streamlit  
- Scikit-learn  
- Pandas  
- Plotly  

Install them with:

```bash
pip install -r requirements.txt
```

---

## Future Enhancements

- Model comparison between multiple classifiers
- Authentication for secure access
- API backend for scaling

---

## Author

Developed by Ahmed Ouazzani ['[AhmedOT22](https://github.com/AhmedOT22)']