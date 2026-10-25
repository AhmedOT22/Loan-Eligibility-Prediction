# Loan Eligibility Prediction

An educational classification project comparing Logistic Regression and Random Forest with a majority-class baseline. It includes a Streamlit demonstration and a separate raw-input training workflow with train-only preprocessing.

[Streamlit demo](https://loan-eligibility-prediction-djdstcprojdapfxugpdqwi.streamlit.app/) · [Training workflow](train_model.py) · [Evaluation report](reports/baseline/metrics.json)

## What changed in the modelling workflow

Raw applications are split before missing-value statistics or category vocabularies are learned. Each candidate contains its own fitted preprocessing:

1. Numeric median imputation and MinMax scaling, fitted on training rows.
2. Categorical mode imputation and one-hot encoding, fitted on training rows; unseen categories are handled at transform time.
3. Logistic Regression or a seeded Random Forest classifier.

`Loan_ID` and the target are excluded from model inputs. The 80/20 split is stratified by the target and uses seed 42. No cross-validation or probability calibration is claimed for this fixed-candidate evaluation.

## Results on the bundled CSV

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 85.37% | 0.832 | 0.988 | 0.903 | 0.852 |
| Random Forest | 82.11% | 0.846 | 0.906 | 0.875 | 0.781 |
| Majority baseline | 69.11% | 0.691 | 1.000 | 0.817 | 0.500 |

Positive class means `Loan_Approved = Y` in the dataset. The holdout contains 123 rows, including 85 positives. Precision/recall/F1 above refer to that positive class. See [metrics and confusion matrices](reports/baseline/metrics.json) and [run metadata](reports/baseline/run.json) for exact aggregate metrics, checksum, class proportions, and package versions. Per-applicant labels and scores are generated locally and are not published in the saved report.

Logistic Regression performs best by accuracy and ROC AUC in this single split. Its high positive recall does not mean equally strong identification of negatives: the confusion matrix records 17 false positives and one false negative. A different split or decision threshold may change the tradeoff. Scores are not calibrated lender approval probabilities.

These results evaluate newly trained raw-input pipelines, not the older artifacts currently loaded by the app. No legacy model, scaler, processed dataset, or stored legacy metrics were overwritten. The original helper functions remain for compatibility; the new training command does not use whole-dataset preprocessing.

## Data scope

The supplied `data/raw/credit.csv` has 614 rows. Features are:

- Categorical: `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`.
- Numeric: `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`.
- Target: `Loan_Approved`, encoded from `Y/N`.

The CSV does not establish its original publisher, source URL, collection period, geography, currency, or representativeness. The earlier UCI attribution has been removed because no supporting source was supplied. Do not treat this dataset as evidence of an actual lender's current policy. Demographic inputs and historical labels also require evaluation before any real decision use; this repository is a modelling demonstration.

## Run locally

Use Python 3.11 or 3.12 from the repository root:

```bash
git clone https://github.com/AhmedOT22/Loan-Eligibility-Prediction.git
cd Loan-Eligibility-Prediction
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
streamlit run app.py
```

The serving environment pins scikit-learn 1.4.2 to match the bundled artifacts, Streamlit 1.50.0 for tested application flows, and PyArrow 20.0.0 for compatibility with NumPy 1.x. Only load pickle/joblib artifacts from trusted sources. Live demo availability was not verified in this code update.

## Train without replacing app artifacts

```bash
python train_model.py
# Optional: save complete raw-input candidate pipelines under the report directory
python train_model.py --output reports/candidate --save-models
```

Default output is `reports/latest/`: metrics and confusion matrices, per-row class/score predictions, and run metadata. `--save-models` writes separate `*_pipeline.joblib` files beneath the selected report directory. These pipelines accept raw feature DataFrames and own their preprocessing; they are not drop-in replacements for the app's separate encoded-model/scaler files.

Training errors propagate instead of being swallowed with a success log. Importing the script does not train or write artifacts.

## Existing app preparation

The app's adapter directly encodes a complete applicant, checks category and feature-order compatibility, and uses the fitted legacy scaler without fitting anything at inference. It no longer concatenates a raw applicant with the processed training table. Missing fields and unsupported categories fail explicitly. The gauge and score-band labels describe an uncalibrated model score. Input tips are illustrative checks. Model, scaler, and CSS paths are anchored to the repository; the feature order comes from the fitted scaler, so startup does not depend on the processed CSV.

## Tests

```bash
python -m pytest -q
```

Tests cover train-only imputation statistics, unseen-category transformation without refitting, disjoint splits, invalid target rejection, legacy applicant encoding, and independence from the supplied reference table. Both bundled classifiers were also smoke-tested with the repaired adapter under scikit-learn 1.4.2. Streamlit application tests also submit forms through both bundled classifiers from a different working directory. Legacy helper failures propagate instead of returning success-shaped defaults. CI runs the tests and full Pylint checks with installed project dependencies.

Further work should evaluate repeatability across splits, threshold selection on validation data, calibration, and an explicit migration of serving artifacts if new pipelines are adopted. No external data collection is needed to continue those code changes.

[MIT License](LICENSE) · [Ahmed Ouazzani](https://github.com/AhmedOT22)

To run the same lint check locally:

```bash
python -m pylint $(git ls-files '*.py')
```
