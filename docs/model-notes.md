# Modelling decisions and serving boundary

## Problem and target

Predict the supplied CSV's `Loan_Approved` Y/N label from eleven raw features. This is a supervised-learning demonstration of a historical label, not a lender decision engine. The positive class is Y; a false positive means predicting Y when the recorded label is N. The data does not establish why any particular label was assigned.

## Two artifact paths

| Path | Preparation | Use |
| --- | --- | --- |
| `models/*_model.pkl` and `models/scaler.pkl` | Legacy numeric imputation/scaling and encoded feature schema | Existing Streamlit demo |
| `train_model.py --save-models` output | Each fitted pipeline owns its training-only imputation, encoding, and scaling | Local candidate evaluation; not loaded by the demo |

The legacy preparation learned statistics before splitting. Repairing inference cannot undo that training history. The application therefore explains the status of its stored accuracy instead of treating it as a clean holdout estimate. Candidate pipelines accept raw DataFrames; they must not replace encoded estimators by simply renaming files.

Before migrating serving, verify raw input validation, artifact schema and package compatibility, model/version labeling, and application tests for each candidate. Preserve a rollback path. No automatic promotion is performed by either evaluation command.

## Evaluation choices

- Reserve a stratified 20% holdout using seed 42 before fitting preprocessing. This is a random split, not temporal or external validation.
- Compare fixed Logistic Regression and Random Forest candidates with a majority-class baseline. Do not search parameters against the holdout.
- Use repeated stratified five-fold CV on only the remaining 80%. Refitting the complete pipeline in every fold prevents validation statistics from entering imputation, scaling, or category learning.
- Report accuracy, precision, recall, specificity, ROC AUC, average precision, Brier score, and confusion matrices. High positive recall alone can hide poor negative-class performance.
- Pool one repeat for threshold diagnostics, avoiding counting each row three times. Threshold 0.5 remains fixed; a useful decision threshold requires an explicit cost objective and separate validation.
- Use fold standard deviation descriptively. Shared observations and overlapping training folds prevent interpreting these values as independent experimental replications.

Logistic Regression scales numeric features because its coefficients depend on feature scale. Random Forest shares the preprocessing contract for simplicity, although tree splits do not require MinMax scaling. Unknown categories are ignored by candidate encoders; production behavior would need monitoring and a defined rejection policy. The legacy adapter instead rejects categories absent from its fixed schema.

## Data and interpretation limits

Source, collection period, geography, currency, and amount units are unverified. Numerical comparisons use the stored scale. A row is not necessarily an independent applicant; no grouping key or repeat-application policy has been verified. Random splitting could therefore be optimistic even with correct preprocessing.

Gender and marital status are retained to reproduce the supplied task. Their presence is not an endorsement of using them for real financial decisions. No fairness, legal suitability, calibration, or deployment-readiness claim is supported by this evaluation. Dataset size also limits conclusions from small subgroup counts.

## Output handling

`evaluate_model.py` exports only aggregate metrics, fold summaries, and figures. It never saves row-level predictions or fitted models. `train_model.py` retains its local per-row diagnostics under the ignored `reports/latest/` path by default. Do not commit applicant labels, scores, or split indices to public reports. The application does not write submitted values or model scores to a local log.

## Next substantive work

Serving migration and calibration are separate modelling changes requiring explicit comparisons, not cosmetic cleanup. Source validation remains outstanding. Existing charts show genuine evaluation output; they are not screenshots of a verified hosted deployment.
