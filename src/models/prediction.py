"""Inference adapter for the existing app's encoded model/scaler artifacts."""
import numpy as np
import pandas as pd
from src.models.pipeline import CATEGORICAL, NUMERIC


def predict(user_input_dict, df_processed, model, scaler, feature_order):
    # df_processed remains in the signature for the existing app; it is never
    # concatenated with a raw applicant or used to fit preprocessing at inference.
    required = set(CATEGORICAL + NUMERIC)
    if required - set(user_input_dict):
        raise ValueError(f'Missing applicant fields: {sorted(required - set(user_input_dict))}')
    if len(feature_order) != len(set(feature_order)):
        raise ValueError('Feature order contains duplicates.')
    frame = pd.DataFrame([user_input_dict])[CATEGORICAL + NUMERIC]
    if frame.isna().any().any():
        raise ValueError('Complete all applicant fields before predicting.')
    frame[NUMERIC] = frame[NUMERIC].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame[NUMERIC].to_numpy(dtype=float)).all() or (frame[NUMERIC] < 0).any().any():
        raise ValueError('Applicant numeric values must be finite and non-negative.')
    encoded = pd.get_dummies(frame, columns=CATEGORICAL, dtype=int)
    unknown = set(encoded.columns) - set(feature_order)
    if unknown:
        raise ValueError(f'Unsupported applicant categories: {sorted(unknown)}')
    encoded = encoded.reindex(columns=feature_order, fill_value=0)
    expected = getattr(scaler, 'feature_names_in_', None)
    if expected is not None and list(expected) != list(feature_order):
        raise ValueError('Scaler and application feature order do not match.')
    scaled = scaler.transform(encoded)
    if hasattr(model, 'feature_names_in_'):
        if list(model.feature_names_in_) != list(feature_order):
            raise ValueError('Model and application feature order do not match.')
        scaled = pd.DataFrame(scaled, columns=feature_order)
    positive_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(scaled)[0][positive_index] * 100)
    return probability, encoded
