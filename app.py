"""Streamlit demonstration using the evaluated train-only raw-input pipelines."""

import streamlit as st

from src.models.serving import load_bundle, predict_score
from src.utils.form import (
    display_model_info,
    fetch_input,
    interpret_probability,
    load_custom_styles,
    model_selector,
)
from src.utils.gauge import generate_gauge_chart


def main():
    """Score a complete form using a verified serving release, without fitting or logging inputs."""
    st.set_page_config(page_title="Loan Eligibility Prediction", layout="centered")
    load_custom_styles()
    st.title("🏦 Loan Eligibility Prediction App")
    st.caption(
        "Educational demonstration. Scores are uncalibrated model outputs, "
        "not lender approval probabilities or decisions."
    )
    model_key = model_selector()
    try:
        model, manifest = load_bundle(model_key)
    except (OSError, ValueError, TypeError, KeyError, ImportError, EOFError, AttributeError):
        st.error(
            "The verified model release is unavailable. Please try again after it is restored."
        )
        st.stop()

    display_model_info(manifest, model_key)
    user_input = fetch_input()
    if user_input:
        try:
            score = predict_score(user_input, model_key, model, manifest)
            interpretation, color = interpret_probability(score)
            st.subheader("📊 Model Score")
            st.plotly_chart(
                generate_gauge_chart(score, interpretation, color), use_container_width=True
            )
            st.caption(
                f"Release: {manifest['release']}. This pipeline owns its fitted preprocessing "
                "and was trained only on the training partition used in the README evaluation."
            )
        except (ValueError, TypeError, KeyError):
            st.error(
                "Check that all form fields contain valid values before calculating a score."
            )


if __name__ == "__main__":
    main()
