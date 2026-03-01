"""
Heart Disease Page — Streamlit Dashboard
Member 5's responsibility

TODO: Follow the same pattern as diabetes_page.py
Key differences:
- Feature names are different (age, sex, cp, trestbps, chol, etc.)
- Model files are ensemble_heart.joblib and scaler_heart.joblib
- What-If analysis — try varying 'chol' (cholesterol) or 'thalach' (max heart rate)
"""

import streamlit as st

def show():
    st.title("❤️ Heart Disease Risk Prediction")
    st.info("🔧 Under Construction — Follow the diabetes_page.py pattern to build this page.")

    # Feature names for heart disease dataset
    FEATURE_NAMES = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    st.markdown("**Features to include as sliders:**")
    st.write(FEATURE_NAMES)
