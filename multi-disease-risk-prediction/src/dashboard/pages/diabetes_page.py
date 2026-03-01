"""
Diabetes Page — Streamlit Dashboard
Member 5's responsibility

Shows:
- Patient input sliders
- Risk prediction + probability
- SHAP waterfall plot
- LIME explanation
- What-If analysis slider
"""

import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Feature names for diabetes dataset (Pima Indians)
FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

MODELS_DIR = "models/saved_models/"


@st.cache_resource
def load_model_and_scaler():
    """Load saved model and scaler. Cached so it only loads once."""
    try:
        model = joblib.load(f"{MODELS_DIR}ensemble_diabetes.joblib")
        scaler = joblib.load(f"{MODELS_DIR}scaler_diabetes.joblib")
        return model, scaler
    except FileNotFoundError:
        return None, None


def show():
    st.title("🩸 Diabetes Risk Prediction")
    st.markdown("Enter patient details below to predict diabetes risk.")
    st.markdown("---")

    model, scaler = load_model_and_scaler()
    if model is None:
        st.error("⚠️ Model not found. Please run `src/models/train_diabetes.py` first.")
        return

    # ── Patient Input ──────────────────────────────────────────────────────────
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Pregnancies", 0, 17, 3)
        glucose = st.slider("Glucose (mg/dL)", 0, 200, 120)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 99, 20)

    with col2:
        insulin = st.slider("Insulin (mu U/ml)", 0, 846, 80)
        bmi = st.slider("BMI", 0.0, 67.0, 25.0, step=0.1)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)
        age = st.slider("Age", 21, 81, 30)

    patient_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                               insulin, bmi, dpf, age]])

    # ── Prediction ────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔍 Predict Risk", type="primary"):
        patient_scaled = scaler.transform(patient_input)
        probability = model.predict_proba(patient_scaled)[0][1]
        prediction = model.predict(patient_scaled)[0]

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error(f"### ⚠️ High Risk of Diabetes\nProbability: **{probability*100:.1f}%**")
            else:
                st.success(f"### ✅ Low Risk of Diabetes\nProbability: **{probability*100:.1f}%**")

        with col2:
            st.metric("Risk Probability", f"{probability*100:.1f}%")
            st.progress(float(probability))

        # ── SHAP Explanation ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Why this prediction? (SHAP)")
        st.markdown("Red bars increase risk, blue bars decrease risk.")

        try:
            from src.explainability.shap_explainer import get_shap_for_patient
            import shap

            # Use the Random Forest for SHAP (TreeExplainer is faster and exact)
            rf_model = joblib.load(f"{MODELS_DIR}rf_diabetes.joblib")
            explanation = get_shap_for_patient(rf_model, patient_scaled, FEATURE_NAMES)

            fig, ax = plt.subplots()
            shap.waterfall_plot(explanation, show=False)
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

        # ── LIME Explanation ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Why this prediction? (LIME)")
        st.markdown("LIME fits a simple model locally to explain this specific prediction.")

        try:
            from src.explainability.lime_explainer import create_lime_explainer, get_lime_for_patient
            # TODO (Member 4+5): Pass actual X_train here
            # For now this is a placeholder — connect with real training data
            st.info("LIME requires training data to be loaded. Connect preprocess_diabetes.py here.")
        except Exception as e:
            st.warning(f"LIME explanation unavailable: {e}")

        # ── What-If Analysis ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔄 What-If Analysis")
        st.markdown("Change one value and see how the risk changes.")

        whatif_glucose = st.slider("What if Glucose was...", 0, 200, int(glucose), key="whatif")
        whatif_input = patient_input.copy()
        whatif_input[0][1] = whatif_glucose  # Glucose is index 1
        whatif_scaled = scaler.transform(whatif_input)
        whatif_prob = model.predict_proba(whatif_scaled)[0][1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Risk", f"{probability*100:.1f}%")
        with col2:
            delta = whatif_prob - probability
            st.metric("New Risk", f"{whatif_prob*100:.1f}%",
                      delta=f"{delta*100:+.1f}%",
                      delta_color="inverse")
