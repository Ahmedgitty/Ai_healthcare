"""
Streamlit Dashboard — Main Entry Point
Member 5's responsibility

Run with: streamlit run src/dashboard/app.py
"""

import streamlit as st

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Disease Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.title("🏥 Disease Risk Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Disease",
    ["🏠 Home", "🩸 Diabetes", "❤️ Heart Disease", "🫘 Kidney Disease", "📊 Model Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack:**")
st.sidebar.markdown("- Scikit-learn, XGBoost")
st.sidebar.markdown("- SHAP + LIME (XAI)")
st.sidebar.markdown("- Streamlit")

# ── Page Routing ──────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🏥 Explainable Multi-Disease Risk Prediction System")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 🩸 Diabetes\nPredicts diabetes risk using glucose, BMI, age and other health markers.")
    with col2:
        st.warning("### ❤️ Heart Disease\nPredicts heart disease risk using cholesterol, blood pressure, and ECG data.")
    with col3:
        st.error("### 🫘 Kidney Disease\nPredicts chronic kidney disease using blood and urine test results.")

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. **Select a disease** from the sidebar
    2. **Enter patient health data** using the input fields
    3. **Get a risk prediction** with probability score
    4. **Understand the prediction** through SHAP and LIME explanations
    5. **Try What-If analysis** to see how changing one value affects risk
    """)

    st.markdown("---")
    st.markdown("### Models Used")
    st.markdown("""
    Each disease uses an ensemble of:
    - **Logistic Regression** — baseline
    - **Random Forest** — primary model
    - **XGBoost** — boosted model
    - **Voting Ensemble** — combines all 3 (best performer)
    """)

elif page == "🩸 Diabetes":
    from src.dashboard.pages.diabetes_page import show
    show()

elif page == "❤️ Heart Disease":
    from src.dashboard.pages.heart_page import show
    show()

elif page == "🫘 Kidney Disease":
    from src.dashboard.pages.kidney_page import show
    show()

elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("---")

    import pandas as pd
    import os

    # Load saved metrics CSVs
    all_metrics = []
    for disease in ["diabetes", "heart", "kidney"]:
        path = f"results/metrics/metrics_{disease}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_metrics.append(df)

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        st.dataframe(combined.style.highlight_max(subset=["AUC-ROC", "F1 Score"], color="lightgreen"))
    else:
        st.warning("No metrics found. Train models first by running the training scripts.")
