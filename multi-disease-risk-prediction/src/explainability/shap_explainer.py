"""
SHAP Explainability
Member 4's responsibility

This script generates:
1. Global SHAP summary plot — which features matter most overall
2. Per-patient waterfall plot — why this specific prediction was made

Run AFTER all models are trained and saved.
"""

import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

MODELS_DIR = "models/saved_models/"
FIGURES_DIR = "results/figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)


def get_shap_explainer(model, X_train, model_type="tree"):
    """
    Create the right SHAP explainer based on model type.

    - tree: for Random Forest and XGBoost (fast, exact)
    - linear: for Logistic Regression
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_train)
    else:
        # Kernel is slowest but works for any model
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
    return explainer


def plot_global_summary(explainer, X_test, feature_names, disease_name, model_name):
    """
    Summary plot showing global feature importance.
    Each dot is one patient — color shows feature value (red=high, blue=low).
    X-axis shows impact on model output.
    """
    shap_values = explainer.shap_values(X_test)

    # For binary classifiers, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use class 1 (disease present)

    plt.figure()
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        show=False
    )
    path = f"{FIGURES_DIR}shap_summary_{disease_name}_{model_name}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP summary plot saved: {path}")
    return shap_values


def plot_waterfall_single_patient(explainer, X_test, shap_values, patient_index, feature_names, disease_name):
    """
    Waterfall plot for a single patient showing which features pushed
    the prediction up (red) or down (blue).
    """
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[patient_index],
            base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                        else explainer.expected_value[1],
            data=X_test[patient_index],
            feature_names=feature_names
        ),
        show=False
    )
    path = f"{FIGURES_DIR}shap_waterfall_{disease_name}_patient{patient_index}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP waterfall plot saved: {path}")


def explain_disease(disease_name, X_test, feature_names):
    """
    Full SHAP explanation pipeline for one disease.
    Loads the saved Random Forest model and generates all plots.
    """
    print(f"\n{'='*50}")
    print(f"  SHAP Explanation — {disease_name.upper()}")
    print(f"{'='*50}")

    model = joblib.load(f"{MODELS_DIR}rf_{disease_name}.joblib")
    explainer = get_shap_explainer(model, X_test, model_type="tree")
    shap_values = plot_global_summary(explainer, X_test, feature_names, disease_name, "rf")

    # Explain first 3 patients as examples
    for i in range(min(3, len(X_test))):
        plot_waterfall_single_patient(explainer, X_test, shap_values, i, feature_names, disease_name)

    return explainer, shap_values


def get_shap_for_patient(model, patient_data, feature_names):
    """
    Used by the dashboard — returns SHAP values for a single patient input.

    Args:
        model: trained model
        patient_data: numpy array of shape (1, n_features)
        feature_names: list of feature names

    Returns:
        shap.Explanation object for use in shap.waterfall_plot()
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    expected = explainer.expected_value
    if isinstance(expected, list):
        expected = expected[1]

    return shap.Explanation(
        values=shap_values[0],
        base_values=expected,
        data=patient_data[0],
        feature_names=feature_names
    )


# ── Run directly to generate all SHAP plots ──────────────────────────────────
if __name__ == "__main__":
    # TODO (Member 4):
    # Import preprocessing functions for each disease and get X_test, feature_names
    # Then call explain_disease() for each disease
    #
    # Example:
    # from src.data.preprocess_diabetes import load_data, preprocess, split_and_smote, scale_features
    # df = load_data(); df = preprocess(df)
    # X_train, X_test, y_train, y_test, features = split_and_smote(df)
    # X_train_s, X_test_s = scale_features(X_train, X_test)
    # explain_disease("diabetes", X_test_s, features)
    pass
