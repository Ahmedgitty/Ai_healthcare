"""
LIME Explainability
Member 4's responsibility

LIME (Local Interpretable Model-agnostic Explanations)
Explains individual predictions by fitting a simple linear model
locally around that data point.

Compare LIME output vs SHAP output — do they agree on which features matter?
"""

import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

MODELS_DIR = "models/saved_models/"
FIGURES_DIR = "results/figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)


def create_lime_explainer(X_train, feature_names, class_names=["No Disease", "Disease"]):
    """
    Create a LIME TabularExplainer.

    Args:
        X_train: Training data (numpy array) — LIME uses this to understand feature ranges
        feature_names: List of feature names
        class_names: Labels for the two classes

    Returns:
        LimeTabularExplainer object
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=42
    )
    return explainer


def explain_patient_lime(lime_explainer, model, patient_data, feature_names, disease_name, patient_index=0):
    """
    Generate a LIME explanation for a single patient.

    Args:
        lime_explainer: LimeTabularExplainer object
        model: Trained sklearn-compatible model
        patient_data: 1D numpy array of one patient's features
        feature_names: List of feature names
        disease_name: e.g. 'diabetes'
        patient_index: for naming the saved file

    Returns:
        lime explanation object
    """
    explanation = lime_explainer.explain_instance(
        data_row=patient_data,
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )

    # Save plot
    fig = explanation.as_pyplot_figure()
    fig.suptitle(f"LIME Explanation — {disease_name} (Patient {patient_index})", fontsize=12)
    path = f"{FIGURES_DIR}lime_{disease_name}_patient{patient_index}.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ LIME explanation saved: {path}")

    # Print top features
    print(f"\nTop LIME features for Patient {patient_index}:")
    for feature, weight in explanation.as_list():
        direction = "↑ risk" if weight > 0 else "↓ risk"
        print(f"  {feature}: {weight:.4f} ({direction})")

    return explanation


def get_lime_for_patient(lime_explainer, model, patient_data, feature_names):
    """
    Used by the dashboard — returns LIME explanation for a single patient.

    Args:
        lime_explainer: LimeTabularExplainer
        model: trained model
        patient_data: 1D numpy array
        feature_names: list of feature names

    Returns:
        List of (feature_condition, weight) tuples for display
    """
    explanation = lime_explainer.explain_instance(
        data_row=patient_data,
        predict_fn=model.predict_proba,
        num_features=len(feature_names)
    )
    return explanation.as_list(), explanation.as_pyplot_figure()


# ── Run directly to generate LIME plots ──────────────────────────────────────
if __name__ == "__main__":
    # TODO (Member 4):
    # Similar to shap_explainer.py — load preprocessed data, create explainer,
    # and run explain_patient_lime() for the first few patients of each disease.
    pass
