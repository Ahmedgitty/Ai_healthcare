# 🏥 Explainable Multi-Disease Risk Prediction System

> AI in Healthcare | AIML Subject Project | Team of 5

---

## 📌 Project Goal
Predict risk for **Diabetes, Heart Disease, and Kidney Disease** from patient health data, evaluate models rigorously, and explain predictions using **SHAP + LIME**, wrapped in an interactive **Streamlit dashboard** with What-If analysis.

---

## 👥 Team & Ownership

| Member | Responsibility | Folder |
|--------|---------------|--------|
| Member 1 | Diabetes — data cleaning, SMOTE, model training | `src/data/`, `src/models/diabetes_*` |
| Member 2 | Heart Disease — data cleaning, SMOTE, model training | `src/data/`, `src/models/heart_*` |
| Member 3 | Kidney Disease — data cleaning, SMOTE, model training | `src/data/`, `src/models/kidney_*` |
| Member 4 | SHAP + LIME explainability for all 3 diseases | `src/explainability/` |
| Member 5 | Streamlit dashboard + What-If analysis + integration | `src/dashboard/` |

---

## 📂 Folder Structure

```
multi-disease-risk-prediction/
│
├── data/
│   ├── raw/                  # Original downloaded datasets (NOT committed to Git)
│   └── processed/            # Cleaned, SMOTE-applied datasets (NOT committed to Git)
│
├── notebooks/                # Jupyter notebooks for EDA and experimentation
│   ├── 01_eda_diabetes.ipynb
│   ├── 02_eda_heart.ipynb
│   ├── 03_eda_kidney.ipynb
│   └── 04_model_comparison.ipynb
│
├── src/
│   ├── data/                 # Data loading and preprocessing scripts
│   │   ├── preprocess_diabetes.py
│   │   ├── preprocess_heart.py
│   │   └── preprocess_kidney.py
│   │
│   ├── models/               # Model training scripts
│   │   ├── train_diabetes.py
│   │   ├── train_heart.py
│   │   ├── train_kidney.py
│   │   └── evaluate.py       # Shared evaluation utilities
│   │
│   ├── explainability/       # SHAP and LIME scripts
│   │   ├── shap_explainer.py
│   │   └── lime_explainer.py
│   │
│   └── dashboard/            # Streamlit app
│       ├── app.py            # Main entry point
│       ├── pages/
│       │   ├── diabetes_page.py
│       │   ├── heart_page.py
│       │   └── kidney_page.py
│       └── utils.py          # Shared dashboard utilities
│
├── models/
│   └── saved_models/         # Trained .joblib model files (NOT committed to Git)
│
├── results/
│   ├── figures/              # SHAP plots, ROC curves, confusion matrices
│   └── metrics/              # CSV files with model evaluation results
│
├── docs/
│   ├── SETUP.md              # Detailed setup instructions
│   ├── DATASETS.md           # Where to download datasets and how to place them
│   └── LEARNINGS.md          # Team notes on concepts learned week by week
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/multi-disease-risk-prediction.git
cd multi-disease-risk-prediction
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Datasets
See `docs/DATASETS.md` for instructions on where to download each dataset and where to place them.

### 5. Run the Dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## 🧠 Models Used
- **Logistic Regression** — Baseline, most interpretable
- **Random Forest** — Primary model, handles non-linear data well
- **XGBoost** — Advanced boosting model, best performer
- **Voting Ensemble** — Combines all 3 for robust predictions

---

## 🔍 Explainability
- **SHAP** — Global feature importance + per-patient waterfall plots
- **LIME** — Local explanation as a second perspective
- Both techniques are compared side by side in the dashboard

---

## 📊 Evaluation Metrics
- AUC-ROC (primary metric)
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (with focus on False Negatives)
- Model comparison table across all algorithms

---

## 📅 Weekly Plan

| Week | Focus |
|------|-------|
| Week 1 | Data cleaning, EDA, SMOTE, train baseline models |
| Week 2 | Random Forest, XGBoost, Voting Ensemble, evaluation metrics |
| Week 3 | SHAP integration, LIME integration, explainability plots |
| Week 4 | Streamlit dashboard, What-If analysis, final integration |

---

## 📖 Resources
- [StatQuest ML Playlist](https://www.youtube.com/c/joshstarmer) — Start here for concepts
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME GitHub](https://github.com/marcotcr/lime)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
