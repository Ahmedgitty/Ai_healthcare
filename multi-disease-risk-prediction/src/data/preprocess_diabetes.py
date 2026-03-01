"""
Diabetes Dataset - Data Preprocessing
Member 1's responsibility

Steps:
1. Load raw data
2. Explore and understand the data (shape, dtypes, nulls)
3. Handle missing values
4. Apply SMOTE for class imbalance
5. Scale features
6. Save processed data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_PATH = "data/raw/diabetes.csv"
PROCESSED_DIR = "data/processed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    """Load raw diabetes dataset."""
    df = pd.read_csv(RAW_PATH)
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nClass distribution:\n{df['Outcome'].value_counts()}")
    return df

def preprocess(df):
    """
    Clean and preprocess the diabetes dataset.

    TODO (Member 1):
    - Some columns like Glucose, BMI have 0 values which are biologically impossible.
      Replace those 0s with NaN and then fill with column median.
    - Check for any other data issues.
    """
    # Columns where 0 is not valid
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # TODO: Replace 0s with NaN in above columns, then fill with median
    # Hint: df[col] = df[col].replace(0, np.nan)
    #       df[col] = df[col].fillna(df[col].median())

    print("\nAfter cleaning:")
    print(df.isnull().sum())
    return df

def split_and_smote(df):
    """Split into train/test and apply SMOTE to training data only."""
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 80% train, 20% test — stratify keeps class ratio same in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nBefore SMOTE — Train class distribution:\n{y_train.value_counts()}")

    # Apply SMOTE only on training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — Train class distribution:\n{pd.Series(y_train_resampled).value_counts()}")

    return X_train_resampled, X_test, y_train_resampled, y_test, X.columns.tolist()

def scale_features(X_train, X_test):
    """Standardize features (mean=0, std=1)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit only on train
    X_test_scaled = scaler.transform(X_test)         # Transform test with same scaler

    # Save scaler for use in dashboard later
    joblib.dump(scaler, "models/saved_models/scaler_diabetes.joblib")
    print("Scaler saved.")
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    X_train, X_test, y_train, y_test, feature_names = split_and_smote(df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    print("\n✅ Diabetes preprocessing complete.")
