"""
Heart Disease Dataset - Data Preprocessing
Member 2's responsibility

Steps:
1. Load raw data
2. Explore and understand the data
3. Handle missing values and encode categoricals
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

RAW_PATH = "data/raw/heart.csv"
PROCESSED_DIR = "data/processed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    """Load raw heart disease dataset."""
    df = pd.read_csv(RAW_PATH)
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nClass distribution:\n{df['target'].value_counts()}")
    return df

def preprocess(df):
    """
    Clean and preprocess the heart disease dataset.

    TODO (Member 2):
    - Check if any categorical columns need encoding (cp, thal, slope etc.)
    - Handle missing values if present
    - The target column is 'target' (0=no disease, 1=disease)
    """
    # TODO: Handle any missing values
    # TODO: If needed, one-hot encode categorical features
    # Hint: pd.get_dummies(df, columns=['cp', 'thal', 'slope'], drop_first=True)

    print("\nAfter cleaning:")
    print(df.isnull().sum())
    return df

def split_and_smote(df):
    """Split into train/test and apply SMOTE to training data only."""
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nBefore SMOTE — Train class distribution:\n{y_train.value_counts()}")

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — Train class distribution:\n{pd.Series(y_train_resampled).value_counts()}")

    return X_train_resampled, X_test, y_train_resampled, y_test, X.columns.tolist()

def scale_features(X_train, X_test):
    """Standardize features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "models/saved_models/scaler_heart.joblib")
    print("Scaler saved.")
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    X_train, X_test, y_train, y_test, feature_names = split_and_smote(df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    print("\n✅ Heart disease preprocessing complete.")
