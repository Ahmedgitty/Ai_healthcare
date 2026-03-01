"""
Kidney Disease Dataset - Data Preprocessing
Member 3's responsibility

⚠️ This is the most complex dataset:
   - 26 columns with many missing values
   - Mix of numeric and categorical columns
   - Target column is text: 'ckd' or 'notckd'

Steps:
1. Load raw data
2. Fix column names and data types
3. Encode categoricals
4. Handle missing values (many!)
5. Encode target variable
6. Apply SMOTE
7. Scale features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

RAW_PATH = "data/raw/kidney.csv"
PROCESSED_DIR = "data/processed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    """Load raw kidney disease dataset."""
    df = pd.read_csv(RAW_PATH)
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nClass distribution:\n{df['classification'].value_counts()}")
    return df

def preprocess(df):
    """
    Clean and preprocess the kidney disease dataset.

    TODO (Member 3):
    1. Drop the 'id' column if present
    2. Strip extra spaces from string columns (common issue in this dataset)
       Hint: df[col] = df[col].str.strip()
    3. Encode binary categorical columns like 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
       Hint: use LabelEncoder or map {'yes': 1, 'no': 0}
    4. For numeric columns that loaded as object due to typos, convert carefully
       Hint: pd.to_numeric(df[col], errors='coerce')
    5. Fill missing values:
       - Numeric columns: fill with median
       - Categorical columns: fill with mode
    6. Encode target: 'ckd' → 1, 'notckd' → 0
    """

    # Step 1: Drop id column if it exists
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # TODO: Complete steps 2-6 above

    print("\nAfter cleaning:")
    print(df.isnull().sum())
    return df

def split_and_smote(df):
    """Split into train/test and apply SMOTE."""
    X = df.drop('classification', axis=1)
    y = df['classification']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nBefore SMOTE:\n{y_train.value_counts()}")

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")

    return X_train_resampled, X_test, y_train_resampled, y_test, X.columns.tolist()

def scale_features(X_train, X_test):
    """Standardize features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "models/saved_models/scaler_kidney.joblib")
    print("Scaler saved.")
    return X_train_scaled, X_test_scaled

if __name__ == "__main__":
    df = load_data()
    df = preprocess(df)
    X_train, X_test, y_train, y_test, feature_names = split_and_smote(df)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    print("\n✅ Kidney disease preprocessing complete.")
