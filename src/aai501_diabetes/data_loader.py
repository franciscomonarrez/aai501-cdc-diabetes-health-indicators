"""
Data loading and preprocessing utilities for CDC Diabetes Health Indicators dataset.

This module handles:
- Loading the raw dataset
- Basic data validation
- Train/test splitting
- Preprocessing pipeline creation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.aai501_diabetes.config import DATA_RAW, TARGET_COL, RANDOM_STATE


def load_data() -> pd.DataFrame:
    """
    Load the CDC Diabetes Health Indicators dataset.

    Returns:
        pd.DataFrame: Raw dataset with all features and target

    Raises:
        FileNotFoundError: If dataset file doesn't exist
    """
    if not DATA_RAW.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_RAW}. "
            "Please ensure the CSV file is in data/raw/"
        )

    df = pd.read_csv(DATA_RAW)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate dataset structure and check for target column.

    Args:
        df: Input dataframe

    Raises:
        ValueError: If target column is missing or data is invalid
    """
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Check for missing target values
    missing_target = df[TARGET_COL].isna().sum()
    if missing_target > 0:
        print(f"Warning: {missing_target} missing values in target column")

    # Check class distribution
    class_counts = df[TARGET_COL].value_counts()
    print(f"\nClass distribution:")
    print(f"  No diabetes (0): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/len(df)*100:.2f}%)")
    print(f"  Diabetes (1): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/len(df)*100:.2f}%)")
    print(f"  Imbalance ratio: {class_counts.get(0, 1) / class_counts.get(1, 1):.2f}:1")


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, stratify: bool = True
) -> tuple:
    """
    Split data into train and test sets.

    Args:
        df: Full dataset
        test_size: Proportion of data for testing
        stratify: Whether to stratify split by target

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    stratify_param = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=stratify_param,
    )

    print(f"\nData split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create preprocessing pipeline for numerical features.

    Note: This dataset appears to be all numerical (binary/categorical encoded),
    so we use a simple imputation + scaling pipeline.

    Args:
        X: Training features to fit preprocessor

    Returns:
        ColumnTransformer: Fitted preprocessing pipeline
    """
    # Identify numerical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Create preprocessing pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_cols)],
        remainder="passthrough",
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """
    Get feature names after preprocessing.

    Args:
        preprocessor: Fitted ColumnTransformer

    Returns:
        list: Feature names
    """
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name != "remainder":
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                feature_names.extend(columns)
    return feature_names

