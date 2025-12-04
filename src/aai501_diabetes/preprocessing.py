"""
Class imbalance handling utilities.

This module provides functions for handling class imbalance using:
- SMOTE (Synthetic Minority Oversampling Technique)
- Class weighting
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

from src.aai501_diabetes.config import RANDOM_STATE


def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Apply SMOTE to balance training data.

    Args:
        X_train: Training features (already preprocessed/scaled)
        y_train: Training labels

    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("\nApplying SMOTE to balance classes...")
    print(f"  Before: {np.bincount(y_train)}")

    smote = SMOTE(random_state=RANDOM_STATE, n_jobs=-1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"  After: {np.bincount(y_resampled)}")
    print(f"  Resampled shape: {X_resampled.shape}")

    return X_resampled, y_resampled


def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    Compute class weights for imbalanced dataset.

    Args:
        y_train: Training labels

    Returns:
        dict: Class weights in format {0: weight_0, 1: weight_1}
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        "balanced", classes=classes, y=y_train
    )
    class_weights = dict(zip(classes, weights))

    print(f"\nComputed class weights:")
    print(f"  Class 0 (no diabetes): {class_weights[0]:.4f}")
    print(f"  Class 1 (diabetes): {class_weights[1]:.4f}")

    return class_weights


def get_imbalance_ratio(y: np.ndarray) -> float:
    """
    Calculate class imbalance ratio.

    Args:
        y: Labels

    Returns:
        float: Ratio of majority to minority class
    """
    counts = np.bincount(y)
    return counts[0] / counts[1] if counts[1] > 0 else float("inf")

