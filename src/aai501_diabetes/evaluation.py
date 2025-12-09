"""
Model evaluation utilities for diabetes prediction.

This module provides functions for:
- Computing comprehensive metrics
- Generating ROC curves
- Creating confusion matrices
- Plotting training curves
- Interpreting results in healthcare context
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from pathlib import Path

from src.aai501_diabetes.config import FIG_DIR


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    threshold: float = 0.5,
) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels (or None to compute from y_proba)
        y_proba: Predicted probabilities (optional)
        threshold: Classification threshold (used if y_pred is None)

    Returns:
        dict: Dictionary of metrics
    """
    # If y_pred not provided, compute from y_proba and threshold
    if y_pred is None:
        if y_proba is None:
            raise ValueError("Either y_pred or y_proba must be provided")
        y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def print_metrics(metrics: dict, model_name: str = "Model") -> None:
    """
    Print formatted metrics.

    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print("\n" + "=" * 80)
    print(f"{model_name.upper()} - EVALUATION METRICS")
    print("=" * 80)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 80)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Path = None,
) -> None:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        save_path: Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve - {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.savefig(FIG_DIR / f"{model_name.lower().replace(' ', '_')}_roc_curve.png", dpi=300, bbox_inches="tight")

    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    save_path: Path = None,
) -> None:
    """
    Plot Precision-Recall curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        save_path: Path to save figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = np.trapz(precision, recall)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"{model_name} (AP = {avg_precision:.4f})")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve - {model_name}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.savefig(FIG_DIR / f"{model_name.lower().replace(' ', '_')}_pr_curve.png", dpi=300, bbox_inches="tight")

    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Path = None,
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Count confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        cbar_kws={"label": "Count"},
    )
    axes[0].set_title(f"Confusion Matrix (Count) - {model_name}", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_xticklabels(["No Diabetes", "Diabetes"])
    axes[0].set_yticklabels(["No Diabetes", "Diabetes"])

    # Normalized confusion matrix
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        ax=axes[1],
        cbar_kws={"label": "Proportion"},
    )
    axes[1].set_title(f"Confusion Matrix (Normalized) - {model_name}", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_xticklabels(["No Diabetes", "Diabetes"])
    axes[1].set_yticklabels(["No Diabetes", "Diabetes"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.savefig(FIG_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=300, bbox_inches="tight")

    plt.close()


def plot_training_curves(history, model_name: str = "Model", save_path: Path = None) -> None:
    """
    Plot training and validation curves.

    Args:
        history: Keras training history or dict with metrics
        model_name: Name of the model
        save_path: Path to save figure
    """
    if hasattr(history, "history"):
        history_dict = history.history
    else:
        history_dict = history

    metrics = ["loss", "accuracy", "precision", "recall", "auc"]
    n_metrics = len([m for m in metrics if m in history_dict])

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    idx = 0
    for metric in metrics:
        if metric in history_dict:
            ax = axes[idx]
            ax.plot(history_dict[metric], label=f"Train {metric}", linewidth=2)
            if f"val_{metric}" in history_dict:
                ax.plot(history_dict[f"val_{metric}"], label=f"Val {metric}", linewidth=2)
            ax.set_title(f"{metric.upper()}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            idx += 1

    plt.suptitle(f"Training Curves - {model_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.savefig(FIG_DIR / f"{model_name.lower().replace(' ', '_')}_training_curves.png", dpi=300, bbox_inches="tight")

    plt.close()


def interpret_results(metrics: dict, model_name: str = "Model") -> None:
    """
    Interpret results in healthcare context.

    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print("\n" + "=" * 80)
    print(f"{model_name.upper()} - HEALTHCARE INTERPRETATION")
    print("=" * 80)

    recall = metrics.get("recall", 0)
    precision = metrics.get("precision", 0)

    print(f"\nRecall ({recall:.2%}):")
    print(
        "  - Percentage of actual diabetes cases correctly identified."
    )
    print(
        "  - HIGH RECALL is critical: Missing a diabetes case (False Negative)"
    )
    print("    can lead to delayed treatment and complications.")

    print(f"\nPrecision ({precision:.2%}):")
    print(
        "  - Percentage of predicted diabetes cases that are actually diabetic."
    )
    print(
        "  - Lower precision means more False Positives, which may lead to"
    )
    print("    unnecessary follow-up tests, but is generally acceptable.")

    print("\nTrade-off Analysis:")
    if recall >= 0.75:
        print("  ✓ Good recall - Model catches most diabetes cases")
    elif recall >= 0.65:
        print("  ⚠ Moderate recall - Model may miss some diabetes cases")
    else:
        print("  ⚠ Low recall - Model may miss many diabetes cases (CONCERN)")

    if precision > 0.7:
        print("  ✓ Good precision - Most predictions are correct")
    elif precision > 0.5:
        print("  ⚠ Moderate precision - Some false alarms, acceptable for screening")
    else:
        print("  ⚠ Lower precision - More false alarms, but acceptable for screening")

    print("\nRecommendation:")
    if recall >= 0.75 and precision >= 0.60:
        print("  ✓ Model is suitable for initial screening with follow-up confirmation.")
    elif recall >= 0.75:
        print("  ✓ Model prioritizes catching cases (high recall) - good for screening.")
    elif recall >= 0.65:
        print("  ⚠ Model has moderate performance - consider threshold tuning or retraining.")
    else:
        print("  ⚠ Model needs improvement - too many missed cases (low recall).")

    print("=" * 80)

