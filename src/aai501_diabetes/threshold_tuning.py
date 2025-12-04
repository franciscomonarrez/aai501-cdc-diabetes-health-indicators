"""
Threshold tuning utilities for optimizing classification thresholds.

This module provides functions for:
- Testing multiple classification thresholds
- Computing metrics at each threshold
- Finding optimal threshold for healthcare (maximizing recall/F2)
- Visualizing threshold vs metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from pathlib import Path

from src.aai501_diabetes.config import FIG_DIR, MODELS_DIR


def f2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F2-score (emphasizes recall over precision).

    F2 = (1 + 2^2) * (precision * recall) / (2^2 * precision + recall)
    F2 = 5 * (precision * recall) / (4 * precision + recall)

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        float: F2-score
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    if precision + recall == 0:
        return 0.0

    f2 = 5 * (precision * recall) / (4 * precision + recall)
    return f2


def evaluate_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """
    Evaluate model performance at a specific threshold.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        dict: Dictionary of metrics at this threshold
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "f2_score": f2_score(y_true, y_pred),
    }

    # Calculate confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics["true_positives"] = cm[1, 1]
        metrics["false_positives"] = cm[0, 1]
        metrics["true_negatives"] = cm[0, 0]
        metrics["false_negatives"] = cm[1, 0]
    else:
        metrics["true_positives"] = 0
        metrics["false_positives"] = 0
        metrics["true_negatives"] = 0
        metrics["false_negatives"] = 0

    return metrics


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold_range: tuple = (0.1, 0.9),
    step: float = 0.01,
) -> pd.DataFrame:
    """
    Test multiple thresholds and compute metrics for each.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        threshold_range: (min, max) threshold range
        step: Step size for threshold testing

    Returns:
        pd.DataFrame: DataFrame with metrics for each threshold
    """
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    results = []

    print(f"Testing {len(thresholds)} thresholds from {threshold_range[0]} to {threshold_range[1]}...")

    for threshold in thresholds:
        metrics = evaluate_threshold(y_true, y_proba, threshold)
        results.append(metrics)

    df = pd.DataFrame(results)
    return df


def find_optimal_threshold(
    results_df: pd.DataFrame,
    strategy: str = "f2_recall",
) -> dict:
    """
    Find optimal threshold based on strategy.

    Strategies:
    - "f2_recall": Maximize F2-score, then recall (healthcare priority)
    - "f2": Maximize F2-score only
    - "recall": Maximize recall (with minimum precision constraint)
    - "balanced": Maximize F1-score

    Args:
        results_df: DataFrame from tune_threshold()
        strategy: Strategy for finding optimal threshold

    Returns:
        dict: Optimal threshold and its metrics
    """
    if strategy == "f2_recall":
        # Prioritize F2, then recall for healthcare
        # Filter to thresholds with recall >= 0.70
        filtered = results_df[results_df["recall"] >= 0.70].copy()
        if len(filtered) > 0:
            optimal_idx = filtered["f2_score"].idxmax()
        else:
            # If no threshold meets recall >= 0.70, just maximize F2
            optimal_idx = results_df["f2_score"].idxmax()
    elif strategy == "f2":
        optimal_idx = results_df["f2_score"].idxmax()
    elif strategy == "recall":
        # Maximize recall with minimum precision >= 0.25
        filtered = results_df[results_df["precision"] >= 0.25].copy()
        if len(filtered) > 0:
            optimal_idx = filtered["recall"].idxmax()
        else:
            optimal_idx = results_df["recall"].idxmax()
    else:  # balanced
        optimal_idx = results_df["f1_score"].idxmax()

    optimal = results_df.loc[optimal_idx].to_dict()
    return optimal


def plot_threshold_metrics(
    results_df: pd.DataFrame,
    optimal_threshold: float = None,
    save_path: Path = None,
) -> None:
    """
    Plot metrics vs threshold.

    Args:
        results_df: DataFrame from tune_threshold()
        optimal_threshold: Optimal threshold to highlight
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    thresholds = results_df["threshold"].values

    # Plot 1: Precision, Recall, F1, F2
    ax1 = axes[0, 0]
    ax1.plot(thresholds, results_df["precision"], label="Precision", linewidth=2, color="#e74c3c")
    ax1.plot(thresholds, results_df["recall"], label="Recall", linewidth=2, color="#3498db")
    ax1.plot(thresholds, results_df["f1_score"], label="F1-Score", linewidth=2, color="#2ecc71")
    ax1.plot(thresholds, results_df["f2_score"], label="F2-Score", linewidth=2, color="#f39c12")
    if optimal_threshold:
        ax1.axvline(x=optimal_threshold, color="black", linestyle="--", linewidth=2, label=f"Optimal ({optimal_threshold:.3f})")
    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Precision, Recall, F1, and F2 vs Threshold", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([thresholds.min(), thresholds.max()])

    # Plot 2: Accuracy
    ax2 = axes[0, 1]
    ax2.plot(thresholds, results_df["accuracy"], linewidth=2, color="#9b59b6")
    if optimal_threshold:
        ax2.axvline(x=optimal_threshold, color="black", linestyle="--", linewidth=2, label=f"Optimal ({optimal_threshold:.3f})")
    ax2.set_xlabel("Threshold", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy vs Threshold", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xlim([thresholds.min(), thresholds.max()])

    # Plot 3: Confusion Matrix Components
    ax3 = axes[1, 0]
    ax3.plot(thresholds, results_df["true_positives"], label="True Positives", linewidth=2, color="#2ecc71")
    ax3.plot(thresholds, results_df["false_positives"], label="False Positives", linewidth=2, color="#e74c3c")
    ax3.plot(thresholds, results_df["false_negatives"], label="False Negatives", linewidth=2, color="#f39c12")
    if optimal_threshold:
        ax3.axvline(x=optimal_threshold, color="black", linestyle="--", linewidth=2, label=f"Optimal ({optimal_threshold:.3f})")
    ax3.set_xlabel("Threshold", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title("Confusion Matrix Components vs Threshold", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_xlim([thresholds.min(), thresholds.max()])

    # Plot 4: Healthcare Focus - Recall and F2 (zoomed)
    ax4 = axes[1, 1]
    ax4.plot(thresholds, results_df["recall"], label="Recall", linewidth=3, color="#3498db")
    ax4.plot(thresholds, results_df["f2_score"], label="F2-Score", linewidth=3, color="#f39c12")
    if optimal_threshold:
        ax4.axvline(x=optimal_threshold, color="black", linestyle="--", linewidth=2, label=f"Optimal ({optimal_threshold:.3f})")
        # Add annotation
        opt_row = results_df[results_df["threshold"] == optimal_threshold].iloc[0]
        ax4.annotate(
            f"Optimal\nRecall: {opt_row['recall']:.3f}\nF2: {opt_row['f2_score']:.3f}",
            xy=(optimal_threshold, opt_row["f2_score"]),
            xytext=(optimal_threshold + 0.1, opt_row["f2_score"] + 0.05),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )
    ax4.set_xlabel("Threshold", fontsize=12)
    ax4.set_ylabel("Score", fontsize=12)
    ax4.set_title("Healthcare Priority: Recall & F2-Score", fontsize=14, fontweight="bold")
    ax4.legend(fontsize=11, loc="best")
    ax4.grid(alpha=0.3)
    ax4.set_xlim([thresholds.min(), thresholds.max()])

    plt.suptitle("Threshold Tuning Analysis - MLP Diabetes Prediction", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.savefig(FIG_DIR / "francisco_threshold_tuning.png", dpi=300, bbox_inches="tight")

    plt.close()


def generate_threshold_summary(
    results_df: pd.DataFrame,
    optimal: dict,
    default_threshold: float = 0.5,
    save_path: Path = None,
) -> None:
    """
    Generate text summary of threshold tuning results.

    Args:
        results_df: DataFrame from tune_threshold()
        optimal: Optimal threshold dict from find_optimal_threshold()
        default_threshold: Default threshold (0.5) for comparison
        save_path: Path to save summary
    """
    default_row = results_df[results_df["threshold"] == default_threshold]
    if len(default_row) == 0:
        # Find closest threshold to 0.5
        default_row = results_df.iloc[(results_df["threshold"] - default_threshold).abs().argsort()[:1]]

    default_metrics = default_row.iloc[0].to_dict()

    summary_lines = [
        "=" * 80,
        "THRESHOLD TUNING SUMMARY - MLP DIABETES PREDICTION",
        "=" * 80,
        "",
        "OBJECTIVE:",
        "  Optimize classification threshold for healthcare screening",
        "  Priority: Maximize Recall (catch diabetes cases) and F2-Score",
        "",
        "=" * 80,
        "DEFAULT THRESHOLD (0.5) PERFORMANCE",
        "=" * 80,
        f"Threshold:        {default_metrics['threshold']:.4f}",
        f"Accuracy:         {default_metrics['accuracy']:.4f} ({default_metrics['accuracy']*100:.2f}%)",
        f"Precision:        {default_metrics['precision']:.4f} ({default_metrics['precision']*100:.2f}%)",
        f"Recall:           {default_metrics['recall']:.4f} ({default_metrics['recall']*100:.2f}%)",
        f"F1-Score:         {default_metrics['f1_score']:.4f}",
        f"F2-Score:         {default_metrics['f2_score']:.4f}",
        f"True Positives:   {default_metrics['true_positives']:,}",
        f"False Positives:  {default_metrics['false_positives']:,}",
        f"False Negatives:  {default_metrics['false_negatives']:,}",
        "",
        "=" * 80,
        "OPTIMAL THRESHOLD PERFORMANCE",
        "=" * 80,
        f"Threshold:        {optimal['threshold']:.4f}",
        f"Accuracy:         {optimal['accuracy']:.4f} ({optimal['accuracy']*100:.2f}%)",
        f"Precision:        {optimal['precision']:.4f} ({optimal['precision']*100:.2f}%)",
        f"Recall:           {optimal['recall']:.4f} ({optimal['recall']*100:.2f}%)",
        f"F1-Score:         {optimal['f1_score']:.4f}",
        f"F2-Score:         {optimal['f2_score']:.4f}",
        f"True Positives:   {optimal['true_positives']:,}",
        f"False Positives:  {optimal['false_positives']:,}",
        f"False Negatives:  {optimal['false_negatives']:,}",
        "",
        "=" * 80,
        "IMPROVEMENT ANALYSIS",
        "=" * 80,
    ]

    # Calculate improvements
    recall_improvement = optimal["recall"] - default_metrics["recall"]
    f2_improvement = optimal["f2_score"] - default_metrics["f2_score"]
    precision_change = optimal["precision"] - default_metrics["precision"]
    fn_reduction = default_metrics["false_negatives"] - optimal["false_negatives"]

    summary_lines.extend([
        f"Recall Improvement:     {recall_improvement:+.4f} ({recall_improvement*100:+.2f}%)",
        f"F2-Score Improvement:  {f2_improvement:+.4f} ({f2_improvement*100:+.2f}%)",
        f"Precision Change:       {precision_change:+.4f} ({precision_change*100:+.2f}%)",
        f"False Negatives Reduced: {fn_reduction:+.0f} ({(fn_reduction/default_metrics['false_negatives']*100):+.1f}%)",
        "",
        "=" * 80,
        "HEALTHCARE INTERPRETATION",
        "=" * 80,
        "",
        f"With optimal threshold ({optimal['threshold']:.3f}):",
        f"  - Recall: {optimal['recall']*100:.2f}% of diabetes cases are correctly identified",
        f"  - This means {optimal['false_negatives']:,} diabetes cases are missed (vs {default_metrics['false_negatives']:,} at default)",
        f"  - False Negatives reduced by {fn_reduction:+.0f} cases",
        "",
        "RECOMMENDATION:",
    ])

    if recall_improvement > 0.05:
        summary_lines.append("  ✓ Significant improvement in recall - optimal threshold recommended")
    elif recall_improvement > 0:
        summary_lines.append("  ✓ Moderate improvement in recall - optimal threshold recommended")
    else:
        summary_lines.append("  ⚠ No significant recall improvement - consider other strategies")

    if optimal["recall"] >= 0.80:
        summary_lines.append("  ✓ Excellent recall (≥80%) - suitable for screening")
    elif optimal["recall"] >= 0.75:
        summary_lines.append("  ✓ Good recall (≥75%) - suitable for screening")
    else:
        summary_lines.append("  ⚠ Recall below 75% - may miss too many cases")

    summary_lines.extend([
        "",
        "=" * 80,
    ])

    summary_text = "\n".join(summary_lines)

    if save_path:
        with open(save_path, "w") as f:
            f.write(summary_text)
        print(f"Summary saved to: {save_path}")
    else:
        print(summary_text)

    return summary_text

