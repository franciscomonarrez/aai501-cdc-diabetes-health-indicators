"""
Threshold tuning script for MLP diabetes prediction model.

This script:
1. Loads the trained MLP model and test predictions
2. Tests thresholds from 0.1 to 0.9
3. Computes metrics for each threshold
4. Finds optimal threshold for healthcare (maximizing recall/F2)
5. Generates visualizations and summary

Author: Francisco (Team Member 3)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.aai501_diabetes.config import MODELS_DIR, FIG_DIR
from src.aai501_diabetes.threshold_tuning import (
    tune_threshold,
    find_optimal_threshold,
    plot_threshold_metrics,
    generate_threshold_summary,
)

# Configuration
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 0.9
THRESHOLD_STEP = 0.01
DEFAULT_THRESHOLD = 0.5


def load_predictions() -> tuple:
    """
    Load test set predictions from saved CSV.

    Returns:
        tuple: (y_true, y_proba)
    """
    predictions_file = MODELS_DIR / "francisco_mlp_predictions.csv"

    if not predictions_file.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_file}\n"
            "Please run francisco_train_mlp.py first to generate predictions."
        )

    df = pd.read_csv(predictions_file)
    y_true = df["y_true"].values
    y_proba = df["y_proba"].values

    print(f"Loaded predictions: {len(y_true)} samples")
    return y_true, y_proba


def main():
    """Main threshold tuning pipeline."""
    print("=" * 80)
    print("THRESHOLD TUNING - MLP DIABETES PREDICTION")
    print("Author: Francisco - Team Member 3")
    print("=" * 80)

    # Step 1: Load predictions
    print("\n[STEP 1] Loading model predictions...")
    y_true, y_proba = load_predictions()

    # Step 2: Tune thresholds
    print(f"\n[STEP 2] Testing thresholds from {THRESHOLD_MIN} to {THRESHOLD_MAX}...")
    results_df = tune_threshold(
        y_true,
        y_proba,
        threshold_range=(THRESHOLD_MIN, THRESHOLD_MAX),
        step=THRESHOLD_STEP,
    )

    # Step 3: Find optimal threshold
    print("\n[STEP 3] Finding optimal threshold for healthcare...")
    optimal = find_optimal_threshold(results_df, strategy="f2_recall")

    print(f"\nOptimal Threshold: {optimal['threshold']:.4f}")
    print(f"  Recall:  {optimal['recall']:.4f} ({optimal['recall']*100:.2f}%)")
    print(f"  F2-Score: {optimal['f2_score']:.4f}")
    print(f"  Precision: {optimal['precision']:.4f} ({optimal['precision']*100:.2f}%)")

    # Step 4: Save results CSV
    print("\n[STEP 4] Saving threshold results...")
    results_file = MODELS_DIR / "francisco_threshold_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved: {results_file}")

    # Step 5: Generate visualizations
    print("\n[STEP 5] Generating threshold tuning plots...")
    plot_threshold_metrics(
        results_df,
        optimal_threshold=optimal["threshold"],
        save_path=FIG_DIR / "francisco_threshold_tuning.png",
    )

    # Step 6: Generate summary
    print("\n[STEP 6] Generating summary report...")
    summary_file = MODELS_DIR / "francisco_threshold_summary.txt"
    generate_threshold_summary(
        results_df,
        optimal,
        default_threshold=DEFAULT_THRESHOLD,
        save_path=summary_file,
    )

    # Step 7: Compare with default threshold
    print("\n" + "=" * 80)
    print("COMPARISON: DEFAULT vs OPTIMAL THRESHOLD")
    print("=" * 80)

    default_row = results_df[results_df["threshold"] == DEFAULT_THRESHOLD]
    if len(default_row) == 0:
        default_row = results_df.iloc[(results_df["threshold"] - DEFAULT_THRESHOLD).abs().argsort()[:1]]
    default_metrics = default_row.iloc[0]

    print(f"\nDefault Threshold ({DEFAULT_THRESHOLD}):")
    print(f"  Recall:  {default_metrics['recall']:.4f} ({default_metrics['recall']*100:.2f}%)")
    print(f"  F2-Score: {default_metrics['f2_score']:.4f}")
    print(f"  Precision: {default_metrics['precision']:.4f} ({default_metrics['precision']*100:.2f}%)")
    print(f"  False Negatives: {default_metrics['false_negatives']:,}")

    print(f"\nOptimal Threshold ({optimal['threshold']:.4f}):")
    print(f"  Recall:  {optimal['recall']:.4f} ({optimal['recall']*100:.2f}%)")
    print(f"  F2-Score: {optimal['f2_score']:.4f}")
    print(f"  Precision: {optimal['precision']:.4f} ({optimal['precision']*100:.2f}%)")
    print(f"  False Negatives: {optimal['false_negatives']:,}")

    recall_improvement = optimal["recall"] - default_metrics["recall"]
    fn_reduction = default_metrics["false_negatives"] - optimal["false_negatives"]

    print(f"\nImprovement:")
    print(f"  Recall: +{recall_improvement:.4f} ({recall_improvement*100:+.2f}%)")
    print(f"  False Negatives Reduced: {fn_reduction:+.0f} cases")

    # Step 8: Save optimal threshold for use in training script
    optimal_threshold_file = MODELS_DIR / "francisco_optimal_threshold.txt"
    with open(optimal_threshold_file, "w") as f:
        f.write(f"{optimal['threshold']:.6f}")
    print(f"\nOptimal threshold saved to: {optimal_threshold_file}")
    print("  (Will be used automatically in future training runs)")

    print("\n" + "=" * 80)
    print("THRESHOLD TUNING COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved:")
    print(f"  - Results CSV: {results_file}")
    print(f"  - Visualization: {FIG_DIR / 'francisco_threshold_tuning.png'}")
    print(f"  - Summary: {summary_file}")
    print(f"  - Optimal Threshold: {optimal_threshold_file}")


if __name__ == "__main__":
    main()

