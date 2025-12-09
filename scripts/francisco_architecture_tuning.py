"""
Architecture tuning script for MLP diabetes prediction model.

This script:
1. Tests multiple MLP architectures
2. Trains each with identical configuration for fair comparison
3. Evaluates at both default (0.5) and optimal (0.44) thresholds
4. Generates comparison plots and summary

Author: Francisco (Team Member 3)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.aai501_diabetes.config import (
    DATA_RAW,
    TARGET_COL,
    RANDOM_STATE,
    FIG_DIR,
    MODELS_DIR,
)
from src.aai501_diabetes.data_loader import (
    load_data,
    validate_data,
    split_data,
    create_preprocessor,
)
from src.aai501_diabetes.preprocessing import (
    apply_smote,
    get_imbalance_ratio,
)
from src.aai501_diabetes.mlp_model import train_mlp, build_mlp, compile_model
from src.aai501_diabetes.threshold_tuning import evaluate_threshold

# Architecture configurations to test
ARCHITECTURES = {
    "A1_Simple": {
        "hidden_layers": [64, 32],
        "dropout_rate": 0.3,
        "description": "Simple 2-layer network",
    },
    "A2_Baseline": {
        "hidden_layers": [128, 64, 32],
        "dropout_rate": 0.3,
        "description": "Current baseline (3-layer)",
    },
    "A3_Deeper": {
        "hidden_layers": [256, 128, 64, 32],
        "dropout_rate": 0.3,
        "description": "Deeper 4-layer network",
    },
    "A4_Wider": {
        "hidden_layers": [256, 256, 128],
        "dropout_rate": 0.3,
        "description": "Wider network with more neurons",
    },
    "A5_HighDropout": {
        "hidden_layers": [128, 64],
        "dropout_rate": 0.5,
        "description": "Higher dropout for regularization",
    },
}

# Training configuration (same for all architectures)
TRAIN_CONFIG = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 256,
    "optimizer": "adam",
}

# Thresholds to evaluate
DEFAULT_THRESHOLD = 0.5
OPTIMAL_THRESHOLD = 0.44  # From threshold tuning


def load_optimal_threshold() -> float:
    """Load optimal threshold from file if available."""
    optimal_file = MODELS_DIR / "francisco_optimal_threshold.txt"
    if optimal_file.exists():
        with open(optimal_file, "r") as f:
            return float(f.read().strip())
    return DEFAULT_THRESHOLD


def evaluate_architecture(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    thresholds: list = [0.5, 0.44],
) -> dict:
    """
    Evaluate model at multiple thresholds.

    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        thresholds: List of thresholds to evaluate

    Returns:
        dict: Metrics at each threshold
    """
    y_proba = model.predict(X_test, verbose=0).flatten()
    results = {}

    for threshold in thresholds:
        metrics = evaluate_threshold(y_test, y_proba, threshold)
        results[f"threshold_{threshold}"] = metrics

    # Add ROC-AUC (threshold-independent)
    from sklearn.metrics import roc_auc_score
    results["roc_auc"] = roc_auc_score(y_test, y_proba)

    return results


def train_and_evaluate_architecture(
    arch_name: str,
    arch_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_weight: dict = None,
) -> dict:
    """
    Train and evaluate a single architecture.

    Args:
        arch_name: Architecture name
        arch_config: Architecture configuration
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        class_weight: Class weights (if not using SMOTE)

    Returns:
        dict: Complete results for this architecture
    """
    print(f"\n{'='*80}")
    print(f"Training Architecture: {arch_name}")
    print(f"Description: {arch_config['description']}")
    print(f"{'='*80}")

    # Build model
    model = build_mlp(
        input_dim=X_train.shape[1],
        hidden_layers=arch_config["hidden_layers"],
        dropout_rate=arch_config["dropout_rate"],
    )

    # Compile model
    model = compile_model(
        model,
        learning_rate=TRAIN_CONFIG["learning_rate"],
        optimizer=TRAIN_CONFIG["optimizer"],
    )

    # Train model
    from src.aai501_diabetes.mlp_model import create_callbacks
    callbacks_list = create_callbacks(f"mlp_{arch_name}")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=TRAIN_CONFIG["epochs"],
        batch_size=TRAIN_CONFIG["batch_size"],
        callbacks=callbacks_list,
        class_weight=class_weight,
        verbose=0,  # Suppress output for cleaner comparison
    )

    # Evaluate at both thresholds
    optimal_threshold = load_optimal_threshold()
    thresholds = [DEFAULT_THRESHOLD, optimal_threshold]

    eval_results = evaluate_architecture(model, X_test, y_test, thresholds)

    # Compile results
    results = {
        "architecture": arch_name,
        "description": arch_config["description"],
        "hidden_layers": str(arch_config["hidden_layers"]),
        "dropout_rate": arch_config["dropout_rate"],
        "num_parameters": model.count_params(),
        "final_epoch": len(history.history["loss"]),
    }

    # Add metrics at default threshold
    default_metrics = eval_results[f"threshold_{DEFAULT_THRESHOLD}"]
    results.update({
        f"default_threshold": DEFAULT_THRESHOLD,
        f"default_accuracy": default_metrics["accuracy"],
        f"default_precision": default_metrics["precision"],
        f"default_recall": default_metrics["recall"],
        f"default_f1_score": default_metrics["f1_score"],
        f"default_f2_score": default_metrics["f2_score"],
        f"default_false_negatives": default_metrics["false_negatives"],
    })

    # Add metrics at optimal threshold
    optimal_metrics = eval_results[f"threshold_{optimal_threshold}"]
    results.update({
        f"optimal_threshold": optimal_threshold,
        f"optimal_accuracy": optimal_metrics["accuracy"],
        f"optimal_precision": optimal_metrics["precision"],
        f"optimal_recall": optimal_metrics["recall"],
        f"optimal_f1_score": optimal_metrics["f1_score"],
        f"optimal_f2_score": optimal_metrics["f2_score"],
        f"optimal_false_negatives": optimal_metrics["false_negatives"],
    })

    # Add ROC-AUC
    results["roc_auc"] = eval_results["roc_auc"]

    print(f"  Default Threshold ({DEFAULT_THRESHOLD}):")
    print(f"    Recall: {default_metrics['recall']:.4f} ({default_metrics['recall']*100:.2f}%)")
    print(f"    F2-Score: {default_metrics['f2_score']:.4f}")
    print(f"    ROC-AUC: {eval_results['roc_auc']:.4f}")
    print(f"    False Negatives: {default_metrics['false_negatives']:.0f}")

    print(f"  Optimal Threshold ({optimal_threshold}):")
    print(f"    Recall: {optimal_metrics['recall']:.4f} ({optimal_metrics['recall']*100:.2f}%)")
    print(f"    F2-Score: {optimal_metrics['f2_score']:.4f}")
    print(f"    False Negatives: {optimal_metrics['false_negatives']:.0f}")

    return results


def main():
    """Main architecture tuning pipeline."""
    print("=" * 80)
    print("ARCHITECTURE TUNING - MLP DIABETES PREDICTION")
    print("Author: Francisco - Team Member 3")
    print("=" * 80)

    # Step 1: Load and preprocess data
    print("\n[STEP 1] Loading and preprocessing data...")
    df = load_data()
    validate_data(df)

    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, stratify=True)

    # Create and fit preprocessor
    preprocessor = create_preprocessor(X_train)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    X_train_scaled = np.array(X_train_scaled)
    X_test_scaled = np.array(X_test_scaled)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Create validation set
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    print(f"  Training: {X_train_final.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test_scaled.shape[0]} samples")

    # Step 2: Handle class imbalance with SMOTE
    print("\n[STEP 2] Applying SMOTE to balance classes...")
    X_train_balanced, y_train_balanced = apply_smote(X_train_final, y_train_final)
    print(f"  Balanced training: {X_train_balanced.shape[0]} samples")

    # Step 3: Train and evaluate each architecture
    print("\n[STEP 3] Training and evaluating architectures...")
    all_results = []

    for arch_name, arch_config in ARCHITECTURES.items():
        results = train_and_evaluate_architecture(
            arch_name,
            arch_config,
            X_train_balanced,
            y_train_balanced,
            X_val,
            y_val,
            X_test_scaled,
            y_test,
        )
        all_results.append(results)

    # Step 4: Save results to CSV
    print("\n[STEP 4] Saving results...")
    results_df = pd.DataFrame(all_results)
    results_file = MODELS_DIR / "francisco_architecture_comparison.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved: {results_file}")

    # Step 5: Generate comparison plot
    print("\n[STEP 5] Generating comparison plots...")
    plot_architecture_comparison(results_df)

    # Step 6: Generate summary
    print("\n[STEP 6] Generating summary report...")
    generate_architecture_summary(results_df)

    print("\n" + "=" * 80)
    print("ARCHITECTURE TUNING COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved:")
    print(f"  - Results CSV: {results_file}")
    print(f"  - Visualization: {FIG_DIR / 'francisco_architecture_tuning.png'}")
    print(f"  - Summary: FRANCISCO_ARCHITECTURE_TUNING_SUMMARY.md")


def plot_architecture_comparison(results_df: pd.DataFrame) -> None:
    """Generate comparison plots for architectures."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    architectures = results_df["architecture"].values
    x_pos = np.arange(len(architectures))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Recall comparison (both thresholds)
    ax1 = axes[0, 0]
    width = 0.35
    ax1.bar(
        x_pos - width / 2,
        results_df["default_recall"].values,
        width,
        label=f"Default Threshold ({DEFAULT_THRESHOLD})",
        color="#3498db",
        alpha=0.7,
    )
    ax1.bar(
        x_pos + width / 2,
        results_df["optimal_recall"].values,
        width,
        label=f"Optimal Threshold ({load_optimal_threshold():.2f})",
        color="#e74c3c",
        alpha=0.7,
    )
    ax1.set_xlabel("Architecture", fontsize=12)
    ax1.set_ylabel("Recall", fontsize=12)
    ax1.set_title("Recall Comparison Across Architectures", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(architectures, rotation=45, ha="right")
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(y=0.80, color="green", linestyle="--", linewidth=1, label="Target (80%)")
    ax1.legend(fontsize=10)

    # Plot 2: F2-Score comparison
    ax2 = axes[0, 1]
    ax2.bar(
        x_pos - width / 2,
        results_df["default_f2_score"].values,
        width,
        label=f"Default Threshold ({DEFAULT_THRESHOLD})",
        color="#3498db",
        alpha=0.7,
    )
    ax2.bar(
        x_pos + width / 2,
        results_df["optimal_f2_score"].values,
        width,
        label=f"Optimal Threshold ({load_optimal_threshold():.2f})",
        color="#e74c3c",
        alpha=0.7,
    )
    ax2.set_xlabel("Architecture", fontsize=12)
    ax2.set_ylabel("F2-Score", fontsize=12)
    ax2.set_title("F2-Score Comparison Across Architectures", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(architectures, rotation=45, ha="right")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # Plot 3: ROC-AUC comparison
    ax3 = axes[1, 0]
    bars = ax3.bar(x_pos, results_df["roc_auc"].values, color="#2ecc71", alpha=0.7)
    ax3.set_xlabel("Architecture", fontsize=12)
    ax3.set_ylabel("ROC-AUC", fontsize=12)
    ax3.set_title("ROC-AUC Comparison Across Architectures", fontsize=14, fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(architectures, rotation=45, ha="right")
    ax3.grid(axis="y", alpha=0.3)
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, results_df["roc_auc"].values)):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 4: False Negatives comparison (healthcare focus)
    ax4 = axes[1, 1]
    ax4.bar(
        x_pos - width / 2,
        results_df["default_false_negatives"].values,
        width,
        label=f"Default Threshold ({DEFAULT_THRESHOLD})",
        color="#f39c12",
        alpha=0.7,
    )
    ax4.bar(
        x_pos + width / 2,
        results_df["optimal_false_negatives"].values,
        width,
        label=f"Optimal Threshold ({load_optimal_threshold():.2f})",
        color="#e74c3c",
        alpha=0.7,
    )
    ax4.set_xlabel("Architecture", fontsize=12)
    ax4.set_ylabel("False Negatives", fontsize=12)
    ax4.set_title("False Negatives Comparison (Healthcare Priority)", fontsize=14, fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(architectures, rotation=45, ha="right")
    ax4.legend(fontsize=10)
    ax4.grid(axis="y", alpha=0.3)
    # Lower is better for false negatives
    ax4.invert_yaxis()

    plt.suptitle(
        "MLP Architecture Comparison - Diabetes Prediction",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "francisco_architecture_tuning.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {FIG_DIR / 'francisco_architecture_tuning.png'}")
    plt.close()


def generate_architecture_summary(results_df: pd.DataFrame) -> None:
    """Generate markdown summary of architecture tuning results."""
    optimal_threshold = load_optimal_threshold()

    # Find best architecture (prioritize recall and F2 at optimal threshold)
    results_df["combined_score"] = (
        results_df["optimal_recall"] * 0.5
        + results_df["optimal_f2_score"] * 0.3
        + results_df["roc_auc"] * 0.2
    )
    best_idx = results_df["combined_score"].idxmax()
    best_arch = results_df.loc[best_idx]

    baseline_idx = results_df[results_df["architecture"] == "A2_Baseline"].index[0]
    baseline = results_df.loc[baseline_idx]

    summary_lines = [
        "# Architecture Tuning Summary - MLP Diabetes Prediction",
        "",
        "**Author**: Francisco (Team Member 3)",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"**Best Architecture**: {best_arch['architecture']} ({best_arch['description']})",
        "",
        f"**Key Metrics at Optimal Threshold ({optimal_threshold}):**",
        f"- Recall: {best_arch['optimal_recall']:.4f} ({best_arch['optimal_recall']*100:.2f}%)",
        f"- F2-Score: {best_arch['optimal_f2_score']:.4f}",
        f"- ROC-AUC: {best_arch['roc_auc']:.4f}",
        f"- False Negatives: {best_arch['optimal_false_negatives']:.0f}",
        "",
        "---",
        "",
        "## Architecture Configurations Tested",
        "",
        "| Architecture | Description | Hidden Layers | Dropout | Parameters |",
        "|--------------|------------|---------------|---------|------------|",
    ]

    for _, row in results_df.iterrows():
        summary_lines.append(
            f"| {row['architecture']} | {row['description']} | {row['hidden_layers']} | {row['dropout_rate']} | {row['num_parameters']:,} |"
        )

    summary_lines.extend([
        "",
        "---",
        "",
        "## Performance Comparison",
        "",
        "### At Default Threshold (0.5)",
        "",
        "| Architecture | Recall | F2-Score | ROC-AUC | False Negatives |",
        "|--------------|--------|----------|---------|-----------------|",
    ])

    for _, row in results_df.iterrows():
        summary_lines.append(
            f"| {row['architecture']} | {row['default_recall']:.4f} | {row['default_f2_score']:.4f} | {row['roc_auc']:.4f} | {row['default_false_negatives']:.0f} |"
        )

    summary_lines.extend([
        "",
        "### At Optimal Threshold ({optimal_threshold})",
        "",
        "| Architecture | Recall | F2-Score | ROC-AUC | False Negatives |",
        "|--------------|--------|----------|---------|-----------------|",
    ])

    for _, row in results_df.iterrows():
        summary_lines.append(
            f"| {row['architecture']} | {row['optimal_recall']:.4f} | {row['optimal_f2_score']:.4f} | {row['roc_auc']:.4f} | {row['optimal_false_negatives']:.0f} |"
        )

    # Comparison with baseline
    recall_improvement = best_arch["optimal_recall"] - baseline["optimal_recall"]
    f2_improvement = best_arch["optimal_f2_score"] - baseline["optimal_f2_score"]
    fn_reduction = baseline["optimal_false_negatives"] - best_arch["optimal_false_negatives"]

    summary_lines.extend([
        "",
        "---",
        "",
        "## Comparison with Baseline (A2_Baseline)",
        "",
        f"**Baseline Performance** (A2_Baseline - {baseline['description']}):",
        f"- Recall: {baseline['optimal_recall']:.4f} ({baseline['optimal_recall']*100:.2f}%)",
        f"- F2-Score: {baseline['optimal_f2_score']:.4f}",
        f"- ROC-AUC: {baseline['roc_auc']:.4f}",
        f"- False Negatives: {baseline['optimal_false_negatives']:.0f}",
        "",
        f"**Best Architecture Performance** ({best_arch['architecture']}):",
        f"- Recall: {best_arch['optimal_recall']:.4f} ({best_arch['optimal_recall']*100:.2f}%)",
        f"- F2-Score: {best_arch['optimal_f2_score']:.4f}",
        f"- ROC-AUC: {best_arch['roc_auc']:.4f}",
        f"- False Negatives: {best_arch['optimal_false_negatives']:.0f}",
        "",
        "**Improvements:**",
        f"- Recall: {recall_improvement:+.4f} ({recall_improvement*100:+.2f} percentage points)",
        f"- F2-Score: {f2_improvement:+.4f}",
        f"- False Negatives Reduced: {fn_reduction:+.0f} cases",
        "",
        "---",
        "",
        "## Healthcare Interpretation",
        "",
        f"### Best Architecture: {best_arch['architecture']}",
        "",
        f"**Recall ({best_arch['optimal_recall']*100:.2f}%)**:",
        f"- {best_arch['optimal_recall']*100:.2f}% of actual diabetes cases are correctly identified",
        f"- This means {best_arch['optimal_false_negatives']:.0f} diabetes cases are missed",
    ])

    if best_arch["optimal_recall"] >= 0.80:
        summary_lines.append("- ✅ **Excellent recall (≥80%)** - Suitable for healthcare screening")
    elif best_arch["optimal_recall"] >= 0.75:
        summary_lines.append("- ✅ **Good recall (≥75%)** - Suitable for healthcare screening")
    else:
        summary_lines.append("- ⚠️ **Recall below 75%** - May miss too many cases")

    summary_lines.extend([
        "",
        f"**False Negatives ({best_arch['optimal_false_negatives']:.0f})**:",
        f"- {best_arch['optimal_false_negatives']:.0f} diabetes cases would be missed",
    ])

    if fn_reduction > 0:
        summary_lines.append(
            f"- ✅ **Improved by {fn_reduction:.0f} cases** compared to baseline"
        )
    elif fn_reduction < 0:
        summary_lines.append(
            f"- ⚠️ **Increased by {abs(fn_reduction):.0f} cases** compared to baseline"
        )
    else:
        summary_lines.append("- ➡️ **No change** compared to baseline")

    summary_lines.extend([
        "",
        "**Recommendation:**",
    ])

    if best_arch["optimal_recall"] >= 0.80 and fn_reduction >= 0:
        summary_lines.append(
            f"- ✅ **Use {best_arch['architecture']}** - Best performance for healthcare screening"
        )
    elif best_arch["optimal_recall"] >= baseline["optimal_recall"]:
        summary_lines.append(
            f"- ✅ **Consider {best_arch['architecture']}** - Better recall than baseline"
        )
    else:
        summary_lines.append(
            "- ⚠️ **Consider keeping baseline** - No significant improvement"
        )

    summary_lines.extend([
        "",
        "---",
        "",
        "## Next Steps",
        "",
        "1. Use best architecture for hyperparameter optimization",
        "2. Further tune learning rate, dropout, batch size",
        "3. Compare final model with teammates' models",
        "4. Prepare final report sections",
        "",
    ])

    summary_text = "\n".join(summary_lines)

    summary_file = Path("FRANCISCO_ARCHITECTURE_TUNING_SUMMARY.md")
    with open(summary_file, "w") as f:
        f.write(summary_text)

    print(f"Summary saved to: {summary_file}")
    print(f"\nBest Architecture: {best_arch['architecture']}")
    print(f"  Recall: {best_arch['optimal_recall']:.4f} ({best_arch['optimal_recall']*100:.2f}%)")
    print(f"  F2-Score: {best_arch['optimal_f2_score']:.4f}")
    print(f"  ROC-AUC: {best_arch['roc_auc']:.4f}")


if __name__ == "__main__":
    main()

