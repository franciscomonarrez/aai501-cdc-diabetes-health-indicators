"""
Hyperparameter optimization script for MLP diabetes prediction model.

This script:
1. Uses fixed architecture A5_HighDropout ([128, 64] with 0.5 dropout)
2. Tests multiple hyperparameter combinations
3. Evaluates at both default (0.5) and optimal (0.44) thresholds
4. Generates comparison plots and summary

Author: Francisco (Team Member 3)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import itertools

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
from src.aai501_diabetes.preprocessing import apply_smote
from src.aai501_diabetes.mlp_model import train_mlp, build_mlp, compile_model
from src.aai501_diabetes.threshold_tuning import evaluate_threshold

# Fixed architecture (A5_HighDropout - best from architecture tuning)
FIXED_ARCHITECTURE = {
    "hidden_layers": [128, 64],
    "dropout_rate": 0.5,
    "description": "A5_HighDropout (best architecture)",
}

# Hyperparameter search space
LEARNING_RATES = [0.001, 0.0005, 0.0001]
BATCH_SIZES = [128, 256, 512]
OPTIMIZERS = ["adam", "rmsprop"]
EPOCHS = 100  # Max epochs (early stopping will stop earlier)

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


def evaluate_hyperparameters(
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


def train_and_evaluate_hyperparameters(
    config_name: str,
    learning_rate: float,
    batch_size: int,
    optimizer: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_weight: dict = None,
) -> dict:
    """
    Train and evaluate a single hyperparameter configuration.

    Args:
        config_name: Configuration name
        learning_rate: Learning rate
        batch_size: Batch size
        optimizer: Optimizer name
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        class_weight: Class weights (if not using SMOTE)

    Returns:
        dict: Complete results for this configuration
    """
    print(f"\n{'='*80}")
    print(f"Training Configuration: {config_name}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Optimizer: {optimizer}")
    print(f"{'='*80}")

    # Build model with fixed architecture
    model = build_mlp(
        input_dim=X_train.shape[1],
        hidden_layers=FIXED_ARCHITECTURE["hidden_layers"],
        dropout_rate=FIXED_ARCHITECTURE["dropout_rate"],
    )

    # Compile model with specified hyperparameters
    model = compile_model(
        model,
        learning_rate=learning_rate,
        optimizer=optimizer,
    )

    # Train model
    from src.aai501_diabetes.mlp_model import create_callbacks
    callbacks_list = create_callbacks(f"mlp_{config_name}")

    try:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=batch_size,
            callbacks=callbacks_list,
            class_weight=class_weight,
            verbose=0,  # Suppress output for cleaner comparison
        )
    except Exception as e:
        print(f"  ⚠️  Error during training: {str(e)}")
        print(f"  Skipping this configuration...")
        return None

    # Evaluate at both thresholds
    optimal_threshold = load_optimal_threshold()
    thresholds = [DEFAULT_THRESHOLD, optimal_threshold]

    eval_results = evaluate_hyperparameters(model, X_test, y_test, thresholds)

    # Compile results
    results = {
        "config_name": config_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": optimizer,
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
    """Main hyperparameter optimization pipeline."""
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION - MLP DIABETES PREDICTION")
    print("Author: Francisco - Team Member 3")
    print("=" * 80)
    print(f"\nFixed Architecture: {FIXED_ARCHITECTURE['description']}")
    print(f"  Hidden Layers: {FIXED_ARCHITECTURE['hidden_layers']}")
    print(f"  Dropout: {FIXED_ARCHITECTURE['dropout_rate']}")

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

    # Step 3: Generate all hyperparameter combinations
    print("\n[STEP 3] Generating hyperparameter combinations...")
    hyperparameter_combinations = list(
        itertools.product(LEARNING_RATES, BATCH_SIZES, OPTIMIZERS)
    )
    print(f"  Total combinations to test: {len(hyperparameter_combinations)}")

    # Step 4: Train and evaluate each configuration
    print("\n[STEP 4] Training and evaluating hyperparameter configurations...")
    all_results = []

    for idx, (lr, batch, opt) in enumerate(hyperparameter_combinations, 1):
        config_name = f"LR{lr}_BS{batch}_{opt.capitalize()}"
        print(f"\n[{idx}/{len(hyperparameter_combinations)}] Testing: {config_name}")

        results = train_and_evaluate_hyperparameters(
            config_name,
            lr,
            batch,
            opt,
            X_train_balanced,
            y_train_balanced,
            X_val,
            y_val,
            X_test_scaled,
            y_test,
        )
        if results is not None:
            all_results.append(results)
        else:
            print(f"  Configuration {config_name} failed and was skipped.")

    # Step 5: Save results to CSV
    print("\n[STEP 5] Saving results...")
    results_df = pd.DataFrame(all_results)
    results_file = MODELS_DIR / "francisco_hyperparameter_comparison.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved: {results_file}")

    # Step 6: Generate comparison plot
    print("\n[STEP 6] Generating comparison plots...")
    plot_hyperparameter_comparison(results_df)

    # Step 7: Generate summary
    print("\n[STEP 7] Generating summary report...")
    generate_hyperparameter_summary(results_df)

    print("\n" + "=" * 80)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved:")
    print(f"  - Results CSV: {results_file}")
    print(f"  - Visualization: {FIG_DIR / 'francisco_hyperparameter_tuning.png'}")
    print(f"  - Summary: FRANCISCO_HYPERPARAMETER_TUNING_SUMMARY.md")


def plot_hyperparameter_comparison(results_df: pd.DataFrame) -> None:
    """Generate comparison plots for hyperparameter configurations."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a more readable config name for plotting
    results_df["config_label"] = (
        results_df["learning_rate"].astype(str) + ", "
        + results_df["batch_size"].astype(str) + ", "
        + results_df["optimizer"].str.capitalize()
    )

    n_configs = len(results_df)
    x_pos = np.arange(n_configs)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

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
    ax1.set_xlabel("Configuration (LR, Batch Size, Optimizer)", fontsize=11)
    ax1.set_ylabel("Recall", fontsize=12)
    ax1.set_title("Recall Comparison Across Hyperparameter Configurations", fontsize=14, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_df["config_label"].values, rotation=45, ha="right", fontsize=9)
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
    ax2.set_xlabel("Configuration (LR, Batch Size, Optimizer)", fontsize=11)
    ax2.set_ylabel("F2-Score", fontsize=12)
    ax2.set_title("F2-Score Comparison Across Hyperparameter Configurations", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_df["config_label"].values, rotation=45, ha="right", fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # Plot 3: ROC-AUC comparison
    ax3 = axes[1, 0]
    bars = ax3.bar(x_pos, results_df["roc_auc"].values, color="#2ecc71", alpha=0.7)
    ax3.set_xlabel("Configuration (LR, Batch Size, Optimizer)", fontsize=11)
    ax3.set_ylabel("ROC-AUC", fontsize=12)
    ax3.set_title("ROC-AUC Comparison Across Hyperparameter Configurations", fontsize=14, fontweight="bold")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(results_df["config_label"].values, rotation=45, ha="right", fontsize=9)
    ax3.grid(axis="y", alpha=0.3)
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, results_df["roc_auc"].values)):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
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
    ax4.set_xlabel("Configuration (LR, Batch Size, Optimizer)", fontsize=11)
    ax4.set_ylabel("False Negatives", fontsize=12)
    ax4.set_title("False Negatives Comparison (Healthcare Priority)", fontsize=14, fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(results_df["config_label"].values, rotation=45, ha="right", fontsize=9)
    ax4.legend(fontsize=10)
    ax4.grid(axis="y", alpha=0.3)
    # Lower is better for false negatives
    ax4.invert_yaxis()

    plt.suptitle(
        "Hyperparameter Optimization - MLP Diabetes Prediction\n"
        f"Architecture: {FIXED_ARCHITECTURE['description']}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "francisco_hyperparameter_tuning.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {FIG_DIR / 'francisco_hyperparameter_tuning.png'}")
    plt.close()


def generate_hyperparameter_summary(results_df: pd.DataFrame) -> None:
    """Generate markdown summary of hyperparameter optimization results."""
    optimal_threshold = load_optimal_threshold()

    # Find best configuration (prioritize recall and F2 at optimal threshold)
    results_df["combined_score"] = (
        results_df["optimal_recall"] * 0.5
        + results_df["optimal_f2_score"] * 0.3
        + results_df["roc_auc"] * 0.2
    )
    best_idx = results_df["combined_score"].idxmax()
    best_config = results_df.loc[best_idx]

    # Get baseline (A5 default config from architecture tuning)
    # This would be: LR=0.001, Batch=256, Optimizer=Adam
    baseline_config = results_df[
        (results_df["learning_rate"] == 0.001)
        & (results_df["batch_size"] == 256)
        & (results_df["optimizer"] == "adam")
    ]

    if len(baseline_config) > 0:
        baseline = baseline_config.iloc[0]
        has_baseline = True
    else:
        # Use first config as baseline if exact match not found
        baseline = results_df.iloc[0]
        has_baseline = False

    summary_lines = [
        "# Hyperparameter Optimization Summary - MLP Diabetes Prediction",
        "",
        "**Author**: Francisco (Team Member 3)",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"**Best Configuration**: {best_config['config_name']}",
        f"- Learning Rate: {best_config['learning_rate']}",
        f"- Batch Size: {best_config['batch_size']}",
        f"- Optimizer: {best_config['optimizer'].capitalize()}",
        "",
        f"**Key Metrics at Optimal Threshold ({optimal_threshold}):**",
        f"- Recall: {best_config['optimal_recall']:.4f} ({best_config['optimal_recall']*100:.2f}%)",
        f"- F2-Score: {best_config['optimal_f2_score']:.4f}",
        f"- ROC-AUC: {best_config['roc_auc']:.4f}",
        f"- False Negatives: {best_config['optimal_false_negatives']:.0f}",
        "",
        "---",
        "",
        "## Fixed Architecture",
        "",
        f"**Architecture**: {FIXED_ARCHITECTURE['description']}",
        f"- Hidden Layers: {FIXED_ARCHITECTURE['hidden_layers']}",
        f"- Dropout Rate: {FIXED_ARCHITECTURE['dropout_rate']}",
        "",
        "---",
        "",
        "## Hyperparameter Search Space",
        "",
        "| Parameter | Values Tested |",
        "|-----------|----------------|",
        f"| Learning Rate | {LEARNING_RATES} |",
        f"| Batch Size | {BATCH_SIZES} |",
        f"| Optimizer | {[o.capitalize() for o in OPTIMIZERS]} |",
        f"| Total Combinations | {len(results_df)} |",
        "",
        "---",
        "",
        "## Performance Comparison",
        "",
        "### At Optimal Threshold ({optimal_threshold})",
        "",
        "| Configuration | LR | Batch | Optimizer | Recall | F2-Score | ROC-AUC | False Negatives |",
        "|---------------|----|----|-----------|--------|----------|---------|-----------------|",
    ]

    # Sort by combined score for table
    sorted_df = results_df.sort_values("combined_score", ascending=False)
    for _, row in sorted_df.iterrows():
        summary_lines.append(
            f"| {row['config_name']} | {row['learning_rate']} | {row['batch_size']} | {row['optimizer'].capitalize()} | "
            f"{row['optimal_recall']:.4f} | {row['optimal_f2_score']:.4f} | {row['roc_auc']:.4f} | {row['optimal_false_negatives']:.0f} |"
        )

    if has_baseline:
        # Comparison with baseline
        recall_improvement = best_config["optimal_recall"] - baseline["optimal_recall"]
        f2_improvement = best_config["optimal_f2_score"] - baseline["optimal_f2_score"]
        fn_reduction = baseline["optimal_false_negatives"] - best_config["optimal_false_negatives"]
        roc_improvement = best_config["roc_auc"] - baseline["roc_auc"]

        summary_lines.extend([
            "",
            "---",
            "",
            "## Comparison with Baseline (A5 Default: LR=0.001, Batch=256, Adam)",
            "",
            f"**Baseline Performance**:",
            f"- Recall: {baseline['optimal_recall']:.4f} ({baseline['optimal_recall']*100:.2f}%)",
            f"- F2-Score: {baseline['optimal_f2_score']:.4f}",
            f"- ROC-AUC: {baseline['roc_auc']:.4f}",
            f"- False Negatives: {baseline['optimal_false_negatives']:.0f}",
            "",
            f"**Best Configuration Performance**:",
            f"- Recall: {best_config['optimal_recall']:.4f} ({best_config['optimal_recall']*100:.2f}%)",
            f"- F2-Score: {best_config['optimal_f2_score']:.4f}",
            f"- ROC-AUC: {best_config['roc_auc']:.4f}",
            f"- False Negatives: {best_config['optimal_false_negatives']:.0f}",
            "",
            "**Improvements:**",
            f"- Recall: {recall_improvement:+.4f} ({recall_improvement*100:+.2f} percentage points)",
            f"- F2-Score: {f2_improvement:+.4f}",
            f"- ROC-AUC: {roc_improvement:+.4f}",
            f"- False Negatives Reduced: {fn_reduction:+.0f} cases",
        ])

    summary_lines.extend([
        "",
        "---",
        "",
        "## Healthcare Interpretation",
        "",
        f"### Best Configuration: {best_config['config_name']}",
        "",
        f"**Recall ({best_config['optimal_recall']*100:.2f}%)**:",
        f"- {best_config['optimal_recall']*100:.2f}% of actual diabetes cases are correctly identified",
        f"- This means {best_config['optimal_false_negatives']:.0f} diabetes cases are missed",
    ])

    if best_config["optimal_recall"] >= 0.80:
        summary_lines.append("- ✅ **Excellent recall (≥80%)** - Suitable for healthcare screening")
    elif best_config["optimal_recall"] >= 0.75:
        summary_lines.append("- ✅ **Good recall (≥75%)** - Suitable for healthcare screening")
    else:
        summary_lines.append("- ⚠️ **Recall below 75%** - May miss too many cases")

    if has_baseline:
        summary_lines.extend([
            "",
            f"**False Negatives ({best_config['optimal_false_negatives']:.0f})**:",
            f"- {best_config['optimal_false_negatives']:.0f} diabetes cases would be missed",
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

    if best_config["optimal_recall"] >= 0.80:
        summary_lines.append(
            f"- ✅ **Use {best_config['config_name']}** - Best performance for healthcare screening"
        )
    else:
        summary_lines.append(
            f"- ⚠️ **Consider {best_config['config_name']}** - Evaluate trade-offs"
        )

    summary_lines.extend([
        "",
        "---",
        "",
        "## Final Model Configuration",
        "",
        "**Architecture**:",
        f"- Hidden Layers: {FIXED_ARCHITECTURE['hidden_layers']}",
        f"- Dropout: {FIXED_ARCHITECTURE['dropout_rate']}",
        "",
        "**Hyperparameters**:",
        f"- Learning Rate: {best_config['learning_rate']}",
        f"- Batch Size: {best_config['batch_size']}",
        f"- Optimizer: {best_config['optimizer'].capitalize()}",
        "",
        "**Performance** (at optimal threshold {optimal_threshold}):",
        f"- Recall: {best_config['optimal_recall']:.4f} ({best_config['optimal_recall']*100:.2f}%)",
        f"- F2-Score: {best_config['optimal_f2_score']:.4f}",
        f"- ROC-AUC: {best_config['roc_auc']:.4f}",
        f"- False Negatives: {best_config['optimal_false_negatives']:.0f}",
        "",
        "---",
        "",
        "## Next Steps",
        "",
        "1. Use this configuration as the final MLP model",
        "2. Compare with teammates' models (Logistic Regression, Random Forest, Gradient Boosting)",
        "3. Prepare final report sections",
        "4. Prepare presentation materials",
        "",
    ])

    summary_text = "\n".join(summary_lines)

    summary_file = Path("FRANCISCO_HYPERPARAMETER_TUNING_SUMMARY.md")
    with open(summary_file, "w") as f:
        f.write(summary_text)

    print(f"Summary saved to: {summary_file}")
    print(f"\nBest Configuration: {best_config['config_name']}")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Optimizer: {best_config['optimizer'].capitalize()}")
    print(f"  Recall: {best_config['optimal_recall']:.4f} ({best_config['optimal_recall']*100:.2f}%)")
    print(f"  F2-Score: {best_config['optimal_f2_score']:.4f}")
    print(f"  ROC-AUC: {best_config['roc_auc']:.4f}")


if __name__ == "__main__":
    main()

