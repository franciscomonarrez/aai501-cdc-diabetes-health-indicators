"""
Main training script for Deep Learning (MLP) model.

This script:
1. Loads and preprocesses the dataset
2. Handles class imbalance (SMOTE or class weights)
3. Trains the MLP model with hyperparameter tuning
4. Evaluates the model comprehensively
5. Generates all required plots and metrics

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
    compute_class_weights,
    get_imbalance_ratio,
)
from src.aai501_diabetes.mlp_model import train_mlp
from src.aai501_diabetes.evaluation import (
    compute_metrics,
    print_metrics,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_training_curves,
    interpret_results,
)

# Configuration
USE_SMOTE = True  # Set to False to use class weights instead
MODEL_CONFIG = {
    "hidden_layers": [128, 64, 32],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
}
EPOCHS = 100
BATCH_SIZE = 256


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("DEEP LEARNING (MLP) MODEL TRAINING - DIABETES PREDICTION")
    print("Author: Francisco - Team Member 3")
    print("=" * 80)

    # Step 1: Load and validate data
    print("\n[STEP 1] Loading and validating dataset...")
    df = load_data()
    validate_data(df)

    # Step 2: Split data
    print("\n[STEP 2] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, stratify=True)

    # Step 3: Create and fit preprocessor
    print("\n[STEP 3] Creating preprocessing pipeline...")
    preprocessor = create_preprocessor(X_train)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Convert to numpy arrays
    X_train_scaled = np.array(X_train_scaled)
    X_test_scaled = np.array(X_test_scaled)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(f"  Preprocessed training shape: {X_train_scaled.shape}")
    print(f"  Preprocessed test shape: {X_test_scaled.shape}")

    # Step 4: Further split for validation
    print("\n[STEP 4] Creating validation set...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    print(f"  Final training: {X_train_final.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test_scaled.shape[0]} samples")

    # Step 5: Handle class imbalance
    print("\n[STEP 5] Handling class imbalance...")
    imbalance_ratio = get_imbalance_ratio(y_train_final)
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

    class_weight = None
    if USE_SMOTE:
        print("  Using SMOTE for oversampling...")
        X_train_balanced, y_train_balanced = apply_smote(X_train_final, y_train_final)
        X_train_final = X_train_balanced
        y_train_final = y_train_balanced
    else:
        print("  Using class weights...")
        class_weight = compute_class_weights(y_train_final)

    # Step 6: Train MLP model
    print("\n[STEP 6] Training MLP model...")
    model, history = train_mlp(
        X_train_final,
        y_train_final,
        X_val,
        y_val,
        model_config=MODEL_CONFIG,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        verbose=1,
    )

    # Step 7: Evaluate on test set
    print("\n[STEP 7] Evaluating model on test set...")
    y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred_default = (y_pred_proba >= 0.5).astype(int)

    # Compute metrics with default threshold (0.5)
    metrics_default = compute_metrics(y_test, y_pred_default, y_pred_proba, threshold=0.5)
    print("\n--- Default Threshold (0.5) ---")
    print_metrics(metrics_default, "MLP")

    # Check if optimal threshold exists from previous tuning
    optimal_threshold_file = MODELS_DIR / "francisco_optimal_threshold.txt"
    if optimal_threshold_file.exists():
        with open(optimal_threshold_file, "r") as f:
            optimal_threshold = float(f.read().strip())
        print(f"\n--- Optimal Threshold ({optimal_threshold:.4f}) ---")
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        metrics_optimal = compute_metrics(y_test, y_pred_optimal, y_pred_proba, threshold=optimal_threshold)
        print_metrics(metrics_optimal, "MLP (Optimal Threshold)")
        metrics = metrics_optimal  # Use optimal for final metrics
    else:
        metrics = metrics_default  # Use default if no optimal threshold
        print("\nNote: Run threshold tuning script to find optimal threshold")

    # Step 8: Generate visualizations
    print("\n[STEP 8] Generating evaluation plots...")
    model_name = "MLP_Diabetes"

    plot_roc_curve(
        y_test,
        y_pred_proba,
        model_name=model_name,
        save_path=FIG_DIR / "francisco_mlp_roc_curve.png",
    )

    plot_precision_recall_curve(
        y_test,
        y_pred_proba,
        model_name=model_name,
        save_path=FIG_DIR / "francisco_mlp_pr_curve.png",
    )

    # Use optimal threshold if available, otherwise default
    optimal_threshold_file = MODELS_DIR / "francisco_optimal_threshold.txt"
    if optimal_threshold_file.exists():
        with open(optimal_threshold_file, "r") as f:
            optimal_threshold = float(f.read().strip())
        y_pred_for_plots = (y_pred_proba >= optimal_threshold).astype(int)
    else:
        y_pred_for_plots = y_pred_default

    plot_confusion_matrix(
        y_test,
        y_pred_for_plots,
        model_name=model_name,
        save_path=FIG_DIR / "francisco_mlp_confusion_matrix.png",
    )

    plot_training_curves(
        history,
        model_name=model_name,
        save_path=FIG_DIR / "francisco_mlp_training_curves.png",
    )

    # Step 9: Interpret results
    interpret_results(metrics, model_name)

    # Step 10: Save predictions and summary
    print("\n[STEP 10] Saving results...")
    results_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred_default,
        "y_proba": y_pred_proba,
    })
    results_df.to_csv(MODELS_DIR / "francisco_mlp_predictions.csv", index=False)

    # Save metrics summary
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(MODELS_DIR / "francisco_mlp_metrics.csv", index=False)

    print(f"\nPredictions saved to: {MODELS_DIR / 'francisco_mlp_predictions.csv'}")
    print(f"Metrics saved to: {MODELS_DIR / 'francisco_mlp_metrics.csv'}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - All outputs saved to models/ and figures/")
    print("=" * 80)


if __name__ == "__main__":
    main()

