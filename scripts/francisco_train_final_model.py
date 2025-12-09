"""
Final MLP model training with optimized hyperparameters.

This script trains the final model using:
- Architecture: A5_HighDropout ([128, 64] with 0.5 dropout)
- Hyperparameters: LR=0.0001, Batch=128, Optimizer=Adam
- Threshold: Optimal (0.44)

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
from src.aai501_diabetes.preprocessing import apply_smote
from src.aai501_diabetes.mlp_model import build_mlp, compile_model, create_callbacks
from src.aai501_diabetes.evaluation import (
    compute_metrics,
    print_metrics,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_training_curves,
    interpret_results,
)

# Final optimized configuration
FINAL_CONFIG = {
    "architecture": {
        "hidden_layers": [128, 64],
        "dropout_rate": 0.5,
        "description": "A5_HighDropout (best architecture)",
    },
    "hyperparameters": {
        "learning_rate": 0.0001,
        "batch_size": 128,
        "optimizer": "adam",
        "epochs": 100,
    },
}

OPTIMAL_THRESHOLD = 0.44


def load_optimal_threshold() -> float:
    """Load optimal threshold from file if available."""
    optimal_file = MODELS_DIR / "francisco_optimal_threshold.txt"
    if optimal_file.exists():
        with open(optimal_file, "r") as f:
            return float(f.read().strip())
    return 0.44


def main():
    """Train final optimized model."""
    print("=" * 80)
    print("FINAL MLP MODEL TRAINING - OPTIMIZED CONFIGURATION")
    print("Author: Francisco - Team Member 3")
    print("=" * 80)

    print("\nFinal Configuration:")
    print(f"  Architecture: {FINAL_CONFIG['architecture']['description']}")
    print(f"    Hidden Layers: {FINAL_CONFIG['architecture']['hidden_layers']}")
    print(f"    Dropout: {FINAL_CONFIG['architecture']['dropout_rate']}")
    print(f"  Hyperparameters:")
    print(f"    Learning Rate: {FINAL_CONFIG['hyperparameters']['learning_rate']}")
    print(f"    Batch Size: {FINAL_CONFIG['hyperparameters']['batch_size']}")
    print(f"    Optimizer: {FINAL_CONFIG['hyperparameters']['optimizer'].capitalize()}")

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

    # Step 3: Build and compile model
    print("\n[STEP 3] Building final model...")
    model = build_mlp(
        input_dim=X_train_balanced.shape[1],
        hidden_layers=FINAL_CONFIG["architecture"]["hidden_layers"],
        dropout_rate=FINAL_CONFIG["architecture"]["dropout_rate"],
    )

    model = compile_model(
        model,
        learning_rate=FINAL_CONFIG["hyperparameters"]["learning_rate"],
        optimizer=FINAL_CONFIG["hyperparameters"]["optimizer"],
    )

    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    model.summary()

    # Step 4: Train model
    print("\n[STEP 4] Training final model...")
    callbacks_list = create_callbacks("mlp_final_optimized")

    history = model.fit(
        X_train_balanced,
        y_train_balanced,
        validation_data=(X_val, y_val),
        epochs=FINAL_CONFIG["hyperparameters"]["epochs"],
        batch_size=FINAL_CONFIG["hyperparameters"]["batch_size"],
        callbacks=callbacks_list,
        verbose=1,
    )

    # Step 5: Save model
    model_path = MODELS_DIR / "francisco_mlp_final_optimized.h5"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Step 6: Evaluate on test set
    print("\n[STEP 6] Evaluating final model on test set...")
    optimal_threshold = load_optimal_threshold()
    y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    y_pred_default = (y_pred_proba >= 0.5).astype(int)

    # Compute metrics at both thresholds
    metrics_optimal = compute_metrics(y_test, y_pred_optimal, y_pred_proba, threshold=optimal_threshold)
    metrics_default = compute_metrics(y_test, y_pred_default, y_pred_proba, threshold=0.5)

    print("\n--- Optimal Threshold (0.44) ---")
    print_metrics(metrics_optimal, "Final MLP (Optimized)")

    print("\n--- Default Threshold (0.5) ---")
    print_metrics(metrics_default, "Final MLP (Optimized)")

    # Step 7: Generate visualizations
    print("\n[STEP 7] Generating evaluation plots...")
    model_name = "MLP_Final_Optimized"

    plot_roc_curve(
        y_test,
        y_pred_proba,
        model_name=model_name,
        save_path=FIG_DIR / "francisco_mlp_final_roc_curve.png",
    )

    plot_precision_recall_curve(
        y_test,
        y_pred_proba,
        model_name=model_name,
        save_path=FIG_DIR / "francisco_mlp_final_pr_curve.png",
    )

    plot_confusion_matrix(
        y_test,
        y_pred_optimal,
        model_name=model_name,
        save_path=FIG_DIR / "francisco_mlp_final_confusion_matrix.png",
    )

    plot_training_curves(
        history,
        model_name=model_name,
        save_path=FIG_DIR / "francisco_mlp_final_training_curves.png",
    )

    # Step 8: Interpret results
    interpret_results(metrics_optimal, model_name)

    # Step 9: Save predictions and metrics
    print("\n[STEP 9] Saving results...")
    results_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred_default": y_pred_default,
        "y_pred_optimal": y_pred_optimal,
        "y_proba": y_pred_proba,
    })
    results_df.to_csv(MODELS_DIR / "francisco_mlp_final_predictions.csv", index=False)

    # Save metrics
    metrics_df = pd.DataFrame([{
        "threshold": "default_0.5",
        **metrics_default,
    }, {
        "threshold": f"optimal_{optimal_threshold}",
        **metrics_optimal,
    }])
    metrics_df.to_csv(MODELS_DIR / "francisco_mlp_final_metrics.csv", index=False)

    print(f"\nPredictions saved to: {MODELS_DIR / 'francisco_mlp_final_predictions.csv'}")
    print(f"Metrics saved to: {MODELS_DIR / 'francisco_mlp_final_metrics.csv'}")

    # Calculate F2-score and false negatives
    from src.aai501_diabetes.threshold_tuning import evaluate_threshold
    optimal_metrics_full = evaluate_threshold(y_test, y_pred_proba, optimal_threshold)
    
    print("\n" + "=" * 80)
    print("FINAL MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nFinal Model Performance (Optimal Threshold {optimal_threshold}):")
    print(f"  Recall: {metrics_optimal['recall']:.4f} ({metrics_optimal['recall']*100:.2f}%)")
    print(f"  F2-Score: {optimal_metrics_full['f2_score']:.4f}")
    print(f"  ROC-AUC: {metrics_optimal['roc_auc']:.4f}")
    print(f"  False Negatives: {optimal_metrics_full['false_negatives']:.0f}")


if __name__ == "__main__":
    main()

