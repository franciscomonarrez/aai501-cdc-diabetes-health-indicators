# Francisco's Component - Deep Learning & Data Pipeline

This document outlines the code structure and usage for Francisco's part of the project.

## Overview

Francisco is responsible for:
1. **Dataset Setup & Preprocessing** - Complete data pipeline
2. **Full EDA** - Comprehensive exploratory data analysis
3. **Deep Learning (MLP)** - Neural network model implementation
4. **Evaluation & Metrics** - All performance metrics and visualizations

## File Structure

```
src/aai501_diabetes/
├── __init__.py
├── config.py              # Central configuration and paths
├── data_loader.py          # Data loading and validation utilities
├── preprocessing.py        # Class imbalance handling (SMOTE/class weights)
├── mlp_model.py           # MLP model architecture and training
└── evaluation.py           # Metrics, plots, and result interpretation

scripts/
├── francisco_eda.py        # Complete EDA script
└── francisco_train_mlp.py  # Main MLP training pipeline
```

## Quick Start

### 1. Run EDA

```bash
make francisco-eda
# or
python scripts/francisco_eda.py
```

This generates:
- Class distribution plots
- Feature distribution histograms
- Correlation heatmap
- Target correlation analysis
- Summary statistics

All figures saved to `figures/` directory.

### 2. Train MLP Model

```bash
make francisco-mlp
# or
python scripts/francisco_train_mlp.py
```

This will:
1. Load and preprocess the dataset
2. Handle class imbalance (SMOTE by default)
3. Train the MLP model with early stopping
4. Evaluate on test set
5. Generate all plots (ROC, PR curve, confusion matrix, training curves)
6. Save model and predictions

## Configuration

### MLP Model Configuration

Edit `scripts/francisco_train_mlp.py` to adjust:

```python
USE_SMOTE = True  # Set to False to use class weights instead
MODEL_CONFIG = {
    "hidden_layers": [128, 64, 32],  # Adjust architecture
    "dropout_rate": 0.3,              # Regularization
    "learning_rate": 0.001,           # Learning rate
}
EPOCHS = 100
BATCH_SIZE = 256
```

### Class Imbalance Handling

Two options available:
- **SMOTE** (default): Synthetic oversampling of minority class
- **Class Weights**: Weighted loss function

Toggle with `USE_SMOTE` flag in training script.

## Output Files

### Models
- `models/mlp_diabetes_model.h5` - Saved Keras model
- `models/francisco_mlp_predictions.csv` - Test set predictions
- `models/francisco_mlp_metrics.csv` - Performance metrics

### Figures
- `figures/francisco_class_distribution.png`
- `figures/francisco_feature_distributions.png`
- `figures/francisco_correlation_heatmap.png`
- `figures/francisco_target_correlations.png`
- `figures/francisco_mlp_roc_curve.png`
- `figures/francisco_mlp_pr_curve.png`
- `figures/francisco_mlp_confusion_matrix.png`
- `figures/francisco_mlp_training_curves.png`

### Reports
- `reports/francisco_eda_summary.txt` - EDA summary statistics

## Metrics Generated

The evaluation includes:
- **Accuracy** - Overall correctness
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under ROC curve

### Healthcare Interpretation

The script automatically provides healthcare-focused interpretation:
- **Recall** is critical: Missing diabetes cases (False Negatives) can be dangerous
- **Precision** trade-off: False Positives lead to follow-up tests but are acceptable
- Recommendations based on metric thresholds

## Code Quality

All code follows:
- PEP 8 style guidelines
- Type hints where appropriate
- Comprehensive docstrings
- Modular, reusable functions

Run linting:
```bash
make lint
```

## Dependencies

Key dependencies (see `requirements.txt`):
- `tensorflow==2.15.0` - Deep learning framework
- `imbalanced-learn==0.12.3` - SMOTE implementation
- `scikit-learn==1.5.1` - Preprocessing and metrics
- `pandas`, `numpy`, `matplotlib`, `seaborn` - Data handling and visualization

## Next Steps for Report

Use the generated outputs for:
1. **Introduction** - Dataset overview from EDA
2. **Data Description** - Summary statistics and feature descriptions
3. **EDA Section** - All generated plots and insights
4. **Preprocessing Section** - SMOTE/class weighting approach
5. **Deep Learning Section** - Model architecture, training, results
6. **Results Interpretation** - Healthcare context and trade-offs

## Troubleshooting

### Dataset Not Found
Ensure the CSV file is in `data/raw/` directory. The script will auto-detect the filename.

### Memory Issues
- Reduce `BATCH_SIZE` in training script
- Use class weights instead of SMOTE (`USE_SMOTE = False`)
- Process data in chunks if needed

### TensorFlow Issues
- Ensure TensorFlow is installed: `pip install tensorflow==2.15.0`
- Check GPU availability if using GPU: `tf.config.list_physical_devices('GPU')`

## Contact

For questions or issues with Francisco's components, refer to this documentation or check the code comments.

