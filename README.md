
# AAI-501 CDC Diabetes Health Indicators

**Team 6 — University of San Diego**  
**Dataset**: UCI ML Repository — CDC Diabetes Health Indicators  
**Link**: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

## Objective

Predict diabetes risk from BRFSS health indicators using multiple machine learning approaches. Compare classification models (Logistic Regression, Random Forest, XGBoost, Deep Learning MLP) and clustering techniques (K-Means). Provide explainability, clear metrics, and address class imbalance challenges.

## Project Structure

```
aai501-cdc-diabetes-health-indicators/
├── data/
│   ├── raw/                          # Raw dataset (CSV files)
│   └── processed/                    # Processed/preprocessed data
├── src/
│   └── aai501_diabetes/              # Shared Python modules
│       ├── __init__.py
│       ├── config.py                 # Configuration and paths
│       ├── data_loader.py            # Data loading utilities
│       ├── preprocessing.py          # Preprocessing and SMOTE
│       ├── mlp_model.py              # MLP architecture
│       ├── evaluation.py             # Metrics and visualization
│       └── threshold_tuning.py       # Threshold optimization
├── scripts/                          # Training and analysis scripts
│   ├── train_models.py              # Baseline models (LR, RF, XGBoost)
│   ├── francisco_eda.py             # EDA pipeline
│   ├── francisco_train_mlp.py       # MLP training
│   ├── francisco_threshold_tuning.py
│   ├── francisco_architecture_tuning.py
│   ├── francisco_hyperparameter_tuning.py
│   ├── francisco_train_final_model.py
│   └── download_data.py             # Optional data download
├── notebooks/                        # Jupyter notebooks for analysis
├── models/                           # Trained model files (.pkl, .h5)
├── figures/                          # Generated visualizations
├── reports/                          # Analysis summaries and reports
├── Makefile                          # Common commands
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd aai501-cdc-diabetes-health-indicators
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

5. **Add dataset**:
   - Download the dataset from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
   - Place the CSV file in `data/raw/`
   - The code will auto-detect common filenames

## Usage

### Quick Start

Train baseline models (Logistic Regression, Random Forest, XGBoost):
```bash
make train
```

### Available Commands

The project uses a `Makefile` for common tasks. Run `make <command>`:

#### General Commands
- `make setup` - Complete setup (venv, install, pre-commit)
- `make train` - Train baseline models (LR, RF, XGBoost)
- `make lint` - Run code quality checks (black, isort, flake8)
- `make eda` - Reminder to create EDA notebook

#### EDA and Preprocessing
- `make francisco-eda` - Run comprehensive EDA pipeline
  - Generates: class distribution, feature distributions, correlation heatmap
  - Outputs: Figures in `figures/`, summary in `reports/`

#### Deep Learning (MLP)
- `make francisco-mlp` - Train baseline MLP model
- `make francisco-threshold` - Optimize classification threshold
- `make francisco-architecture` - Compare different MLP architectures
- `make francisco-hyperparameter` - Hyperparameter optimization
- `make francisco-final` - Train final optimized MLP model

### Running Scripts Directly

You can also run scripts directly with Python:

```bash
# Set PYTHONPATH for scripts that import from src/
export PYTHONPATH=$(pwd)  # On Windows: set PYTHONPATH=%CD%

# Run scripts
python scripts/train_models.py
python scripts/francisco_eda.py
python scripts/francisco_train_mlp.py
```

## Output Locations

- **Models**: Saved to `models/` directory
  - Baseline models: `*.pkl` files (joblib format)
  - MLP models: `*.h5` files (Keras format)
  - Metrics and predictions: `*.csv` files

- **Figures**: Saved to `figures/` directory
  - EDA visualizations
  - ROC curves, confusion matrices
  - Training curves
  - Feature importance plots

- **Reports**: Saved to `reports/` directory
  - Analysis summaries
  - Evaluation metrics

## Code Quality

The project follows PEP 8 style guidelines enforced by:
- **black** - Code formatting
- **isort** - Import sorting
- **flake8** - Linting
- **pre-commit** - Git hooks for automatic checks

Run all checks:
```bash
make lint
```

## Dataset

The CDC Diabetes Health Indicators dataset contains:
- **253,680 samples** with 22 features
- **Target**: Binary diabetes classification
- **Class imbalance**: ~6.18:1 (No Diabetes : Diabetes)
- **Source**: BRFSS 2015 survey data

Key features include:
- Demographics (age, gender, education, income)
- Health behaviors (exercise, smoking, alcohol)
- Health conditions (high BP, high cholesterol, stroke history)
- Healthcare access (insurance, checkup frequency)

## Models

### Baseline Models
- **Logistic Regression**: Linear baseline
- **Random Forest**: Ensemble method with 300 trees
- **XGBoost**: Gradient boosting with optimized hyperparameters

### Deep Learning
- **MLP (Multilayer Perceptron)**: Neural network with:
  - Architecture: [128, 64] hidden layers
  - Dropout: 0.5 for regularization
  - Optimizer: Adam (LR=0.0001)
  - SMOTE for class imbalance handling
  - Optimal threshold: 0.44 (healthcare-optimized)

### Performance Highlights
- **MLP Recall**: 83.56% (healthcare-optimized)
- **MLP ROC-AUC**: 0.88
- **False Negatives**: 1,162 (32% reduction from baseline)

## License

See [LICENSE](LICENSE) file for details.

## References

- UCI Machine Learning Repository. CDC Diabetes Health Indicators. [Dataset Link](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR.
- Chen, T., & Guestrin, C. (2016). XGBoost. KDD.
- Chollet, F. (2015). Keras. GitHub.
