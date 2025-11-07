# AAI-501 CDC Diabetes Health Indicators

Team 6 — University of San Diego  
Dataset: UCI ML Repository — CDC Diabetes Health Indicators  
Link: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

## Objective
Predict diabetes risk from BRFSS health indicators. Compare classification models. Provide explainability and clear metrics.

## Structure
See the repo tree. Raw CSV should be saved to:
`data/raw/diabetes_health_indicators.csv`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pre-commit install
```

## Train

```bash
make train
```

Models are saved in `models/`
Figures in `figures/`
