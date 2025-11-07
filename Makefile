
PY=python

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pre-commit install

train:
	$(PY) scripts/train_models.py
eda:
	@echo "Create notebooks/01_eda.ipynb and explore data/raw/diabetes_health_indicators.csv"

lint:
	pre-commit run -a

