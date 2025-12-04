
PY=python

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pre-commit install

train:
	$(PY) scripts/train_models.py
eda:
	@echo "Create notebooks/01_eda.ipynb and explore data/raw/diabetes_health_indicators.csv"

francisco-eda:
	$(PY) scripts/francisco_eda.py

francisco-mlp:
	$(PY) scripts/francisco_train_mlp.py

francisco-threshold:
	PYTHONPATH=$(PWD) $(PY) scripts/francisco_threshold_tuning.py

francisco-architecture:
	PYTHONPATH=$(PWD) $(PY) scripts/francisco_architecture_tuning.py

francisco-hyperparameter:
	PYTHONPATH=$(PWD) $(PY) scripts/francisco_hyperparameter_tuning.py

francisco-final:
	PYTHONPATH=$(PWD) $(PY) scripts/francisco_train_final_model.py

lint:
	pre-commit run -a

