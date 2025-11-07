from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "diabetes_health_indicators.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"

for p in [DATA_PROCESSED, MODELS_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)
