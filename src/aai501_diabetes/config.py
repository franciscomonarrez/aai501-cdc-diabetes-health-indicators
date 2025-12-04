from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Dataset paths - handles both possible filenames
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_RAW = DATA_RAW_DIR / "diabetes_health_indicators.csv"
# Fallback to actual filename if default doesn't exist
if not DATA_RAW.exists():
    actual_file = list(DATA_RAW_DIR.glob("*.csv"))
    if actual_file:
        DATA_RAW = actual_file[0]

DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "figures"
NOTEBOOKS_DIR = ROOT / "notebooks"

# Create directories if they don't exist
for p in [DATA_PROCESSED, MODELS_DIR, FIG_DIR, NOTEBOOKS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Constants
RANDOM_STATE = 42
TARGET_COL = "Diabetes_binary"
