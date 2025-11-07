from pathlib import Path
import requests

DIRECT_URL = ""  # set if a direct CSV URL is available
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "diabetes_health_indicators.csv"

if not DIRECT_URL:
    print("Download CSV from UCI in a browser and save to:")
    print(OUT)
else:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(DIRECT_URL, timeout=60)
    r.raise_for_status()
    OUT.write_bytes(r.content)
    print(f"Saved: {OUT}")
