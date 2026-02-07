import json
from pathlib import Path
import pandas as pd

from snapml_observability.contract_parser import parse_snapml_contract
from snapml_observability.drift_detection import detect_preprocessing_drift

CONTRACT_PATH = "snapml_preprocessing.json"
DATA_PATH = "data/Base.csv"

contract = parse_snapml_contract(CONTRACT_PATH)

batch = (
    pd.read_csv(DATA_PATH)
    .drop(columns=["fraud_bool"])
    .sample(500, random_state=4)
)

# INTENTIONAL CONTRACT VIOLATION
batch = batch.sample(frac=1, axis=1)

print("=== COLUMN SHUFFLE WITHOUT SNAPML-AWARE ALIGNMENT ===")

snapshot = {}

try:
    report = detect_preprocessing_drift(contract, batch)
    print("UNEXPECTED SUCCESS (potential silent failure):")
    print(report)

    snapshot = {
        "status": "unexpected_success",
        "report": report,
    }

except Exception as e:
    print("EXPECTED CONTRACT VIOLATION (fail-loud behavior)")
    print("ERROR:", e)

    snapshot = {
        "status": "contract_violation",
        "error_type": type(e).__name__,
        "error_message": str(e),
    }

# -----------------------------
# Snapshot persistence
# -----------------------------
Path("demo/snapshot_reports").mkdir(parents=True, exist_ok=True)

with open(
    "demo/snapshot_reports/column_shuffle_unguarded.json", "w"
) as f:
    json.dump(snapshot, f, indent=2)
