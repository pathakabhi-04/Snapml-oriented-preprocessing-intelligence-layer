import json
from pathlib import Path
import pandas as pd


from snapml_observability.contract_parser import parse_snapml_contract
from snapml_observability.input_alignment import align_to_snapml_order
from snapml_observability.drift_detection import detect_preprocessing_drift

CONTRACT_PATH = "snapml_preprocessing.json"
FEATURE_ORDER_PATH = "snapml_training/feature_order.json"
DATA_PATH = "data/Base.csv"

contract = parse_snapml_contract(CONTRACT_PATH)

with open(FEATURE_ORDER_PATH) as f:
    feature_order = json.load(f)

batch = (
    pd.read_csv(DATA_PATH)
    .drop(columns=["fraud_bool"])
    .sample(500, random_state=2)
)

# Make most categorical values unseen
for col in ["payment_type", "employment_status", "housing_status"]:
    batch[col] = "UNKNOWN"

batch = align_to_snapml_order(batch, feature_order)

report = detect_preprocessing_drift(contract, batch)

print("=== CATEGORICAL SPARSITY DRIFT ===")
print(report)


# -----------------------------
# Snapshot persistence (NEW)
# -----------------------------
Path("demo/snapshot_reports").mkdir(parents=True, exist_ok=True)

with open("demo/snapshot_reports/drift_categorical_sparsity.json", "w") as f:
    json.dump(report, f, indent=2)
