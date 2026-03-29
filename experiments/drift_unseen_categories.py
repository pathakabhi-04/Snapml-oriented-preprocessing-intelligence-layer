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
    .sample(500, random_state=1)
)

# Inject unseen category into categorical column
batch.loc[:, "payment_type"] = "ZZ_UNKNOWN"

batch = align_to_snapml_order(batch, feature_order)

report = detect_preprocessing_drift(
    contract,
    batch,
    feature_order
)

print("=== UNSEEN CATEGORY DRIFT ===")
print(json.dumps(report, indent=2))  # ✅ improved

# -----------------------------
# Snapshot persistence
# -----------------------------
Path("demo/snapshot_reports").mkdir(parents=True, exist_ok=True)

with open("demo/snapshot_reports/drift_unseen_categories.json", "w") as f:
    json.dump(report, f, indent=2)