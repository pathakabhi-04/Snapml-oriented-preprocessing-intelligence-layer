import json
from pathlib import Path
import pandas as pd

from snapml_observability.contract_parser import parse_snapml_contract
from snapml_observability.input_alignment import align_to_snapml_order
from snapml_observability.drift_detection import detect_preprocessing_drift

# -----------------------------
# Paths
# -----------------------------
CONTRACT_PATH = "snapml_preprocessing.json"
FEATURE_ORDER_PATH = "snapml_training/feature_order.json"
BASELINE_PATH = "snapml_training/numeric_baseline.json"
DATA_PATH = "data/Base.csv"

# -----------------------------
# Load contract
# -----------------------------
contract = parse_snapml_contract(CONTRACT_PATH)

with open(FEATURE_ORDER_PATH) as f:
    feature_order = json.load(f)

with open(BASELINE_PATH) as f:
    numeric_baseline = json.load(f)

# -----------------------------
# Load batch
# -----------------------------
batch = (
    pd.read_csv(DATA_PATH)
    .drop(columns=["fraud_bool"])
    .sample(500, random_state=3)
)

# -----------------------------
# Inject numeric drift (TARGETED ✅)
# -----------------------------
batch.loc[:, "income"] = batch["income"] * 10

# -----------------------------
# Align
# -----------------------------
batch = align_to_snapml_order(batch, feature_order)

# -----------------------------
# Detect drift (FIXED ✅)
# -----------------------------
report = detect_preprocessing_drift(
    contract,
    batch,
    feature_order,
    numeric_baseline
)

# -----------------------------
# Output
# -----------------------------
print("=== NUMERIC DISTRIBUTION DRIFT ===")
print(json.dumps(report, indent=2))

# -----------------------------
# Save snapshot
# -----------------------------
Path("demo/snapshot_reports").mkdir(parents=True, exist_ok=True)

with open("demo/snapshot_reports/numeric_distribution_drift.json", "w") as f:
    json.dump(report, f, indent=2)