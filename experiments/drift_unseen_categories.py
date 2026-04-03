import json
from pathlib import Path
import pandas as pd

from snapml_observability.contract_parser import parse_snapml_contract
from snapml_observability.input_alignment import align_to_snapml_order
from snapml_observability.drift_detection import detect_preprocessing_drift

from simulation.utils import load_numeric_baseline


# -----------------------------
# Configuration
# -----------------------------
CONTRACT_PATH = "snapml_preprocessing.json"
FEATURE_ORDER_PATH = "snapml_training/feature_order.json"
DATA_PATH = "data/Base.csv"


# -----------------------------
# Load artifacts
# -----------------------------
contract = parse_snapml_contract(CONTRACT_PATH)

with open(FEATURE_ORDER_PATH) as f:
    feature_order = json.load(f)

numeric_baseline = load_numeric_baseline()


# -----------------------------
# Load batch (IMPORTANT: remove target for drift detection)
# -----------------------------
df = pd.read_csv(DATA_PATH)

batch = (
    df
    .drop(columns=["fraud_bool"])   # ✅ drift detection uses only features
    .sample(500, random_state=1)
)


# -----------------------------
# Inject drift
# -----------------------------
batch.loc[:, "payment_type"] = "ZZ_UNKNOWN"


# -----------------------------
# Align to SnapML feature order
# -----------------------------
batch = align_to_snapml_order(batch, feature_order)


# -----------------------------
# Detect drift
# -----------------------------
report = detect_preprocessing_drift(
    contract,
    batch,
    feature_order,
    numeric_baseline   # ✅ FIXED
)


# -----------------------------
# Print result
# -----------------------------
print("=== UNSEEN CATEGORY DRIFT ===")
print(json.dumps(report, indent=2))


# -----------------------------
# Snapshot persistence
# -----------------------------
output_path = Path("demo/snapshot_reports")
output_path.mkdir(parents=True, exist_ok=True)

with open(output_path / "drift_unseen_categories.json", "w") as f:
    json.dump(report, f, indent=2)