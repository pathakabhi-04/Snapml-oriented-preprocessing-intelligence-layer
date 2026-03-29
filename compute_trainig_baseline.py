import json
import pandas as pd

DATA_PATH = "data/Base.csv"
FEATURE_ORDER_PATH = "snapml_training/feature_order.json"

# Load data
df = pd.read_csv(DATA_PATH).drop(columns=["fraud_bool"])

with open(FEATURE_ORDER_PATH) as f:
    feature_order = json.load(f)

baseline = {}

for feature in feature_order:
    if pd.api.types.is_numeric_dtype(df[feature]):
        baseline[feature] = {
            "mean": float(df[feature].mean()),
            "std": float(df[feature].std()),
        }

# Save baseline
with open("snapml_training/numeric_baseline.json", "w") as f:
    json.dump(baseline, f, indent=2)

print("✅ Training baseline saved.")