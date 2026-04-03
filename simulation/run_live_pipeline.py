import json
import time
from pathlib import Path

import pandas as pd
import joblib

from simulation.stream_generator import stream_batches
from simulation.metrics_engine import MetricsEngine
from simulation.utils import load_numeric_baseline

from snapml_observability.contract_parser import parse_snapml_contract
from snapml_observability.input_alignment import align_to_snapml_order
from snapml_observability.drift_detection import detect_preprocessing_drift


# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/Base.csv"
MODEL_PATH = "snapml_training/trained_pipeline.joblib"
CONTRACT_PATH = "snapml_preprocessing.json"
FEATURE_ORDER_PATH = "snapml_training/feature_order.json"

OUTPUT_PATH = "logs/live_results.json"


# -----------------------------
# Load everything
# -----------------------------
print("Loading artifacts...")

data = pd.read_csv(DATA_PATH)

pipeline = joblib.load(MODEL_PATH)

contract = parse_snapml_contract(CONTRACT_PATH)

with open(FEATURE_ORDER_PATH) as f:
    feature_order = json.load(f)

numeric_baseline = load_numeric_baseline()

metrics_engine = MetricsEngine(
    pipeline,
    label_delay=25,
    window_size=15
)

Path("logs").mkdir(exist_ok=True)

results = []

# -----------------------------
# Run simulation
# -----------------------------
print("Starting live simulation...\n")

for batch in stream_batches(data, num_batches=60, delay=0.5):

    batch_id = batch["batch_id"]
    scenario = batch["scenario"]
    batch_df = batch["data"]
    injection_time = batch["injection_time"]

    print(f"[Batch {batch_id}] Scenario: {scenario}")

    # -----------------------------
    # Drift Detection
    # -----------------------------
    drift_input = batch_df.drop(columns=["fraud_bool"])
    drift_input = align_to_snapml_order(drift_input, feature_order)

    drift_report = detect_preprocessing_drift(
        contract,
        drift_input,
        feature_order,
        numeric_baseline
    )

    drift_time = drift_report["timestamp"]
    drift_detected = len(drift_report["alerts"]) > 0

    if drift_detected:
        print(f"  🚨 Drift detected at {drift_time}")

    # -----------------------------
    # Metrics Engine
    # -----------------------------
    metrics_engine.add_batch(batch_id, batch_df)
    metrics_report = metrics_engine.evaluate_if_ready()

    metrics_time = None
    accuracy = None
    lead_time = None

    if metrics_report:
        metrics_time = metrics_report["timestamp"]
        accuracy = metrics_report["accuracy"]

        print(f"  📉 Accuracy computed: {accuracy:.4f} at {metrics_time}")

        # -----------------------------
        # Proper comparison (same batch)
        # -----------------------------
        metrics_batch_id = metrics_report["batch_id"]

        for prev in results:
            if prev["batch_id"] == metrics_batch_id:
                if prev["drift_detected"]:
                    lead_time = metrics_time - prev["drift_time"]
                break

        if lead_time is not None:
            print(f"  ⏱ Lead time: {lead_time:.2f} sec")

    # -----------------------------
    # Store result
    # -----------------------------
    result = {
        "batch_id": batch_id,
        "scenario": scenario,
        "injection_time": injection_time,
        "drift_detected": drift_detected,
        "drift_time": drift_time,
        "metrics_time": metrics_time,
        "accuracy": accuracy,
        "lead_time": lead_time,
    }

    results.append(result)

    print("")


# -----------------------------
# Save results
# -----------------------------
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("Simulation complete.")
print(f"Results saved to {OUTPUT_PATH}")

# -----------------------------
# Evaluation Metrics
# -----------------------------

TP = FP = FN = TN = 0

for r in results:
    is_actual_drift = r["scenario"] != "none"
    detected = r["drift_detected"]

    if detected and is_actual_drift:
        TP += 1
    elif detected and not is_actual_drift:
        FP += 1
    elif not detected and is_actual_drift:
        FN += 1
    else:
        TN += 1

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
detection_accuracy = (TP + TN) / len(results)

print("\n=== Drift Detection Evaluation ===")
print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Detection Accuracy: {detection_accuracy:.4f}")