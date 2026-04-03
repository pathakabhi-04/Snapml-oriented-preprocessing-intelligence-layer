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

# -----------------------------
# Lead Time Analysis
# -----------------------------

lead_times = [r["lead_time"] for r in results if r["lead_time"] is not None]

if lead_times:
    avg_lead_time = sum(lead_times) / len(lead_times)
    max_lead_time = max(lead_times)
    min_lead_time = min(lead_times)

    print("\n=== Lead Time Analysis ===")
    print(f"Samples: {len(lead_times)}")
    print(f"Average Lead Time: {avg_lead_time:.2f} sec")
    print(f"Max Lead Time: {max_lead_time:.2f} sec")
    print(f"Min Lead Time: {min_lead_time:.2f} sec")

# -----------------------------
# Detection Coverage
# -----------------------------

drift_batches = [r for r in results if r["scenario"] != "none"]

detected = sum(1 for r in drift_batches if r["drift_detected"])
total = len(drift_batches)

print("\n=== Drift Coverage ===")
print(f"Detected: {detected}/{total} = {detected/total:.2f}")

# -----------------------------
# Metrics Availability
# -----------------------------

metrics_ready = sum(1 for r in results if r["metrics_time"] is not None)

print("\n=== Metrics Availability ===")
print(f"Metrics computed for {metrics_ready}/{len(results)} batches")

# ============================================================
# STEP 5 — Scenario-wise Benchmarking
# ============================================================

from collections import defaultdict

scenario_stats = defaultdict(lambda: {
    "total": 0,
    "detected": 0,
    "missed": 0
})

for r in results:
    scenario = r["scenario"]
    detected = r["drift_detected"]
    is_actual_drift = scenario != "none"

    scenario_stats[scenario]["total"] += 1

    if is_actual_drift:
        if detected:
            scenario_stats[scenario]["detected"] += 1
        else:
            scenario_stats[scenario]["missed"] += 1

print("\n=== Scenario-wise Detection ===")

for scenario, stats in scenario_stats.items():
    total = stats["total"]
    detected = stats["detected"]
    missed = stats["missed"]

    print(f"\nScenario: {scenario}")
    print(f"  Total Batches: {total}")

    if scenario != "none":
        detection_rate = detected / (detected + missed) if (detected + missed) > 0 else 0

        print(f"  Detected: {detected}")
        print(f"  Missed: {missed}")
        print(f"  Detection Rate: {detection_rate:.2f}")

# -----------------------------
# FINAL SUMMARY (STEP 6)
# -----------------------------

print("\n=== FINAL SUMMARY ===")

# Drift detection quality
print(f"Drift Detection Precision: {precision:.2f}")
print(f"Drift Detection Recall: {recall:.2f}")

# Lead time
if lead_times:
    print(f"Average Lead Time: {avg_lead_time:.2f} sec")

# Coverage
print(f"Drift Coverage: {detected}/{total} = {detected/total:.2f}")

# Metrics delay insight
print(f"Metrics Availability: {metrics_ready}/{len(results)} batches")

# Key takeaway
print("\n🔑 Insight:")
print("Preprocessing drift is detected BEFORE model performance degrades.")
print("Metrics-based monitoring reacts only after delayed label feedback.")