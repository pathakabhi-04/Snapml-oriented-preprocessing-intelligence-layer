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


def run_pipeline():
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

        print(f"[Batch {batch_id}] Scenario: {scenario}")

        # Drift Detection
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

        # Metrics
        metrics_engine.add_batch(batch_id, batch_df)
        metrics_report = metrics_engine.evaluate_if_ready()

        metrics_time = None
        accuracy = None
        lead_time = None

        if metrics_report:
            metrics_time = metrics_report["timestamp"]
            accuracy = metrics_report["accuracy"]

            print(f"  📉 Accuracy computed: {accuracy:.4f}")

            metrics_batch_id = metrics_report["batch_id"]

            for prev in results:
                if prev["batch_id"] == metrics_batch_id:
                    if prev["drift_detected"]:
                        lead_time = metrics_time - prev["drift_time"]
                    break

        # Store result
        logs = results[-1]["logs"].copy() if results else []
        logs.append(f"[Batch {batch_id}] Scenario: {scenario}")

        if drift_detected:
            logs.append("🚨 Drift detected")

        if metrics_report:
            logs.append(f"📉 Accuracy: {accuracy:.4f}")

        result = {
            "batch_id": batch_id,
            "scenario": scenario,
            "drift_detected": drift_detected,
            "accuracy": accuracy,
            "metrics_time": metrics_time,
            "drift_time": drift_time,
            "lead_time": lead_time,
            "logs": logs
        }

        results.append(result)
        yield result
        print("")

    # -----------------------------
    # Save results
    # -----------------------------
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("Simulation complete.")

    # -----------------------------
    # Evaluation Metrics
    # -----------------------------
    TP = FP = FN = TN = 0

    for r in results:
        is_actual_drift = r["scenario"] != "none"
        detected_flag = r["drift_detected"]

        if detected_flag and is_actual_drift:
            TP += 1
        elif detected_flag and not is_actual_drift:
            FP += 1
        elif not detected_flag and is_actual_drift:
            FN += 1
        else:
            TN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    detection_accuracy = (TP + TN) / len(results)

    # -----------------------------
    # Lead Time
    # -----------------------------
    lead_times = [r["lead_time"] for r in results if r["lead_time"] is not None]

    avg_lead_time = sum(lead_times) / len(lead_times) if lead_times else None
    max_lead_time = max(lead_times) if lead_times else None
    min_lead_time = min(lead_times) if lead_times else None

    # -----------------------------
    # Coverage
    # -----------------------------
    drift_batches = [r for r in results if r["scenario"] != "none"]

    coverage_detected = sum(1 for r in drift_batches if r["drift_detected"])
    coverage_total = len(drift_batches)

    # -----------------------------
    # Metrics availability
    # -----------------------------
    metrics_ready = sum(1 for r in results if r["metrics_time"] is not None)

    # -----------------------------
    # Scenario stats
    # -----------------------------
    from collections import defaultdict

    scenario_stats = defaultdict(lambda: {
        "total": 0,
        "detected": 0,
        "missed": 0
    })

    for r in results:
        scenario = r["scenario"]
        detected_flag = r["drift_detected"]
        is_actual_drift = scenario != "none"

        scenario_stats[scenario]["total"] += 1

        if is_actual_drift:
            if detected_flag:
                scenario_stats[scenario]["detected"] += 1
            else:
                scenario_stats[scenario]["missed"] += 1

    print("\n=== Scenario-wise Detection ===")

    for scenario, stats in scenario_stats.items():
        print(f"\nScenario: {scenario}")
        print(f"  Total: {stats['total']}")

        if scenario != "none":
            print(f"  Detected: {stats['detected']}")
            print(f"  Missed: {stats['missed']}")

            if (stats["detected"] + stats["missed"]) > 0:
                rate = stats["detected"] / (stats["detected"] + stats["missed"])
                print(f"  Detection Rate: {rate:.2f}")
    # -----------------------------
    # FINAL SUMMARY
    # -----------------------------
    summary = {
        "type": "summary",

        "drift_detection": {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "precision": precision,
            "recall": recall,
            "accuracy": detection_accuracy,
        },

        "lead_time": {
            "samples": len(lead_times),
            "avg": avg_lead_time,
            "max": max_lead_time,
            "min": min_lead_time,
        },

        "coverage": {
            "detected": coverage_detected,
            "total": coverage_total,
            "rate": coverage_detected / coverage_total if coverage_total > 0 else 0
        },

        "metrics_availability": {
            "available": metrics_ready,
            "total": len(results)
        },

        "scenario_stats": dict(scenario_stats)
    }

    yield summary