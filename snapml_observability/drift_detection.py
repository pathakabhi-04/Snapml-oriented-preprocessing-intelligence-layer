from typing import Dict, Any
import numpy as np
import pandas as pd
import time

from snapml_observability.contract_models import SnapMLPreprocessingContract


# ------------------------------------------------------------
# Alert Builder
# ------------------------------------------------------------
def build_alert(
    alert_type: str,
    feature: str,
    severity: str,
    metrics: Dict[str, Any],
    explanation: str,
    action: str,
) -> Dict[str, Any]:

    severity_scores = {
        "low": 0.3,
        "medium": 0.6,
        "high": 1.0,
    }

    return {
        "type": alert_type,
        "feature": feature,
        "severity": {
            "level": severity,
            "score": severity_scores.get(severity, 0.0),
        },
        "metrics": metrics,
        "explanation": explanation,
        "recommended_action": action,
    }


def detect_preprocessing_drift(
    contract: SnapMLPreprocessingContract,
    inference_batch: pd.DataFrame,
    feature_order: list,
    numeric_baseline: dict,
    *,
    unseen_category_threshold: float = 0.10,
    sparsity_threshold: float = 0.05,
    norm_anomaly_threshold: float = 0.2,   # 🔥 changed
    numeric_shift_threshold: float = 0.5,  # 🔥 changed
) -> Dict[str, Any]:

    report: Dict[str, Any] = {
        "alerts": [],
        "metrics": {},
        "timestamp": time.time()   # 🔥 ADD THIS
    }

    # ------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------
    def find_steps(step_type: str):
        steps = []
        for block in contract.transformers:
            for step in block.steps:
                if step.step_type == step_type:
                    steps.append(step)
        return steps

    # ============================================================
    # 1) CATEGORICAL DRIFT
    # ============================================================

    ohe_steps = find_steps("OneHotEncoder")

    if ohe_steps:
        step = ohe_steps[0]
        cat_indices = step.columns
        known_categories = step.data.get("categories", [])

        categorical_metrics = {}

        for idx, categories in zip(cat_indices, known_categories):

            feature_name = feature_order[idx]
            unseen_count = 0

            for _, row in inference_batch.iterrows():
                if row.iloc[idx] not in categories:
                    unseen_count += 1

            unseen_rate = unseen_count / len(inference_batch)

            categorical_metrics[feature_name] = {
                "unseen_rate": unseen_rate
            }

            if unseen_rate > unseen_category_threshold:
                severity = "high" if unseen_rate > 0.5 else "medium"

                report["alerts"].append(
                    build_alert(
                        "unseen_category",
                        feature_name,
                        severity,
                        {
                            "unseen_rate": unseen_rate,
                            "threshold": unseen_category_threshold,
                        },
                        f"Feature '{feature_name}' contains unseen values. These will be encoded as zero-vectors, reducing signal.",
                        "retrain_pipeline" if severity == "high" else "monitor_and_update_categories",
                    )
                )

        # Global sparsity
        all_cat_unseen_rows = 0

        for _, row in inference_batch.iterrows():
            unseen_in_row = sum(
                row.iloc[idx] not in categories
                for idx, categories in zip(cat_indices, known_categories)
            )

            if unseen_in_row == len(cat_indices):
                all_cat_unseen_rows += 1

        sparsity_rate = all_cat_unseen_rows / len(inference_batch)

        categorical_metrics["_global"] = {
            "sparsity_rate": sparsity_rate
        }

        if sparsity_rate > sparsity_threshold:
            report["alerts"].append(
                build_alert(
                    "categorical_sparsity",
                    "ALL_CATEGORICAL",
                    "high",
                    {
                        "sparsity_rate": sparsity_rate,
                        "threshold": sparsity_threshold,
                    },
                    "Many rows collapse to zero-vector due to unseen categories.",
                    "investigate_data_pipeline",
                )
            )

        report["metrics"]["categorical"] = categorical_metrics

    # ============================================================
    # 2) NORMALIZATION DRIFT
    # ============================================================

    norm_steps = find_steps("Normalizer")

    if norm_steps:
        step = norm_steps[0]
        num_indices = step.columns

        norms = []

        for _, row in inference_batch.iterrows():
            values = row.iloc[num_indices]

            try:
                vec = values.astype(float).values
            except ValueError:
                raise ValueError("Column mismatch with contract")

            norms.append(np.linalg.norm(vec, ord=2))

        norms = np.array(norms)
        median_norm = np.median(norms)

        anomaly_rate = np.mean(norms > (3 * median_norm))

        report.setdefault("metrics", {}).setdefault("numeric", {})
        report["metrics"]["numeric"]["norm_anomaly_rate"] = anomaly_rate

        if anomaly_rate > norm_anomaly_threshold:
            report["alerts"].append(
                build_alert(
                    "normalization_drift",
                    "numeric_features",
                    "high",
                    {
                        "anomaly_rate": anomaly_rate,
                        "threshold": norm_anomaly_threshold,
                    },
                    "Numeric vectors deviate from expected L2 norms.",
                    "inspect_numeric_distribution",
                )
            )

    # ============================================================
    # 3) NUMERIC DISTRIBUTION DRIFT (FIXED BASELINE)
    # ============================================================

    num_indices = contract.data_schema.numeric_indices
    numeric_dist_metrics = {}

    for idx in num_indices:

        feature_name = feature_order[idx]
        values = inference_batch.iloc[:, idx].astype(float)

        mean = values.mean()
        std = values.std()

        baseline = numeric_baseline.get(feature_name, None)

        if not baseline:
            continue

        baseline_mean = baseline["mean"]
        baseline_std = baseline["std"]

        # ✅ NORMALIZED DRIFT (FIXED)
        mean_shift = abs(mean - baseline_mean) / (baseline_std + 1e-6)
        std_shift = abs(std - baseline_std) / (baseline_std + 1e-6)

        numeric_dist_metrics[feature_name] = {
            "mean": mean,
            "std": std,
            "mean_shift": mean_shift,
            "std_shift": std_shift,
        }

        # Strong drift (either mean OR std very high)
        if max(mean_shift, std_shift) > 1.5:

            severity = "high"

            report["alerts"].append(
                build_alert(
                    "numeric_distribution_drift",
                    feature_name,
                    severity,
                    {
                        "mean_shift": mean_shift,
                        "std_shift": std_shift,
                    },
                    f"Feature '{feature_name}' distribution shifted significantly.",
                    "inspect_numeric_distribution",
                )
            )

        # Moderate drift (both mean AND std moderately shifted)
        elif (
            mean_shift > numeric_shift_threshold
            or std_shift > numeric_shift_threshold
        ):
            severity = "medium"

            report["alerts"].append(
                build_alert(
                    "numeric_distribution_drift",
                    feature_name,
                    severity,
                    {
                        "mean_shift": mean_shift,
                        "std_shift": std_shift,
                    },
                    f"Feature '{feature_name}' distribution moderately shifted.",
                    "inspect_numeric_distribution",
                )
            )

    report.setdefault("metrics", {}).setdefault("numeric", {})
    report["metrics"]["numeric"]["distribution"] = numeric_dist_metrics

    # ============================================================
    # 4) SUMMARY
    # ============================================================

    drifted_features = []
    top_drift = []

    for alert in report["alerts"]:
        feature = alert["feature"]

        if feature not in ["ALL_CATEGORICAL", "numeric_features"]:
            drifted_features.append(feature)

            metric_values = alert.get("metrics", {})
            drift_value = (
                metric_values.get("unseen_rate")
                or metric_values.get("anomaly_rate")
                or metric_values.get("mean_shift")
                or 0.0
            )

            top_drift.append({
                "feature": feature,
                "drift_score": drift_value
            })

    top_drift = sorted(top_drift, key=lambda x: x["drift_score"], reverse=True)

    report["summary"] = {
        "num_alerts": len(report["alerts"]),
        "drifted_features": list(set(drifted_features)),
        "top_drifted_features": top_drift[:3],
    }

    return report

    