from typing import Dict, Any
import numpy as np
import pandas as pd

from snapml_observability.contract_models import SnapMLPreprocessingContract


def detect_preprocessing_drift(
    contract: SnapMLPreprocessingContract,
    inference_batch: pd.DataFrame,
    *,
    unseen_category_threshold: float = 0.10,
    sparsity_threshold: float = 0.05,
    norm_anomaly_threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Detect drift as violations of SnapML preprocessing assumptions.

    This function:
    - does NOT run preprocessing
    - does NOT refit anything
    - uses ONLY the SnapML preprocessing contract
    """

    report: Dict[str, Any] = {
        "alerts": [],
        "metrics": {},
    }

    # ------------------------------------------------------------
    # Helper: find preprocessing steps by type
    # ------------------------------------------------------------
    def find_steps(step_type: str):
        steps = []
        for block in contract.transformers:
            for step in block.steps:
                if step.step_type == step_type:
                    steps.append(step)
        return steps

    # ============================================================
    # 1) Unseen Category Drift & Sparsity Explosion
    # ============================================================

    ohe_steps = find_steps("OneHotEncoder")

    if ohe_steps:
        step = ohe_steps[0]  # SnapML contract guarantees fixed behavior
        cat_indices = step.columns
        known_categories = step.data.get("categories", [])

        unseen_rows = 0
        all_cat_unseen_rows = 0

        for _, row in inference_batch.iterrows():
            unseen_in_row = 0
            for idx, categories in zip(cat_indices, known_categories):
                value = row.iloc[idx]
                if value not in categories:
                    unseen_in_row += 1

            if unseen_in_row > 0:
                unseen_rows += 1
            if unseen_in_row == len(cat_indices):
                all_cat_unseen_rows += 1

        unseen_rate = unseen_rows / len(inference_batch)
        sparsity_rate = all_cat_unseen_rows / len(inference_batch)

        report["metrics"]["unseen_category_rate"] = unseen_rate
        report["metrics"]["categorical_sparsity_rate"] = sparsity_rate

        if unseen_rate > unseen_category_threshold:
            report["alerts"].append(
                f"⚠ Unseen category rate {unseen_rate:.2%} exceeds threshold "
                f"{unseen_category_threshold:.2%}."
            )

        if sparsity_rate > sparsity_threshold:
            report["alerts"].append(
                f"⚠ Categorical sparsity rate {sparsity_rate:.2%} exceeds threshold "
                f"{sparsity_threshold:.2%} (risk of silent information loss)."
            )

    # ============================================================
    # 2) Normalization Assumption Drift (L2)
    # ============================================================

    norm_steps = find_steps("Normalizer")

    if norm_steps:
        step = norm_steps[0]
        num_indices = step.columns

        norms = []

        for _, row in inference_batch.iterrows():
            values = row.iloc[num_indices]

            # --- HARD GUARD: SnapML contract alignment check ---
            try:
                vec = values.astype(float).values
            except ValueError as e:
                raise ValueError(
                    "Inference data column order does not match SnapML preprocessing "
                    "contract. Numeric indices refer to non-numeric values."
                ) from e
            # --------------------------------------------------

            norm = np.linalg.norm(vec, ord=2)
            norms.append(norm)

        norms = np.array(norms)
        median_norm = np.median(norms)

        anomaly_rate = np.mean(norms > (3 * median_norm))
        report["metrics"]["norm_anomaly_rate"] = anomaly_rate

        if anomaly_rate > norm_anomaly_threshold:
            report["alerts"].append(
                f"⚠ Normalization assumption drift detected: "
                f"{anomaly_rate:.2%} of samples exceed expected L2 norm range."
            )

    return report
