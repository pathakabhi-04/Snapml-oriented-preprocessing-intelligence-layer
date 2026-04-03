import time
import pandas as pd
from simulation.drift_controller import apply_drift


# 🔥 Label Corruption
def corrupt_labels(batch, prob=0.2):
    batch = batch.copy()

    if "fraud_bool" not in batch.columns:
        return batch

    mask = batch.sample(frac=prob).index
    batch.loc[mask, "fraud_bool"] = 1 - batch.loc[mask, "fraud_bool"]

    return batch


def stream_batches(
    data: pd.DataFrame,
    batch_size: int = 100,
    num_batches: int = 20,
    delay: float = 1.0,
    random_seed: int = 42
):
    """
    Simulates a streaming API that yields drifted + degraded batches.
    """

    data = data.copy()
    data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    total_rows = len(data)

    for batch_id in range(num_batches):
        start_idx = (batch_id * batch_size) % total_rows
        end_idx = start_idx + batch_size

        batch = data.iloc[start_idx:end_idx].copy()

        # -----------------------------
        # Apply drift
        # -----------------------------
        drifted_batch, scenario = apply_drift(batch)

        # -----------------------------
        # 🔥 Apply label corruption (NEW)
        # -----------------------------
        if scenario != "none":
            # gradual increase → realistic degradation
            corruption_prob = min(0.05 + batch_id * 0.02, 0.5)
            drifted_batch = corrupt_labels(drifted_batch, prob=corruption_prob)

        # -----------------------------
        # Timestamp (drift injection time)
        # -----------------------------
        injection_time = time.time()

        yield {
            "batch_id": batch_id,
            "data": drifted_batch,
            "scenario": scenario,
            "injection_time": injection_time  # ✅ fixed (no duplicate time.time())
        }

        # Simulate real-time streaming
        time.sleep(delay)