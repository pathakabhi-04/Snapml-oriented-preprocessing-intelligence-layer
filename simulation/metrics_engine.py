import time
from collections import deque
from typing import Optional, Dict

import numpy as np
from sklearn.metrics import accuracy_score


class MetricsEngine:
    """
    Simulates REALISTIC model monitoring:
    - label delay
    - rolling accuracy window
    """

    def __init__(
        self,
        pipeline,
        label_delay: int = 5,
        window_size: int = 5
    ):
        self.pipeline = pipeline

        self.label_delay = label_delay
        self.window_size = window_size

        # Buffers
        self.prediction_buffer = deque()
        self.label_buffer = deque()
        self.eval_window = deque()

    # -----------------------------
    # Add batch (predict only)
    # -----------------------------
    def add_batch(self, batch_id, batch_df):
        timestamp = time.time()

        X = batch_df.drop(columns=["fraud_bool"])
        y_true = batch_df["fraud_bool"]

        y_pred = self.pipeline.predict(X)

        # Store predictions
        self.prediction_buffer.append({
            "batch_id": batch_id,
            "y_pred": y_pred,
            "timestamp": timestamp
        })

        # Store labels separately (simulate delayed arrival)
        self.label_buffer.append({
            "batch_id": batch_id,
            "y_true": y_true.values
        })

    # -----------------------------
    # Evaluate when labels "arrive"
    # -----------------------------
    def evaluate_if_ready(self) -> Optional[Dict]:

        # Wait until enough delayed labels exist
        if len(self.label_buffer) <= self.label_delay:
            return None

        # Simulate label arrival
        label_record = self.label_buffer.popleft()
        batch_id = label_record["batch_id"]
        y_true = label_record["y_true"]

        # Find matching prediction
        pred_record = None
        for record in list(self.prediction_buffer):
            if record["batch_id"] == batch_id:
                pred_record = record
                break

        if pred_record is None:
            return None

        y_pred = pred_record["y_pred"]

        # Add to evaluation window
        self.eval_window.append((y_true, y_pred))

        # Maintain window size
        if len(self.eval_window) > self.window_size:
            self.eval_window.popleft()

        # Only compute when window is full
        if len(self.eval_window) < self.window_size:
            return None

        # Compute rolling accuracy
        all_true = np.concatenate([x[0] for x in self.eval_window])
        all_pred = np.concatenate([x[1] for x in self.eval_window])

        accuracy = accuracy_score(all_true, all_pred)

        return {
            "batch_id": batch_id,
            "accuracy": float(accuracy),
            "timestamp": time.time()
        }