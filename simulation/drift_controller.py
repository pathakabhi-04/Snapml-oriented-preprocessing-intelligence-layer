import random
import pandas as pd
from simulation.config import CATEGORICAL_FEATURES


# ----------- Drift Injection Functions -----------

def inject_numeric_scale(batch, feature, factor=10):
    batch = batch.copy()
    if feature in batch.columns:
        batch[feature] = batch[feature] * factor
    return batch


def inject_numeric_noise(batch, feature, noise_level=0.5):
    batch = batch.copy()
    if feature in batch.columns:
        noise = batch[feature].std() * noise_level
        batch[feature] = batch[feature] + noise
    return batch


def inject_unseen_category(batch, feature=None):
    batch = batch.copy()

    if feature is None:
        feature = random.choice(CATEGORICAL_FEATURES)

    if feature not in batch.columns:
        return batch

    # 🔥 FIX: handle dtype correctly
    if batch[feature].dtype == "O":
        # string categorical
        batch[feature] = "ZZ_UNKNOWN"
    else:
        # numeric categorical
        batch[feature] = batch[feature].max() + 100

    return batch


def inject_column_shuffle(batch):
    batch = batch.copy()
    cols = list(batch.columns)
    random.shuffle(cols)
    return batch[cols]


def inject_mixed_drift(batch):
    batch = inject_numeric_scale(batch, "income", factor=10)
    batch = inject_unseen_category(batch)  # 🔥 no hardcoding
    return batch


# ----------- Scenario Registry -----------

DRIFT_SCENARIOS = {
    "none": lambda b: b,
    "income_scale": lambda b: inject_numeric_scale(b, "income", 10),
    "income_noise": lambda b: inject_numeric_noise(b, "income", 0.5),
    "unseen_payment": lambda b: inject_unseen_category(b, "payment_type"),
    "column_shuffle": inject_column_shuffle,
    "mixed": inject_mixed_drift,
}


# ----------- Main API -----------

def apply_drift(batch: pd.DataFrame, scenario: str = None):
    if scenario is None:
        scenario = random.choice(list(DRIFT_SCENARIOS.keys()))

    transformed = DRIFT_SCENARIOS[scenario](batch)

    return transformed, scenario