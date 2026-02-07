import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.pipeline import Pipeline

from snapml import LogisticRegression

import joblib


# -----------------------------
# Configuration
# -----------------------------

DATA_PATH = "data/Base.csv"
TARGET_COL = "fraud_bool"
RANDOM_STATE = 42


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


# -----------------------------
# Feature Groups
# -----------------------------
# IMPORTANT:
# These are intentionally explicit.
# No automatic inference. No schema abstraction.

NUMERICAL_FEATURES = [
    "customer_age",
    "income",
    "days_since_request",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "device_fraud_count",
    "proposed_credit_limit",
    "session_length_in_minutes"
]

CATEGORICAL_FEATURES = [
    "payment_type",
    "employment_status",
    "housing_status",
    "device_os",
    "source"
]


# -----------------------------
# Train / Validation Split
# -----------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


# -----------------------------
# Preprocessing Pipeline
# -----------------------------
# THIS IS THE PREPROCESSING CONTRACT.
# Every transformer here MUST be SnapML-exportable.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            # SnapML DOES NOT support StandardScaler.
            # Normalizer IS supported and exportable.
            Normalizer(norm="l2"),
            NUMERICAL_FEATURES,
        ),
        (
            "cat",
            # SnapML requires:
            # - handle_unknown="ignore"
            # - sparse=False (for sklearn 1.1.x)
            OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            ),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
)


# -----------------------------
# SnapML Model
# -----------------------------

model = LogisticRegression(
    max_iter=100,
    penalty="l2",
    regularizer=1.0,
    random_state=RANDOM_STATE,
)


# -----------------------------
# Full Training Pipeline
# -----------------------------
# NOTE:
# This pipeline is used ONLY for training.
# The preprocessing is later exported separately via SnapML.

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model),
    ]
)


# -----------------------------
# Train
# -----------------------------

pipeline.fit(X_train, y_train)


# -----------------------------
# Basic Sanity Check
# -----------------------------
# Accuracy is NOT the goal.
# This is only to confirm the system works.

train_score = pipeline.score(X_train, y_train)
val_score = pipeline.score(X_val, y_val)

print(f"Train Accuracy: {train_score:.4f}")
print(f"Validation Accuracy: {val_score:.4f}")


# -----------------------------
# Persist Training Artifact
# -----------------------------
# This is NOT the production contract.
# The production contract is the SnapML-exported preprocessing JSON.

joblib.dump(pipeline, "snapml_training/trained_pipeline.joblib")
