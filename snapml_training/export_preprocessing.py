import os
import joblib

from snapml import export_preprocessing_pipeline


# -----------------------------
# Paths
# -----------------------------

MODEL_PIPELINE_PATH = "snapml_training/trained_pipeline.joblib"
EXPORT_PATH = "snapml_preprocessing.json"


# -----------------------------
# Load Trained Pipeline
# -----------------------------
# IMPORTANT:
# This pipeline is ONLY used here to extract the preprocessing
# and export it via SnapML.
# It must not be used anywhere else.

pipeline = joblib.load(MODEL_PIPELINE_PATH)

preprocessing_pipeline = pipeline.named_steps["preprocessing"]


# -----------------------------
# Export SnapML Preprocessing
# -----------------------------
# This is the SINGLE source of truth for downstream systems.

export_preprocessing_pipeline(
    preprocessing_pipeline,
    EXPORT_PATH
)

print(f"SnapML preprocessing exported to: {EXPORT_PATH}")
