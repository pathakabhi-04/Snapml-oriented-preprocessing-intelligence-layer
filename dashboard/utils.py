import pandas as pd
from pathlib import Path

from snapml_observability.contract_loader import load_snapml_contract
from snapml_observability.contract_parser import parse_snapml_contract
from snapml_observability.drift_detection import detect_preprocessing_drift


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DATA_PATH = Path("data/Base.csv")
CONTRACT_PATH = Path("snapml_preprocessing.json")


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

def load_dataset():
    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# Load raw contract (for dashboard visualization)
# ---------------------------------------------------------

def load_contract():
    return load_snapml_contract(CONTRACT_PATH)


# ---------------------------------------------------------
# Load parsed contract (for drift detection engine)
# ---------------------------------------------------------

def load_parsed_contract():
    contract = parse_snapml_contract(CONTRACT_PATH)
    return contract


# ---------------------------------------------------------
# Run drift detection
# ---------------------------------------------------------

import json


FEATURE_ORDER_PATH = Path("snapml_training/feature_order.json")


def load_feature_order():
    with open(FEATURE_ORDER_PATH, "r") as f:
        return json.load(f)


def align_columns(df):
    """
    Align dataframe columns to training feature order.
    """
    feature_order = load_feature_order()

    df = df.copy()

    # ensure correct order
    df = df[feature_order]

    return df


def run_drift_check(df):

    contract = load_parsed_contract()

    df = align_columns(df)



    report = detect_preprocessing_drift(
        contract,
        df
    )

    return report