import pandas as pd
from typing import List


def align_to_snapml_order(
    df: pd.DataFrame,
    feature_order: List[str],
) -> pd.DataFrame:
    """
    Align inference data to SnapML training-time column order.

    This function:
    - does NOT infer schema
    - does NOT change values
    - does NOT run preprocessing
    - ONLY reorders columns

    Raises an error if required features are missing.
    """

    missing = [f for f in feature_order if f not in df.columns]
    if missing:
        raise ValueError(
            f"Inference data missing required SnapML features: {missing}"
        )

    return df[feature_order]
