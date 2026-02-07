# snapml_observability/contract_loader.py

import json
from pathlib import Path


REQUIRED_TOP_LEVEL_KEYS = {
    "data_schema",
    "transformers",
    "remainder",
}


def load_snapml_contract(path: str | Path) -> dict:
    """
    Load and validate a SnapML preprocessing contract.

    This function treats the SnapML JSON as a frozen, authoritative artifact.
    No assumptions beyond structural validation are made.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"SnapML contract not found: {path}")

    with path.open("r") as f:
        contract = json.load(f)

    missing = REQUIRED_TOP_LEVEL_KEYS - contract.keys()
    if missing:
        raise ValueError(
            f"Invalid SnapML contract. Missing keys: {missing}"
        )

    return contract
