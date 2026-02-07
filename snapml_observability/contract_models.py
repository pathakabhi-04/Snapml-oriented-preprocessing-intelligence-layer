# snapml_observability/contract_models.py

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class DataSchema:
    """
    Positional schema as defined by SnapML.
    All indices refer to positions AFTER preprocessing.
    """
    numeric_indices: List[int]
    categorical_indices: List[int]


@dataclass(frozen=True)
class PreprocessingStep:
    """
    One atomic preprocessing operation as emitted by SnapML.
    """
    step_type: str                 # e.g. "Normalizer", "OneHotEncoder"
    params: Dict[str, Any]         # hyperparameters
    data: Dict[str, Any]           # learned state (categories, bins, etc.)
    columns: List[int]             # positional column indices


@dataclass(frozen=True)
class TransformerBlock:
    """
    A logical transformer block in the SnapML pipeline.
    """
    name: str                      # transformer1, transformer2, ...
    steps: List[PreprocessingStep]


@dataclass(frozen=True)
class SnapMLPreprocessingContract:
    """
    Full SnapML preprocessing contract.
    This is the authoritative, parsed representation.
    """
    data_schema: DataSchema
    transformers: List[TransformerBlock]
    remainder: str
