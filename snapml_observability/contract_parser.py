# snapml_observability/contract_parser.py

from typing import List

from snapml_observability.contract_loader import load_snapml_contract
from snapml_observability.contract_models import (
    DataSchema,
    PreprocessingStep,
    TransformerBlock,
    SnapMLPreprocessingContract,
)


def parse_snapml_contract(path: str) -> SnapMLPreprocessingContract:
    """
    Parse a SnapML preprocessing JSON into a typed contract.

    This function performs a structural, lossless translation.
    No interpretation or enrichment is performed.
    """

    raw = load_snapml_contract(path)

    # -------------------------
    # Parse data schema
    # -------------------------
    schema = DataSchema(
        numeric_indices=raw["data_schema"]["num_indices"],
        categorical_indices=raw["data_schema"]["cat_indices"],
    )

    # -------------------------
    # Parse transformers
    # -------------------------
    transformer_blocks: List[TransformerBlock] = []

    for transformer_name, steps in raw["transformers"].items():
        parsed_steps: List[PreprocessingStep] = []

        for step in steps:
            parsed_steps.append(
                PreprocessingStep(
                    step_type=step["type"],
                    params=step.get("params", {}),
                    data=step.get("data", {}),
                    columns=step["columns"],
                )
            )

        transformer_blocks.append(
            TransformerBlock(
                name=transformer_name,
                steps=parsed_steps,
            )
        )

    # -------------------------
    # Assemble full contract
    # -------------------------
    contract = SnapMLPreprocessingContract(
        data_schema=schema,
        transformers=transformer_blocks,
        remainder=raw["remainder"],
    )

    return contract
