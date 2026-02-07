from typing import Dict, List, Any

from snapml_observability.contract_models import (
    SnapMLPreprocessingContract,
)


def introspect_contract(
    contract: SnapMLPreprocessingContract,
) -> List[Dict[str, Any]]:
    """
    Introspect a SnapML preprocessing contract.

    This function produces a structured, ordered, human-inspectable
    representation of preprocessing behavior WITHOUT interpretation,
    assumptions, or data access.
    """

    introspection_output: List[Dict[str, Any]] = []

    for block in contract.transformers:
        block_description = {
            "transformer_name": block.name,
            "steps": [],
        }

        for step in block.steps:
            step_description = {
                "step_type": step.step_type,
                "applies_to_columns": step.columns,
                "parameters": step.params,
                "learned_state": list(step.data.keys()),
            }

            block_description["steps"].append(step_description)

        introspection_output.append(block_description)

    return introspection_output
