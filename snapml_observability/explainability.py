from typing import List, Dict, Any


def explain_introspection(
    introspection: List[Dict[str, Any]]
) -> List[str]:
    """
    Convert SnapML preprocessing introspection into
    human-readable explanations.

    This function explains WHAT preprocessing is configured to do,
    not WHETHER it is correct or beneficial.

    No data is accessed. No preprocessing is executed.
    """

    explanations: List[str] = []

    for block in introspection:
        transformer_name = block["transformer_name"]
        steps = block["steps"]

        if not steps:
            explanations.append(
                f"{transformer_name}: No preprocessing steps applied."
            )
            continue

        for step in steps:
            step_type = step["step_type"]
            cols = step["applies_to_columns"]
            params = step["parameters"]
            learned = step["learned_state"]

            if step_type == "Normalizer":
                explanations.append(
                    f"Numerical features at positions {cols} are normalized "
                    f"using {params.get('norm')} normalization. "
                    f"This normalization is applied consistently at inference time."
                )

            elif step_type == "OneHotEncoder":
                explanations.append(
                    f"Categorical features at positions {cols} are one-hot encoded. "
                    f"Categories are fixed from training time. "
                    f"Any unseen category at inference will be ignored, "
                    f"producing a zero-vector for that feature."
                )

            else:
                explanations.append(
                    f"{step_type} is applied to columns {cols} "
                    f"with parameters {params}."
                )

    return explanations
