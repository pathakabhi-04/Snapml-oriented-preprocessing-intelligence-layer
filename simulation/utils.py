import json


def load_numeric_baseline(path="snapml_training/numeric_baseline.json"):
    with open(path) as f:
        return json.load(f)