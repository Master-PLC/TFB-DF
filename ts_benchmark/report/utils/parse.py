import json
import os
import yaml

import pandas as pd

def _parse_model_column(col_name: str):
    if ";" not in col_name:
        return col_name, {}
    model_name, params_str = col_name.split(";", 1)
    params_str = params_str.replace("'", '"')
    try:
        params = json.loads(params_str)
    except (json.JSONDecodeError, ValueError):
        params = {"raw": params_str}
    return model_name, params


def _sanitize_value(v):
    if isinstance(v, float):
        if pd.isna(v):
            return None
        if v == float("inf") or v == float("-inf"):
            return None
    return v


def _yaml_dump(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)