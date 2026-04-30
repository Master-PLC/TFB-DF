# -*- coding: utf-8 -*-

import json
import os
from typing import Union, List

import pandas as pd
import yaml

from common.constant import ROOT_PATH
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.recording import load_record_data
from ts_benchmark.report.utils.leaderboard import get_leaderboard

# currently we do not support showing or processing artifact columns
# these columns are dropped as soon as data is loaded in order to save memory
ARTIFACT_COLUMNS = [
    FieldNames.ACTUAL_DATA,
    FieldNames.INFERENCE_DATA,
    FieldNames.LOG_INFO,
]

# Columns that are metadata, not model result columns
META_COLUMNS = {"metric_name", "strategy_args"}


def _parse_model_column(col_name: str):
    """
    Parse a model column name like 'iTransformer;{...params...}' into model name and params dict.
    """
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
    """Replace NaN/inf with None for proper YAML null representation."""
    if isinstance(v, float):
        if pd.isna(v):
            return None
        if v == float("inf") or v == float("-inf"):
            return None
    return v


def report(report_config: dict) -> None:
    """
    Generate a YAML report based on specified configuration parameters.

    Parameters:
    - report_config (dict): A dictionary containing the following keys and their respective values:
        - log_files_list (List[str]): A list of file paths for log files.
        - leaderboard_file_name (str): The name for the saved report file (without extension,
          or with .yaml / .yml extension).
        - aggregate_type (str): The aggregation type used when reporting the final results of evaluation metrics.
        - report_metrics (Union[str, List[str]]): The metrics for the report, can be a string or a list of strings.
        - fill_type (str): The type of fill for missing values.
        - null_value_threshold (float): The threshold value for null metrics.

    Raises:
    - ValueError: If no log files are provided.

    Returns:
    - None: The function does not return a value, but generates and saves a report to a YAML file.
    """
    log_files: Union[List[str], pd.DataFrame] = report_config.get("log_files_list")
    if not log_files:
        raise ValueError("No log files to report")

    log_data = (
        log_files
        if isinstance(log_files, pd.DataFrame)
        else load_record_data(log_files, drop_columns=ARTIFACT_COLUMNS)
    )

    leaderboard_df = get_leaderboard(
        log_data,
        report_config["report_metrics"],
        report_config.get("aggregate_type", "mean"),
        report_config.get("fill_type", "mean_value"),
        report_config.get("null_value_threshold", 0.3),
    )

    num_rows = leaderboard_df.shape[0]
    leaderboard_df.insert(0, "strategy_args", [log_data.iloc[0, 1]] * num_rows)

    # Build clean nested YAML structure:
    # Each row: metric_name, strategy_args, models: [{name, params, value}, ...]
    yaml_data = []
    model_cols = [c for c in leaderboard_df.columns if c not in META_COLUMNS]

    for _, row in leaderboard_df.iterrows():
        entry = {
            "metric_name": row["metric_name"],
            "strategy_args": row["strategy_args"],
        }
        models = []
        for col in model_cols:
            model_name, params = _parse_model_column(col)
            value = _sanitize_value(row[col])
            models.append({
                "name": model_name,
                "params": params,
                "value": value,
            })
        entry["models"] = models
        yaml_data.append(entry)

    # Ensure the file name ends with .yaml
    file_name = report_config["leaderboard_file_name"]

    # Save to YAML file
    if report_config.get("save_path", None) is not None:
        save_path = report_config.get("save_path", None)
        output_path = os.path.join(save_path, file_name)
    else:
        output_path = os.path.join(ROOT_PATH, "result", file_name)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
