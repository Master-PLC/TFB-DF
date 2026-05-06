# -*- coding: utf-8 -*-

import json
import os

import pandas as pd
from typing import Union, List

from common.constant import ROOT_PATH
from ts_benchmark.evaluation.strategy.constants import FieldNames
from ts_benchmark.recording import load_record_data
from ts_benchmark.report.utils.leaderboard import get_leaderboard
from ts_benchmark.report.utils.parse import _parse_model_column, _sanitize_value, _yaml_dump

# currently we do not support showing or processing artifact columns
# these columns are dropped as soon as data is loaded in order to save memory
ARTIFACT_COLUMNS = [
    FieldNames.ACTUAL_DATA,
    FieldNames.INFERENCE_DATA,
    FieldNames.LOG_INFO,
]

# Columns that are metadata, not model result columns
META_COLUMNS = {"metric_name", "strategy_args"}


def report(report_config: dict, simpler_save_name: bool = False) -> None:
    """
    Generate per-model YAML reports: performance.yaml + config.yaml for each model.

    Parameters:
    - report_config (dict): see report_csv.report for full spec.
    - simpler_save_name (bool): Whether to use simpler file names for the output YAML files.

    Output per model (under save_path):
      - {model_name}.performance.yaml : columns: metric_name, value
      - {model_name}.config.yaml      : flattened key-value pairs
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

    model_cols = [c for c in leaderboard_df.columns if c not in META_COLUMNS]

    save_path = report_config.get("save_path", None)
    if save_path is None:
        save_path = os.path.join(ROOT_PATH, "result")
    file_name = report_config["leaderboard_file_name"]

    for col in model_cols:
        model_name, params = _parse_model_column(col)

        # performance.yaml — metric_name → value only
        performance = {
            row["metric_name"]: _sanitize_value(row[col])
            for _, row in leaderboard_df.iterrows()
        }

        # config.yaml — model info + strategy (write once, not repeated per metric)
        config = {"model_name": model_name}
        strategy_args = json.loads(leaderboard_df.iloc[0]["strategy_args"])
        config.update(params)
        config.update(strategy_args)

        _yaml_dump(os.path.join(save_path, f"{model_name}.{file_name}" if not simpler_save_name else "performance.yaml"), performance)
        _yaml_dump(os.path.join(save_path, f"{model_name}.config.yaml" if not simpler_save_name else "config.yaml"), config)