# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ts_benchmark.report import report_dash, report_csv, report_yaml


def report(report_config: dict, report_method: str = "csv", simpler_save_name: bool = False) -> None:
    if report_method == "dash":
        report_dash.report(report_config)
    elif report_method == "csv":
        report_csv.report(report_config, simpler_save_name=simpler_save_name)
    elif report_method == "yaml":
        report_yaml.report(report_config, simpler_save_name=simpler_save_name)
    else:
        raise ValueError(f"Unknown report method {report_method}")
