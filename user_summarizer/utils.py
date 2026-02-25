"""Shared utilities for user_summarizer notebooks."""

import os
from pathlib import Path

import yaml
from pyspark.sql import SparkSession


def get_spark() -> SparkSession:
    """Return the active SparkSession — works on Databricks and locally via Databricks Connect."""
    if "spark" in globals():
        return globals()["spark"]
    from databricks.connect import DatabricksSession

    return DatabricksSession.builder.getOrCreate()


def load_config() -> dict:
    """Load config.yaml from the user_summarizer directory."""
    config_path = (
        Path(__file__).parent / "config.yaml"
        if "__file__" in dir()
        else Path(os.getcwd()) / "config.yaml"
    )
    with open(config_path) as f:
        return yaml.safe_load(f)
