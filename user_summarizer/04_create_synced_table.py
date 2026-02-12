# Databricks notebook source
# MAGIC %md
# MAGIC # Create Synced Table for User Preferences
# MAGIC
# MAGIC Creates a Lakebase synced table from the `user_preferences` Unity Catalog table
# MAGIC produced in step 03. This enables low-latency reads of user profiles for the
# MAGIC agent at serving time.

# COMMAND ----------

import os
from pathlib import Path

import yaml
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession
from databricks.sdk.service.database import (
    NewPipelineSpec,
    SyncedDatabaseTable,
    SyncedTableSchedulingPolicy,
    SyncedTableSpec,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load configuration

# COMMAND ----------

config_path = Path(__file__).parent / "config.yaml" if "__file__" in dir() else Path(os.getcwd()) / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

uc_cfg = config["unity_catalog"]
SOURCE_TABLE = f"{uc_cfg['catalog']}.{uc_cfg['schema']}.{uc_cfg['user_profiles_table']}"

lb_cfg = config["lakebase"]
SYNCED_TABLE = f"{lb_cfg['database_catalog']}.{lb_cfg['database_schema']}.{lb_cfg['synced_table_name']}"
PRIMARY_KEY_COLUMNS = lb_cfg["primary_key_columns"]

print(f"Source table : {SOURCE_TABLE}")
print(f"Synced table : {SYNCED_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable Change Data Feed on source table

# COMMAND ----------

def get_spark() -> SparkSession:
    """Return the active SparkSession — works on Databricks and locally via Databricks Connect."""
    if "spark" in globals():
        return globals()["spark"]
    from databricks.connect import DatabricksSession
    return DatabricksSession.builder.getOrCreate()

spark = get_spark()

spark.sql(f"ALTER TABLE {SOURCE_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"Enabled Change Data Feed on {SOURCE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create synced table

# COMMAND ----------

w = WorkspaceClient()

synced_table = w.database.create_synced_database_table(
    SyncedDatabaseTable(
        name=SYNCED_TABLE,
        spec=SyncedTableSpec(
            source_table_full_name=SOURCE_TABLE,
            primary_key_columns=PRIMARY_KEY_COLUMNS,
            scheduling_policy=SyncedTableSchedulingPolicy.TRIGGERED,
            new_pipeline_spec=NewPipelineSpec(
                storage_catalog=uc_cfg["catalog"],
                storage_schema=uc_cfg["schema"],
            ),
        ),
    )
)

print(f"Created synced table: {synced_table.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check sync status

# COMMAND ----------

status = w.database.get_synced_database_table(name=SYNCED_TABLE)
print(f"Synced table status: {status.data_synchronization_status.detailed_state}")
