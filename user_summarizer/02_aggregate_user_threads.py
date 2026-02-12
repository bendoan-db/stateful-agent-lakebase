# Databricks notebook source
# MAGIC %md
# MAGIC # Aggregate User Threads
# MAGIC
# MAGIC Reads per-thread conversations from the output of step 01, then aggregates all
# MAGIC threads for each user into a single text field with thread markers separating
# MAGIC each conversation. Writes results to the `user_conversations_aggregated` table.

# COMMAND ----------

import os
from pathlib import Path

import yaml
from pyspark.sql import SparkSession, functions as F
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment setup

# COMMAND ----------

def get_spark() -> SparkSession:
    """Return the active SparkSession — works on Databricks and locally via Databricks Connect."""
    if "spark" in globals():
        return globals()["spark"]
    from databricks.connect import DatabricksSession
    return DatabricksSession.builder.getOrCreate()

spark = get_spark()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load configuration

# COMMAND ----------

config_path = Path(__file__).parent / "config.yaml" if "__file__" in dir() else Path(os.getcwd()) / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

uc_cfg = config["unity_catalog"]
INPUT_TABLE = f"{uc_cfg['catalog']}.{uc_cfg['schema']}.{uc_cfg['aggregated_conversations_table']}"
OUTPUT_TABLE = f"{uc_cfg['catalog']}.{uc_cfg['schema']}.{uc_cfg['aggregated_threads_table']}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read per-thread conversations

# COMMAND ----------

threads_df = spark.table(INPUT_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate all threads per user

# COMMAND ----------

THREAD_SEPARATOR = "--- THREAD {thread_num}: {thread_id} (started {started_at}) ---"

# Build a formatted block for each thread: marker + conversation text
thread_block_df = (
    threads_df
    .withColumn(
        "thread_rank",
        F.row_number().over(
            Window.partitionBy("prefix").orderBy("started_at")
        ),
    )
    .withColumn(
        "thread_block",
        F.concat(
            F.lit("\n"),
            F.format_string(
                "--- THREAD %d: %s (started %s) ---",
                F.col("thread_rank"),
                F.col("thread_id"),
                F.col("started_at").cast("string"),
            ),
            F.lit("\n"),
            F.col("conversation"),
            F.lit("\n"),
            F.lit("--- END THREAD ---"),
        ),
    )
)

# Collect threads in chronological order per user and concatenate
user_conversations_df = (
    thread_block_df
    .groupBy("prefix")
    .agg(
        F.concat_ws(
            "\n",
            F.sort_array(
                F.collect_list(
                    F.struct(F.col("thread_rank"), F.col("thread_block"))
                ),
            ).getField("thread_block"),
        ).alias("all_conversations"),
        F.sum("message_count").alias("total_messages"),
        F.count("*").alias("thread_count"),
        F.min("started_at").alias("first_conversation_at"),
        F.max("ended_at").alias("last_conversation_at"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to output table

# COMMAND ----------

(
    user_conversations_df
    .orderBy("prefix")
    .write
    .mode("overwrite")
    .saveAsTable(OUTPUT_TABLE)
)

print(f"Wrote aggregated user threads to {OUTPUT_TABLE}")

# COMMAND ----------

result_df = spark.table(OUTPUT_TABLE)
if "display" in dir():
    display(result_df)
else:
    result_df.show(truncate=False)
