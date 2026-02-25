# Databricks notebook source
# MAGIC %md
# MAGIC # Aggregate Conversations from Agent Memory
# MAGIC
# MAGIC Reads messages from the store table configured in `config.yaml`, groups them by
# MAGIC user (prefix) and thread (extracted from key), then concatenates messages in
# MAGIC chronological order into a single conversation string per thread. Writes the
# MAGIC results to the configured Unity Catalog output table.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

from user_summarizer.utils import get_spark, load_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment setup

# COMMAND ----------

spark = get_spark()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load configuration

# COMMAND ----------

config = load_config()

store_cfg = config["store_table"]
STORE_TABLE = f"{store_cfg['catalog']}.{store_cfg['schema']}.{store_cfg['name']}"

uc_cfg = config["unity_catalog"]
OUTPUT_TABLE = f"{uc_cfg['catalog']}.{uc_cfg['schema']}.{uc_cfg['aggregated_conversations_table']}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read from store table

# COMMAND ----------

store_df = spark.table(STORE_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract thread ID and parse message values

# COMMAND ----------

# The key column has format "{thread_id}-{role}" where role is "user" or "assistant".
# Extract the thread_id by stripping the last "-user" or "-assistant" suffix.
parsed_df = (
    store_df
    .withColumn(
        "thread_id",
        F.regexp_replace(F.col("key"), "-(user|assistant)$", ""),
    )
    .withColumn("role", F.get_json_object(F.col("value"), "$.role"))
    .withColumn("content", F.get_json_object(F.col("value"), "$.content"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregate conversations by user and thread

# COMMAND ----------

# Build a formatted message line: "[timestamp] role: content"
messages_df = (
    parsed_df
    .withColumn(
        "message_line",
        F.concat(
            F.lit("["),
            F.col("updated_at").cast("string"),
            F.lit("] "),
            F.col("role"),
            F.lit(": "),
            F.col("content"),
        ),
    )
)

# Order by updated_at within each (prefix, thread_id) group, then concatenate
window = Window.partitionBy("prefix", "thread_id").orderBy("updated_at")

ordered_df = messages_df.withColumn("row_num", F.row_number().over(window))

conversations_df = (
    ordered_df
    .groupBy("prefix", "thread_id")
    .agg(
        F.concat_ws(
            "\n",
            F.sort_array(
                F.collect_list(
                    F.struct(F.col("row_num"), F.col("message_line"))
                ),
            ).getField("message_line"),
        ).alias("conversation"),
        F.min("updated_at").alias("started_at"),
        F.max("updated_at").alias("ended_at"),
        F.count("*").alias("message_count"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to output table

# COMMAND ----------

(
    conversations_df
    .orderBy("prefix", "started_at")
    .write
    .mode("overwrite")
    .saveAsTable(OUTPUT_TABLE)
)

print(f"Wrote aggregated conversations to {OUTPUT_TABLE}")

# COMMAND ----------

result_df = spark.table(OUTPUT_TABLE)
if "display" in dir():
    display(result_df)
else:
    result_df.show(truncate=False)
