# Databricks notebook source
# MAGIC %md
# MAGIC # Summarize User Profiles
# MAGIC
# MAGIC Reads aggregated user conversation threads from step 02, then uses `ai_query`
# MAGIC to extract key interests, preferences, and behavioral patterns for each user.
# MAGIC Writes the resulting user profiles to the configured output table.

# COMMAND ----------

from pyspark.sql import functions as F

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

uc_cfg = config["unity_catalog"]
INPUT_TABLE = f"{uc_cfg['catalog']}.{uc_cfg['schema']}.{uc_cfg['aggregated_threads_table']}"
OUTPUT_TABLE = f"{uc_cfg['catalog']}.{uc_cfg['schema']}.{uc_cfg['user_profiles_table']}"

summarizer_cfg = config["summarizer"]
LLM_ENDPOINT = summarizer_cfg["llm_endpoint_name"]
SUMMARIZER_PROMPT = summarizer_cfg["prompt"].strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read aggregated user threads

# COMMAND ----------

users_df = spark.table(INPUT_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summarize each user with ai_query

# COMMAND ----------

import json

escaped_prompt = SUMMARIZER_PROMPT.replace("'", "\\'")

RESPONSE_FORMAT = json.dumps({
    "type": "json_schema",
    "json_schema": {
        "name": "user_profile",
        "schema": {
            "type": "object",
            "properties": {
                "interests": {"type": "array", "items": {"type": "string"}},
                "preferences": {"type": "array", "items": {"type": "string"}},
                "behavioral_notes": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
            },
        },
        "strict": True,
    },
})

# Escape single quotes in the JSON schema for SQL embedding
escaped_response_format = RESPONSE_FORMAT.replace("'", "\\'")

ai_query_sql = (
    f"ai_query('{LLM_ENDPOINT}', "
    f"CONCAT('{escaped_prompt}', '\\n\\nConversation history:\\n', all_conversations), "
    f"responseFormat => '{escaped_response_format}')"
)

profiles_df = (
    users_df
    .withColumn("profile", F.from_json(F.expr(ai_query_sql), "interests ARRAY<STRING>, preferences ARRAY<STRING>, behavioral_notes ARRAY<STRING>, summary STRING"))
    .select(
        F.col("prefix").alias("user_id"),
        "profile",
        "profile.interests",
        "profile.preferences",
        "profile.behavioral_notes",
        "profile.summary",
        "total_messages",
        "thread_count",
        "first_conversation_at",
        "last_conversation_at",
        F.current_timestamp().alias("summarized_at"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write user profiles

# COMMAND ----------

(
    profiles_df
    .write
    .mode("overwrite")
    .option("mergeSchema", True)
    .saveAsTable(OUTPUT_TABLE)
)

print(f"Wrote user profiles to {OUTPUT_TABLE}")

# COMMAND ----------

result_df = spark.table(OUTPUT_TABLE)
if "display" in dir():
    display(result_df)
else:
    result_df.show(truncate=False)
