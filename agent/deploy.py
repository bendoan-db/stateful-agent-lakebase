# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Agent to Model Serving
# MAGIC
# MAGIC This notebook logs the agent to MLflow, registers it in Unity Catalog,
# MAGIC and deploys it to a Databricks Model Serving endpoint.

# COMMAND ----------

import os
import sys
import yaml

import mlflow
from databricks.sdk import WorkspaceClient

# Add the current directory to the path for local imports
notebook_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if notebook_dir not in sys.path:
    sys.path.insert(0, notebook_dir)

config_path = os.path.join(notebook_dir, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

w = WorkspaceClient()
current_user = w.current_user.me().user_name

# MLflow experiment
experiment_base_name = config["databricks"]["mlflow_experiment_name"]
experiment_name = f"/Users/{current_user}/{experiment_base_name}"

uc_config = config["unity_catalog"]
REGISTERED_MODEL_NAME = f"{uc_config['catalog']}.{uc_config['schema']}.{uc_config['model_name']}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up MLflow

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    print(f"Creating new MLflow experiment: {experiment_name}")
    mlflow.create_experiment(experiment_name)
else:
    print(f"Using existing MLflow experiment: {experiment_name}")

mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the Agent Model

# COMMAND ----------

pip_requirements = []
requirements_path = os.path.join(notebook_dir, "..", "requirements.txt")
if os.path.exists(requirements_path):
    with open(requirements_path) as f:
        pip_requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

resources = [
    # Declare the model serving endpoint the agent uses for LLM calls
    mlflow.models.resources.DatabricksServingEndpoint(
        endpoint_name=config["llm"]["endpoint_name"],
    ),
    # Declare the embedding endpoint used for semantic search
    mlflow.models.resources.DatabricksServingEndpoint(
        endpoint_name=config["embeddings"]["endpoint"],
    ),
]

# Add any UC function tools as resources
for uc_fn in config["tools"].get("uc_function_names", []):
    if uc_fn:
        resources.append(mlflow.models.resources.DatabricksUCFunction(function_name=uc_fn))

agent_path = os.path.join(notebook_dir, "agent.py")
agent_config_path = os.path.join(notebook_dir, "config.yaml")

input_example = {
    "input": [{"type": "message", "role": "user", "content": "Hello"}],
    "custom_inputs": {"thread_id": "example-thread-001"},
    "context": {"user_id": "example-user-001"},
}

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        name="stateful_agent",
        python_model=agent_path,
        model_config=agent_config_path,
        pip_requirements=pip_requirements,
        resources=resources,
        input_example=input_example,
        registered_model_name=REGISTERED_MODEL_NAME,
    )

print(f"Model logged: {model_info.model_uri}")
print(f"Registered as: {REGISTERED_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy to Model Serving

# COMMAND ----------

from databricks.agents import deploy

deployment = deploy(
    model_name=REGISTERED_MODEL_NAME,
    model_version=model_info.registered_model_version,
)

print(f"Endpoint name : {deployment.endpoint_name}")
print(f"Endpoint URL  : {deployment.endpoint_url}")
print(f"Query endpoint: {deployment.query_endpoint}")
