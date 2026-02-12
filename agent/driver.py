# Databricks notebook source
# MAGIC %md
# MAGIC # Agent Driver
# MAGIC
# MAGIC This notebook tests the LangGraph agent with a sample message.
# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Agent

# COMMAND ----------
import mlflow
import os
import sys
import yaml

from mlflow.types.responses import ResponsesAgentRequest
from mlflow.types.agent import ChatContext

# Load config using path relative to this notebook's location
# Add the current directory to the path for local imports
notebook_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if notebook_dir not in sys.path:
    sys.path.insert(0, notebook_dir)

config_path = os.path.join(notebook_dir, "config.yaml")
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# COMMAND ----------
from databricks.sdk import WorkspaceClient

# Configure MLflow for Databricks
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")  # Use Unity Catalog for model registry

# Set up MLflow experiment from config
# Experiment name must be an absolute path in Databricks workspace
experiment_base_name = config['databricks']['mlflow_experiment_name']

# Get current user to construct the experiment path
w = WorkspaceClient()
current_user = w.current_user.me().user_name
experiment_name = f"/Users/{current_user}/{experiment_base_name}"

# Check if experiment exists, create if it doesn't
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    print(f"Creating new MLflow experiment: {experiment_name}")
    mlflow.create_experiment(experiment_name)
else:
    print(f"Using existing MLflow experiment: {experiment_name}")

mlflow.set_experiment(experiment_name)
# COMMAND ----------
# Import the agent from the agent module
from importlib import import_module

# Load the agent module - this will execute the module and create the AGENT
agent_module = import_module("agent")
AGENT = agent_module.AGENT

print(f"Agent loaded: {type(AGENT).__name__}")
# COMMAND ----------
# Create test request with user context
request = ResponsesAgentRequest(
    #input=[{"type": "message", "role": "user", "content": "who is the most powerful anime character from a scientific basis?"}],
    input=[{"type": "message", "role": "user", "content": "what was the last question I asked you?"}],
    custom_inputs={"thread_id": "test-thread-002"},
    context=ChatContext(user_id="test-user-3"),
)
# COMMAND ----------

# MAGIC %md
# MAGIC ### Test predict()

# COMMAND ----------

response = AGENT.predict(request)
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test predict_stream()

# COMMAND ----------

# for event in AGENT.predict_stream(request):
#     print(event)
