"""Centralized configuration constants for the agent server.

All shared constants live here so that agent.py, utils_memory.py,
and scripts/ import from a single source of truth.  Values are read
from environment variables where appropriate, with sensible defaults.
"""

import os

# ---------------------------------------------------------------------------
# Lakebase / storage
# ---------------------------------------------------------------------------

LAKEBASE_INSTANCE_NAME = os.environ.get("LAKEBASE_INSTANCE_NAME", "doan-langgraph-memory")

STORE_TABLES = [
    "checkpoint_migrations",
    "checkpoint_writes",
    "checkpoints",
    "checkpoint_blobs",
    "store_migrations",
    "vector_migrations",
    "store",
    "store_vectors",
]

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "databricks"
MLFLOW_REGISTRY_URI = "databricks-uc"
MLFLOW_EXPERIMENT_NAME = "agents-on-apps"

# ---------------------------------------------------------------------------
# Model endpoints
# ---------------------------------------------------------------------------

LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
EMBEDDING_DIMS = 1024

# ---------------------------------------------------------------------------
# App identity
# ---------------------------------------------------------------------------

APP_NAME = "stateful-agent-app"

# ---------------------------------------------------------------------------
# User profile table (Lakebase-synced)
# ---------------------------------------------------------------------------

USER_PROFILES_TABLE = "user_preferences_online"

# ---------------------------------------------------------------------------
# Memory namespace
# ---------------------------------------------------------------------------

MEMORY_NAMESPACE_PREFIX = "user_memories"


def get_memory_namespace(user_id: str) -> tuple[str, str]:
    """Return the Lakebase store namespace tuple for a given user."""
    return (MEMORY_NAMESPACE_PREFIX, user_id.replace(".", "-"))


# ---------------------------------------------------------------------------
# Unity Catalog tool integration
# ---------------------------------------------------------------------------

UC_TOOL_NAMES: list[str] = []

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools to answer questions. "
    "All conversations are automatically saved to long-term memory. "
    "When the user asks about previous conversations, prior questions, or anything "
    "they've told you before, you MUST use the get_user_memory tool to search for "
    "relevant memories before responding."
)
