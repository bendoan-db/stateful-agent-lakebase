"""Async memory tools, profile lookup, and message persistence helpers."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import mlflow
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from databricks_ai_bridge.lakebase import LakebaseClient
from mlflow.types.responses import ResponsesAgentRequest

from agent_server.config import (
    LAKEBASE_INSTANCE_NAME,
    USER_PROFILES_TABLE,
    get_memory_namespace,
)

logger = logging.getLogger(__name__)


def get_user_id(request: ResponsesAgentRequest) -> Optional[str]:
    """Extract user_id from custom_inputs or request context."""
    ci = dict(request.custom_inputs or {})
    if ci.get("user_id"):
        return ci["user_id"]
    if request.context and getattr(request.context, "user_id", None):
        return request.context.user_id
    return None


# ---------------------------------------------------------------------------
# Memory tools (returned as a list; store & user_id via config["configurable"])
# ---------------------------------------------------------------------------


def memory_tools():
    """Return the three async memory tool functions."""

    @tool
    async def get_user_memory(query: str, config: RunnableConfig) -> str:
        """Search for relevant information about the user from long-term memory using semantic search via vector embeddings.

        Use this tool to retrieve previously saved information about the user,
        such as their preferences, facts they've shared, or other personal details.
        """
        store = config.get("configurable", {}).get("store")
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id or not store:
            return "Memory not available - no user_id provided."

        namespace = get_memory_namespace(user_id)
        results = await store.asearch(namespace, query=query, limit=5)

        if not results:
            return "No memories found for this user."

        memory_items = []
        for item in results:
            memory_items.append(f"- [{item.key}]: {json.dumps(item.value)}")

        return f"Found {len(results)} relevant memories (ranked by semantic similarity):\n" + "\n".join(memory_items)

    @tool
    async def save_user_memory(key: str, value: str, config: RunnableConfig) -> str:
        """Save a piece of information to the user's long-term memory.

        Use this tool to remember something the user has told you,
        such as their preferences, facts about them, or other personal details.

        Args:
            key: A short descriptive key for the memory (e.g., "favorite_color", "job_title")
            value: The information to remember
        """
        store = config.get("configurable", {}).get("store")
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id or not store:
            return "Cannot save memory - no user_id provided."

        namespace = get_memory_namespace(user_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        await store.aput(namespace, key, {"value": value, "timestamp": timestamp})
        return f"Successfully saved memory with key '{key}' for user."

    @tool
    async def delete_user_memory(memory_key: str, config: RunnableConfig) -> str:
        """Delete a specific memory from the user's long-term memory.

        Use this tool when the user asks you to forget something or remove
        a piece of information from their memory.

        Args:
            memory_key: The key of the memory to delete (e.g., "preferences", "likes", "background_info")
        """
        store = config.get("configurable", {}).get("store")
        user_id = config.get("configurable", {}).get("user_id")
        if not user_id or not store:
            return "Cannot delete memory - no user_id provided."

        namespace = get_memory_namespace(user_id)
        await store.adelete(namespace, memory_key)
        return f"Successfully deleted memory with key '{memory_key}' for user."

    return [get_user_memory, save_user_memory, delete_user_memory]


# ---------------------------------------------------------------------------
# Message persistence
# ---------------------------------------------------------------------------


async def save_message_to_store(store, user_id: str, thread_id: str, role: str, content: str) -> None:
    """Save a message to the Lakebase store for automatic conversation persistence."""
    namespace = get_memory_namespace(user_id)
    key = f"{thread_id}-{role}"
    timestamp = datetime.now(timezone.utc).isoformat()
    value = {"role": role, "content": content, "timestamp": timestamp}
    try:
        await store.aput(namespace, key, value)
        logger.info(f"Auto-saved {role} message for user {user_id} with key {key}")
    except Exception as e:
        logger.error(f"Failed to auto-save {role} message for user {user_id}: {e}")


# ---------------------------------------------------------------------------
# Profile lookup
# ---------------------------------------------------------------------------


async def lookup_user_profile(user_id: str, workspace_client=None) -> Optional[dict]:
    """Look up user profile from the Lakebase-synced Postgres table.

    Wraps the synchronous ``LakebaseClient`` query in ``asyncio.to_thread``
    so it doesn't block the event loop.
    """
    ns = get_memory_namespace(user_id)
    store_user_id = f"{ns[0]}.{ns[1]}"

    def _sync_lookup():
        with mlflow.start_span(name="lookup_user_profile", span_type="RETRIEVER") as span:
            span.set_inputs({"user_id": user_id, "store_user_id": store_user_id})
            try:
                client = LakebaseClient(instance_name=LAKEBASE_INSTANCE_NAME, workspace_client=workspace_client)
                rows = client.execute(
                    f"SELECT summary, interests, preferences, behavioral_notes "
                    f"FROM {USER_PROFILES_TABLE} WHERE user_id = '{store_user_id}' LIMIT 1",
                )
                client.close()

                if not rows:
                    logger.info(f"No profile found for user_id={user_id}")
                    span.set_outputs({"profile": None})
                    return None

                profile = {k: str(v) if v is not None else None for k, v in dict(rows[0]).items()}
                logger.info(f"Loaded profile for user_id={user_id}")
                span.set_outputs({"profile": profile})
                return profile
            except Exception as e:
                logger.error(f"Error looking up user profile for {user_id}: {e}")
                span.set_status("ERROR")
                span.set_outputs({"error": str(e)})
                return None

    return await asyncio.to_thread(_sync_lookup)


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


@mlflow.trace(name="build_system_prompt")
def build_system_prompt(base_prompt: str, profile: Optional[dict] = None) -> str:
    """Build the system prompt, optionally augmented with user profile context."""
    if not profile:
        logger.info("No profile found.")
        return base_prompt

    profile_section = "\n\n## User Profile Context\n"
    if profile.get("summary"):
        profile_section += f"**Summary:** {profile['summary']}\n"
    if profile.get("interests"):
        profile_section += f"**Interests:** {profile['interests']}\n"
    if profile.get("preferences"):
        profile_section += f"**Preferences:** {profile['preferences']}\n"
    if profile.get("behavioral_notes"):
        profile_section += f"**Behavioral Notes:** {profile['behavioral_notes']}\n"
    profile_section += (
        "\nUse this context to personalize your responses. "
        "Do not volunteer this information unprompted — use it implicitly to be more helpful."
    )

    return base_prompt + profile_section
