"""Stream processing and auth helpers for the AgentServer."""

import os

from databricks.sdk import WorkspaceClient
from langchain_core.messages import AIMessageChunk
from mlflow.types.responses import (
    ResponsesAgentStreamEvent,
    create_text_delta,
    output_to_responses_items_stream,
)


def get_user_workspace_client() -> WorkspaceClient:
    """Create a WorkspaceClient using the forwarded user token (on-behalf-of-user auth).

    The ``x-forwarded-access-token`` header is injected by the Databricks Apps
    proxy and carries the end-user's OAuth token.
    """
    from mlflow.genai.agent_server.utils import get_request_headers

    headers = get_request_headers()
    user_token = headers.get("x-forwarded-access-token")
    host = get_databricks_host_from_env()
    return WorkspaceClient(host=host, token=user_token)


def get_databricks_host_from_env() -> str:
    """Return the Databricks workspace host URL from environment variables."""
    for key in ("DATABRICKS_HOST", "DB_HOST"):
        host = os.environ.get(key)
        if host:
            return host
    # Fall back to workspace client's default config
    return WorkspaceClient().config.host


async def process_agent_astream_events(astream):
    """Convert LangGraph async stream events into ``ResponsesAgentStreamEvent`` items.

    This async generator handles both ``updates`` (complete node outputs) and
    ``messages`` (incremental token chunks) produced by
    ``graph.astream(..., stream_mode=["updates", "messages"])``.
    """
    async for event in astream:
        kind = event[0]
        payload = event[1]

        if kind == "updates":
            for node_data in payload.values():
                if node_data and len(node_data.get("messages", [])) > 0:
                    for stream_event in output_to_responses_items_stream(
                        node_data["messages"]
                    ):
                        yield stream_event

        elif kind == "messages":
            try:
                chunk = payload[0]
                if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                    yield ResponsesAgentStreamEvent(
                        **create_text_delta(delta=content, item_id=chunk.id),
                    )
            except Exception:
                pass
