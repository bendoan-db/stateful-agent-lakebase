"""Core LangGraph agent with @invoke/@stream decorators for AgentServer."""

import asyncio
import logging
import os
import uuid
from typing import Annotated, Any, Optional, Sequence, TypedDict

import mlflow
from databricks_langchain import (
    AsyncDatabricksStore,
    ChatDatabricks,
    UCFunctionToolkit,
)
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    to_chat_completions_input,
)

from agent_server.utils import get_user_workspace_client, process_agent_astream_events
from agent_server.utils_memory import (
    build_system_prompt,
    get_user_id,
    lookup_user_profile,
    memory_tools,
    save_message_to_store,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LAKEBASE_INSTANCE_NAME = os.environ.get("LAKEBASE_INSTANCE_NAME", "doan-langgraph-memory")
LLM_ENDPOINT_NAME = "databricks-claude-sonnet-4-5"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
EMBEDDING_DIMS = 1024

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools to answer questions. "
    "All conversations are automatically saved to long-term memory. "
    "When the user asks about previous conversations, prior questions, or anything "
    "they've told you before, you MUST use the get_user_memory tool to search for "
    "relevant memories before responding."
)

UC_TOOL_NAMES: list[str] = []

# ---------------------------------------------------------------------------
# Static tools (created once at module level)
# ---------------------------------------------------------------------------

_static_tools = []

if UC_TOOL_NAMES:
    uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
    _static_tools.extend(uc_toolkit.tools)

from langchain_community.tools import DuckDuckGoSearchResults

_static_tools.append(DuckDuckGoSearchResults())

# ---------------------------------------------------------------------------
# MLflow autolog
# ---------------------------------------------------------------------------

mlflow.langchain.autolog()

# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]
    user_id: Optional[str]


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

_memory_tools = memory_tools()

# ---------------------------------------------------------------------------
# One-time store setup (uses SP credentials which have CREATE permission)
# ---------------------------------------------------------------------------

_store_setup_done = False
_store_setup_lock = asyncio.Lock()


async def _ensure_store_setup():
    """Run store.setup() once using the app's default (SP) credentials."""
    global _store_setup_done
    if _store_setup_done:
        return
    async with _store_setup_lock:
        if _store_setup_done:
            return
        async with AsyncDatabricksStore(
            instance_name=LAKEBASE_INSTANCE_NAME,
            embedding_endpoint=EMBEDDING_ENDPOINT,
            embedding_dims=EMBEDDING_DIMS,
        ) as store:
            await store.setup()
        _store_setup_done = True
        logger.info("Store tables initialized (one-time setup)")


def _create_graph(system_prompt: str):
    """Create the LangGraph workflow.

    ``store``, ``user_id``, and ``thread_id`` are read from
    ``config["configurable"]`` at runtime so the graph itself is stateless.
    """
    model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
    all_tools = _static_tools + _memory_tools
    model_with_tools = model.bind_tools(all_tools) if all_tools else model

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])

    model_runnable = preprocessor | model_with_tools

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "end"

    async def call_model(state: AgentState, config: RunnableConfig):
        response = await model_runnable.ainvoke(state, config)
        return {"messages": [response]}

    async def save_messages(state: AgentState, config: RunnableConfig):
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id")
        thread_id = configurable.get("thread_id")
        store = configurable.get("store")
        if user_id and thread_id and store:
            messages = state["messages"]
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if content:
                        await save_message_to_store(store, user_id, thread_id, "assistant", content)
                    break
        return {}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("save_messages", save_messages)

    if all_tools:
        workflow.add_node("tools", ToolNode(all_tools))
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"continue": "tools", "end": "save_messages"},
        )
        workflow.add_edge("tools", "agent")
    else:
        workflow.add_edge("agent", "save_messages")

    workflow.add_edge("save_messages", END)
    workflow.set_entry_point("agent")

    return workflow.compile()


# ---------------------------------------------------------------------------
# @invoke — non-streaming entry point
# ---------------------------------------------------------------------------


@invoke()
async def invoke_agent(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Non-streaming prediction — delegates to streaming and collects results."""
    outputs = []
    async for event in stream_agent(request):
        if event.type == "response.output_item.done":
            outputs.append(event.item)
    return ResponsesAgentResponse(output=outputs)


# ---------------------------------------------------------------------------
# @stream — streaming entry point
# ---------------------------------------------------------------------------


@stream()
async def stream_agent(request: ResponsesAgentRequest):
    """Streaming prediction — main entry point for the AgentServer."""
    user_id = get_user_id(request)

    if not user_id:
        logger.error("Cannot store or retrieve memories without a user_id.")

    # Thread management
    ci = dict(request.custom_inputs or {})
    thread_id = ci.get("thread_id")
    if not thread_id:
        thread_id = uuid.uuid4().hex
        logger.info(f"No thread_id provided, generated: {thread_id}")

    # Convert input messages
    cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])

    # Ensure store tables exist (one-time, uses SP credentials with CREATE permission)
    await _ensure_store_setup()

    # Authenticate as the end user (forwarded OAuth token) or fall back to local creds
    user_client = get_user_workspace_client()

    # Open an async store connection for the lifetime of this request
    async with AsyncDatabricksStore(
        instance_name=LAKEBASE_INSTANCE_NAME,
        embedding_endpoint=EMBEDDING_ENDPOINT,
        embedding_dims=EMBEDDING_DIMS,
        workspace_client=user_client,
    ) as store:

        # Look up user profile and build augmented system prompt
        profile = await lookup_user_profile(user_id, workspace_client=user_client) if user_id else None
        augmented_prompt = build_system_prompt(SYSTEM_PROMPT, profile)

        # Build graph
        graph = _create_graph(augmented_prompt)

        # Config passed through to graph nodes and tools
        run_config = {
            "configurable": {
                "store": store,
            }
        }
        if user_id:
            run_config["configurable"]["user_id"] = user_id
        if thread_id:
            run_config["configurable"]["thread_id"] = thread_id

        state_input = {"messages": cc_msgs}
        if user_id:
            state_input["user_id"] = user_id

        # Auto-save the user's last message before graph execution
        if user_id and thread_id and cc_msgs:
            last_user_msg = next(
                (m for m in reversed(cc_msgs) if m.get("role") == "user"),
                None,
            )
            if last_user_msg and last_user_msg.get("content"):
                await save_message_to_store(store, user_id, thread_id, "user", last_user_msg["content"])

        # Stream the graph execution
        async for event in process_agent_astream_events(
            graph.astream(state_input, run_config, stream_mode=["updates", "messages"])
        ):
            yield event
