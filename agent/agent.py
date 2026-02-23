# Databricks notebook source
# MAGIC %md
# MAGIC # Stateful LangGraph Agent with Long-Term Memory
# MAGIC
# MAGIC This notebook implements a LangGraph agent using the MLflow `ResponsesAgent` interface
# MAGIC with user-based long-term memory persistence via Databricks Lakebase.

# COMMAND ----------

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Generator, Optional, Sequence, TypedDict

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    DatabricksStore,
    UCFunctionToolkit,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from databricks_ai_bridge.lakebase import LakebaseClient
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "config.yaml")

CONFIG = mlflow.models.ModelConfig(development_config=_CONFIG_PATH)

# Extract configuration values
llm_config = CONFIG.get("llm")
LLM_ENDPOINT_NAME = llm_config.get("endpoint_name")
SYSTEM_PROMPT = llm_config.get("system_prompt")
LAKEBASE_INSTANCE_NAME = CONFIG.get("lakebase").get("instance_name")
embeddings_config = CONFIG.get("embeddings")
EMBEDDING_ENDPOINT = embeddings_config.get("endpoint")
EMBEDDING_DIMS = embeddings_config.get("dims")
UC_TOOL_NAMES: list[str] = CONFIG.get("tools").get("uc_function_names")

user_profiles_config = CONFIG.get("user_profiles")
USER_PROFILES_TABLE = user_profiles_config.get("table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tools Configuration
# MAGIC
# MAGIC Define tools for your agent, enabling it to retrieve data or take actions beyond text generation.
# MAGIC See https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html

# COMMAND ----------

tools = []

# UC tools from config
if UC_TOOL_NAMES:
    uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
    tools.extend(uc_toolkit.tools)

# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html
VECTOR_SEARCH_TOOLS = []

tools.extend(VECTOR_SEARCH_TOOLS)

# DuckDuckGo web search tool
from langchain_community.tools import DuckDuckGoSearchResults

search_tool = DuckDuckGoSearchResults()
tools.append(search_tool)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent State Definition

# COMMAND ----------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]
    user_id: Optional[str]

# COMMAND ----------

# MAGIC %md
# MAGIC ## LangGraph Responses Agent

# COMMAND ----------

class LangGraphResponsesAgent(ResponsesAgent):
    """Stateless agent using ResponsesAgent with user-based long-term memory.

    Features:
    - Connection pooling with credential rotation via DatabricksStore
    - User-based long-term memory persistence (memories stored under "users".user_id) in "store" table
    - Tool support with UC functions
    - Automatic connection management - borrows connections per operation for scalability
    """

    def __init__(self):
        self.lakebase_instance_name = LAKEBASE_INSTANCE_NAME
        self.system_prompt = SYSTEM_PROMPT
        self.model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

        self._store = None
        self._memory_tools = None

    @property
    def store(self):
        """Lazy initialization of DatabricksStore with semantic search support."""
        if self._store is None:
            logger.info(f"Initializing DatabricksStore with instance: {self.lakebase_instance_name} and embedding endpoint {EMBEDDING_ENDPOINT} with dims {EMBEDDING_DIMS}")
            self._store = DatabricksStore(
                instance_name=self.lakebase_instance_name,
                embedding_endpoint=EMBEDDING_ENDPOINT,
                embedding_dims=EMBEDDING_DIMS,
            )
            self._store.setup()
        return self._store

    @property
    def memory_tools(self):
        """Lazy initialization of memory tools."""
        if self._memory_tools is None:
            logger.info("Creating memory tools")
            self._memory_tools = self._create_memory_tools()
        return self._memory_tools

    @property
    def model_with_all_tools(self):
        all_tools = tools + self.memory_tools
        return self.model.bind_tools(all_tools) if all_tools else self.model

    def _create_memory_tools(self):
        """Create tools for reading and writing long-term memory."""

        @tool
        def get_user_memory(query: str, config: RunnableConfig) -> str:
            """Search for relevant information about the user from long-term memory using semantic search via vector embeddings.

            Use this tool to retrieve previously saved information about the user,
            such as their preferences, facts they've shared, or other personal details.

            Args:
            """
            user_id = config.get("configurable", {}).get("user_id")
            if not user_id:
                return "Memory not available - no user_id provided."

            namespace = ("user_memories", user_id.replace(".", "-"))

            results = self.store.search(namespace, query=query, limit=5)

            if not results:
                return "No memories found for this user."

            memory_items = []
            for item in results:
                memory_items.append(f"- [{item.key}]: {json.dumps(item.value)}")

            return f"Found {len(results)} relevant memories (ranked by semantic similarity):\n" + "\n".join(memory_items)

        @tool
        def delete_user_memory(memory_key: str, config: RunnableConfig) -> str:
            """Delete a specific memory from the user's long-term memory.

            Use this tool when the user asks you to forget something or remove
            a piece of information from their memory.

            Args:
                memory_key: The key of the memory to delete (e.g., "preferences", "likes", "background_info")
            """
            user_id = config.get("configurable", {}).get("user_id")
            if not user_id:
                return "Cannot delete memory - no user_id provided."

            namespace = ("user_memories", user_id.replace(".", "-"))

            self.store.delete(namespace, memory_key)
            return f"Successfully deleted memory with key '{memory_key}' for user."

        return [get_user_memory, delete_user_memory]

    def _save_message_to_store(self, user_id: str, thread_id: str, role: str, content: str) -> None:
        """Save a message to the Lakebase store for automatic conversation persistence.

        Args:
            user_id: The user identifier for namespacing.
            thread_id: The thread identifier used as the storage key prefix.
            role: Message role ("user" or "assistant").
            content: The message text content.
        """
        namespace = ("user_memories", user_id.replace(".", "-"))
        key = f"{thread_id}-{role}"
        timestamp = datetime.now(timezone.utc).isoformat()
        value = {"role": role, "content": content, "timestamp": timestamp}
        try:
            self.store.put(namespace, key, value)
            logger.info(f"Auto-saved {role} message for user {user_id} with key {key}")
        except Exception as e:
            logger.error(f"Failed to auto-save {role} message for user {user_id}: {e}")

    def _create_graph(self, system_prompt: Optional[str] = None):
        """Create the LangGraph workflow"""
        effective_prompt = system_prompt if system_prompt is not None else self.system_prompt

        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "continue"
            return "end"

        model_with_tools = self.model_with_all_tools

        if effective_prompt:
            preprocessor = RunnableLambda(
                lambda state: [{"role": "system", "content": effective_prompt}] + state["messages"]
            )
        else:
            preprocessor = RunnableLambda(lambda state: state["messages"])

        model_runnable = preprocessor | model_with_tools

        def call_model(state: AgentState, config: RunnableConfig):
            response = model_runnable.invoke(state, config)
            return {"messages": [response]}

        def save_messages(state: AgentState, config: RunnableConfig):
            configurable = config.get("configurable", {})
            user_id = configurable.get("user_id")
            thread_id = configurable.get("thread_id")
            if user_id and thread_id:
                messages = state["messages"]
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and not msg.tool_calls:
                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        if content:
                            self._save_message_to_store(user_id, thread_id, "assistant", content)
                        break
            return {}

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", RunnableLambda(call_model))
        workflow.add_node("save_messages", RunnableLambda(save_messages))

        active_tools = (tools + self.memory_tools)

        if active_tools:
            workflow.add_node("tools", ToolNode(active_tools))
            workflow.add_conditional_edges(
                "agent",
                should_continue,
                {"continue": "tools", "end": "save_messages"}
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", "save_messages")

        workflow.add_edge("save_messages", END)
        workflow.set_entry_point("agent")

        return workflow.compile()

    def _get_user_id(self, request: ResponsesAgentRequest) -> Optional[str]:
        """
        Use user_id from chat context if available, return None if not provided
        """
        # User id from chat context as user id to store memories
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.types.html#mlflow.types.agent.ChatContext
        if request.context and getattr(request.context, "user_id", None):
            return request.context.user_id
        return None

    def _lookup_user_profile(self, user_id: str) -> Optional[dict]:
        """Look up user profile from the Lakebase-synced Postgres table.

        Returns a dict with profile fields on success, or None if not found/error.
        """
        store_user_id = f"user_memories.{user_id.replace('.', '-')}"
        with mlflow.start_span(name="lookup_user_profile", span_type="RETRIEVER") as span:
            span.set_inputs({"user_id": user_id, "store_user_id": store_user_id})
            try:
                client = LakebaseClient(instance_name=self.lakebase_instance_name)
                rows = client.execute(
                    f"SELECT summary, interests, preferences, behavioral_notes FROM {USER_PROFILES_TABLE} WHERE user_id = '{store_user_id}' LIMIT 1",
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

    @mlflow.trace(name="build_system_prompt")
    def _build_system_prompt(self, profile: Optional[dict] = None) -> str:
        """Build the system prompt, optionally augmented with user profile context."""
        if not profile:
            logger.info(f"No profile found.")
            return self.system_prompt

        profile_section = "\n\n## User Profile Context\n"
        if profile.get("summary"):
            profile_section += f"**Summary:** {profile['summary']}\n"
        if profile.get("interests"):
            profile_section += f"**Interests:** {profile['interests']}\n"
        if profile.get("preferences"):
            profile_section += f"**Preferences:** {profile['preferences']}\n"
        if profile.get("behavioral_notes"):
            profile_section += f"**Behavioral Notes:** {profile['behavioral_notes']}\n"
        profile_section += "\nUse this context to personalize your responses. Do not volunteer this information unprompted — use it implicitly to be more helpful."

        return self.system_prompt + profile_section

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming prediction"""
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs)

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming prediction"""
        user_id = self._get_user_id(request)

        # If there is no user_id, we cannot retrieve memories
        if not user_id:
            logger.error(
                "Cannot store or retrieve memories without a user_id."
            )

        ci = dict(request.custom_inputs or {})
        thread_id = ci.get("thread_id")
        if not thread_id:
            thread_id = uuid.uuid4().hex
            ci["thread_id"] = thread_id
            logger.info(f"No thread_id provided, generated: {thread_id}")

        if user_id:
            ci["user_id"] = user_id
        request.custom_inputs = ci

        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])

        run_config = {"configurable": {}}
        if user_id:
            run_config["configurable"]["user_id"] = user_id
        if thread_id:
            run_config["configurable"]["thread_id"] = thread_id

        profile = self._lookup_user_profile(user_id) if user_id else None
        augmented_prompt = self._build_system_prompt(profile)
        graph = self._create_graph(system_prompt=augmented_prompt)

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
                self._save_message_to_store(user_id, thread_id, "user", last_user_msg["content"])

        # Stream the graph execution
        for event in graph.stream(
            state_input,
            run_config,
            stream_mode=["updates", "messages"]
        ):
            if event[0] == "updates":
                for node_data in event[1].values():
                    if node_data and len(node_data.get("messages", [])) > 0:
                        yield from output_to_responses_items_stream(node_data["messages"])
            # Stream message chunks for real-time text generation
            elif event[0] == "messages":
                try:
                    chunk = event[1][0]
                    if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(delta=content, item_id=chunk.id),
                        )
                except Exception as e:
                    logger.error(f"Error streaming chunk: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Model

# COMMAND ----------

mlflow.langchain.autolog()
AGENT = LangGraphResponsesAgent()
mlflow.models.set_model(AGENT)
