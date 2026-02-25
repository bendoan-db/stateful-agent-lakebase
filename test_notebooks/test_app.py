# Databricks notebook source
# MAGIC %md
# MAGIC # Test Stateful Agent App
# MAGIC
# MAGIC Sends test messages directly to the agent backend (`/invocations`), bypassing the
# MAGIC frontend. Works against:
# MAGIC - **Local backend**: `uv run start-server` on `http://localhost:8000`
# MAGIC - **Deployed app**: the Databricks App URL (uses SDK auth)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import json
import requests
from databricks.sdk import WorkspaceClient

# Toggle target: "local" for localhost:8000, "deployed" for the Databricks App URL
TARGET = "deployed"  # change to "local" for local dev

w = WorkspaceClient()
current_user = w.current_user.me().user_name

if TARGET == "local":
    base_url = "http://localhost:8000"
    auth_headers = {}
else:
    app = w.apps.get("stateful-agent-app")
    base_url = app.url.rstrip("/")
    auth_headers = w.config.authenticate()

print(f"Target: {TARGET}")
print(f"Backend URL: {base_url}")
print(f"Current user: {current_user}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def send_message(content: str, thread_id: str = None, user_id: str = None, stream: bool = False):
    """Send a message directly to the backend /invocations endpoint.

    Args:
        content: The message text.
        thread_id: Optional thread ID for conversation continuity.
        user_id: Override user ID (defaults to current workspace user).
        stream: If True, use SSE streaming and print tokens as they arrive.
    """
    payload = {
        "input": [{"type": "message", "role": "user", "content": content}],
        "context": {"user_id": user_id or current_user},
    }
    if thread_id:
        payload["custom_inputs"] = {"thread_id": thread_id}

    url = f"{base_url}/invocations"
    headers = {**auth_headers, "Content-Type": "application/json"}

    if stream:
        resp = requests.post(url, json=payload, headers=headers, stream=True)
        resp.raise_for_status()
        print("Streaming response:")
        for line in resp.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data = decoded[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                        if event.get("type") == "response.output_text.delta":
                            print(event.get("delta", ""), end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        print()
    else:
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()
        for item in result.get("output", []):
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        print(block.get("text", ""))
        return result


def health_check():
    """Ping the backend to verify it's reachable."""
    try:
        resp = requests.get(f"{base_url}/health", headers=auth_headers, timeout=5)
        print(f"Health: {resp.status_code} — {resp.text[:200]}")
    except requests.ConnectionError:
        print(f"ERROR: Cannot reach {base_url}. Is the backend running?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Health Check

# COMMAND ----------

health_check()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test: Basic Message

# COMMAND ----------

send_message("Hello! What can you help me with?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test: Streaming

# COMMAND ----------

send_message("Tell me a short joke.", stream=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test: Memory Recall

# COMMAND ----------

send_message("What do you remember about me?", thread_id="test-memory-001")

# COMMAND ----------
