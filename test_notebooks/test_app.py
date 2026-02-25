# Databricks notebook source
# MAGIC %md
# MAGIC # Test Stateful Agent App
# MAGIC
# MAGIC Sends test messages to the `stateful-agent-app` deployed as a Databricks App.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import json
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
current_user = w.current_user.me().user_name

# Get the app URL
app = w.apps.get("stateful-agent-app")
app_url = app.url.rstrip("/")
print(f"App URL: {app_url}")
print(f"Current user: {current_user}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Function

# COMMAND ----------

def send_message(content: str, thread_id: str = None, stream: bool = False):
    """Send a message to the agent app and print the response.

    Uses the Databricks SDK's ApiClient for auth (handles PAT, OAuth, etc.)
    instead of raw requests, which avoids sign-in redirects.
    """
    payload = {
        "input": [{"type": "message", "role": "user", "content": content}],
        "context": {"user_id": current_user},
    }
    if thread_id:
        payload["custom_inputs"] = {"thread_id": thread_id}

    url = f"{app_url}/invocations"

    if stream:
        # Use the SDK's API client for streaming
        import requests

        headers = w.api_client.default_headers
        headers["Content-Type"] = "application/json"

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
        print()  # newline after stream
    else:
        # Use the SDK's API client which handles auth automatically
        resp = w.api_client.do("POST", url, body=payload)
        print(f"Response: {json.dumps(resp, indent=2)[:1000]}")
        for item in resp.get("output", []):
            if item.get("type") == "message":
                for content_block in item.get("content", []):
                    if content_block.get("type") == "output_text":
                        print(content_block.get("text", ""))
        return resp

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
