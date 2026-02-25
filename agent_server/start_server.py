"""Bootstrap the MLflow AgentServer."""

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

# Importing agent registers the @invoke/@stream decorators
import agent_server.agent  # noqa: F401

from mlflow.genai.agent_server import AgentServer

agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
app = agent_server.app


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
