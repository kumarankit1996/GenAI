
from typing import Any, Generator, Optional
import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
import dspy
import uuid

# Autolog DSPy traces to MLflow
mlflow.dspy.autolog()

# Set up DSPy with a Databricks-hosted LLM
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
lm = dspy.LM(model=f"databricks/{LLM_ENDPOINT_NAME}", max_tokens=250)
dspy.settings.configure(lm=lm)


class DSPyChatAgent(ChatAgent):     
    def __init__(self):
        self.agent = dspy.ChainOfThought("question,history -> answer")


    def prepare_message_history(self, messages: list[ChatAgentMessage]):
        history_entries = []
        # Assume the last message in the input is the most recent user question.
        for i in range(0, len(messages) - 1, 2):
            history_entries.append({"question": messages[i].content, "answer": messages[i + 1].content})
        return dspy.History(messages=history_entries)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        latest_question = messages[-1].content
        response = self.agent(question=latest_question, history=self.prepare_message_history(messages)).answer
        return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=response, id=uuid.uuid4().hex)]
        )

# Set model for logging or interactive testing
from mlflow.models import set_model
AGENT = DSPyChatAgent()
set_model(AGENT)
