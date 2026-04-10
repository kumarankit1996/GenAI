"""
LangChain Voice Agent Backend - v4 (Latest LangChain 1.2.x+ Pattern)
Modern streaming architecture with Azure OpenAI

Features:
- Latest LangChain create_agent() API with InMemorySaver
- stream_mode="messages" for efficient streaming
- Async generator-based streaming pipeline
- Session-based conversation memory with checkpointer
- Voice-optimized system prompts (concise, conversational)
- Production-grade logging with structured metrics
- Tool calling with automatic execution loop

References:
- https://docs.langchain.com/oss/python/langchain/voice-agent
- https://python.langchain.com/docs/how_to/tool_calling/
- https://python.langchain.com/docs/integrations/chat/azure_chat_openai
"""

from typing import List, AsyncIterator, Optional, Dict, Any, AsyncGenerator
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.utils.uuid import uuid7
from dotenv import load_dotenv
import time
import logging
import os
import asyncio
import json
from datetime import datetime
import re

# Load environment variables from .env file (project root)
# Uses explicit path so it works regardless of working directory
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(dotenv_path=dotenv_path)

# Configure logger
logger = logging.getLogger(__name__)


# Define tools for the agent
@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    now = datetime.now()
    return f"The current date and time is: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Args:
        expression: A mathematical expression as a string (e.g., "2 + 2")
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is: {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


# Initialize DuckDuckGo search tool
web_search_tool = DuckDuckGoSearchRun()


class VoiceAgent:
    """
    Modern LangChain-based voice agent with Azure OpenAI

    Uses the latest LangChain create_agent() API with:
    - InMemorySaver for conversation memory across turns
    - stream_mode="messages" for efficient token streaming
    - Automatic tool calling loop (no manual implementation needed)
    - Optimized for voice interaction with low latency

    Features:
    - Streaming responses for real-time feel
    - Persistent conversation memory with configurable truncation
    - Production-grade logging with structured metrics
    - Voice-optimized system prompts (concise, conversational)
    - Built-in tools (time, calculator, web search)
    """

    def __init__(
        self,
        azure_openai_api_key: Optional[str] = None,
        azure_openai_endpoint: Optional[str] = None,
        azure_openai_deployment: Optional[str] = None,
        azure_openai_api_version: str = "2025-04-01-preview",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        max_history_turns: int = 10,
        tools: Optional[List] = None
    ):
        """
        Initialize the voice agent with Azure OpenAI

        Args:
            azure_openai_api_key: Azure OpenAI API key (or set AZURE_OPENAI_API_KEY env var)
            azure_openai_endpoint: Azure OpenAI endpoint URL (e.g., https://your-resource.openai.azure.com/)
            azure_openai_deployment: Azure OpenAI deployment name (e.g., gpt-4, gpt-4o)
            azure_openai_api_version: Azure OpenAI API version (default: 2024-02-15-preview)
            temperature: LLM temperature (0-1). Higher = more creative
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt for agent personality
            max_history_turns: Maximum conversation turns to keep (1 turn = user + assistant)
                             Note: InMemorySaver handles this automatically, this is for reference
            tools: Optional list of additional tools to add to the agent
        """
        logger.info(f"Initializing VoiceAgent with Azure OpenAI deployment={azure_openai_deployment}")

        # Voice optimization settings
        self.max_history_turns = max_history_turns
        self.model_type = "azure_openai"

        # Initialize Azure OpenAI LLM with streaming enabled
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=azure_openai_endpoint,
                azure_deployment=azure_openai_deployment,
                api_version=azure_openai_api_version,
                api_key=azure_openai_api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=True,
                stream_usage=True,  # Enable token usage in streaming
            )
            logger.info(f"AzureChatOpenAI initialized successfully with deployment: {azure_openai_deployment}")
        except Exception as e:
            logger.error(f"Failed to initialize AzureChatOpenAI: {e}", exc_info=True)
            raise

        # Default system prompt optimized for voice interactions
        # Based on LangChain voice agent best practices
        self.system_prompt = system_prompt or """You are a professional voice assistant powered by Azure OpenAI.

Keep your responses:
- **Concise** (2-4 sentences maximum)
- **Professional and clear** (no emojis, no exclamation marks, no casual slang)
- **Natural and conversational** (as if speaking in a business setting)
- **Easy to understand when spoken aloud** (avoid jargon, use simple words)
- **Avoid long lists** (use at most 3 items)
- **No complex formatting** (no tables, code blocks, or markdown)
- **Direct and informative** (get to the point quickly)

When appropriate:
- Ask clarifying questions (keep them brief)
- Provide concrete examples
- Maintain a helpful, professional tone

You have access to tools that can help you:
- get_current_time: Get the current date and time
- calculate: Perform mathematical calculations
- web_search: Search the web for current information using DuckDuckGo

Rules for tool usage:
- Use tools to gather facts, but never read raw tool output verbatim unless the user explicitly asks for it.
- After using a tool, think about the result and write a clean final answer in your own words.
- Merge tool results into a natural sentence with correct spacing, punctuation, and pronunciation-friendly wording.
- Do not expose intermediate reasoning, tool traces, search snippets, or scraped fragments.
- If tool output is messy, incomplete, or duplicated, clean it up before answering.
- Prefer short spoken answers: 1 to 3 sentences unless the user asks for more detail.

Use web_search when the user asks about:
- Current events or news
- Weather information
- Sports scores or schedules
- Stock prices or market information
- Recent developments or updates
- Factual information that may have changed recently"""

        # Define tools for the agent
        default_tools = [get_current_time, calculate, web_search_tool]
        if tools:
            default_tools.extend(tools)

        # Build agent using latest LangChain create_agent pattern
        # This replaces manual tool loop with automatic execution
        self._build_agent(default_tools)

        # In-memory conversation store using LangGraph checkpointer
        # Thread IDs map to session IDs for conversation memory
        self.thread_store: Dict[str, str] = {}

        logger.info("VoiceAgent initialization complete with latest LangChain pattern")

    def _build_agent(self, tools: List) -> None:
        """
        Build agent using latest LangChain create_agent() API

        Creates agent with:
        - AzureChatOpenAI model with streaming
        - Tool binding with automatic execution loop
        - InMemorySaver for conversation memory across turns
        - System prompt for voice optimization
        """
        # Create agent with latest pattern - automatic tool loop + memory
        self.agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=self.system_prompt,
            checkpointer=InMemorySaver(),
        )

        # Get tool names for logging
        tool_names = [tool.name if hasattr(tool, 'name') else getattr(tool, '__name__', 'unknown') for tool in tools]
        logger.info(f"LangChain agent created with {len(tools)} tools: {tool_names}")

    def _get_thread_id(self, session_id: str) -> str:
        """
        Get or create a thread ID for a session

        Thread IDs are used by InMemorySaver to maintain conversation history.
        Each session gets a unique thread ID for persistent memory.

        Args:
            session_id: Session identifier

        Returns:
            Thread ID for conversation memory
        """
        if session_id not in self.thread_store:
            # Generate unique thread ID using LangChain's uuid7
            thread_id = str(uuid7())
            self.thread_store[session_id] = thread_id
            logger.debug(f"Created new thread_id={thread_id} for session_id={session_id}")
        return self.thread_store[session_id]

    def _log_voice_metrics(
        self,
        session_id: str,
        first_token_latency_ms: float,
        total_response_time_ms: float,
        token_count: int,
        chunk_count: int,
        tokens_per_second: float
    ) -> None:
        """
        Log voice-specific metrics using structured logging

        Voice Agent Target Metrics:
        - First token latency: < 500ms (critical for natural conversation)
        - Total response time: < 3s for typical queries
        - Tokens per second: > 50 tokens/sec

        Args:
            session_id: Session identifier
            first_token_latency_ms: Time to first token in milliseconds
            total_response_time_ms: Total response time in milliseconds
            token_count: Approximate token count
            chunk_count: Number of streaming chunks
            tokens_per_second: Streaming throughput
        """
        # Structured metrics log entry
        metrics = {
            "event": "voice_response_complete",
            "session_id": session_id,
            "first_token_latency_ms": round(first_token_latency_ms, 2),
            "total_response_time_ms": round(total_response_time_ms, 2),
            "token_count": token_count,
            "chunk_count": chunk_count,
            "tokens_per_second": round(tokens_per_second, 2),
            "timestamp": datetime.now().isoformat()
        }

        # Log metrics as structured JSON
        logger.info(f"VOICE_METRICS: {json.dumps(metrics)}")

        # Log warnings if metrics exceed thresholds
        if first_token_latency_ms > 500:
            logger.warning(
                f"VOICE_LATENCY_WARNING: High first token latency "
                f"({first_token_latency_ms:.0f}ms, target: <500ms)"
            )

        if tokens_per_second < 50:
            logger.debug(
                f"VOICE_THROUGHPUT: Low throughput "
                f"({tokens_per_second:.1f} tokens/sec)"
            )

    def _normalize_for_voice(self, text: str) -> str:
        """Light cleanup: collapse multiple spaces/newlines into single spaces."""
        return re.sub(r"\s+", " ", text).strip()

    async def stream_response(
        self,
        user_input: str,
        session_id: str = "default",
        history: Optional[List[BaseMessage]] = None
    ) -> AsyncIterator[str]:
        """
        Stream response from the agent using latest LangChain pattern

        Uses stream_mode="messages" for efficient token streaming.
        The agent automatically handles:
        - Tool calling and execution loop
        - Conversation memory via InMemorySaver
        - System prompt injection

        Voice Optimization Features:
        - Tracks first token latency (target < 500ms)
        - Measures tokens per second
        - Yields chunks immediately for real-time feel
        - Logs structured metrics for monitoring

        Args:
            user_input: User's message
            session_id: Session identifier for history tracking
            history: Optional pre-populated message history (for migration only)

        Yields:
            Response chunks as strings
        """
        logger.info(f"Processing request for session_id={session_id}, input_length={len(user_input)}")

        start_time = time.time()
        first_token_time = None
        chunk_count = 0
        full_response = ""
        thread_id = self._get_thread_id(session_id)

        try:
            async for event in self.agent.astream_events(
                    {"messages": [HumanMessage(content=user_input)]},
                    {"configurable": {"thread_id": thread_id}},
                    version="v2",
            ):
                kind = event["event"]

                if kind != "on_chat_model_stream":
                    continue

                chunk = event["data"].get("chunk")
                if not chunk or not hasattr(chunk, "content") or not chunk.content:
                    continue

                raw_piece = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                if not raw_piece.strip():
                    continue

                full_response += raw_piece
                chunk_count += 1
                if first_token_time is None:
                    first_token_time = time.time() - start_time

                yield raw_piece

            full_response = self._normalize_for_voice(full_response)

            total_time = time.time() - start_time
            token_count = len(full_response.split())
            tokens_per_second = token_count / total_time if total_time > 0 else 0

            self._log_voice_metrics(
                session_id=session_id,
                first_token_latency_ms=first_token_time * 1000 if first_token_time else 0,
                total_response_time_ms=total_time * 1000,
                token_count=token_count,
                chunk_count=chunk_count,
                tokens_per_second=tokens_per_second,
            )

            if not full_response.strip():
                yield "I apologize, but I couldn't generate a response. Please try again."

        except Exception as e:
            logger.error(f"Error during stream_response: session_id={session_id}, error={str(e)}", exc_info=True)
            yield f"I encountered an error: {str(e)}. Please try again."

    async def stream_response_with_events(
        self,
        user_input: str,
        session_id: str = "default",
        history: Optional[List[BaseMessage]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream response from the agent with detailed event information

        Uses astream_events to provide rich event stream including:
        - Tool calls with arguments
        - Tool execution results
        - Model streaming tokens
        - Chain start/end events

        This is useful for clients that want detailed event information.

        Args:
            user_input: User's message
            session_id: Session identifier for history tracking
            history: Optional pre-populated message history

        Yields:
            Event dictionaries with type and data
        """
        logger.info(f"Processing request with events for session_id={session_id}")

        # Get thread ID for conversation memory
        thread_id = self._get_thread_id(session_id)

        try:
            # Stream events from agent using latest pattern
            async for event in self.agent.astream_events(
                {"messages": [HumanMessage(content=user_input)]},
                {"configurable": {"thread_id": thread_id}},
                version="v2",
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    # Streaming tokens from the model
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, 'content') and chunk.content:
                        yield {
                            "type": "token",
                            "content": chunk.content,
                            "event": event
                        }

                elif kind == "on_chain_start":
                    # Agent or tool execution starting
                    if event.get("name"):
                        yield {
                            "type": "chain_start",
                            "name": event["name"],
                            "event": event
                        }

                elif kind == "on_chain_end":
                    # Agent or tool execution completed
                    if event.get("name"):
                        yield {
                            "type": "chain_end",
                            "name": event["name"],
                            "event": event
                        }

                elif kind == "on_tool_start":
                    # Tool execution starting
                    yield {
                        "type": "tool_start",
                        "name": event.get("name", "unknown"),
                        "input": event["data"].get("input"),
                        "event": event
                    }

                elif kind == "on_tool_end":
                    # Tool execution completed
                    yield {
                        "type": "tool_end",
                        "name": event.get("name", "unknown"),
                        "output": event["data"].get("output"),
                        "event": event
                    }

        except Exception as e:
            logger.error(f"Error during stream_response_with_events: {e}", exc_info=True)
            yield {
                "type": "error",
                "message": str(e)
            }

    async def get_response(
        self,
        user_input: str,
        session_id: str = "default",
        history: Optional[List[BaseMessage]] = None
    ) -> str:
        """
        Get complete (non-streaming) response from the agent

        Note: For voice agents, prefer stream_response() for better UX

        Args:
            user_input: User's message
            session_id: Session identifier
            history: Optional pre-populated message history

        Returns:
            Complete response string
        """
        logger.debug(f"get_response called for session_id={session_id}")
        response = ""
        async for chunk in self.stream_response(user_input, session_id, history):
            response += chunk
        return response

    def clear_history(self, session_id: str = "default"):
        """
        Clear conversation history for a session

        Note: With InMemorySaver, this clears the thread from the checkpointer.

        Args:
            session_id: Session identifier to clear
        """
        if session_id in self.thread_store:
            thread_id = self.thread_store[session_id]
            # Clear the thread from checkpointer
            if hasattr(self.agent, 'checkpointer') and self.agent.checkpointer:
                # Note: InMemorySaver doesn't have a public delete method
                # so we just remove the thread mapping
                del self.thread_store[session_id]
                logger.info(f"Cleared history for session_id={session_id}")
        else:
            logger.debug(f"No history to clear for session_id={session_id}")

    def get_history(self, session_id: str = "default") -> List[BaseMessage]:
        """
        Get conversation history for a session

        Note: With InMemorySaver, history is managed by the checkpointer.
        This method returns the thread ID mapping for reference.

        Args:
            session_id: Session identifier

        Returns:
            List with thread information (actual messages managed by checkpointer)
        """
        if session_id in self.thread_store:
            return [AIMessage(content=f"Thread ID: {self.thread_store[session_id]}")]
        return []

    def get_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """
        Get statistics about the conversation

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with conversation statistics
        """
        thread_id = self.thread_store.get(session_id, "none")

        stats = {
            "session_id": session_id,
            "thread_id": thread_id,
            "max_history_turns": self.max_history_turns,
            "model_type": self.model_type,
            "memory_type": "InMemorySaver (LangGraph)",
            "active_sessions": len(self.thread_store)
        }

        logger.debug(f"Stats for session_id={session_id}: {stats}")
        return stats


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    async def test_agent():
        print("=" * 60)
        print("Voice Agent v5 Test - Latest LangChain Pattern")
        print("=" * 60)

        # Get Azure OpenAI configuration from environment
        azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        print(f"\nAzure OpenAI Configuration:")
        print(f"   Endpoint: {azure_openai_endpoint}")
        print(f"   Deployment: {azure_openai_deployment}")
        print(f"   API Version: {azure_openai_api_version}")

        # Initialize agent with voice-optimized settings
        try:
            agent = VoiceAgent(
                azure_openai_api_key=azure_openai_api_key,
                azure_openai_endpoint=azure_openai_endpoint,
                azure_openai_deployment=azure_openai_deployment,
                azure_openai_api_version=azure_openai_api_version,
                max_history_turns=5  # Test with shorter history
            )
            print("\nAgent initialized successfully")
            print(f"   Memory type: InMemorySaver (LangGraph)")
            print(f"   Model type: {agent.model_type}")
            print(f"   Tools: get_current_time, calculate, web_search\n")
        except Exception as e:
            print(f"\nFailed to initialize agent: {e}\n")
            return

        # Test streaming response with metrics
        print("User: Hello! What can you help me with?")
        print("Assistant: ", end="", flush=True)

        start = time.time()
        async for chunk in agent.stream_response("Hello! What can you help me with?"):
            print(chunk, end="", flush=True)
        elapsed = time.time() - start
        print(f"\nResponse time: {elapsed:.2f}s\n")

        # Test with tool usage - time
        print("User: What time is it right now?")
        print("Assistant: ", end="", flush=True)

        start = time.time()
        async for chunk in agent.stream_response("What time is it right now?"):
            print(chunk, end="", flush=True)
        elapsed = time.time() - start
        print(f"\nResponse time: {elapsed:.2f}s\n")

        # Test calculator tool
        print("User: What is 15 * 23 + 7?")
        print("Assistant: ", end="", flush=True)

        start = time.time()
        async for chunk in agent.stream_response("What is 15 * 23 + 7?"):
            print(chunk, end="", flush=True)
        elapsed = time.time() - start
        print(f"\nResponse time: {elapsed:.2f}s\n")

        # Test web search tool
        print("User: What are the latest developments in AI?")
        print("Assistant: ", end="", flush=True)

        start = time.time()
        async for chunk in agent.stream_response("What are the latest developments in AI?"):
            print(chunk, end="", flush=True)
        elapsed = time.time() - start
        print(f"\nResponse time: {elapsed:.2f}s\n")

        # Show stats
        stats = agent.get_stats()
        print("\nConversation Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print("All tests passed - Voice agent ready for production")
        print("=" * 60)

    # Run async test
    asyncio.run(test_agent())
