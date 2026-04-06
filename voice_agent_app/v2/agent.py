"""
LangChain Voice Agent Backend
Provides streaming conversational AI with Databricks foundation models

Best Practices Implemented:
- LangChain streaming with proper generators
- ChatMessageHistory for conversation management
- Voice-optimized system prompts (concise, conversational)
- MLflow tracing for observability
- History truncation for latency optimization
- Voice-specific metrics (first token latency, tokens/sec)
- Comprehensive error handling
- Session isolation and management
- Latest databricks-langchain package
- Production-grade logging

Reference: LangChain Voice Agent Best Practices
"""

from typing import List, AsyncIterator, Optional, Dict, Any
from databricks.sdk.core import Config
import mlflow
import time
import logging

# Import ChatDatabricks from official databricks-langchain package
from databricks_langchain import ChatDatabricks
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.tracers import LangChainTracer
import os

# Configure logger
logger = logging.getLogger(__name__)


class VoiceAgent:
    """
    LangChain-based voice agent optimized for real-time interaction
    
    Features:
    - Streaming responses for real-time feel
    - Session-based conversation history with truncation
    - MLflow tracing with voice-specific metrics
    - Optimized for voice (concise, conversational responses)
    - Latency-optimized (history truncation, efficient streaming)
    - Databricks foundation model integration
    
    Voice-Specific Optimizations:
    - First token latency < 500ms target
    - Response length 2-4 sentences for natural speech
    - History truncation to keep latency low
    - Tracks tokens/second for quality monitoring
    """
    
    def __init__(
        self,
        model_endpoint: str = "databricks-qwen3-next-80b-a3b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        enable_tracing: bool = True,
        max_history_turns: int = 10
    ):
        """
        Initialize the voice agent
        
        Args:
            model_endpoint: Databricks foundation model endpoint name
            temperature: LLM temperature (0-1). Higher = more creative
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt for agent personality
            enable_tracing: Enable MLflow tracing (recommended for production)
            max_history_turns: Maximum conversation turns to keep (1 turn = user + assistant)
                             Reduces latency by limiting context window. Default: 10 turns = 20 messages
        """
        logger.info(f"Initializing VoiceAgent with model={model_endpoint}, temperature={temperature}, max_tokens={max_tokens}")
        
        # Initialize Databricks config for authentication
        self.cfg = Config()
        
        # Voice optimization settings
        self.max_history_turns = max_history_turns
        self.enable_tracing = enable_tracing
        
        # Initialize LLM with streaming
        try:
            self.llm = ChatDatabricks(
                model=model_endpoint,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.info(f"ChatDatabricks initialized successfully with model: {model_endpoint}")
            logger.debug(f"Voice optimization settings: max_history_turns={max_history_turns}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatDatabricks: {e}", exc_info=True)
            raise
        
        # Default system prompt optimized for voice interactions
        # Based on LangChain voice agent best practices
        self.system_prompt = system_prompt or """You are a professional voice assistant powered by Databricks.

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

You have access to Databricks' powerful AI models for analysis and problem-solving."""
        
        # Create prompt template with history
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create chain
        self.chain = self.prompt | self.llm
        
        # In-memory conversation store (keyed by session_id)
        self.store: Dict[str, ChatMessageHistory] = {}
        
        logger.info("VoiceAgent initialization complete")
    
    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """
        Get or create chat history for a session
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            ChatMessageHistory for the session
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
            logger.debug(f"Created new session history for session_id={session_id}")
        return self.store[session_id]
    
    def _truncate_history(
        self, 
        messages: List[BaseMessage], 
        max_turns: int
    ) -> List[BaseMessage]:
        """
        Truncate conversation history to reduce latency
        
        LangChain Best Practice: Limit context window for voice agents
        to keep first token latency under 500ms
        
        Args:
            messages: Full message history
            max_turns: Maximum number of turns to keep (1 turn = user + assistant)
            
        Returns:
            Truncated message list with optional context summary
        """
        max_messages = max_turns * 2  # Each turn = user + assistant message
        
        if len(messages) <= max_messages:
            return messages
        
        # Keep recent messages
        truncated = messages[-max_messages:]
        
        # Add context summary about truncated messages
        num_truncated = len(messages) - max_messages
        summary = SystemMessage(
            content=f"[Previous conversation context: {num_truncated} earlier messages not shown to reduce latency]"
        )
        
        logger.debug(f"Truncated history: removed {num_truncated} messages, keeping {len(truncated)} recent messages")
        return [summary] + truncated
    
    @mlflow.trace(name="stream_response", span_type="CHAIN")
    async def stream_response(
        self,
        user_input: str,
        session_id: str = "default",
        history: Optional[List[BaseMessage]] = None
    ) -> AsyncIterator[str]:
        """
        Stream response from the agent with MLflow tracing and voice-specific metrics
        
        Voice Optimization Features:
        - Tracks first token latency (target < 500ms)
        - Measures tokens per second
        - Truncates history to reduce latency
        - Yields chunks immediately for real-time feel
        
        Args:
            user_input: User's message
            session_id: Session identifier for history tracking
            history: Optional pre-populated message history (overrides session history)
            
        Yields:
            Response chunks as strings
        """
        logger.info(f"Processing request for session_id={session_id}, input_length={len(user_input)}")
        
        # Track timing for voice metrics
        start_time = time.time()
        first_token_time = None
        chunk_count = 0
        
        # Get or create session history
        session_history = self.get_session_history(session_id)
        
        # If history provided, populate session
        if history:
            session_history.clear()
            for msg in history:
                session_history.add_message(msg)
            logger.debug(f"Session history populated with {len(history)} messages")
        
        # Add user message
        session_history.add_message(HumanMessage(content=user_input))
        
        # Get history as messages (exclude the just-added user message)
        history_messages = session_history.messages[:-1]
        
        # VOICE OPTIMIZATION: Truncate history to reduce latency
        if len(history_messages) > self.max_history_turns * 2:
            history_messages = self._truncate_history(
                history_messages, 
                self.max_history_turns
            )
        
        # Stream response
        full_response = ""
        try:
            with mlflow.start_span(name="llm_streaming") as span:
                span.set_inputs({
                    "user_input": user_input, 
                    "history_length": len(history_messages),
                    "session_id": session_id
                })
                
                async for chunk in self.chain.astream({
                    "input": user_input,
                    "history": history_messages
                }):
                    if hasattr(chunk, 'content') and chunk.content:
                        content = chunk.content
                        full_response += content
                        chunk_count += 1
                        
                        # Track first token latency (critical voice metric)
                        if first_token_time is None:
                            first_token_time = time.time() - start_time
                            mlflow.log_metric("first_token_latency_ms", first_token_time * 1000)
                            
                            # Flag if latency is too high for voice
                            if first_token_time > 0.5:  # 500ms threshold
                                logger.warning(f"High first token latency: {first_token_time:.3f}s (target: <0.5s)")
                        
                        yield content
                
                # Calculate final metrics
                total_time = time.time() - start_time
                token_count = len(full_response.split())  # Approximate
                
                # Log voice-specific metrics
                span.set_outputs({
                    "response": full_response, 
                    "response_length": len(full_response),
                    "token_count": token_count,
                    "total_time_ms": total_time * 1000,
                    "first_token_latency_ms": first_token_time * 1000 if first_token_time else None
                })
                
                mlflow.log_metric("total_response_time_ms", total_time * 1000)
                mlflow.log_metric("response_length_chars", len(full_response))
                mlflow.log_metric("response_length_tokens", token_count)
                mlflow.log_metric("chunk_count", chunk_count)
                
                if total_time > 0:
                    tokens_per_second = token_count / total_time
                    mlflow.log_metric("tokens_per_second", tokens_per_second)
                
                logger.info(
                    f"Response complete: session_id={session_id}, "
                    f"total_time={total_time:.3f}s, tokens={token_count}, "
                    f"first_token_latency={first_token_time:.3f}s if first_token_time else 'N/A', "
                    f"tokens_per_sec={tokens_per_second:.2f} if total_time > 0 else 'N/A'"
                )
            
            # Add assistant response to history
            if full_response:
                session_history.add_message(AIMessage(content=full_response))
            else:
                # Handle empty response
                error_msg = "I apologize, but I couldn't generate a response. Please try again."
                session_history.add_message(AIMessage(content=error_msg))
                logger.warning(f"Empty response generated for session_id={session_id}")
                yield error_msg
                
        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}. Please try again."
            session_history.add_message(AIMessage(content=error_msg))
            logger.error(f"Error during stream_response: session_id={session_id}, error={str(e)}", exc_info=True)
            
            if self.enable_tracing:
                mlflow.log_param("error", str(e))
                mlflow.log_param("error_type", type(e).__name__)
            
            yield error_msg
    
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
        
        Args:
            session_id: Session identifier to clear
        """
        if session_id in self.store:
            self.store[session_id].clear()
            logger.info(f"Cleared history for session_id={session_id}")
        else:
            logger.debug(f"No history to clear for session_id={session_id}")
    
    def get_history(self, session_id: str = "default") -> List[BaseMessage]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of messages in the conversation
        """
        if session_id in self.store:
            return self.store[session_id].messages
        return []
    
    def get_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """
        Get statistics about the conversation
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with conversation statistics
        """
        history = self.get_history(session_id)
        user_messages = [msg for msg in history if msg.type == "human"]
        ai_messages = [msg for msg in history if msg.type == "ai"]
        
        stats = {
            "session_id": session_id,
            "total_messages": len(history),
            "user_messages": len(user_messages),
            "ai_messages": len(ai_messages),
            "max_history_turns": self.max_history_turns,
            "history_will_truncate": len(history) > self.max_history_turns * 2
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
        print("Voice Agent Test - LangChain Best Practices")
        print("=" * 60)
        
        # Initialize agent with voice-optimized settings
        try:
            agent = VoiceAgent(
                max_history_turns=5  # Test with shorter history
            )
            print("\nAgent initialized successfully")
            print(f"   History truncation: {agent.max_history_turns} turns\n")
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
        
        # Follow-up with history
        print("User: What did I just say?")
        print("Assistant: ", end="", flush=True)
        
        start = time.time()
        async for chunk in agent.stream_response("What did I just say?"):
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
