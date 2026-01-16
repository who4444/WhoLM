

import sys
import os
# Add the back_end directory to sys.path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.chatbot.memory import ConversationMemory 
from services.chatbot.gemini_client import generate_response
from services.chatbot.prompts.document_prompts import doc_prompt
from services.rag.qdrant_rag_pipeline import QdrantRAGPipeline
from typing import Optional, Dict, Any
from config.config import Config

class WhoLM:
    """
    Enhanced chatbot with conversation memory.

    Features:
    - Persistent conversation context
    - Semantic search across chat history
    - Session management
    - Integration with RAG pipeline
    """

    def __init__(self,
                 qdrant_url: str = "http://localhost:6333",
                 short_term_limit: int = 50,
                 max_session_age_days: int = 30):
        """
        Initialize the chatbot with memory system.

        Args:
            qdrant_url: Qdrant server URL
            short_term_limit: Max messages in RAM per session
            max_session_age_days: Days before cleanup
        """
        self.memory = ConversationMemory(
            qdrant_url=qdrant_url,
            short_term_limit=short_term_limit,
            max_session_age_days=max_session_age_days
        )

        # Initialize RAG pipeline
        self.rag_pipeline = QdrantRAGPipeline(
            qdrant_url=qdrant_url,
            text_collection=Config.QDRANT_DOC_COLLECTION,
            frame_collection=Config.QDRANT_VD_COLLECTION,
            embedding_dim=Config.QDRANT_TEXT_EMBEDDING_DIM,
            frame_embedding_dim=Config.QDRANT_IMAGE_EMBEDDING_DIM
        )

        # System prompts 
        self.system_prompts = {
            "video": """
            You are an expert assistant for video content analysis and Q&A.
            Help users find and understand video content using the provided frames.
            Be helpful, accurate, and cite your sources when possible.
            """,
            "document": doc_prompt.template,
            "general": """
            You are a helpful AI assistant with access to video and document content.
            Answer questions based on the available context and maintain conversation flow.
            """
        }

    def start_conversation(self, session_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Start a new conversation session.

        Args:
            session_id: Unique session identifier
            user_id: Optional user identifier

        Returns:
            Session context dictionary
        """
        context = self.memory.start_session(session_id, user_id=user_id)

        # Initialize with default preferences
        self.memory.update_context(session_id, {
            "content_type": "general",
            "topic": None,
            "difficulty_level": "intermediate"
        })

        return context

    def chat(self, session_id: str, user_input: str,
             content_type: str = "general") -> str:
        """
        Process a chat message with memory context.

        Args:
            session_id: Conversation session ID
            user_input: User's message
            content_type: Type of content to focus on ("video", "document", "general")

        Returns:
            Assistant's response
        """
        # Ensure session exists
        context = self.memory.get_context(session_id)
        if not context:
            self.start_conversation(session_id)

        # Add user message to memory
        self.memory.add_message(session_id, "user", user_input)

        # Get conversation history
        recent_messages = self.memory.get_recent_messages(session_id, limit=10)

        # Build conversation context
        conversation_history = "\n".join([
            f"{msg.role}: {msg.content}" for msg in recent_messages[-6:]  # Last 6 messages
        ])

        # Update content type preference
        if content_type != "general":
            self.memory.update_context(session_id, {"content_type": content_type})

        # Get system prompt
        sys_prompt = self.system_prompts.get(content_type, self.system_prompts["general"])

        # Search for relevant content using RAG
        try:
            # Combine conversation history with current query
            enhanced_query = f"{conversation_history}\n\nCurrent: {user_input}"

            # Query the RAG pipeline
            rag_results = self.rag_pipeline.query(enhanced_query)

            # Format context from RAG results
            if rag_results:
                context_parts = []
                for i, result in enumerate(rag_results[:5]):  # Top 5 results
                    modality = result.get("metadata", {}).get("modality", "text")
                    if modality == "frame":
                        # Handle frame results
                        frame_info = f"Frame from video '{result.get('metadata', {}).get('video_name', 'unknown')}' at {result.get('metadata', {}).get('timestamp', 0)}s"
                        context_parts.append(f"[Source {i+1}] {frame_info}")
                    else:
                        # Handle text results
                        text_content = result.get('text', '')
                        context_parts.append(f"[Source {i+1}] {text_content}")
                context_str = "\n\n".join(context_parts)
            else:
                context_str = "No relevant content found in the database."

        except Exception as e:
            print(f"RAG query failed: {e}")
            context_str = "Unable to retrieve relevant content at this time."

        # Generate response using Gemini
        try:
            response = generate_response(
                sys_prompt=sys_prompt,
                context=context_str,
                query=user_input
            )
        except Exception as e:
            print(f"Response generation failed: {e}")
            response = "I'm sorry, I encountered an error generating a response. Please try again."

        # Add assistant response to memory
        self.memory.add_message(session_id, "assistant", response)

        return response

    def get_conversation_history(self, session_id: str, limit: int = 20) -> list:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return

        Returns:
            List of message dictionaries
        """
        messages = self.memory.get_recent_messages(session_id, limit=limit)
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in messages
        ]

    def search_similar_conversations(self, query: str, limit: int = 5) -> list:
        """
        Search for similar past conversations.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of similar conversation results
        """
        return self.memory.search_similar_conversations(query, limit=limit)

    def update_user_preferences(self, session_id: str, preferences: Dict[str, Any]):
        """
        Update user preferences for the session.

        Args:
            session_id: Session identifier
            preferences: Dictionary of preferences to update
        """
        self.memory.update_context(session_id, preferences)

    def end_conversation(self, session_id: str):
        """
        End a conversation session.

        Args:
            session_id: Session identifier
        """
        self.memory.end_session(session_id)

    def cleanup_old_sessions(self) -> int:
        """
        Clean up old conversation sessions.

        Returns:
            Number of sessions cleaned up
        """
        return self.memory.cleanup_old_sessions()

    def export_conversation(self, session_id: str, filepath: str) -> bool:
        """
        Export conversation history to a file.

        Args:
            session_id: Session identifier
            filepath: Path to export file

        Returns:
            True if export successful, False otherwise
        """
        return self.memory.export_conversation(session_id, filepath)


# Example usage
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = WhoLM()

    # Start a conversation
    session_id = "example_session_001"
    chatbot.start_conversation(session_id, user_id="user123")

    # Example conversation
    print("=== WhoLM Chatbot with Memory ===\n")

    # First message
    user_msg = "Hello! Can you help me find videos about machine learning?"
    print(f"User: {user_msg}")
    response = chatbot.chat(session_id, user_msg, content_type="video")
    print(f"Assistant: {response}\n")

    # Follow-up message
    user_msg = "What about specifically neural networks?"
    print(f"User: {user_msg}")
    response = chatbot.chat(session_id, user_msg, content_type="video")
    print(f"Assistant: {response}\n")

    # Get conversation history
    history = chatbot.get_conversation_history(session_id)
    print(f"Conversation has {len(history)} messages\n")

    # Search for similar conversations
    similar = chatbot.search_similar_conversations("machine learning videos")
    print(f"Found {len(similar)} similar conversations\n")

    # Update preferences
    chatbot.update_user_preferences(session_id, {
        "topic": "machine learning",
        "difficulty_level": "beginner"
    })

    # Export conversation
    chatbot.export_conversation(session_id, "example_conversation.json")

    # End conversation
    chatbot.end_conversation(session_id)

    print("Conversation ended and exported.")