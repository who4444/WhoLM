import sys
import os
# Add the back_end directory to sys.path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from database.qdrant import QdrantDB
from ingestion.embeddings.text_encoder import encode_texts

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    message_id: str
    session_id: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'metadata' not in data:
            data['metadata'] = {}
        return cls(**data)


@dataclass
class ConversationContext:
    """Context information for the current conversation."""
    session_id: str
    user_id: Optional[str] = None
    topic: Optional[str] = None
    last_activity: datetime = None
    preferences: Dict[str, Any] = None
    custom_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.now()
        if self.preferences is None:
            self.preferences = {}
        if self.custom_data is None:
            self.custom_data = {}


class ConversationMemory:
    """
    Manages conversation memory with multiple storage layers.

    Features:
    - Short-term memory (recent messages in RAM)
    - Long-term memory (persistent storage in Qdrant)
    - Session management
    - Context-aware retrieval
    - Automatic cleanup of old conversations
    """

    def __init__(self,
                 qdrant_url: str = "http://localhost:6333",
                 short_term_limit: int = 50,
                 max_session_age_days: int = 30,
                 embedding_dim: int = 1024):
        """
        Initialize conversation memory.

        Args:
            qdrant_url: Qdrant server URL
            short_term_limit: Max messages to keep in short-term memory
            max_session_age_days: Days before old sessions are cleaned up
            embedding_dim: Dimension of text embeddings
        """
        self.qdrant_url = qdrant_url
        self.short_term_limit = short_term_limit
        self.max_session_age_days = max_session_age_days
        self.embedding_dim = embedding_dim

        # Initialize Qdrant collections for different memory types
        self.conversations_db = QdrantDB(
            url=qdrant_url,
            vector_collection="conversations",
            vector_dim=embedding_dim
        )

        self.contexts_db = QdrantDB(
            url=qdrant_url,
            vector_collection="conversation_contexts",
            vector_dim=embedding_dim
        )

        # Short-term memory (in-memory)
        self.short_term_memory: Dict[str, deque] = {}  # session_id -> deque of Messages

        # Active contexts
        self.active_contexts: Dict[str, ConversationContext] = {}

        # Message ID counter
        self.message_counter = 0

        logger.info("Conversation memory initialized")

    def start_session(self, session_id: str, user_id: Optional[str] = None,
                     initial_context: Dict[str, Any] = None) -> ConversationContext:
        """
        Start a new conversation session.

        Args:
            session_id: Unique session identifier
            user_id: Optional user identifier
            initial_context: Initial context data

        Returns:
            ConversationContext for the session
        """
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            custom_data=initial_context or {}
        )

        # Initialize short-term memory for this session
        self.short_term_memory[session_id] = deque(maxlen=self.short_term_limit)
        self.active_contexts[session_id] = context

        # Store context in Qdrant
        self._store_context(context)

        logger.info(f"Started conversation session: {session_id}")
        return context

    def add_message(self, session_id: str, role: str, content: str,
                   metadata: Dict[str, Any] = None) -> Message:
        """
        Add a message to the conversation.

        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            metadata: Additional metadata

        Returns:
            The created Message object
        """
        if session_id not in self.active_contexts:
            raise ValueError(f"Session {session_id} not found. Call start_session() first.")

        # Create message
        self.message_counter += 1
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            message_id=f"msg_{self.message_counter}",
            session_id=session_id,
            metadata=metadata or {}
        )

        # Add to short-term memory
        if session_id not in self.short_term_memory:
            self.short_term_memory[session_id] = deque(maxlen=self.short_term_limit)

        self.short_term_memory[session_id].append(message)

        # Update context
        context = self.active_contexts[session_id]
        context.last_activity = datetime.now()

        # Store in long-term memory (Qdrant)
        self._store_message(message)

        # Update context in Qdrant
        self._store_context(context)

        logger.debug(f"Added {role} message to session {session_id}")
        return message

    def get_recent_messages(self, session_id: str, limit: int = 10) -> List[Message]:
        """
        Get recent messages from short-term memory.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return

        Returns:
            List of recent messages
        """
        if session_id not in self.short_term_memory:
            return []

        messages = list(self.short_term_memory[session_id])
        return messages[-limit:]  # Return most recent

    def get_conversation_history(self, session_id: str, hours_back: int = 24) -> List[Message]:
        """
        Get conversation history from long-term memory.

        Args:
            session_id: Session identifier
            hours_back: How many hours of history to retrieve

        Returns:
            List of messages from the time period
        """
        # Query Qdrant for messages from this session within time range
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        # For now, return short-term memory (Qdrant query would be more complex)
        # In a full implementation, you'd query Qdrant with filters
        return self.get_recent_messages(session_id, limit=100)

    def search_similar_conversations(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar conversations using semantic similarity.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of similar conversation results
        """
        # Encode query
        query_embedding = encode_texts([query])[0]

        # Search conversations collection
        results = self.conversations_db.search(query_embedding, limit=limit)

        # Format results
        formatted_results = []
        for result in results:
            payload = result["payload"]
            formatted_results.append({
                "session_id": payload.get("session_id"),
                "message_id": payload.get("message_id"),
                "content": payload.get("content"),
                "role": payload.get("role"),
                "timestamp": payload.get("timestamp"),
                "score": result["score"]
            })

        return formatted_results

    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """
        Get the current context for a session.

        Args:
            session_id: Session identifier

        Returns:
            ConversationContext or None if not found
        """
        return self.active_contexts.get(session_id)

    def update_context(self, session_id: str, updates: Dict[str, Any]) -> None:
        """
        Update context for a session.

        Args:
            session_id: Session identifier
            updates: Dictionary of updates to apply
        """
        if session_id not in self.active_contexts:
            raise ValueError(f"Session {session_id} not found")

        context = self.active_contexts[session_id]

        # Update context fields
        for key, value in updates.items():
            if hasattr(context, key):
                setattr(context, key, value)
            else:
                context.custom_data[key] = value

        context.last_activity = datetime.now()

        # Store updated context
        self._store_context(context)

    def end_session(self, session_id: str) -> None:
        """
        End a conversation session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.active_contexts:
            context = self.active_contexts[session_id]
            context.last_activity = datetime.now()

            # Store final context
            self._store_context(context)

            # Clean up memory
            if session_id in self.short_term_memory:
                del self.short_term_memory[session_id]
            del self.active_contexts[session_id]

            logger.info(f"Ended conversation session: {session_id}")

    def cleanup_old_sessions(self) -> int:
        """
        Clean up old conversation sessions.

        Returns:
            Number of sessions cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=self.max_session_age_days)
        cleaned_count = 0

        # Clean up active contexts
        to_remove = []
        for session_id, context in self.active_contexts.items():
            if context.last_activity < cutoff_date:
                to_remove.append(session_id)

        for session_id in to_remove:
            self.end_session(session_id)
            cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old sessions")

        return cleaned_count

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.

        Returns:
            Dictionary with memory statistics
        """
        total_messages = sum(len(msgs) for msgs in self.short_term_memory.values())

        return {
            "active_sessions": len(self.active_contexts),
            "short_term_messages": total_messages,
            "max_session_age_days": self.max_session_age_days,
            "short_term_limit": self.short_term_limit,
            "qdrant_url": self.qdrant_url
        }

    def _store_message(self, message: Message) -> None:
        """Store a message in Qdrant."""
        try:
            # Create embedding for the message
            embedding = encode_texts([message.content])[0]

            # Prepare payload
            payload = message.to_dict()

            # Store in Qdrant
            point_id = hash(f"{message.session_id}_{message.message_id}") % 2**63
            self.conversations_db.upsert([point_id], [embedding], [payload])

        except Exception as e:
            logger.error(f"Failed to store message in Qdrant: {e}")

    def _store_context(self, context: ConversationContext) -> None:
        """Store conversation context in Qdrant."""
        try:
            # Create a text representation for embedding
            context_text = f"Session {context.session_id} topic: {context.topic or 'general'}"
            if context.user_id:
                context_text += f" user: {context.user_id}"

            embedding = encode_texts([context_text])[0]

            # Prepare payload
            payload = {
                "session_id": context.session_id,
                "user_id": context.user_id,
                "topic": context.topic,
                "last_activity": context.last_activity.isoformat(),
                "preferences": json.dumps(context.preferences),
                "custom_data": json.dumps(context.custom_data)
            }

            # Store in Qdrant
            point_id = hash(context.session_id) % 2**63
            self.contexts_db.upsert([point_id], [embedding], [payload])

        except Exception as e:
            logger.error(f"Failed to store context in Qdrant: {e}")

    def export_conversation(self, session_id: str, filepath: str) -> bool:
        """
        Export conversation history to a JSON file.

        Args:
            session_id: Session identifier
            filepath: Path to export file

        Returns:
            Success status
        """
        try:
            messages = self.get_recent_messages(session_id, limit=1000)

            conversation_data = {
                "session_id": session_id,
                "exported_at": datetime.now().isoformat(),
                "messages": [msg.to_dict() for msg in messages]
            }

            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported conversation {session_id} to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export conversation: {e}")
            return False