import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from config.config import Config

logger = logging.getLogger(__name__)


class PostgresDB:
    """PostgreSQL database connection and operations for chatbot memory."""

    def __init__(self):
        self.config = Config()
        self.connection = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Establish connection to PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                host=self.config.POSTGRES_HOST,
                port=self.config.POSTGRES_PORT,
                database=self.config.POSTGRES_DB,
                user=self.config.POSTGRES_USER,
                password=self.config.POSTGRES_PASSWORD
            )
            self.connection.autocommit = True
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def create_tables(self):
        """Create necessary tables for chatbot memory."""
        with self.connection.cursor() as cursor:
            # Enable pgvector extension if not already enabled
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("pgvector extension enabled")
            except Exception as e:
                logger.warning(f"Could not enable pgvector extension: {e}. Make sure it's installed.")

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    message_id VARCHAR(255) UNIQUE NOT NULL,
                    session_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    metadata JSONB,
                    embedding VECTOR(1024),  -- Assuming 1024 dim embeddings
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Conversation contexts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_contexts (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    user_id VARCHAR(255),
                    topic VARCHAR(255),
                    last_activity TIMESTAMP NOT NULL,
                    preferences JSONB DEFAULT '{}',
                    custom_data JSONB DEFAULT '{}',
                    embedding VECTOR(1024),  -- Assuming 1024 dim embeddings
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_contexts_session_id ON conversation_contexts(session_id);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_contexts_last_activity ON conversation_contexts(last_activity);
            """)

            logger.info("Database tables created successfully")

    def insert_message(self, message_id: str, session_id: str, role: str,
                      content: str, timestamp: datetime, metadata: Dict[str, Any],
                      embedding: List[float]) -> None:
        """Insert a message into the database."""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO messages (message_id, session_id, role, content, timestamp, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (message_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
            """, (message_id, session_id, role, content, timestamp, json.dumps(metadata), embedding))

    def insert_context(self, session_id: str, user_id: Optional[str], topic: Optional[str],
                      last_activity: datetime, preferences: Dict[str, Any],
                      custom_data: Dict[str, Any], embedding: List[float]) -> None:
        """Insert or update conversation context."""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO conversation_contexts (session_id, user_id, topic, last_activity, preferences, custom_data, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    topic = EXCLUDED.topic,
                    last_activity = EXCLUDED.last_activity,
                    preferences = EXCLUDED.preferences,
                    custom_data = EXCLUDED.custom_data,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP;
            """, (session_id, user_id, topic, last_activity, json.dumps(preferences), json.dumps(custom_data), embedding))

    def get_messages_by_session(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages for a session, ordered by timestamp."""
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT message_id, session_id, role, content, timestamp, metadata
                FROM messages
                WHERE session_id = %s
                ORDER BY timestamp DESC
                LIMIT %s;
            """, (session_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation context for a session."""
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT session_id, user_id, topic, last_activity, preferences, custom_data
                FROM conversation_contexts
                WHERE session_id = %s;
            """, (session_id,))
            result = cursor.fetchone()
            return dict(result) if result else None

    def search_similar_messages(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar messages using vector similarity."""
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT message_id, session_id, role, content, timestamp, metadata,
                       1 - (embedding <=> %s::vector) as similarity
                FROM messages
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (embedding, embedding, limit))
            return [dict(row) for row in cursor.fetchall()]

    def search_similar_contexts(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar conversation contexts using vector similarity."""
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT session_id, user_id, topic, last_activity, preferences, custom_data,
                       1 - (embedding <=> %s::vector) as similarity
                FROM conversation_contexts
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (embedding, embedding, limit))
            return [dict(row) for row in cursor.fetchall()]

    def cleanup_old_sessions(self, max_age_days: int) -> int:
        """Clean up old conversation sessions and their messages."""
        with self.connection.cursor() as cursor:
            # Delete old contexts and get count
            cursor.execute("""
                DELETE FROM conversation_contexts
                WHERE last_activity < CURRENT_TIMESTAMP - INTERVAL '%s days';
            """, (max_age_days,))
            deleted_contexts = cursor.rowcount

            # Delete orphaned messages (messages without contexts)
            cursor.execute("""
                DELETE FROM messages
                WHERE session_id NOT IN (
                    SELECT session_id FROM conversation_contexts
                );
            """)
            deleted_messages = cursor.rowcount

            logger.info(f"Cleaned up {deleted_contexts} old contexts and {deleted_messages} orphaned messages")
            return deleted_contexts

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("PostgreSQL connection closed")
