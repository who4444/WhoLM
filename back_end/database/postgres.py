import time
import random
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import json
from config.config import Config

logger = logging.getLogger(__name__)

# Retry config
MAX_RETRIES = 3
BASE_DELAY = 0.5  # seconds
MAX_DELAY = 5.0


class PostgresDB:
    """PostgreSQL database with connection pooling and retry logic."""

    def __init__(self, min_conn: int = 1, max_conn: int = 5):
        self.config = Config()
        self._pool: Optional[pool.ThreadedConnectionPool] = None
        self._connect_pool(min_conn, max_conn)
        self.create_tables()

    def _connect_pool(self, min_conn: int, max_conn: int):
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                host=self.config.POSTGRES_HOST,
                port=self.config.POSTGRES_PORT,
                database=self.config.POSTGRES_DB,
                user=self.config.POSTGRES_USER,
                password=self.config.POSTGRES_PASSWORD,
            )
            logger.info(f"PostgreSQL pool created (min={min_conn}, max={max_conn})")
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL pool: {e}")
            raise

    def _get_conn(self):
        """Get a connection from the pool with retry on pool exhaustion."""
        for attempt in range(MAX_RETRIES):
            try:
                conn = self._pool.getconn()
                conn.autocommit = True
                return conn
            except psycopg2.pool.PoolError:
                if attempt < MAX_RETRIES - 1:
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.2), MAX_DELAY)
                    logger.warning(f"Pool exhausted, retrying in {delay:.1f}s (attempt {attempt + 1})")
                    time.sleep(delay)
                else:
                    raise

    def _put_conn(self, conn):
        self._pool.putconn(conn)

    def _execute(self, fn: Callable, read_only: bool = False):
        """Execute a database operation with connection management and retry."""
        last_exc = None
        for attempt in range(MAX_RETRIES):
            conn = None
            try:
                conn = self._get_conn()
                result = fn(conn)
                return result
            except psycopg2.errors.OperationalError as e:
                last_exc = e
                if conn:
                    try:
                        self._pool.putconn(conn, close=True)
                    except Exception:
                        pass
                    conn = None
                if attempt < MAX_RETRIES - 1:
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.3), MAX_DELAY)
                    logger.warning(f"DB operational error, retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
                else:
                    raise
            except Exception:
                raise
            finally:
                if conn:
                    self._pool.putconn(conn)

        raise last_exc

    def create_tables(self):
        """Create necessary tables for chatbot memory."""
        def _do(conn):
            with conn.cursor() as cursor:
                try:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    logger.info("pgvector extension enabled")
                except Exception as e:
                    logger.warning(f"Could not enable pgvector extension: {e}. Make sure it's installed.")

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        message_id VARCHAR(255) UNIQUE NOT NULL,
                        session_id VARCHAR(255) NOT NULL,
                        role VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        metadata JSONB,
                        embedding VECTOR(1024),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_contexts (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) UNIQUE NOT NULL,
                        user_id VARCHAR(255),
                        topic VARCHAR(255),
                        last_activity TIMESTAMP NOT NULL,
                        preferences JSONB DEFAULT '{}',
                        custom_data JSONB DEFAULT '{}',
                        embedding VECTOR(1024),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
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
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS content_uploads (
                        id SERIAL PRIMARY KEY,
                        content_id VARCHAR(255) UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        content_type VARCHAR(100) NOT NULL,
                        storage_path TEXT,
                        youtube_url TEXT,
                        status VARCHAR(50) DEFAULT 'pending',
                        error TEXT,
                        processing_result JSONB,
                        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_content_uploads_status ON content_uploads(status);
                """)

                # Add retry_count column if it doesn't exist (safe migration)
                try:
                    cursor.execute("""
                        ALTER TABLE content_uploads ADD COLUMN retry_count INTEGER DEFAULT 0;
                    """)
                    logger.info("Added retry_count column to content_uploads")
                except Exception:
                    conn.rollback()
                    logger.debug("retry_count column already exists")

                logger.info("Database tables created successfully")

        self._execute(_do)

    # ── Content Uploads CRUD ──

    def insert_content(self, content_id: str, name: str, content_type: str,
                       storage_path: str = None, youtube_url: str = None,
                       status: str = "processing") -> None:
        def _do(conn):
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO content_uploads (content_id, name, content_type, storage_path, youtube_url, status)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (content_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        status = EXCLUDED.status,
                        updated_at = CURRENT_TIMESTAMP;
                """, (content_id, name, content_type, storage_path, youtube_url, status))
        self._execute(_do)

    def update_content_status(self, content_id: str, status: str,
                               error: str = None, processing_result: dict = None) -> None:
        def _do(conn):
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE content_uploads
                    SET status = %s, error = %s, processing_result = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE content_id = %s;
                """, (status, error, json.dumps(processing_result) if processing_result else None,
                      content_id))
        self._execute(_do)

    def get_all_content(self) -> List[Dict[str, Any]]:
        def _do(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT content_id, name, content_type, storage_path, youtube_url,
                           status, error, processing_result, retry_count, upload_time
                    FROM content_uploads
                    ORDER BY upload_time DESC;
                """)
                return [dict(row) for row in cursor.fetchall()]
        return self._execute(_do, read_only=True)

    def get_content_by_id(self, content_id: str) -> Optional[Dict[str, Any]]:
        def _do(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT content_id, name, content_type, storage_path, youtube_url,
                           status, error, processing_result, retry_count, upload_time
                    FROM content_uploads
                    WHERE content_id = %s;
                """, (content_id,))
                result = cursor.fetchone()
                return dict(result) if result else None
        return self._execute(_do, read_only=True)

    def delete_content(self, content_id: str) -> bool:
        def _do(conn):
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM content_uploads WHERE content_id = %s;", (content_id,))
                return cursor.rowcount > 0
        return self._execute(_do)

    def recover_stuck_tasks(self, stuck_minutes: int = 30) -> int:
        """Mark content stuck in transient states as failed (e.g., after restart).
        Only fails content that hasn't exceeded max retries."""
        MAX_RETRIES = 3

        def _do(conn):
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE content_uploads
                    SET status = 'failed',
                        error = 'Task interrupted (server restart or crash)',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE status IN ('processing', 'downloading')
                      AND retry_count >= %s
                      AND updated_at < CURRENT_TIMESTAMP - make_interval(mins => %s);
                """, (MAX_RETRIES, stuck_minutes))
                failed = cursor.rowcount

                if failed > 0:
                    logger.warning(f"Marked {failed} exhausted tasks as failed")

                cursor.execute("""
                    SELECT content_id, name, content_type, retry_count
                    FROM content_uploads
                    WHERE status IN ('processing', 'downloading')
                      AND retry_count < %s
                      AND updated_at < CURRENT_TIMESTAMP - make_interval(mins => %s);
                """, (MAX_RETRIES, stuck_minutes))
                stuck = [dict(row) for row in cursor.fetchall()]

                if stuck:
                    logger.warning(f"Found {len(stuck)} stuck task(s) eligible for retry")
                    for task in stuck:
                        logger.info(f"  Stuck: {task['content_id']} ({task['name']}) "
                                    f"retries={task['retry_count']}")

                return failed + len(stuck)

        return self._execute(_do)

    def increment_retry_count(self, content_id: str) -> int:
        """Increment retry count and return the new value."""
        def _do(conn):
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE content_uploads
                    SET retry_count = retry_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE content_id = %s
                    RETURNING retry_count;
                """, (content_id,))
                result = cursor.fetchone()
                return result[0] if result else 0
        return self._execute(_do)

    # ── Messages ──

    def insert_message(self, message_id: str, session_id: str, role: str,
                      content: str, timestamp: datetime, metadata: Dict[str, Any],
                      embedding: List[float]) -> None:
        def _do(conn):
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO messages (message_id, session_id, role, content, timestamp, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (message_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding;
                """, (message_id, session_id, role, content, timestamp, json.dumps(metadata), embedding))
        self._execute(_do)

    def insert_context(self, session_id: str, user_id: Optional[str], topic: Optional[str],
                      last_activity: datetime, preferences: Dict[str, Any],
                      custom_data: Dict[str, Any], embedding: List[float]) -> None:
        def _do(conn):
            with conn.cursor() as cursor:
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
        self._execute(_do)

    def get_messages_by_session(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        def _do(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT message_id, session_id, role, content, timestamp, metadata
                    FROM messages
                    WHERE session_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s;
                """, (session_id, limit))
                return [dict(row) for row in cursor.fetchall()]
        return self._execute(_do, read_only=True)

    def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        def _do(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT session_id, user_id, topic, last_activity, preferences, custom_data
                    FROM conversation_contexts
                    WHERE session_id = %s;
                """, (session_id,))
                result = cursor.fetchone()
                return dict(result) if result else None
        return self._execute(_do, read_only=True)

    def search_similar_messages(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        def _do(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT message_id, session_id, role, content, timestamp, metadata,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM messages
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (embedding, embedding, limit))
                return [dict(row) for row in cursor.fetchall()]
        return self._execute(_do, read_only=True)

    def search_similar_contexts(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        def _do(conn):
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT session_id, user_id, topic, last_activity, preferences, custom_data,
                           1 - (embedding <=> %s::vector) as similarity
                    FROM conversation_contexts
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (embedding, embedding, limit))
                return [dict(row) for row in cursor.fetchall()]
        return self._execute(_do, read_only=True)

    def cleanup_old_sessions(self, max_age_days: int) -> int:
        def _do(conn):
            with conn.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM conversation_contexts
                    WHERE last_activity < CURRENT_TIMESTAMP - make_interval(days => %s);
                """, (max_age_days,))
                deleted_contexts = cursor.rowcount

                cursor.execute("""
                    DELETE FROM messages
                    WHERE session_id NOT IN (
                        SELECT session_id FROM conversation_contexts
                    );
                """)
                deleted_messages = cursor.rowcount

                logger.info(f"Cleaned up {deleted_contexts} old contexts and {deleted_messages} orphaned messages")
                return deleted_contexts
        return self._execute(_do)

    def close(self):
        """Close all pool connections."""
        if self._pool:
            self._pool.closeall()
            logger.info("PostgreSQL pool closed")
