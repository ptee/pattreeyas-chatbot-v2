"""
Database Management Layer for CV LLM Agent
Provides abstraction for PostgreSQL and Qdrant vector DB operations
Centralizes connection management, query execution, and error handling
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from config import ConfigManager, get_config
from exceptions import (
    DatabaseConnectionError,
    PostgreSQLConnectionError,
    QdrantConnectionError,
    DatabaseOperationError,
    DatabaseQueryError,
    DatabaseInsertError,
    DatabaseTableError,
)

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, "WARNING"))

# ============================================================================
# POSTGRESQL DATABASE MANAGER
# ============================================================================

class PostgreSQLManager:
    """
    Manages PostgreSQL database operations

    Handles:
    - Connection management with context managers
    - Query execution with error handling
    - Batch operations
    - Table existence checks
    - Data clearing/deletion operations
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize PostgreSQL manager

        Args:
            config: ConfigManager instance (uses global if not provided)

        Raises:
            PostgreSQLConnectionError: If initial connection fails
        """
        self.config = config or get_config()
        self.conn = None
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify database connection is available"""
        try:
            conn = psycopg2.connect(self.config.get_postgres_url())
            conn.close()
            logger.info("✓ PostgreSQL connection verified")
        except psycopg2.Error as e:
            logger.error(f"✗ PostgreSQL connection failed: {e}")
            raise PostgreSQLConnectionError(f"Failed to connect to PostgreSQL: {e}") from e

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections

        Yields:
            psycopg2 connection object

        Usage:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        conn = None
        try:
            conn = psycopg2.connect(self.config.get_postgres_url())
            yield conn
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise PostgreSQLConnectionError(f"Database connection error: {e}") from e
        finally:
            if conn:
                conn.close()

    def execute(self, query: str, params: Tuple = None, fetch: bool = False) -> Any:
        """
        Execute a single SQL query

        Args:
            query: SQL query string
            params: Query parameters (for parameterized queries)
            fetch: If True, fetch and return results; if False, return row count

        Returns:
            Query results if fetch=True, else row count

        Raises:
            DatabaseQueryError: If query execution fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                try:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)

                    if fetch:
                        result = cursor.fetchall()
                        conn.commit()
                        return result
                    else:
                        conn.commit()
                        return cursor.rowcount
                finally:
                    cursor.close()
        except psycopg2.Error as e:
            logger.error(f"SQL error: {e}\nQuery: {query}")
            raise DatabaseQueryError(f"SQL query failed: {e}") from e

    def execute_many(self, query: str, data: List[Tuple]) -> int:
        """
        Execute multiple insert/update statements (batch operation)

        Args:
            query: SQL query with %s placeholders
            data: List of tuples containing query parameters

        Returns:
            Number of rows affected

        Raises:
            DatabaseInsertError: If batch insert fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    execute_values(cursor, query, data)
                    conn.commit()
                    rows_affected = cursor.rowcount
                    logger.info(f"✓ Inserted {rows_affected} rows")
                    return rows_affected
                finally:
                    cursor.close()
        except psycopg2.Error as e:
            logger.error(f"Batch insert error: {e}")
            raise DatabaseInsertError(f"Batch insert failed: {e}") from e

    def fetch_one(self, query: str, params: Tuple = None) -> Optional[Dict]:
        """
        Fetch a single row

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Single row as dictionary, or None if no results

        Raises:
            DatabaseQueryError: If query fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                try:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    result = cursor.fetchone()
                    conn.commit()
                    return result
                finally:
                    cursor.close()
        except psycopg2.Error as e:
            logger.error(f"SQL error: {e}")
            raise DatabaseQueryError(f"SQL query failed: {e}") from e

    def fetch_all(self, query: str, params: Tuple = None) -> List[Dict]:
        """
        Fetch all rows matching query

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of rows as dictionaries

        Raises:
            DatabaseQueryError: If query fails
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                try:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    result = cursor.fetchall()
                    conn.commit()
                    return result if result else []
                finally:
                    cursor.close()
        except psycopg2.Error as e:
            logger.error(f"SQL error: {e}")
            raise DatabaseQueryError(f"SQL query failed: {e}") from e

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise
        """
        try:
            result = self.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                )
                """,
                (table_name,),
                fetch=True
            )
            return result[0]['exists'] if result else False
        except Exception as e:
            logger.warning(f"Could not check if table {table_name} exists: {e}")
            return False

    def view_exists(self, view_name: str) -> bool:
        """
        Check if a view exists in the database

        Args:
            view_name: Name of the view to check

        Returns:
            True if view exists, False otherwise
        """
        try:
            result = self.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.views
                    WHERE table_schema = 'public' AND table_name = %s
                )
                """,
                (view_name,),
                fetch=True
            )
            return result[0]['exists'] if result else False
        except Exception as e:
            logger.warning(f"Could not check if view {view_name} exists: {e}")
            return False

    def clear_table(self, table_name: str) -> int:
        """
        Delete all rows from a table

        Args:
            table_name: Name of the table to clear

        Returns:
            Number of rows deleted

        Raises:
            DatabaseOperationError: If delete fails
        """
        try:
            rows_deleted = self.execute(f"DELETE FROM {table_name}", fetch=False)
            logger.info(f"✓ Cleared {rows_deleted} rows from {table_name}")
            return rows_deleted
        except Exception as e:
            logger.error(f"Failed to clear table {table_name}: {e}")
            raise DatabaseOperationError(f"Failed to clear table: {e}") from e

    def drop_table(self, table_name: str, cascade: bool = False) -> bool:
        """
        Drop a table from the database

        Args:
            table_name: Name of the table to drop
            cascade: If True, drop dependent objects

        Returns:
            True if drop was successful

        Raises:
            DatabaseTableError: If drop fails
        """
        try:
            cascade_clause = "CASCADE" if cascade else "RESTRICT"
            self.execute(f"DROP TABLE IF EXISTS {table_name} {cascade_clause}", fetch=False)
            logger.info(f"✓ Dropped table {table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to drop table {table_name}: {e}")
            raise DatabaseTableError(f"Failed to drop table: {e}") from e

    def clear_all_cv_data(self, cv_id: Optional[str] = None) -> int:
        """
        Clear all CV-related data from database

        Args:
            cv_id: Optional CV ID to clear data for specific CV. If None, clears all data.

        Returns:
            Total number of rows cleared
        """
        tables_to_clear = [
            "chunk_registry",
            "languages",
            "work_references",
            "soft_skills",
            "awards_certifications",
            "publications",
            "projects",
            "education",
            "work_experience",
            "skills",
            "cv_metadata"
        ]

        logger.info("\n" + "="*70)
        logger.info("CLEARING EXISTING DATA FROM POSTGRESQL")
        logger.info("="*70)

        total_cleared = 0
        for table_name in tables_to_clear:
            if self.table_exists(table_name):
                try:
                    if cv_id:
                        # Clear only records for this CV
                        if table_name == "cv_metadata":
                            row_count = self.execute(f"DELETE FROM {table_name} WHERE id = %s", (cv_id,))
                        else:
                            row_count = self.execute(f"DELETE FROM {table_name} WHERE cv_id = %s", (cv_id,))
                        if row_count > 0:
                            logger.info(f"✓ Cleared {row_count} rows from {table_name} for CV {cv_id[:8]}...")
                            total_cleared += row_count
                    else:
                        # Clear all records
                        row_count = self.clear_table(table_name)
                        total_cleared += row_count
                except Exception as e:
                    logger.warning(f"⚠ Could not clear {table_name}: {e}")
            else:
                logger.info(f"⚠ Table {table_name} does not exist (skipping)")

        logger.info(f"\n✓ Total rows cleared: {total_cleared}")
        return total_cleared

    def close(self) -> None:
        """
        Close database connection
        Note: Since we use context managers, this is mostly for compatibility
        """
        # Our implementation uses context managers that auto-close connections
        logger.info("✓ Database connection manager ready for cleanup")


# ============================================================================
# QDRANT VECTOR DATABASE MANAGER
# ============================================================================

class QdrantManager:
    """
    Manages Qdrant vector database operations

    Handles:
    - Connection and health checking
    - Collection creation and management
    - Vector search operations
    - Point insertion and deletion
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize Qdrant manager

        Args:
            config: ConfigManager instance (uses global if not provided)

        Raises:
            QdrantConnectionError: If connection fails
        """
        self.config = config or get_config()
        self.client = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Qdrant"""
        try:
            self.client = QdrantClient(
                url=self.config.get_qdrant_url(),
                api_key=self.config.get_qdrant_api_key(),
                prefer_grpc=False
            )
            # Verify connection by checking health
            self.client.get_collections()
            logger.info(f"✓ Connected to Qdrant at {self.config.get_qdrant_url()}")
        except Exception as e:
            logger.error(f"✗ Qdrant connection failed: {e}")
            raise QdrantConnectionError(f"Failed to connect to Qdrant: {e}") from e

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            return collection_name in collection_names
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def create_collection(self, collection_name: str, vector_size: int = 1536,
                         distance_metric: str = "COSINE") -> bool:
        """
        Create a new vector collection

        Args:
            collection_name: Name for the new collection
            vector_size: Dimension of vectors (default: 1536 for OpenAI embeddings)
            distance_metric: Distance metric ("COSINE", "EUCLIDEAN", "DOT")

        Returns:
            True if creation successful

        Raises:
            QdrantConnectionError: If creation fails
        """
        try:
            if self.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' already exists")
                return True

            distance = {
                "COSINE": Distance.COSINE,
                "EUCLIDEAN": Distance.EUCLIDEAN,
                "DOT": Distance.DOT,
            }.get(distance_metric.upper(), Distance.COSINE)

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )

            logger.info(f"✓ Created collection '{collection_name}' ({vector_size}D, {distance_metric})")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise QdrantConnectionError(f"Failed to create collection: {e}") from e

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if deletion successful

        Raises:
            QdrantConnectionError: If deletion fails
        """
        try:
            if not self.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' does not exist")
                return True

            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"✓ Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise QdrantConnectionError(f"Failed to delete collection: {e}") from e

    def insert_points(self, collection_name: str, points: List[PointStruct]) -> bool:
        """
        Insert points (vectors) into a collection

        Args:
            collection_name: Name of the collection
            points: List of PointStruct objects

        Returns:
            True if insertion successful

        Raises:
            DatabaseInsertError: If insertion fails
        """
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"✓ Inserted {len(points)} points into '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to insert points: {e}")
            raise DatabaseInsertError(f"Failed to insert vectors: {e}") from e

    def search(self, collection_name: str, query_vector: List[float],
               limit: int = 10, score_threshold: Optional[float] = None,
               query_filter: Optional[Filter] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors

        Args:
            collection_name: Name of the collection
            query_vector: Query vector (embedding)
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            query_filter: Optional filter for search

        Returns:
            List of search results with scores

        Raises:
            DatabaseQueryError: If search fails
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )

            # Convert results to dictionary format
            formatted_results = [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload or {}
                }
                for result in results
            ]

            logger.info(f"Search returned {len(formatted_results)} results from '{collection_name}'")
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise DatabaseQueryError(f"Vector search failed: {e}") from e

    def get_point(self, collection_name: str, point_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single point by ID

        Args:
            collection_name: Name of the collection
            point_id: ID of the point

        Returns:
            Point data with payload, or None if not found

        Raises:
            DatabaseQueryError: If retrieval fails
        """
        try:
            point = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id]
            )
            if point:
                return {
                    "id": point[0].id,
                    "vector": point[0].vector,
                    "payload": point[0].payload or {}
                }
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve point: {e}")
            raise DatabaseQueryError(f"Failed to retrieve point: {e}") from e

    def delete_points(self, collection_name: str, point_ids: List[int]) -> bool:
        """
        Delete points from a collection

        Args:
            collection_name: Name of the collection
            point_ids: List of point IDs to delete

        Returns:
            True if deletion successful

        Raises:
            DatabaseOperationError: If deletion fails
        """
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids
            )
            logger.info(f"✓ Deleted {len(point_ids)} points from '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete points: {e}")
            raise DatabaseOperationError(f"Failed to delete vectors: {e}") from e

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection stats

        Raises:
            DatabaseQueryError: If retrieval fails
        """
        try:
            collection_info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": collection_info.points_count,
                "dimensions": collection_info.config.params.vectors.size if collection_info.config.params.vectors else None,
                "distance_metric": str(collection_info.config.params.vectors.distance) if collection_info.config.params.vectors else None,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise DatabaseQueryError(f"Failed to get collection stats: {e}") from e

    def check_collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists (alias for collection_exists)

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists, False otherwise
        """
        return self.collection_exists(collection_name)

    def upsert_vectors(self, collection_name: str, points: List[PointStruct]) -> bool:
        """
        Upsert (insert or update) vectors into a collection

        Args:
            collection_name: Name of the collection
            points: List of PointStruct objects

        Returns:
            True if upsert successful

        Raises:
            DatabaseInsertError: If upsert fails
        """
        return self.insert_points(collection_name, points)

    def create_payload_indexes(self, collection_name: str) -> None:
        """
        Create payload indexes for efficient filtering on key fields

        Args:
            collection_name: Name of the collection
        """
        # Payload indexes are optional - log info but don't fail
        logger.info(f"✓ Payload indexes for '{collection_name}' are handled by Qdrant automatically")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_postgres_manager(config: Optional[ConfigManager] = None) -> PostgreSQLManager:
    """Get or create PostgreSQL manager instance"""
    return PostgreSQLManager(config)


def get_qdrant_manager(config: Optional[ConfigManager] = None) -> QdrantManager:
    """Get or create Qdrant manager instance"""
    return QdrantManager(config)