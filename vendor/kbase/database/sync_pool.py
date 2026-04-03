"""
Synchronous database pool for MCP server compatibility.

Uses psycopg (sync) instead of asyncpg (async) for simpler integration
with MCP servers that don't require full async.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
import structlog

logger = structlog.get_logger(__name__)

# Global pool instance
_sync_pool: Optional["SyncDatabasePool"] = None


class SyncDatabasePool:
    """
    Synchronous PostgreSQL connection pool using psycopg.

    Provides a simpler sync interface for MCP servers.
    """

    def __init__(
        self,
        dsn: str,
        min_size: int = 5,
        max_size: int = 10,
    ):
        """
        Initialize sync database pool.

        Args:
            dsn: PostgreSQL connection string
            min_size: Minimum connections
            max_size: Maximum connections
        """
        self.dsn = self._convert_dsn(dsn)
        self.min_size = min_size
        self.max_size = max_size
        self._pool = None

    def _convert_dsn(self, dsn: str) -> str:
        """Convert asyncpg DSN format to psycopg format if needed."""
        # Handle postgresql+psycopg:// prefix
        if dsn.startswith("postgresql+psycopg://"):
            dsn = dsn.replace("postgresql+psycopg://", "postgresql://")
        return dsn

    def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._pool is not None:
            logger.warning("Sync pool already initialized")
            return

        try:
            logger.info(
                "Initializing sync database pool",
                min_size=self.min_size,
                max_size=self.max_size,
            )

            self._pool = ConnectionPool(
                self.dsn,
                min_size=self.min_size,
                max_size=self.max_size,
                kwargs={"row_factory": dict_row},
            )

            logger.info("Sync database pool initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize sync database pool", error=str(e))
            raise

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool is None:
            logger.warning("Sync pool not initialized")
            return

        try:
            logger.info("Closing sync database pool")
            self._pool.close()
            self._pool = None
            logger.info("Sync database pool closed")

        except Exception as e:
            logger.error("Error closing sync database pool", error=str(e))
            raise

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.

        Usage:
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
                results = cursor.fetchall()
        """
        if self._pool is None:
            raise RuntimeError("Sync pool not initialized. Call initialize() first.")

        with self._pool.connection() as conn:
            yield conn

    def execute(self, query: str, params: tuple = None) -> None:
        """Execute a query."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)

    def fetch(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute a query and fetch all results."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

    def fetchrow(self, query: str, params: tuple = None) -> Optional[Dict[str, Any]]:
        """Execute a query and fetch one result."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchone()

    def fetchval(self, query: str, params: tuple = None, column: int = 0) -> Any:
        """Execute a query and fetch a single value."""
        row = self.fetchrow(query, params)
        if row is None:
            return None
        # dict_row returns a dict, get value by index from values
        values = list(row.values())
        return values[column] if column < len(values) else None


# Global pool management functions

def get_sync_pool() -> SyncDatabasePool:
    """
    Get or create the global sync database pool.

    Returns:
        SyncDatabasePool instance
    """
    global _sync_pool

    if _sync_pool is None:
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            raise RuntimeError("DATABASE_URL environment variable not set")

        _sync_pool = SyncDatabasePool(dsn)
        _sync_pool.initialize()

    return _sync_pool


def close_sync_pool() -> None:
    """Close the global sync database pool."""
    global _sync_pool

    if _sync_pool is not None:
        _sync_pool.close()
        _sync_pool = None


# Alias for backward compatibility with MCP server
get_db_pool = get_sync_pool
close_db_pool = close_sync_pool

__all__ = [
    "SyncDatabasePool",
    "get_sync_pool",
    "close_sync_pool",
    "get_db_pool",
    "close_db_pool",
]
