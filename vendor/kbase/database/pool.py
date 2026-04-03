"""
Database connection pool management for PostgreSQL
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import asyncpg
from asyncpg import Pool
import structlog

logger = structlog.get_logger(__name__)

# Global pool instance
_pool: Optional[Pool] = None
_pool_lock = asyncio.Lock()


class DatabasePool:
    """
    PostgreSQL connection pool manager.

    Provides async connection pooling with automatic lifecycle management.
    """

    def __init__(
        self,
        dsn: str,
        min_size: int = 10,
        max_size: int = 20,
        command_timeout: float = 60.0,
        max_queries: int = 50000,
        max_inactive_connection_lifetime: float = 300.0,
    ):
        """
        Initialize database pool configuration.

        Args:
            dsn: PostgreSQL connection string
            min_size: Minimum number of connections in pool
            max_size: Maximum number of connections in pool
            command_timeout: Timeout for SQL commands (seconds)
            max_queries: Max queries per connection before recycling
            max_inactive_connection_lifetime: Max idle time before connection recycling (seconds)
        """
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.command_timeout = command_timeout
        self.max_queries = max_queries
        self.max_inactive_connection_lifetime = max_inactive_connection_lifetime
        self._pool: Optional[Pool] = None

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._pool is not None:
            logger.warning("Pool already initialized")
            return

        try:
            logger.info(
                "Initializing database pool",
                min_size=self.min_size,
                max_size=self.max_size,
            )

            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=self.command_timeout,
                max_queries=self.max_queries,
                max_inactive_connection_lifetime=self.max_inactive_connection_lifetime,
            )

            logger.info("Database pool initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize database pool", error=str(e))
            raise

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is None:
            logger.warning("Pool not initialized")
            return

        try:
            logger.info("Closing database pool")
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed")

        except Exception as e:
            logger.error("Error closing database pool", error=str(e))
            raise

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """
        Acquire a connection from the pool.

        Usage:
            async with pool.acquire() as conn:
                result = await conn.fetch("SELECT * FROM table")
        """
        if self._pool is None:
            raise RuntimeError("Pool not initialized. Call initialize() first.")

        async with self._pool.acquire() as connection:
            yield connection

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        """
        Acquire a connection with an automatic transaction.

        The transaction is automatically committed on success or rolled back on error.

        Usage:
            async with pool.transaction() as conn:
                await conn.execute("INSERT INTO table VALUES ($1)", value)
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def execute(self, query: str, *args) -> str:
        """
        Execute a query and return status.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Query execution status
        """
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list:
        """
        Execute a query and fetch all results.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            List of records
        """
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """
        Execute a query and fetch one result.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single record or None
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args, column: int = 0):
        """
        Execute a query and fetch a single value.

        Args:
            query: SQL query
            *args: Query parameters
            column: Column index to return

        Returns:
            Single value
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args, column=column)

    @property
    def pool(self) -> Pool:
        """Get the underlying asyncpg pool."""
        if self._pool is None:
            raise RuntimeError("Pool not initialized")
        return self._pool


# Global pool management functions


async def create_pool(
    dsn: str,
    min_size: int = 10,
    max_size: int = 20,
    command_timeout: float = 60.0,
    **kwargs,
) -> DatabasePool:
    """
    Create and initialize a global database pool.

    Args:
        dsn: PostgreSQL connection string
        min_size: Minimum pool size
        max_size: Maximum pool size
        command_timeout: Command timeout in seconds
        **kwargs: Additional pool configuration

    Returns:
        Initialized DatabasePool instance
    """
    global _pool

    async with _pool_lock:
        if _pool is not None:
            logger.warning("Global pool already exists")
            return _pool

        pool = DatabasePool(
            dsn=dsn,
            min_size=min_size,
            max_size=max_size,
            command_timeout=command_timeout,
            **kwargs,
        )
        await pool.initialize()
        _pool = pool
        return pool


async def get_pool() -> DatabasePool:
    """
    Get the global database pool.

    Returns:
        Global DatabasePool instance

    Raises:
        RuntimeError: If pool not initialized
    """
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call create_pool() first.")
    return _pool


async def close_pool() -> None:
    """Close the global database pool."""
    global _pool

    async with _pool_lock:
        if _pool is not None:
            await _pool.close()
            _pool = None
