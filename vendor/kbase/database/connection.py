"""
Backward compatibility layer for database connection management.

Re-exports sync pool functions with legacy names used by mcp-cerebellum.
This module provides get_db_pool and close_db_pool as aliases for
the sync pool functions for MCP server compatibility.
"""

from kbase.database.sync_pool import get_sync_pool as get_db_pool, close_sync_pool as close_db_pool

__all__ = ["get_db_pool", "close_db_pool"]
