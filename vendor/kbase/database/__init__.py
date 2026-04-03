"""
Database managers and utilities for kbase-core
"""

from kbase.database.pool import DatabasePool, create_pool, get_pool
from kbase.database.collections import CollectionManager
from kbase.database.documents import DocumentManager
from kbase.database.users import UserManager
from kbase.database.api_keys import APIKeyManager

# Supabase and Neo4j clients use optional extras — import lazily to avoid
# forcing all consumers to install these heavy dependencies.
def get_supabase_client():
    from kbase.database.supabase import get_supabase_client as _get
    return _get()

def reset_supabase_client():
    from kbase.database.supabase import reset_supabase_client as _reset
    return _reset()

__all__ = [
    # PostgreSQL pool
    "DatabasePool",
    "create_pool",
    "get_pool",
    # Managers
    "CollectionManager",
    "DocumentManager",
    "UserManager",
    "APIKeyManager",
    # Supabase (lazy — requires kbase-core[supabase])
    "get_supabase_client",
    "reset_supabase_client",
]
