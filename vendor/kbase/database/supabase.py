"""Generic Supabase client singleton.

Provides a shared client initialized from SUPABASE_URL + SUPABASE_SERVICE_KEY.
MCPs build their own domain-specific managers on top using get_supabase_client().

Requires the supabase optional extra:
    pip install kbase-core[supabase]
"""

import os
from typing import Optional

try:
    from supabase import create_client, Client
except ImportError:
    raise ImportError(
        "supabase package is required. Install with: pip install kbase-core[supabase]"
    )

_client: Optional["Client"] = None


def get_supabase_client() -> "Client":
    """Get or create Supabase client (singleton).

    Reads SUPABASE_URL and SUPABASE_SERVICE_KEY from environment.

    Returns:
        Supabase Client instance

    Raises:
        ValueError: If required environment variables are not set
    """
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        _client = create_client(url, key)
    return _client


def reset_supabase_client() -> "Client":
    """Force-create a fresh Supabase client (call after connection errors).

    Returns:
        New Supabase Client instance
    """
    global _client
    _client = None
    return get_supabase_client()
