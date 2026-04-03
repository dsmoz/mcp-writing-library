"""Generic in-process LRU+TTL cache for expensive operations (embeddings, API calls).

Thread-safe via threading.Lock. Entries expire lazily on access.
When max_size is set, oldest entries are evicted on overflow (LRU order).
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class TTLCache:
    """
    In-process key-value cache with time-to-live expiry and optional LRU eviction.

    Thread-safe. Entries expire lazily on access — no background cleanup thread.
    When max_size is set, the least-recently-used entry is evicted on overflow.

    Args:
        ttl:      Time-to-live in seconds (default 300 = 5 minutes)
        max_size: Maximum number of entries. None = unbounded (default).
                  Set a bound to prevent unbounded memory growth in long-running
                  processes (e.g. max_size=1000 for embedding caches).

    Example:
        cache = TTLCache(ttl=300, max_size=500)
        cache.set("key", value)
        value = cache.get("key")   # None if expired or missing
    """

    def __init__(self, ttl: int = 300, max_size: Optional[int] = None):
        self._ttl = ttl
        self._max_size = max_size
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Return cached value, or None if missing or expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, ts = entry
            if time.time() - ts > self._ttl:
                del self._store[key]
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Store value with current timestamp, evicting LRU entry if at capacity."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = (value, time.time())
                return
            if self._max_size is not None and len(self._store) >= self._max_size:
                self._store.popitem(last=False)  # evict oldest
            self._store[key] = (value, time.time())

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        """Return number of entries (including potentially expired ones)."""
        with self._lock:
            return len(self._store)
