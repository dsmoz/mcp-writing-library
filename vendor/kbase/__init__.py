"""
kbase-core - Core library for knowledge base systems

Provides reusable components for building knowledge base applications.

Submodules:
- kbase.models - Pydantic data models
- kbase.database - Database connection and managers
- kbase.core - Core utilities (cache, logger, document processing)
- kbase.jobs - Job queue and processing utilities
- kbase.utils - General utilities
- kbase.vector - Vector operations (embeddings, search)

Note: Submodules are not imported at the top level to avoid unnecessary
dependency loading. Import them directly when needed:

    from kbase.jobs import RetryManager
    from kbase.models import Collection
"""

__version__ = "1.3.0"

# Don't import submodules at the top level - let consumers import what they need
# This avoids pulling in dependencies that may not be installed (like email-validator)

__all__ = [
    "__version__",
    # Submodule names for documentation - import them directly
    # e.g., from kbase.jobs import ...
    # e.g., from kbase.models import ...
]
