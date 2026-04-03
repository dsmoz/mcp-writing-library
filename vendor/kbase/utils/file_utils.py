"""SHA-256 file hashing utilities for document change detection."""

import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB


def compute_sha256(file_path: str) -> Optional[str]:
    """
    Compute SHA-256 hash of a file.

    Args:
        file_path: Absolute path to the file

    Returns:
        Hex digest string, or None if file doesn't exist or can't be read
    """
    path = Path(file_path)
    if not path.exists():
        return None

    try:
        if path.stat().st_size > _MAX_FILE_SIZE:
            logger.warning(f"File too large to hash (>{_MAX_FILE_SIZE // 1024 // 1024}MB): {file_path}")
            return None
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError as e:
        logger.warning(f"Could not hash file {file_path}: {e}")
        return None


def check_file_changed(file_path: str, stored_hash: Optional[str]) -> bool:
    """
    Check whether a file has changed since it was last indexed.

    Args:
        file_path: Absolute path to the file
        stored_hash: Previously computed SHA-256 hash

    Returns:
        True if the file has changed or stored_hash is None
    """
    if not stored_hash:
        return True  # No hash stored — assume changed

    current_hash = compute_sha256(file_path)
    if current_hash is None:
        return True  # File missing — treat as changed

    return current_hash != stored_hash
