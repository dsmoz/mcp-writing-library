#!/usr/bin/env python3
"""
Setup Qdrant collections for the writing library.

Run once before first use:
    uv run python scripts/setup_collections.py
"""
import sys
from pathlib import Path

# Ensure imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
kbase_core_path = project_root.parent / 'libraries' / 'kbase-core'
if kbase_core_path.exists():
    sys.path.insert(0, str(kbase_core_path))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from src.tools.collections import setup_collections, get_stats

if __name__ == "__main__":
    print("Setting up writing library collections...")
    results = setup_collections()

    for key, info in results.items():
        status = info.get("status", "unknown")
        collection = info.get("collection")
        if status == "error":
            print(f"  ❌ {key}: {collection} — {info.get('error')}")
        else:
            print(f"  ✅ {key}: {collection} — {status}")

    print("\nCollection stats:")
    stats = get_stats()
    for key, info in stats.items():
        print(f"  {key}: {info.get('points_count', 0)} points")
