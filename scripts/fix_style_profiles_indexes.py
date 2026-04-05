#!/usr/bin/env python3
"""
One-off fix: create missing 'name' and 'channel' payload indexes
on existing style_profiles collections.

    uv run python scripts/fix_style_profiles_indexes.py
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
kbase_core_path = project_root.parent / 'libraries' / 'kbase-core'
if kbase_core_path.exists():
    sys.path.insert(0, str(kbase_core_path))

from dotenv import load_dotenv
load_dotenv(project_root / '.env')

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
import os


def main():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        print("ERROR: QDRANT_URL not set")
        sys.exit(1)

    client = QdrantClient(url=url, api_key=api_key)

    # Find all style_profiles collections
    collections = client.get_collections().collections
    targets = [c.name for c in collections if c.name.endswith("_writing_style_profiles")]

    if not targets:
        print("No style_profiles collections found.")
        return

    for coll in targets:
        print(f"\nFixing indexes on: {coll}")
        for field in ("name", "channel"):
            try:
                client.create_payload_index(
                    collection_name=coll,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                print(f"  ✓ Created '{field}' index")
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"  – '{field}' index already exists, skipping")
                else:
                    print(f"  ✗ Failed to create '{field}' index: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
