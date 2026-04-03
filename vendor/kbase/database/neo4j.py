"""Neo4j graph manager for document relationship graphs.

Stores lightweight Document nodes (id only) and SIMILAR_TO / CITES edges.
All document metadata lives in your primary database (e.g. Supabase);
Neo4j is the relationship layer only.

Requires the neo4j optional extra:
    pip install kbase-core[neo4j]

Environment variables:
    NEO4J_URI       bolt://localhost:7687
    NEO4J_USER      neo4j
    NEO4J_PASSWORD  (required)
    NEO4J_DATABASE  (optional — uses driver default if unset)
"""

import logging
import os
from datetime import datetime, timezone

try:
    from neo4j import GraphDatabase
except ImportError:
    raise ImportError(
        "neo4j package is required. Install with: pip install kbase-core[neo4j]"
    )

logger = logging.getLogger(__name__)


class Neo4jGraph:
    """Manages a document relationship graph in Neo4j.

    Nodes
    -----
    :Document  {id: str}  — primary documents (UUID or any stable key)
    :ZoteroItem {id: str, title: str, item_type: str}  — Zotero library items

    Edges
    -----
    SIMILAR_TO  {score: float, computed_at: ISO8601}
    CITES       {detected_at: ISO8601, evidence: str}
    """

    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        if not password:
            raise ValueError("NEO4J_PASSWORD must be set")
        self._database = os.getenv("NEO4J_DATABASE") or None
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self._driver.close()

    # =========================================================================
    # Node management
    # =========================================================================

    def ensure_node(self, doc_id: str) -> None:
        """Create a Document node if it doesn't already exist (idempotent MERGE)."""
        with self._driver.session(database=self._database) as session:
            session.run("MERGE (d:Document {id: $id})", id=doc_id)

    def upsert_zotero_node(self, zotero_key: str, title: str = "", item_type: str = "") -> None:
        """Create or update a ZoteroItem node with metadata stored on the node."""
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MERGE (z:ZoteroItem {id: $id})
                SET z.title = $title, z.item_type = $item_type,
                    z.updated_at = $updated_at
                """,
                id=zotero_key,
                title=title or zotero_key,
                item_type=item_type or "",
                updated_at=datetime.now(timezone.utc).isoformat(),
            )

    # =========================================================================
    # Similarity edges
    # =========================================================================

    def upsert_similarity_edges(
        self,
        from_id: str,
        neighbors: list[dict],
        threshold: float = 0.65,
        max_neighbors: int = 10,
    ) -> int:
        """Write SIMILAR_TO edges from from_id to each qualifying neighbor.

        Filters out neighbors below `threshold` and skips self-loops.
        Uses MERGE so reindexing the same document is safe.

        Args:
            from_id:       Source document ID
            neighbors:     List of dicts with "document_id" (or "id") and "score" keys
            threshold:     Minimum score to write an edge (default 0.65)
            max_neighbors: Max edges to write per document (default 10)

        Returns:
            Number of edges written.
        """
        now = datetime.now(timezone.utc).isoformat()
        count = 0

        with self._driver.session(database=self._database) as session:
            session.run("MERGE (d:Document {id: $id})", id=from_id)

            for neighbor in neighbors[:max_neighbors]:
                neighbor_id = neighbor.get("document_id") or neighbor.get("id")
                score = float(neighbor.get("score", 0))

                if not neighbor_id or neighbor_id == from_id:
                    continue
                if score < threshold:
                    continue

                session.run(
                    """
                    MERGE (a:Document {id: $from_id})
                    MERGE (b:Document {id: $to_id})
                    MERGE (a)-[r:SIMILAR_TO]->(b)
                    SET r.score = $score, r.computed_at = $computed_at
                    """,
                    from_id=from_id,
                    to_id=neighbor_id,
                    score=score,
                    computed_at=now,
                )
                count += 1

        return count

    def upsert_zotero_similarity_edge(
        self,
        cerebellum_doc_id: str,
        zotero_key: str,
        score: float,
        title: str = "",
        item_type: str = "",
        threshold: float = 0.65,
    ) -> bool:
        """Write a SIMILAR_TO edge from a Document to a ZoteroItem.

        Returns True if the edge was written, False if below threshold.
        """
        if score < threshold:
            return False
        now = datetime.now(timezone.utc).isoformat()
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MERGE (a:Document {id: $from_id})
                MERGE (z:ZoteroItem {id: $zotero_key})
                SET z.title = $title, z.item_type = $item_type, z.updated_at = $now
                MERGE (a)-[r:SIMILAR_TO]->(z)
                SET r.score = $score, r.computed_at = $now
                """,
                from_id=cerebellum_doc_id,
                zotero_key=zotero_key,
                title=title or zotero_key,
                item_type=item_type or "",
                score=score,
                now=now,
            )
        return True

    # =========================================================================
    # Citation edges
    # =========================================================================

    def upsert_citation_edge(
        self,
        from_id: str,
        to_id: str,
        evidence: str = "",
    ) -> None:
        """Write a CITES edge from from_id to to_id (idempotent MERGE)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MERGE (a:Document {id: $from_id})
                MERGE (b:Document {id: $to_id})
                MERGE (a)-[r:CITES]->(b)
                SET r.detected_at = $detected_at, r.evidence = $evidence
                """,
                from_id=from_id,
                to_id=to_id,
                detected_at=now,
                evidence=evidence[:500],
            )

    def upsert_zotero_citation_edge(
        self,
        from_id: str,
        to_id: str,
        from_is_zotero: bool = False,
        to_is_zotero: bool = False,
        evidence: str = "",
        from_title: str = "",
        to_title: str = "",
        from_item_type: str = "",
        to_item_type: str = "",
    ) -> None:
        """Write a CITES edge between a Document and a ZoteroItem (either direction).

        Args:
            from_id / to_id:         IDs of the endpoints
            from_is_zotero / to_is_zotero: True if the ID is a Zotero key
            evidence:                Optional evidence text (truncated to 500 chars)
            from_title / to_title:   Stored on ZoteroItem nodes
        """
        now = datetime.now(timezone.utc).isoformat()
        from_label = "ZoteroItem" if from_is_zotero else "Document"
        to_label = "ZoteroItem" if to_is_zotero else "Document"

        with self._driver.session(database=self._database) as session:
            if from_is_zotero:
                session.run(
                    "MERGE (n:ZoteroItem {id: $id}) SET n.title = $title, n.item_type = $itype, n.updated_at = $now",
                    id=from_id, title=from_title or from_id, itype=from_item_type or "", now=now,
                )
            if to_is_zotero:
                session.run(
                    "MERGE (n:ZoteroItem {id: $id}) SET n.title = $title, n.item_type = $itype, n.updated_at = $now",
                    id=to_id, title=to_title or to_id, itype=to_item_type or "", now=now,
                )
            session.run(
                f"""
                MERGE (a:{from_label} {{id: $from_id}})
                MERGE (b:{to_label} {{id: $to_id}})
                MERGE (a)-[r:CITES]->(b)
                SET r.detected_at = $detected_at, r.evidence = $evidence
                """,
                from_id=from_id,
                to_id=to_id,
                detected_at=now,
                evidence=evidence[:500],
            )

    # =========================================================================
    # Queries
    # =========================================================================

    def get_related(
        self,
        doc_id: str,
        hops: int = 1,
        relation_type: str = "any",
        limit: int = 10,
    ) -> list[str]:
        """Return document IDs related to doc_id within N hops.

        Args:
            doc_id:        Anchor document ID
            hops:          1 = direct neighbors only, 2 = neighbors of neighbors (max 3)
            relation_type: "similar", "cites", "cited_by", or "any"
            limit:         Max number of related IDs to return

        Returns:
            List of document ID strings (not the anchor itself)
        """
        hops = max(1, min(hops, 3))

        if relation_type == "similar":
            cypher = (
                f"MATCH (a:Document {{id: $doc_id}})-[:SIMILAR_TO*1..{hops}]-(b) "
                f"WHERE (b:Document OR b:ZoteroItem) AND b.id <> $doc_id "
                f"RETURN DISTINCT b.id AS related_id LIMIT $limit"
            )
        elif relation_type == "cites":
            cypher = (
                f"MATCH (a:Document {{id: $doc_id}})-[:CITES*1..{hops}]->(b) "
                f"WHERE (b:Document OR b:ZoteroItem) AND b.id <> $doc_id "
                f"RETURN DISTINCT b.id AS related_id LIMIT $limit"
            )
        elif relation_type == "cited_by":
            cypher = (
                f"MATCH (a:Document {{id: $doc_id}})<-[:CITES*1..{hops}]-(b) "
                f"WHERE (b:Document OR b:ZoteroItem) AND b.id <> $doc_id "
                f"RETURN DISTINCT b.id AS related_id LIMIT $limit"
            )
        else:
            cypher = (
                f"MATCH (a:Document {{id: $doc_id}})-[:SIMILAR_TO|CITES*1..{hops}]-(b) "
                f"WHERE (b:Document OR b:ZoteroItem) AND b.id <> $doc_id "
                f"RETURN DISTINCT b.id AS related_id LIMIT $limit"
            )

        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, doc_id=doc_id, limit=limit)
            return [row["related_id"] for row in result]

    def get_all_edges(self, limit: int = 300) -> list[dict]:
        """Return all edges in the graph up to limit.

        Returns:
            List of dicts with keys: source, target, type, score,
            source_node_type, target_node_type, source_title, target_title,
            source_item_type, target_item_type.
        """
        cypher = """
            MATCH (a)-[r:SIMILAR_TO|CITES]->(b)
            WHERE (a:Document OR a:ZoteroItem) AND (b:Document OR b:ZoteroItem)
            RETURN a.id AS source, b.id AS target, type(r) AS rel_type,
                   coalesce(r.score, 0.0) AS score,
                   CASE WHEN a:ZoteroItem THEN 'zotero' ELSE 'document' END AS source_node_type,
                   CASE WHEN b:ZoteroItem THEN 'zotero' ELSE 'document' END AS target_node_type,
                   coalesce(a.title, a.id) AS source_title,
                   coalesce(b.title, b.id) AS target_title,
                   coalesce(a.item_type, '') AS source_item_type,
                   coalesce(b.item_type, '') AS target_item_type
            LIMIT $limit
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, limit=limit)
            return [
                {
                    "source": row["source"],
                    "target": row["target"],
                    "type": "similar" if row["rel_type"] == "SIMILAR_TO" else "cites",
                    "score": float(row["score"]),
                    "source_node_type": row["source_node_type"],
                    "target_node_type": row["target_node_type"],
                    "source_title": row["source_title"],
                    "target_title": row["target_title"],
                    "source_item_type": row["source_item_type"],
                    "target_item_type": row["target_item_type"],
                }
                for row in result
            ]

    def get_edges_for_documents(self, doc_ids: list[str], limit: int = 500) -> list[dict]:
        """Return all edges where at least one endpoint is in doc_ids.

        More efficient than get_all_edges when filtering by a known document set.
        """
        cypher = """
            MATCH (a)-[r:SIMILAR_TO|CITES]->(b)
            WHERE (a:Document OR a:ZoteroItem) AND (b:Document OR b:ZoteroItem)
              AND (a.id IN $doc_ids OR b.id IN $doc_ids)
            RETURN a.id AS source, b.id AS target, type(r) AS rel_type,
                   coalesce(r.score, 0.0) AS score,
                   CASE WHEN a:ZoteroItem THEN 'zotero' ELSE 'document' END AS source_node_type,
                   CASE WHEN b:ZoteroItem THEN 'zotero' ELSE 'document' END AS target_node_type,
                   coalesce(a.title, a.id) AS source_title,
                   coalesce(b.title, b.id) AS target_title,
                   coalesce(a.item_type, '') AS source_item_type,
                   coalesce(b.item_type, '') AS target_item_type
            LIMIT $limit
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, doc_ids=doc_ids, limit=limit)
            return [
                {
                    "source": row["source"],
                    "target": row["target"],
                    "type": "similar" if row["rel_type"] == "SIMILAR_TO" else "cites",
                    "score": float(row["score"]),
                    "source_node_type": row["source_node_type"],
                    "target_node_type": row["target_node_type"],
                    "source_title": row["source_title"],
                    "target_title": row["target_title"],
                    "source_item_type": row["source_item_type"],
                    "target_item_type": row["target_item_type"],
                }
                for row in result
            ]

    # =========================================================================
    # Deletion
    # =========================================================================

    def delete_document_edges(self, doc_id: str) -> None:
        """Remove all edges for a document (call before reindexing)."""
        with self._driver.session(database=self._database) as session:
            session.run(
                "MATCH (d:Document {id: $id})-[r]-() DELETE r",
                id=doc_id,
            )

    def delete_document_node(self, doc_id: str) -> None:
        """Delete Document node and all its edges from Neo4j."""
        with self._driver.session(database=self._database) as session:
            session.run(
                "MATCH (d:Document {id: $id}) DETACH DELETE d",
                id=doc_id,
            )
