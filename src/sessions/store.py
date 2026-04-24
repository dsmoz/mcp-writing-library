"""SQLite-backed review session store.

Database path is taken from the SESSIONS_DB_PATH env var (Railway Volume mount).
Falls back to a local .sessions.db file for stdio/dev mode.

Schema
------
sessions(
    id          TEXT PRIMARY KEY,
    client_id   TEXT NOT NULL,
    name        TEXT NOT NULL,
    items       TEXT NOT NULL,   -- JSON list[ReviewItem]
    decisions   TEXT NOT NULL,   -- JSON list[Decision]
    status      TEXT NOT NULL,   -- open | completed
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
)
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from src.sessions.models import Decision, ReviewItem

_EXPIRE_DAYS = 30
_DB_PATH: Optional[Path] = None


def _db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is not None:
        return _DB_PATH
    env = os.getenv("SESSIONS_DB_PATH", "").strip()
    _DB_PATH = Path(env) if env else Path(__file__).parent.parent.parent / ".sessions.db"
    return _DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id         TEXT PRIMARY KEY,
            client_id  TEXT NOT NULL,
            name       TEXT NOT NULL,
            items      TEXT NOT NULL,
            decisions  TEXT NOT NULL,
            status     TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session(
    client_id: str,
    items: list[ReviewItem],
    name: Optional[str] = None,
) -> str:
    session_id = str(uuid4())
    session_name = name or f"Review — {datetime.now(timezone.utc).strftime('%b %d, %Y %H:%M')} UTC"
    now = _now()
    with _connect() as conn:
        _ensure_schema(conn)
        _purge_expired(conn)
        conn.execute(
            "INSERT INTO sessions (id, client_id, name, items, decisions, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                client_id,
                session_name,
                json.dumps([i.model_dump() for i in items]),
                json.dumps([]),
                "open",
                now,
                now,
            ),
        )
    return session_id


def load_session(session_id: str, client_id: str) -> Optional[dict]:
    with _connect() as conn:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ? AND client_id = ?",
            (session_id, client_id),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "client_id": row["client_id"],
        "name": row["name"],
        "items": json.loads(row["items"]),
        "decisions": json.loads(row["decisions"]),
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def save_decisions(session_id: str, client_id: str, decisions: list[Decision]) -> bool:
    now = _now()
    with _connect() as conn:
        _ensure_schema(conn)
        row = conn.execute(
            "SELECT decisions FROM sessions WHERE id = ? AND client_id = ?",
            (session_id, client_id),
        ).fetchone()
        if row is None:
            return False
        existing = {d["item_id"]: d for d in json.loads(row["decisions"])}
        for d in decisions:
            existing[d.item_id] = d.model_dump()
        conn.execute(
            "UPDATE sessions SET decisions = ?, status = ?, updated_at = ? WHERE id = ? AND client_id = ?",
            (json.dumps(list(existing.values())), "completed", now, session_id, client_id),
        )
    return True


def list_sessions(client_id: str, status: Optional[str] = "open") -> list[dict]:
    with _connect() as conn:
        _ensure_schema(conn)
        _purge_expired(conn)
        if status and status != "all":
            rows = conn.execute(
                "SELECT id, name, status, items, decisions, created_at FROM sessions "
                "WHERE client_id = ? AND status = ? ORDER BY created_at DESC",
                (client_id, status),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name, status, items, decisions, created_at FROM sessions "
                "WHERE client_id = ? ORDER BY created_at DESC",
                (client_id,),
            ).fetchall()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "status": r["status"],
            "item_count": len(json.loads(r["items"])),
            "decision_count": len(json.loads(r["decisions"])),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def _purge_expired(conn: sqlite3.Connection) -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_EXPIRE_DAYS)).isoformat()
    conn.execute("DELETE FROM sessions WHERE created_at < ?", (cutoff,))
    conn.commit()
