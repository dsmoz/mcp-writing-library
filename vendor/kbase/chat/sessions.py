"""Chat session storage — SQLite backend.

Database path: CHAT_DB_PATH env var (default: .data/chats.db in cwd)

Schema:
    sessions(id TEXT PK, title TEXT, created_at TEXT, updated_at TEXT,
             messages TEXT (JSON), zotero_keys TEXT (JSON, nullable))

Migration: on first run (DB absent) existing .data/chats/*.json files are
imported automatically from cwd/.data/chats/. Old files are left untouched.
"""

import json
import os
import sqlite3
import time
from pathlib import Path

def _get_db_path() -> Path:
    override = os.getenv("CHAT_DB_PATH")
    return Path(override) if override else Path.cwd() / ".data" / "chats.db"


_db_initialized = False

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    messages    TEXT NOT NULL DEFAULT '[]',
    zotero_keys TEXT
);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_get_db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_db() -> None:
    global _db_initialized
    if _db_initialized:
        return
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    first_run = not db_path.exists()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        conn.execute(_CREATE_TABLE)
        conn.commit()
    finally:
        conn.close()
    if first_run:
        _migrate_json_files()
    _db_initialized = True


def _migrate_json_files() -> None:
    """One-time import of existing .data/chats/*.json files into SQLite."""
    chats_dir = _get_db_path().parent / "chats"
    if not chats_dir.exists():
        return
    conn = _get_conn()
    try:
        for path in chats_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                sid = data.get("id") or path.stem
                conn.execute(
                    "INSERT OR IGNORE INTO sessions "
                    "(id, title, created_at, updated_at, messages, zotero_keys) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        sid,
                        data.get("title", sid),
                        data.get("created_at", ""),
                        data.get("updated_at", ""),
                        json.dumps(data.get("messages", []), ensure_ascii=False),
                        json.dumps(data["zotero_keys"], ensure_ascii=False)
                        if "zotero_keys" in data else None,
                    ),
                )
            except Exception:
                pass
        conn.commit()
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["messages"] = json.loads(d["messages"] or "[]")
    if d["zotero_keys"]:
        d["zotero_keys"] = json.loads(d["zotero_keys"])
    else:
        d.pop("zotero_keys", None)
    return d


# ── Public API ────────────────────────────────────────────────────────────────

def load_session(session_id: str) -> dict | None:
    """Load a session by ID. Returns None if not found."""
    _ensure_db()
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def save_session(session: dict) -> None:
    """Upsert a session, updating updated_at."""
    _ensure_db()
    session["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, messages, zotero_keys) "
            "VALUES (:id, :title, :created_at, :updated_at, :messages, :zotero_keys) "
            "ON CONFLICT(id) DO UPDATE SET "
            "  title       = excluded.title, "
            "  updated_at  = excluded.updated_at, "
            "  messages    = excluded.messages, "
            "  zotero_keys = excluded.zotero_keys",
            {
                "id":          session["id"],
                "title":       session.get("title", session["id"]),
                "created_at":  session.get("created_at", session["updated_at"]),
                "updated_at":  session["updated_at"],
                "messages":    json.dumps(session.get("messages", []), ensure_ascii=False),
                "zotero_keys": json.dumps(session["zotero_keys"], ensure_ascii=False)
                               if session.get("zotero_keys") else None,
            },
        )
        conn.commit()
    finally:
        conn.close()


def new_session(session_id: str, title: str) -> dict:
    """Create a new blank session dict (not yet saved to DB)."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "id":         session_id,
        "title":      title,
        "created_at": now,
        "updated_at": now,
        "messages":   [],
    }


def list_library_sessions(limit: int = 10) -> list[dict]:
    """Return the most recent library chat sessions (summary only)."""
    _ensure_db()
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM sessions "
            "WHERE id LIKE 'library_%' "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"id": r["id"], "title": r["title"], "updated_at": r["updated_at"]}
                for r in rows]
    finally:
        conn.close()


def delete_session(session_id: str) -> None:
    """Delete a session by ID (no-op if not found)."""
    _ensure_db()
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()
