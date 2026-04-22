"""Two-layer pattern loader: core defaults (repo JSON) + per-user overrides (volume).

Effective items   = core_items + user_added - user_removed
Effective values  = core_values | user_value_overrides

Writes only ever touch the per-user layer. Core files under data/patterns/ are
read-only from this module.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = _REPO_ROOT / "data" / "patterns"
_DEFAULT_USER_DIR = _REPO_ROOT / "data" / "patterns" / "users"


def _user_dir() -> Path:
    return Path(os.getenv("PATTERNS_USER_DIR") or _DEFAULT_USER_DIR)


def _safe_client_id(client_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", client_id or "default")


def _normalise_filename(filename: str) -> str:
    return filename[:-5] if filename.endswith(".json") else filename


def _core_path(filename: str) -> Path:
    return CORE_DIR / f"{_normalise_filename(filename)}.json"


def _user_path(filename: str, client_id: str) -> Path:
    return _user_dir() / _safe_client_id(client_id) / f"{_normalise_filename(filename)}.json"


_lock = threading.Lock()
_cache: dict[tuple, tuple[tuple[float, float], Any]] = {}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_user_overrides(filename: str, client_id: str) -> dict:
    data = _read_json(_user_path(filename, client_id)) or {}
    return {
        "added": list(data.get("added", [])),
        "removed": list(data.get("removed", [])),
        "value_overrides": dict(data.get("value_overrides", {})),
    }


def _cached(kind: str, filename: str, client_id: str, builder):
    cp = _core_path(filename)
    up = _user_path(filename, client_id)
    key = (kind, _normalise_filename(filename), _safe_client_id(client_id))
    stamps = (_mtime(cp), _mtime(up))
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == stamps:
            return hit[1]
        value = builder()
        _cache[key] = (stamps, value)
        return value


def load_items(filename: str, client_id: str = "default") -> list[str]:
    """Return merged items list for this client (core + added - removed)."""
    def build():
        core = _read_json(_core_path(filename))
        if not core:
            raise FileNotFoundError(f"Unknown pattern file: {filename}")
        items = list(core.get("items", []))
        ov = _load_user_overrides(filename, client_id)
        removed = set(ov["removed"])
        merged = [x for x in items if x not in removed]
        seen = set(merged)
        for x in ov["added"]:
            if x not in seen:
                merged.append(x)
                seen.add(x)
        return merged
    return _cached("items", filename, client_id, build)


def load_values(filename: str, client_id: str = "default") -> dict:
    """Return merged values dict (core values overlaid with user value_overrides)."""
    def build():
        core = _read_json(_core_path(filename))
        if not core:
            raise FileNotFoundError(f"Unknown pattern file: {filename}")
        values = dict(core.get("values", {}))
        ov = _load_user_overrides(filename, client_id)
        values.update(ov["value_overrides"])
        return values
    return _cached("values", filename, client_id, build)


def load_description(filename: str) -> str:
    core = _read_json(_core_path(filename)) or {}
    return core.get("description", "")


def _save_user_overrides(filename: str, client_id: str, data: dict) -> None:
    payload = {
        "added": list(data.get("added", [])),
        "removed": list(data.get("removed", [])),
        "value_overrides": dict(data.get("value_overrides", {})),
    }
    _write_json_atomic(_user_path(filename, client_id), payload)


def add_user_item(filename: str, value: str, client_id: str) -> list[str]:
    """Add `value` to user's added list; ensure it is not in removed."""
    core = _read_json(_core_path(filename))
    if not core or "items" not in core:
        raise ValueError(f"{filename} is not an items-style pattern file")
    ov = _load_user_overrides(filename, client_id)
    if value in ov["removed"]:
        ov["removed"].remove(value)
    if value not in core.get("items", []) and value not in ov["added"]:
        ov["added"].append(value)
    _save_user_overrides(filename, client_id, ov)
    return load_items(filename, client_id)


def remove_user_item(filename: str, value: str, client_id: str) -> list[str]:
    """Remove `value`: drop from added if user-added, else add to removed."""
    core = _read_json(_core_path(filename))
    if not core or "items" not in core:
        raise ValueError(f"{filename} is not an items-style pattern file")
    ov = _load_user_overrides(filename, client_id)
    if value in ov["added"]:
        ov["added"].remove(value)
    elif value in core.get("items", []) and value not in ov["removed"]:
        ov["removed"].append(value)
    _save_user_overrides(filename, client_id, ov)
    return load_items(filename, client_id)


def set_user_value(filename: str, key: str, value: float, client_id: str) -> dict:
    """Upsert a per-doc-type / per-key numeric override."""
    core = _read_json(_core_path(filename))
    if not core or "values" not in core:
        raise ValueError(f"{filename} is not a values-style pattern file")
    ov = _load_user_overrides(filename, client_id)
    ov["value_overrides"][key] = value
    _save_user_overrides(filename, client_id, ov)
    return load_values(filename, client_id)


def reset_user_overrides(filename: str, client_id: str) -> None:
    """Delete the user's override file for this pattern."""
    p = _user_path(filename, client_id)
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def list_pattern_files() -> list[str]:
    """Return sorted list of all core JSON filenames (without .json)."""
    if not CORE_DIR.exists():
        return []
    return sorted(p.stem for p in CORE_DIR.glob("*.json"))


def list_user_overrides(client_id: str) -> dict:
    """Return a summary of all pattern overrides for this client_id."""
    udir = _user_dir() / _safe_client_id(client_id)
    if not udir.exists():
        return {}
    out: dict[str, dict] = {}
    for p in sorted(udir.glob("*.json")):
        data = _read_json(p) or {}
        out[p.stem] = {
            "added": list(data.get("added", [])),
            "removed": list(data.get("removed", [])),
            "value_overrides": dict(data.get("value_overrides", {})),
        }
    return out


def clear_cache() -> None:
    with _lock:
        _cache.clear()
