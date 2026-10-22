"""Persistent memory store for cross-session agent memory.
Simple SQLite backend. Supports save/retrieve/clear.
"""
import re
import sqlite3, os, time

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "agent_memory.db")
_CURRENT_SCOPE = "default"


def set_scope(scope: str):
    """Select the long-term memory namespace used by the active agent."""
    global _CURRENT_SCOPE
    _CURRENT_SCOPE = str(scope or "default")


def current_scope() -> str:
    return _CURRENT_SCOPE


def _get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories ("
        "id INTEGER PRIMARY KEY, content TEXT, created_at REAL, "
        "scope TEXT NOT NULL DEFAULT 'default')"
    )
    # Keep old databases usable after introducing per-agent namespaces.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    if "scope" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT 'default'")
    conn.commit()
    return conn


def save_memory(content: str) -> int:
    """Save a memory. Returns the memory id."""
    conn = _get_db()
    cur = conn.execute(
        "INSERT INTO memories (content, created_at, scope) VALUES (?, ?, ?)",
        (str(content), time.time(), _CURRENT_SCOPE),
    )
    conn.commit()
    conn.close()
    return cur.lastrowid


def get_all_memories() -> list[str]:
    """Retrieve all stored memories."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT content FROM memories WHERE scope=? ORDER BY created_at DESC",
        (_CURRENT_SCOPE,),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def retrieve_memories(query: str = "", limit: int = 8) -> list[str]:
    """Retrieve recent memories, preferring lexical matches to the query."""
    memories = get_all_memories()
    if not memories:
        return []
    terms = set(re.findall(r"\w+", (query or "").lower()))
    if not terms:
        return memories[:limit]
    ranked = []
    for index, content in enumerate(memories):
        words = set(re.findall(r"\w+", content.lower()))
        ranked.append((len(terms & words), -index, content))
    ranked.sort(reverse=True)
    return [content for _, _, content in ranked[:limit]]


def clear_all():
    """Clear all memories."""
    conn = _get_db()
    conn.execute("DELETE FROM memories WHERE scope=?", (_CURRENT_SCOPE,))
    conn.commit()
    conn.close()


def count() -> int:
    conn = _get_db()
    n = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE scope=?", (_CURRENT_SCOPE,)
    ).fetchone()[0]
    conn.close()
    return n
