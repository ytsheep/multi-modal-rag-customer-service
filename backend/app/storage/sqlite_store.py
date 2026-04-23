import sqlite3
from pathlib import Path
from typing import Any


class SQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS uploaded_documents (
                    document_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    cleaning_report TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                )
                """
            )
            conn.commit()

    def save_message(self, user_id: str, session_id: str, role: str, content: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages (user_id, session_id, role, content) VALUES (?, ?, ?, ?)",
                (user_id, session_id, role, content),
            )
            conn.commit()

    def recent_messages(self, user_id: str, session_id: str, limit: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, created_at
                FROM chat_messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, session_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            sessions = conn.execute(
                """
                SELECT session_id, MAX(created_at) AS updated_at
                FROM chat_messages
                WHERE user_id = ?
                GROUP BY session_id
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
            result = []
            for session in sessions:
                messages = conn.execute(
                    """
                    SELECT id, session_id, role, content, created_at
                    FROM chat_messages
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY id ASC
                    """,
                    (user_id, session["session_id"]),
                ).fetchall()
                msg_dicts = [dict(row) for row in messages]
                first_user_msg = next((m["content"] for m in msg_dicts if m["role"] == "user"), "新会话")
                result.append(
                    {
                        "session_id": session["session_id"],
                        "title": first_user_msg[:30],
                        "updated_at": session["updated_at"],
                        "messages": msg_dicts,
                    }
                )
        return result

    def delete_session(self, user_id: str, session_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM chat_messages WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            conn.commit()

    def save_document(
        self,
        document_id: str,
        file_name: str,
        file_path: str,
        chunk_count: int,
        cleaning_report: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO uploaded_documents
                (document_id, file_name, file_path, chunk_count, cleaning_report)
                VALUES (?, ?, ?, ?, ?)
                """,
                (document_id, file_name, file_path, chunk_count, cleaning_report),
            )
            conn.commit()
