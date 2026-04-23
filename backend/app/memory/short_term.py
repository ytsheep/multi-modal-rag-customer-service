from app.core.config import get_settings
from app.storage.sqlite_store import SQLiteStore


class ShortTermMemory:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = SQLiteStore(self.settings.sqlite_path)

    def load(self, user_id: str, session_id: str) -> list[dict[str, str]]:
        rows = self.store.recent_messages(
            user_id=user_id,
            session_id=session_id,
            limit=self.settings.short_term_message_limit,
        )
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def save_turn(self, user_id: str, session_id: str, question: str, answer: str) -> None:
        self.store.save_message(user_id, session_id, "user", question)
        self.store.save_message(user_id, session_id, "assistant", answer)

