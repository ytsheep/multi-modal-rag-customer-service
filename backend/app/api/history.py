from fastapi import APIRouter, Query

from app.models import HistorySession
from app.storage.sqlite_store import SQLiteStore
from app.core.config import get_settings


router = APIRouter(tags=["history"])


@router.get("/history", response_model=list[HistorySession])
def list_history(user_id: str = Query(default="default")) -> list[HistorySession]:
    store = SQLiteStore(get_settings().sqlite_path)
    return store.list_sessions(user_id)


@router.delete("/history/{session_id}")
def delete_history(session_id: str, user_id: str = Query(default="default")) -> dict[str, bool]:
    store = SQLiteStore(get_settings().sqlite_path)
    store.delete_session(user_id, session_id)
    return {"ok": True}

