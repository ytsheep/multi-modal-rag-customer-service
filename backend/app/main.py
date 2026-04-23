from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, history, upload
from app.core.config import get_settings
from app.storage.sqlite_store import SQLiteStore


settings = get_settings()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    settings.ensure_dirs()
    SQLiteStore(settings.sqlite_path).init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(upload.router, prefix=settings.api_prefix)
app.include_router(chat.router, prefix=settings.api_prefix)
app.include_router(history.router, prefix=settings.api_prefix)

