from functools import lru_cache
from pathlib import Path
import os

from pydantic import BaseModel


def _load_env_file() -> None:
    env_path = Path("backend/.env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()


class Settings(BaseModel):
    app_name: str = "产品资料问答助手"
    api_prefix: str = "/api"
    data_dir: Path = Path("backend/data")
    upload_dir: Path = Path("backend/data/uploads")
    chroma_dir: Path = Path("backend/data/chroma")
    sqlite_path: Path = Path("backend/data/db/app.sqlite3")

    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    dashscope_chat_model: str = os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")
    dashscope_embedding_model: str = os.getenv(
        "DASHSCOPE_EMBEDDING_MODEL",
        "text-embedding-v4",
    )
    embedding_dimensions: int = int(os.getenv("DASHSCOPE_EMBEDDING_DIMENSIONS", "1024"))

    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "product_manual_chunks")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    retrieval_candidates: int = int(os.getenv("RETRIEVAL_CANDIDATES", "12"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "3"))
    short_term_message_limit: int = int(os.getenv("SHORT_TERM_MESSAGE_LIMIT", "8"))

    aliyun_ocr_enabled: bool = os.getenv("ALIYUN_OCR_ENABLED", "false").lower() == "true"
    aliyun_access_key_id: str = os.getenv("ALIYUN_ACCESS_KEY_ID", "")
    aliyun_access_key_secret: str = os.getenv("ALIYUN_ACCESS_KEY_SECRET", "")
    aliyun_ocr_endpoint: str = os.getenv(
        "ALIYUN_OCR_ENDPOINT",
        "ocr-api.cn-hangzhou.aliyuncs.com",
    )

    def ensure_dirs(self) -> None:
        for path in [self.data_dir, self.upload_dir, self.chroma_dir, self.sqlite_path.parent]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
