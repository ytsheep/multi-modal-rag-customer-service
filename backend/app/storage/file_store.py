from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import get_settings


class FileStore:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def save_upload(self, upload: UploadFile) -> tuple[str, str, Path]:
        document_id = uuid4().hex
        safe_name = Path(upload.filename or "uploaded_file").name
        target_dir = self.settings.upload_dir / document_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name
        content = await upload.read()
        target_path.write_bytes(content)
        return document_id, safe_name, target_path

