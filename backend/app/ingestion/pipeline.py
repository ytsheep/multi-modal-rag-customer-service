import json

from fastapi import UploadFile

from app.core.config import get_settings
from app.ingestion.chunker import Chunker
from app.ingestion.cleaner import CleaningPipeline
from app.ingestion.parser import DocumentParser
from app.storage.chroma_store import ChromaStore
from app.storage.file_store import FileStore
from app.storage.sqlite_store import SQLiteStore


class IngestionPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.file_store = FileStore()
        self.parser = DocumentParser()
        self.cleaner = CleaningPipeline()
        self.chunker = Chunker()
        self.chroma = ChromaStore()
        self.sqlite = SQLiteStore(self.settings.sqlite_path)

    async def ingest_upload(self, upload: UploadFile) -> dict:
        document_id, file_name, path = await self.file_store.save_upload(upload)
        parsed = self.parser.parse(path)
        cleaned, report = self.cleaner.clean(parsed)
        chunks = self.chunker.split(document_id, file_name, cleaned)
        self.chroma.add_chunks(chunks)
        self.sqlite.save_document(
            document_id=document_id,
            file_name=file_name,
            file_path=str(path),
            chunk_count=len(chunks),
            cleaning_report=json.dumps(report, ensure_ascii=False),
        )
        return {
            "document_id": document_id,
            "file_name": file_name,
            "chunks": len(chunks),
            "cleaning_report": report,
        }

