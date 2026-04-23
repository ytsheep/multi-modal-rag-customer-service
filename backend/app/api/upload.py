from fastapi import APIRouter, File, UploadFile

from app.ingestion.pipeline import IngestionPipeline
from app.models import UploadResponse


router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    pipeline = IngestionPipeline()
    result = await pipeline.ingest_upload(file)
    return UploadResponse(**result)

