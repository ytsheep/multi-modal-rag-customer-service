from pydantic import BaseModel, Field
from typing import Any


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    file_name: str
    title_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    score: float = 0.0
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str = "default"
    user_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    citations: list[Citation] = []
    intent: str = "rag_qa"
    intent_reason: str = ""


class UploadResponse(BaseModel):
    document_id: str
    file_name: str
    chunks: int
    cleaning_report: dict[str, Any]


class HistoryMessage(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    created_at: str


class HistorySession(BaseModel):
    session_id: str
    title: str
    updated_at: str
    messages: list[HistoryMessage] = []


class DocumentChunk(BaseModel):
    id: str
    document_id: str
    file_name: str
    title_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    content: str
    metadata: dict[str, Any] = {}
