import re

from app.core.config import get_settings
from app.ingestion.parser import ParsedBlock
from app.models import DocumentChunk


class Chunker:
    def __init__(self) -> None:
        self.settings = get_settings()

    def split(self, document_id: str, file_name: str, blocks: list[ParsedBlock]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        buffer = ""
        current_page: int | None = None
        current_title = ""

        def flush(use_overlap: bool = False) -> None:
            nonlocal buffer, current_page, current_title
            content = buffer.strip()
            if not content:
                buffer = ""
                return
            chunk_id = f"{document_id}-{len(chunks) + 1:04d}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    file_name=file_name,
                    title_path=current_title,
                    page_start=current_page,
                    page_end=current_page,
                    content=content,
                    metadata={
                        "document_id": document_id,
                        "file_name": file_name,
                        "title_path": current_title,
                        "page_start": current_page,
                        "page_end": current_page,
                    },
                )
            )
            if use_overlap and self.settings.chunk_overlap > 0:
                buffer = self._tail_overlap(content)
            else:
                buffer = ""

        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            page = block.page
            title_path = block.title_path.strip() or (f"第 {page} 页" if page is not None else "")
            pieces = self._split_to_pieces(text)

            for piece in pieces:
                if not piece:
                    continue
                changed_scope = (
                    buffer
                    and (
                        page != current_page
                        or title_path != current_title
                    )
                )
                if changed_scope:
                    flush(use_overlap=False)

                current_page = page
                current_title = title_path

                if not buffer:
                    buffer = piece
                    continue

                candidate = f"{buffer}\n\n{piece}".strip()
                if len(candidate) <= self.settings.chunk_size:
                    buffer = candidate
                else:
                    flush(use_overlap=True)
                    if buffer:
                        candidate = f"{buffer}\n\n{piece}".strip()
                        if len(candidate) <= self.settings.chunk_size:
                            buffer = candidate
                        else:
                            flush(use_overlap=False)
                            buffer = piece
                    else:
                        buffer = piece

        flush(use_overlap=False)
        return chunks

    def _split_to_pieces(self, text: str) -> list[str]:
        max_size = self.settings.chunk_size
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
        pieces: list[str] = []
        for paragraph in paragraphs or [text.strip()]:
            if len(paragraph) <= max_size:
                pieces.append(paragraph)
                continue
            pieces.extend(self._split_long_paragraph(paragraph, max_size))
        return pieces

    def _split_long_paragraph(self, paragraph: str, max_size: int) -> list[str]:
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[。！？；;.!?])\s+|\n+", paragraph)
            if part.strip()
        ]
        pieces: list[str] = []
        buffer = ""
        for sentence in sentences or [paragraph]:
            if len(sentence) > max_size:
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                pieces.extend(self._hard_split(sentence, max_size))
                continue
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) <= max_size:
                buffer = candidate
            else:
                pieces.append(buffer)
                buffer = sentence
        if buffer:
            pieces.append(buffer)
        return pieces

    def _hard_split(self, text: str, max_size: int) -> list[str]:
        return [text[index : index + max_size].strip() for index in range(0, len(text), max_size)]

    def _tail_overlap(self, content: str) -> str:
        overlap = min(self.settings.chunk_overlap, max(self.settings.chunk_size // 4, 0))
        if overlap <= 0 or len(content) <= overlap:
            return ""
        tail = content[-overlap:]
        sentence_start = max(tail.rfind("。"), tail.rfind("；"), tail.rfind("\n"))
        if sentence_start >= 0 and sentence_start + 1 < len(tail):
            tail = tail[sentence_start + 1 :]
        return tail.strip()
