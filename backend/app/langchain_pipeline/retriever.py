from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from app.models import Citation
from app.retrieval.hybrid import HybridRetriever


class HybridLangChainRetriever(BaseRetriever):
    """LangChain adapter for the project's hybrid retrieval algorithm."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid_retriever: HybridRetriever = Field(default_factory=HybridRetriever)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        citations = self.hybrid_retriever.retrieve(query)
        return [self._citation_to_document(citation) for citation in citations]

    def _citation_to_document(self, citation: Citation) -> Document:
        metadata: dict[str, Any] = {
            "chunk_id": citation.chunk_id,
            "document_id": citation.document_id,
            "file_name": citation.file_name,
            "title_path": citation.title_path,
            "page_start": citation.page_start,
            "page_end": citation.page_end,
            "score": citation.score,
        }
        return Document(page_content=citation.content, metadata=metadata)


def document_to_citation(document: Document) -> Citation:
    metadata = document.metadata or {}
    return Citation(
        chunk_id=str(metadata.get("chunk_id", "")),
        document_id=str(metadata.get("document_id", "")),
        file_name=str(metadata.get("file_name", "")),
        title_path=str(metadata.get("title_path", "")),
        page_start=_maybe_int(metadata.get("page_start")),
        page_end=_maybe_int(metadata.get("page_end")),
        score=float(metadata.get("score") or 0.0),
        content=document.page_content,
    )


def _maybe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

