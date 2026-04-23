from typing import Any

import chromadb

from app.core.config import get_settings
from app.llm.embeddings import DashScopeEmbeddingFunction
from app.models import DocumentChunk


class ChromaStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_function = DashScopeEmbeddingFunction()
        self.client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._normalize_metadata(chunk.metadata) for chunk in chunks],
        )

    def vector_query(self, query: str, n_results: int) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return self._flatten_query_result(result, source="vector")

    def all_chunks(self) -> list[dict[str, Any]]:
        result = self.collection.get(include=["documents", "metadatas"])
        rows = []
        for idx, chunk_id in enumerate(result.get("ids", [])):
            metadata = result.get("metadatas", [])[idx] or {}
            rows.append(
                {
                    "id": chunk_id,
                    "document": result.get("documents", [])[idx] or "",
                    "metadata": metadata,
                }
            )
        return rows

    def _flatten_query_result(self, result: dict[str, Any], source: str) -> list[dict[str, Any]]:
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        rows = []
        for idx, chunk_id in enumerate(ids):
            distance = distances[idx] if idx < len(distances) else 1.0
            rows.append(
                {
                    "id": chunk_id,
                    "document": docs[idx] if idx < len(docs) else "",
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "score": max(0.0, 1.0 - float(distance)),
                    "source": source,
                }
            )
        return rows

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        normalized: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                normalized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                normalized[key] = value
            else:
                normalized[key] = str(value)
        return normalized

