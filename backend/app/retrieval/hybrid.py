import math
import re
from collections import Counter, defaultdict
from typing import Any

from app.core.config import get_settings
from app.models import Citation
from app.storage.chroma_store import ChromaStore


def tokenize(text: str) -> list[str]:
    lower = text.lower()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lower)
    return [word for word in words if word.strip()]


class HybridRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = ChromaStore()

    def retrieve(self, query: str) -> list[Citation]:
        candidates = self.settings.retrieval_candidates
        vector_rows = self.store.vector_query(query, candidates)
        sparse_rows = self._sparse_query(query, candidates)
        merged = self._merge_scores(vector_rows, sparse_rows)
        reranked = self._rerank(query, merged)
        return [self._to_citation(row) for row in reranked[: self.settings.rerank_top_k]]

    def _sparse_query(self, query: str, n_results: int) -> list[dict[str, Any]]:
        all_rows = self.store.all_chunks()
        if not all_rows:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []
        doc_freq: Counter[str] = Counter()
        doc_terms: dict[str, Counter[str]] = {}
        for row in all_rows:
            terms = Counter(tokenize(row["document"]))
            doc_terms[row["id"]] = terms
            for term in terms:
                doc_freq[term] += 1

        scores: list[tuple[float, dict[str, Any]]] = []
        avg_len = sum(sum(counter.values()) for counter in doc_terms.values()) / max(len(doc_terms), 1)
        k1 = 1.5
        b = 0.75
        total_docs = len(all_rows)
        for row in all_rows:
            terms = doc_terms[row["id"]]
            doc_len = sum(terms.values()) or 1
            score = 0.0
            for term in query_terms:
                tf = terms.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log(1 + (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
            if score > 0:
                item = dict(row)
                item["score"] = score
                item["source"] = "sparse"
                scores.append((score, item))
        scores.sort(key=lambda item: item[0], reverse=True)
        max_score = scores[0][0] if scores else 1.0
        return [{**row, "score": score / max_score} for score, row in scores[:n_results]]

    def _merge_scores(self, vector_rows: list[dict[str, Any]], sparse_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        score_parts: dict[str, dict[str, float]] = defaultdict(lambda: {"vector": 0.0, "sparse": 0.0})

        for row in vector_rows:
            merged[row["id"]] = row
            score_parts[row["id"]]["vector"] = max(score_parts[row["id"]]["vector"], row["score"])
        for row in sparse_rows:
            merged.setdefault(row["id"], row)
            score_parts[row["id"]]["sparse"] = max(score_parts[row["id"]]["sparse"], row["score"])

        output = []
        for chunk_id, row in merged.items():
            row = dict(row)
            row["score"] = 0.65 * score_parts[chunk_id]["vector"] + 0.35 * score_parts[chunk_id]["sparse"]
            output.append(row)
        return output

    def _rerank(self, query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_terms = set(tokenize(query))
        product_hints = self._extract_product_hints(query)
        for row in rows:
            doc_terms = set(tokenize(row["document"]))
            overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
            metadata = row.get("metadata", {}) or {}
            title = str(metadata.get("title_path", ""))
            title_hit = 0.08 if any(term in title.lower() for term in query_terms) else 0.0
            product_boost = self._product_match_boost(product_hints, row)
            row["score"] = row["score"] * 0.85 + overlap * 0.15 + title_hit
            row["score"] += product_boost
        return sorted(rows, key=lambda item: item["score"], reverse=True)

    def _extract_product_hints(self, query: str) -> set[str]:
        lowered = query.lower()
        hints: set[str] = set()
        if any(token in lowered for token in ["产品a", "a产品", "西门子产品a"]):
            hints.add("产品a")
        if any(token in lowered for token in ["产品b", "b产品", "西门子产品b"]):
            hints.add("产品b")
        return hints

    def _product_match_boost(self, hints: set[str], row: dict[str, Any]) -> float:
        if not hints or len(hints) > 1:
            return 0.0
        metadata = row.get("metadata", {}) or {}
        haystack = " ".join(
            [
                str(row.get("document", "")),
                str(metadata.get("file_name", "")),
                str(metadata.get("title_path", "")),
                str(metadata.get("document_id", "")),
            ]
        ).lower()
        hint = next(iter(hints))
        if hint == "产品a":
            return 0.18 if "产品a" in haystack or "manual_a" in haystack else -0.12
        if hint == "产品b":
            return 0.18 if "产品b" in haystack or "manual_b" in haystack else -0.12
        return 0.0

    def _to_citation(self, row: dict[str, Any]) -> Citation:
        metadata = row.get("metadata", {}) or {}
        return Citation(
            chunk_id=row["id"],
            document_id=str(metadata.get("document_id", "")),
            file_name=str(metadata.get("file_name", "")),
            title_path=str(metadata.get("title_path", "")),
            page_start=self._maybe_int(metadata.get("page_start")),
            page_end=self._maybe_int(metadata.get("page_end")),
            score=float(row.get("score", 0.0)),
            content=row.get("document", ""),
        )

    def _maybe_int(self, value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
