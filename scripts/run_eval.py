from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
EVAL_DIR = ROOT / "eval"
REPORT_DIR = EVAL_DIR / "reports"

sys.path.insert(0, str(BACKEND))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_eval_env(collection_name: str) -> None:
    os.environ["CHROMA_COLLECTION"] = collection_name


def has_dashscope_key() -> bool:
    from app.core.config import get_settings

    return bool(get_settings().dashscope_api_key)


def seed_eval_knowledge(collection_name: str) -> dict[str, Any]:
    ensure_eval_env(collection_name)

    import chromadb
    from app.core.config import get_settings
    from app.llm.embeddings import DashScopeEmbeddingFunction

    settings = get_settings()
    if not settings.dashscope_api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for seeding eval vectors.")

    knowledge = load_jsonl(EVAL_DIR / "rag_knowledge.jsonl")
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=DashScopeEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=[item["chunk_id"] for item in knowledge],
        documents=[item["content"] for item in knowledge],
        metadatas=[
            {
                "document_id": item["document_id"],
                "file_name": item["file_name"],
                "title_path": item["title_path"],
                "page_start": item["page_start"],
                "page_end": item["page_end"],
            }
            for item in knowledge
        ],
    )
    return {"collection": collection_name, "chunks": len(knowledge)}


def evaluate_intent() -> dict[str, Any]:
    from app.agent.intent import IntentRouter

    router = IntentRouter()
    cases = load_jsonl(EVAL_DIR / "intent_cases.jsonl")
    details = []
    correct = 0
    for case in cases:
        result = router.classify(case["question"])
        ok = result.intent.value == case["expected_intent"]
        correct += int(ok)
        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected": case["expected_intent"],
                "actual": result.intent.value,
                "confidence": result.confidence,
                "reason": result.reason,
                "ok": ok,
            }
        )
    return {
        "name": "intent",
        "total": len(cases),
        "accuracy": correct / max(len(cases), 1),
        "details": details,
    }


def evaluate_retrieval(collection_name: str) -> dict[str, Any]:
    ensure_eval_env(collection_name)

    from app.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever()
    cases = load_jsonl(EVAL_DIR / "rag_cases.jsonl")
    details = []
    hit_count = 0
    rr_scores = []

    for case in cases:
        citations = retriever.retrieve(case["question"])
        returned_ids = [item.chunk_id for item in citations]
        expected_ids = set(case["expected_chunk_ids"])
        hit_positions = [idx for idx, chunk_id in enumerate(returned_ids, start=1) if chunk_id in expected_ids]
        hit = bool(hit_positions)
        hit_count += int(hit)
        rr_scores.append(1 / hit_positions[0] if hit_positions else 0)
        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expected_chunk_ids": case["expected_chunk_ids"],
                "returned_chunk_ids": returned_ids,
                "scores": [round(item.score, 4) for item in citations],
                "hit_at_3": hit,
                "reciprocal_rank": rr_scores[-1],
            }
        )

    total = len(cases)
    return {
        "name": "retrieval",
        "total": total,
        "recall_at_3": hit_count / max(total, 1),
        "mrr": statistics.mean(rr_scores) if rr_scores else 0,
        "details": details,
    }


def normalized(text: str) -> str:
    return "".join(text.lower().split())


def fact_coverage(answer: str, expected_facts: list[str]) -> float:
    if not expected_facts:
        return 1.0
    normalized_answer = normalized(answer)
    hits = sum(1 for fact in expected_facts if normalized(fact) in normalized_answer)
    return hits / len(expected_facts)


def page_citation_ok(citations: list[Any], case: dict[str, Any]) -> bool:
    expected_pages = set(case.get("expected_pages", []))
    expected_files = set(case.get("expected_files", []))
    if not expected_pages:
        return True

    if expected_files:
        matched_files: set[str] = set()
        for citation in citations:
            if citation.file_name not in expected_files:
                continue
            citation_pages = {citation.page_start, citation.page_end} - {None}
            if citation_pages & expected_pages:
                matched_files.add(citation.file_name)
        return expected_files <= matched_files

    for citation in citations:
        citation_pages = {citation.page_start, citation.page_end} - {None}
        if not citation_pages & expected_pages:
            continue
        return True
    return False


def evaluate_content(collection_name: str, threshold: float) -> dict[str, Any]:
    ensure_eval_env(collection_name)

    from app.agent.service import AgentService
    from app.models import ChatRequest

    service = AgentService()
    cases = load_jsonl(EVAL_DIR / "content_cases.jsonl")
    details = []
    passed = 0
    fact_scores = []
    page_scores = []
    latencies = []
    timestamp = int(time.time())

    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        response = service.answer(
            ChatRequest(
                question=case["question"],
                user_id="eval",
                session_id=f"eval-content-{timestamp}-{index}",
            )
        )
        latency = time.perf_counter() - started
        latencies.append(latency)

        intent_ok = response.intent == "rag_qa"
        coverage = fact_coverage(response.answer, case.get("expected_facts", []))
        page_ok = page_citation_ok(response.citations, case)
        content_ok = intent_ok and coverage >= threshold and page_ok
        passed += int(content_ok)
        fact_scores.append(coverage)
        page_scores.append(1.0 if page_ok else 0.0)

        details.append(
            {
                "id": case["id"],
                "question": case["question"],
                "actual_intent": response.intent,
                "intent_ok": intent_ok,
                "expected_facts": case.get("expected_facts", []),
                "fact_coverage": coverage,
                "expected_pages": case.get("expected_pages", []),
                "expected_files": case.get("expected_files", []),
                "returned_citations": [
                    {
                        "chunk_id": citation.chunk_id,
                        "file_name": citation.file_name,
                        "page_start": citation.page_start,
                        "page_end": citation.page_end,
                        "score": round(citation.score, 4),
                    }
                    for citation in response.citations
                ],
                "page_citation_ok": page_ok,
                "content_ok": content_ok,
                "latency_seconds": round(latency, 3),
                "answer_preview": response.answer[:180],
            }
        )

    total = len(cases)
    return {
        "name": "content",
        "total": total,
        "accuracy": passed / max(total, 1),
        "avg_fact_coverage": statistics.mean(fact_scores) if fact_scores else 0,
        "page_citation_accuracy": statistics.mean(page_scores) if page_scores else 0,
        "avg_latency_seconds": statistics.mean(latencies) if latencies else 0,
        "p95_latency_seconds": percentile(latencies, 95),
        "fact_threshold": threshold,
        "details": details,
    }


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def save_report(report: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORT_DIR / f"eval_{stamp}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_md = REPORT_DIR / "latest.md"
    latest_md.write_text(render_markdown(report), encoding="utf-8")
    return json_path


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 产品资料问答助手评估报告",
        "",
        f"- 时间：{report['created_at']}",
        f"- Collection：`{report['collection']}`",
        f"- 是否重新灌入评估知识库：{report['seeded']}",
        "",
        "## 汇总",
        "",
    ]
    for result in report["results"]:
        lines.append(f"### {result['name']}")
        for key, value in result.items():
            if key in {"name", "details"}:
                continue
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run product manual QA evaluation.")
    parser.add_argument("--mode", choices=["intent", "retrieval", "content", "all"], default="all")
    parser.add_argument("--collection", default="eval_product_manual_chunks")
    parser.add_argument("--seed", action="store_true", help="Reset and seed eval knowledge into ChromaDB.")
    parser.add_argument("--content-threshold", type=float, default=0.6)
    args = parser.parse_args()

    ensure_eval_env(args.collection)
    seeded_info: dict[str, Any] | None = None
    if args.seed:
        seeded_info = seed_eval_knowledge(args.collection)

    results = []
    if args.mode in {"intent", "all"}:
        results.append(evaluate_intent())

    needs_model = args.mode in {"retrieval", "content", "all"}
    if needs_model and not has_dashscope_key():
        results.append(
            {
                "name": "model_required",
                "skipped": True,
                "reason": "DASHSCOPE_API_KEY 未配置，已跳过 retrieval/content。",
            }
        )
    else:
        if args.mode in {"retrieval", "all"}:
            results.append(evaluate_retrieval(args.collection))
        if args.mode in {"content", "all"}:
            results.append(evaluate_content(args.collection, args.content_threshold))

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "collection": args.collection,
        "seeded": bool(args.seed),
        "seeded_info": seeded_info,
        "results": results,
    }
    report_path = save_report(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {report_path}")
    print(f"Markdown summary: {REPORT_DIR / 'latest.md'}")


if __name__ == "__main__":
    main()
