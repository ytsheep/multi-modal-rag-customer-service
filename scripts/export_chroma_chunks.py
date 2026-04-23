from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DEFAULT_EXPORT_DIR = ROOT / "eval" / "exports"

sys.path.insert(0, str(BACKEND))


def maybe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_row(chunk_id: str, content: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "document_id": str(metadata.get("document_id", "")),
        "file_name": str(metadata.get("file_name", "")),
        "title_path": str(metadata.get("title_path", "")),
        "page_start": maybe_int(metadata.get("page_start")),
        "page_end": maybe_int(metadata.get("page_end")),
        "content": content or "",
    }


def list_collection_names(client: Any) -> list[str]:
    names = []
    for collection in client.list_collections():
        names.append(getattr(collection, "name", str(collection)))
    return names


def load_rows(collection_name: str, page_size: int) -> list[dict[str, Any]]:
    import chromadb
    from app.core.config import get_settings

    settings = get_settings()
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:
        names = ", ".join(list_collection_names(client)) or "无"
        raise RuntimeError(f"找不到 Chroma collection：{collection_name}。当前可用 collection：{names}") from exc

    total = collection.count()
    rows: list[dict[str, Any]] = []
    for offset in range(0, total, page_size):
        result = collection.get(
            limit=page_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        for index, chunk_id in enumerate(ids):
            content = docs[index] if index < len(docs) else ""
            metadata = metadatas[index] if index < len(metadatas) else {}
            rows.append(normalize_row(chunk_id, content, metadata or {}))

    return sorted(
        rows,
        key=lambda row: (
            row["file_name"],
            row["page_start"] if row["page_start"] is not None else 10**9,
            row["page_end"] if row["page_end"] is not None else 10**9,
            row["title_path"],
            row["chunk_id"],
        ),
    )


def collection_summaries() -> list[dict[str, Any]]:
    import chromadb
    from app.core.config import get_settings

    settings = get_settings()
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    summaries = []
    for name in list_collection_names(client):
        try:
            collection = client.get_collection(name)
            count = collection.count()
        except Exception:
            count = None
        summaries.append({"name": name, "count": count})
    return summaries


def apply_filters(
    rows: list[dict[str, Any]],
    file_name: str | None,
    contains: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    output = rows
    if file_name:
        needle = file_name.lower()
        output = [row for row in output if needle in row["file_name"].lower()]
    if contains:
        needle = contains.lower()
        output = [
            row
            for row in output
            if needle in row["content"].lower()
            or needle in row["title_path"].lower()
            or needle in row["file_name"].lower()
        ]
    if limit:
        output = output[:limit]
    return output


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "chunk_id",
        "document_id",
        "file_name",
        "title_path",
        "page_start",
        "page_end",
        "content",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_paths(collection: str, export_dir: Path) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_collection = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in collection)
    base = export_dir / f"{safe_collection}_{stamp}"
    return base.with_suffix(".jsonl"), base.with_suffix(".csv")


def main() -> None:
    from app.core.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Export Chroma chunks for manual evaluation labeling.")
    parser.add_argument("--collection", default=settings.chroma_collection, help="Chroma collection name.")
    parser.add_argument("--format", choices=["jsonl", "csv", "both"], default="both")
    parser.add_argument("--out", type=Path, default=None, help="Output file path. For --format both, pass a directory.")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--file-name", default=None, help="Filter by partial uploaded file name.")
    parser.add_argument("--contains", default=None, help="Filter rows containing this keyword in file/title/content.")
    parser.add_argument("--limit", type=int, default=None, help="Export at most N rows after filtering.")
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--list-collections", action="store_true", help="List Chroma collections and exit.")
    args = parser.parse_args()

    if args.list_collections:
        print(json.dumps({"collections": collection_summaries()}, ensure_ascii=False, indent=2))
        return

    try:
        rows = load_rows(args.collection, args.page_size)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        print("可先运行：python scripts\\export_chroma_chunks.py --list-collections", file=sys.stderr)
        raise SystemExit(2) from exc
    rows = apply_filters(rows, args.file_name, args.contains, args.limit)

    if args.out and args.format != "both":
        output_paths = [args.out]
    else:
        export_dir = args.out if args.out and args.format == "both" else args.export_dir
        export_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path, csv_path = default_paths(args.collection, export_dir)
        output_paths = [jsonl_path, csv_path] if args.format == "both" else [
            jsonl_path if args.format == "jsonl" else csv_path
        ]

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".csv":
            write_csv(rows, path)
        else:
            write_jsonl(rows, path)

    print(
        json.dumps(
            {
                "collection": args.collection,
                "exported_rows": len(rows),
                "outputs": [str(path) for path in output_paths],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
