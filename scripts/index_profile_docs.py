#!/usr/bin/env python3
"""Index profile documents (resume, bio, etc.) into Supabase for RAG."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

try:
    import pypdf
except ImportError:  # pragma: no cover
    pypdf = None

from app import (
    SUPABASE_PROFILE_CHUNKS_TABLE,
    chunk_text,
    get_embeddings,
    get_supabase,
    supabase_response_data,
)


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
BULLET = "\u2022"


def iter_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                files.extend(path.rglob(f"*{ext}"))
        elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(set(files))


def normalize_profile_text(text: str) -> str:
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"(?<=\w)-\n(?=\w)", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    lines = cleaned.split("\n")
    parts: List[str] = []
    current: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                parts.append(" ".join(current))
                current = []
            continue
        if stripped.startswith(BULLET):
            if current:
                parts.append(" ".join(current))
                current = []
            parts.append(f"{BULLET} {stripped.lstrip(BULLET).strip()}")
            continue
        if stripped.isupper() and len(stripped) <= 60:
            if current:
                parts.append(" ".join(current))
                current = []
            parts.append(stripped)
            continue
        current.append(stripped)
    if current:
        parts.append(" ".join(current))
    normalized = "\n".join(parts)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        if pypdf is None:
            raise RuntimeError("pypdf is required to read PDF files.")
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return normalize_profile_text("\n".join(pages))
    return path.read_text(encoding="utf-8", errors="ignore")


def index_file(path: Path) -> int:
    text = read_text(path)
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(chunks)
    client = get_supabase()
    source = str(path)
    delete_response = (
        client.table(SUPABASE_PROFILE_CHUNKS_TABLE).delete().eq("source", source).execute()
    )
    supabase_response_data(delete_response, "delete_profile_chunks")
    now = datetime.now(timezone.utc).isoformat()
    payload = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload.append(
            {
                "source": source,
                "content": chunk,
                "embedding": vector,
                "chunk_index": idx,
                "updated_at": now,
            }
        )
    insert_response = client.table(SUPABASE_PROFILE_CHUNKS_TABLE).insert(payload).execute()
    supabase_response_data(insert_response, "insert_profile_chunks")
    return len(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index profile documents into Supabase for RAG."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories containing profile docs.",
    )
    args = parser.parse_args()
    files = iter_files(args.paths)
    if not files:
        raise SystemExit("No supported files found to index.")
    total_chunks = 0
    for path in files:
        chunk_count = index_file(path)
        print(f"Indexed {path}: {chunk_count} chunks")
        total_chunks += chunk_count
    print(f"Total chunks indexed: {total_chunks}")


if __name__ == "__main__":
    main()
