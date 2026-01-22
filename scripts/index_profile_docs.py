#!/usr/bin/env python3
"""Index profile documents (resume, bio, etc.) into Supabase for RAG."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import pypdf
except ImportError:  # pragma: no cover
    pypdf = None
try:
    from docling.document_converter import DocumentConverter
except Exception:  # pragma: no cover
    DocumentConverter = None

from app import (
    SUPABASE_PROFILE_CHUNKS_TABLE,
    chunk_text,
    get_embeddings,
    get_supabase,
    supabase_response_data,
)


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
BULLET = "\u2022"
SECTION_TITLES = {
    "education",
    "certifications",
    "certifications & courses",
    "data & ai project",
    "projects",
    "professional experience",
    "leadership & honors/awards",
    "leadership & honors",
    "honors/awards",
    "skills",
    "languages",
}


def iter_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                files.extend(path.rglob(f"*{ext}"))
        elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return prefer_markdown(sorted(set(files)))


def prefer_markdown(files: List[Path]) -> List[Path]:
    md_keys = {(path.parent, path.stem) for path in files if path.suffix.lower() == ".md"}
    preferred: List[Path] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".pdf" and (path.parent, path.stem) in md_keys:
            continue
        preferred.append(path)
    return preferred


def load_docling_markdown(path: Path) -> Optional[str]:
    if DocumentConverter is None:
        return None
    try:
        converter = DocumentConverter()
        result = converter.convert(str(path))
        document = getattr(result, "document", result)
        for method_name in ("export_to_markdown", "to_markdown", "render_markdown"):
            method = getattr(document, method_name, None)
            if callable(method):
                return method()
        for attr_name in ("markdown", "md"):
            value = getattr(document, attr_name, None)
            if isinstance(value, str):
                return value
    except Exception as exc:  # noqa: BLE001
        print(f"Docling parse failed for {path}: {exc}")
    return None


def format_section(title: Optional[str], lines: List[str]) -> str:
    body = "\n".join(line for line in lines if line)
    if not title:
        return body.strip()
    return f"Section: {title}\n{body}".strip()


def split_markdown_sections(markdown: str) -> List[str]:
    sections: List[str] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            if current_lines:
                current_lines.append("")
            continue
        if stripped.startswith("#"):
            if current_lines:
                sections.append(format_section(current_title, current_lines))
                current_lines = []
            current_title = stripped.lstrip("#").strip() or None
            continue
        current_lines.append(stripped)
    if current_lines:
        sections.append(format_section(current_title, current_lines))
    return [section for section in sections if section.strip()]


def is_section_heading(line: str) -> bool:
    cleaned = re.sub(r"[:\-\u2013\u2014]+$", "", line).strip()
    if not cleaned:
        return False
    if cleaned.lower() in SECTION_TITLES:
        return True
    if cleaned.isupper() and len(cleaned) <= 60:
        return True
    return False


def is_divider_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    if all(ch in "-_=*~." for ch in stripped):
        return True
    non_alnum = sum(1 for ch in stripped if not ch.isalnum())
    return non_alnum / len(stripped) >= 0.8


def extract_divider_heading(lines: List[str]) -> Optional[str]:
    for idx in range(len(lines) - 1, -1, -1):
        candidate = lines[idx].strip()
        if not candidate:
            continue
        if len(candidate) > 80:
            return None
        if candidate.endswith((".", ",")):
            return None
        return candidate
    return None


def split_text_sections(text: str) -> List[str]:
    sections: List[str] = []
    current_title: Optional[str] = None
    current_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current_lines:
                current_lines.append("")
            continue
        if is_divider_line(stripped):
            heading = extract_divider_heading(current_lines)
            if heading:
                heading_index = None
                for idx in range(len(current_lines) - 1, -1, -1):
                    if current_lines[idx].strip() == heading:
                        heading_index = idx
                        break
                if heading_index is not None:
                    previous_lines = current_lines[:heading_index]
                    if previous_lines:
                        sections.append(format_section(current_title, previous_lines))
                    current_title = heading
                    current_lines = []
                    continue
            if current_lines:
                sections.append(format_section(current_title, current_lines))
                current_lines = []
            continue
        if is_section_heading(stripped):
            if current_lines:
                sections.append(format_section(current_title, current_lines))
                current_lines = []
            current_title = stripped
            continue
        current_lines.append(stripped)
    if current_lines:
        sections.append(format_section(current_title, current_lines))
    return [section for section in sections if section.strip()]


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


def split_section_header(section: str) -> Tuple[Optional[str], str]:
    lines = section.splitlines()
    if not lines:
        return None, ""
    first_line = lines[0].strip()
    if first_line.lower().startswith("section:"):
        title = first_line.split(":", 1)[1].strip() or None
        body = "\n".join(lines[1:]).strip()
        return title, body
    return None, section.strip()


def build_chunk_header(title: Optional[str]) -> str:
    if title:
        return f"Section: {title}"
    return "Section: Profile"


def chunk_profile_section(section: str) -> List[str]:
    title, body = split_section_header(section)
    header = build_chunk_header(title)
    if not body:
        return [header]
    chunks = chunk_text(body)
    if not chunks:
        return [header]
    return [f"{header}\n{chunk}".strip() for chunk in chunks]


def read_pdf_text(path: Path) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf is required to read PDF files.")
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def read_profile_sections(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        markdown = load_docling_markdown(path)
        if markdown:
            sections = split_markdown_sections(markdown)
            if sections:
                return sections
        normalized = normalize_profile_text(read_pdf_text(path))
        sections = split_text_sections(normalized)
        return sections or [normalized]
    if suffix == ".md":
        markdown = path.read_text(encoding="utf-8", errors="ignore")
        sections = split_markdown_sections(markdown)
        return sections or [markdown.strip()]
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    normalized = normalize_profile_text(raw_text)
    sections = split_text_sections(normalized)
    return sections or [normalized]


def index_file(path: Path) -> int:
    sections = read_profile_sections(path)
    chunks: List[str] = []
    for section in sections:
        chunks.extend(chunk_profile_section(section))
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
