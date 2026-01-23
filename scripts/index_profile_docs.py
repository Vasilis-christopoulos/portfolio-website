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
SECTION_TAGS = [
    (
        ("professional experience", "work experience", "experience"),
        ["experience", "work history", "employers", "companies", "roles"],
    ),
    (
        ("professional summary", "summary", "profile"),
        ["summary", "overview", "profile"],
    ),
    (
        ("technical skills", "skills"),
        ["skills", "tools", "stack", "technologies"],
    ),
    (
        ("high-impact ai projects", "projects"),
        ["projects", "case studies", "work samples"],
    ),
    (
        ("education",),
        ["education", "degrees", "university"],
    ),
    (
        (
            "leadership & honors/awards",
            "leadership & honors",
            "honors/awards",
            "leadership",
            "honors",
            "awards",
        ),
        ["leadership", "awards", "honors", "achievements"],
    ),
    (
        ("certifications & courses", "certifications", "courses"),
        ["certifications", "courses"],
    ),
    (("languages",), ["languages"]),
]


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


def format_section(
    section_title: Optional[str],
    entry_title: Optional[str],
    lines: List[str],
) -> str:
    body = "\n".join(line for line in lines if line)
    header_lines: List[str] = []
    if section_title:
        header_lines.append(f"Section: {section_title}")
    if entry_title:
        header_lines.append(f"Entry: {entry_title}")
    if header_lines:
        if body:
            return "\n".join(header_lines + [body]).strip()
        return "\n".join(header_lines).strip()
    return body.strip()


def split_markdown_sections(markdown: str) -> List[str]:
    sections: List[str] = []
    current_section_title: Optional[str] = None
    current_entry_title: Optional[str] = None
    current_lines: List[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            if current_lines:
                current_lines.append("")
            continue
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip() or None
            if level <= 2:
                if current_lines:
                    sections.append(
                        format_section(
                            current_section_title,
                            current_entry_title,
                            current_lines,
                        )
                    )
                    current_lines = []
                current_section_title = title
                current_entry_title = None
                continue
            if level == 3:
                if current_lines:
                    sections.append(
                        format_section(
                            current_section_title,
                            current_entry_title,
                            current_lines,
                        )
                    )
                    current_lines = []
                current_entry_title = title
                continue
        current_lines.append(stripped)
    if current_lines:
        sections.append(
            format_section(
                current_section_title,
                current_entry_title,
                current_lines,
            )
        )
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
                        sections.append(format_section(current_title, None, previous_lines))
                    current_title = heading
                    current_lines = []
                    continue
            if current_lines:
                sections.append(format_section(current_title, None, current_lines))
                current_lines = []
            continue
        if is_section_heading(stripped):
            if current_lines:
                sections.append(format_section(current_title, None, current_lines))
                current_lines = []
            current_title = stripped
            continue
        current_lines.append(stripped)
    if current_lines:
        sections.append(format_section(current_title, None, current_lines))
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


def split_section_header(section: str) -> Tuple[Optional[str], Optional[str], str]:
    lines = section.splitlines()
    if not lines:
        return None, None, ""
    section_title: Optional[str] = None
    entry_title: Optional[str] = None
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        lowered = line.lower()
        if lowered.startswith("section:") and section_title is None:
            section_title = line.split(":", 1)[1].strip() or None
            idx += 1
            continue
        if lowered.startswith("entry:") and entry_title is None:
            entry_title = line.split(":", 1)[1].strip() or None
            idx += 1
            continue
        break
    body = "\n".join(lines[idx:]).strip()
    return section_title, entry_title, body


def get_section_tags(section_title: Optional[str]) -> List[str]:
    if not section_title:
        return []
    lowered = section_title.lower()
    tags: List[str] = []
    seen = set()
    for keywords, values in SECTION_TAGS:
        if any(keyword in lowered for keyword in keywords):
            for tag in values:
                if tag not in seen:
                    tags.append(tag)
                    seen.add(tag)
    return tags


def build_chunk_header(
    section_title: Optional[str],
    entry_title: Optional[str],
    tags: List[str],
) -> str:
    header = f"Section: {section_title}" if section_title else "Section: Profile"
    if entry_title:
        header = f"{header} | Entry: {entry_title}"
    if tags:
        header = f"{header} | Tags: {', '.join(tags)}"
    return header


def chunk_profile_section(section: str) -> List[str]:
    section_title, entry_title, body = split_section_header(section)
    tags = get_section_tags(section_title)
    header = build_chunk_header(section_title, entry_title, tags)
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
