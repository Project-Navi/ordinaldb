"""Markdown-aware chunking with source/section metadata.

Each chunk gets a deterministic ID of the form ``{relative_path}#{index:04d}``
so that re-ingesting the same file is trackable across runs: the caller can
diff old vs. new chunk IDs for a source file and decide what to delete before
re-adding (see demo.section_delete_and_reupsert for the workflow this
enables).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from docspilot.corpus import CorpusFile

_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 120


@dataclass(frozen=True)
class Chunk:
    id: str
    document: Document


def chunk_file(corpus_file: CorpusFile) -> list[Chunk]:
    """Split one markdown file into (id, Document) chunks with metadata.

    Metadata written per chunk:
      - source: relative path from repo root, e.g. "docs/api.md"
      - section: nearest markdown header above the chunk ("h3" > "h2" > "h1")
      - doc_type: "docs" if under docs/, else "root" -- used for filtering
      - file_name: basename of the logical source path (relative_path), for
        display in source attribution -- derived from relative_path rather
        than absolute_path so it stays correct even when absolute_path
        points somewhere else (e.g. a temp file simulating an edit)
    """
    text = corpus_file.absolute_path.read_text(encoding="utf-8")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    header_sections = header_splitter.split_text(text)
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )

    doc_type = "docs" if corpus_file.is_docs_subtree else "root"
    chunks: list[Chunk] = []
    index = 0
    for section_doc in header_sections:
        section = (
            section_doc.metadata.get("h3")
            or section_doc.metadata.get("h2")
            or section_doc.metadata.get("h1")
            or "(no heading)"
        )
        for piece in sub_splitter.split_text(section_doc.page_content):
            chunk_id = f"{corpus_file.relative_path}#{index:04d}"
            metadata = {
                "source": corpus_file.relative_path,
                "section": section,
                "doc_type": doc_type,
                "file_name": Path(corpus_file.relative_path).name,
            }
            chunks.append(Chunk(id=chunk_id, document=Document(page_content=piece, metadata=metadata)))
            index += 1
    return chunks


def chunk_files(corpus_files: list[CorpusFile]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for corpus_file in corpus_files:
        all_chunks.extend(chunk_file(corpus_file))
    return all_chunks
