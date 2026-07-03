"""Corpus definition: the real markdown files DocsPilot ingests.

The corpus is OrdinalDB's own documentation: the root README plus every
project-level markdown doc and the nested Python package README, in
addition to `docs/*.md` (recursively, including `docs/roadmap/`). Using a
fixed, real corpus instead of a folder of synthetic snippets means the demo
exercises the same headings, code fences, and cross-references a real docs
site would have.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Paths are relative to the OrdinalDB repo root.
CORPUS_RELATIVE_PATHS: tuple[str, ...] = (
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "RELEASING.md",
    "SECURITY.md",
    "THIRD_PARTY.md",
    "THREAT_MODEL.md",
    "ordinaldb-python/README.md",
    "docs/api.md",
    "docs/edge-deployment.md",
    "docs/persistence.md",
    "docs/provenance.md",
    "docs/raspberry-pi.md",
    "docs/roadmap/0.2.0-feature-parity-spec.md",
    "docs/roadmap/0.3.0-api-async-streaming-spec.md",
)


@dataclass(frozen=True)
class CorpusFile:
    """A single markdown file in the corpus, resolved to an absolute path."""

    relative_path: str
    absolute_path: Path

    @property
    def is_docs_subtree(self) -> bool:
        """True for files under docs/ -- used as the `doc_type` metadata filter."""
        return self.relative_path.startswith("docs/")


def find_repo_root(start: Path) -> Path:
    """Walk upward from `start` looking for the OrdinalDB repo root marker.

    We identify the root by the presence of both README.md and a docs/
    directory, which is stable regardless of how deep this project lives
    inside the repository checkout.
    """
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "README.md").is_file() and (candidate / "docs").is_dir():
            return candidate
    raise FileNotFoundError(
        f"could not locate OrdinalDB repo root walking up from {start}"
    )


def load_corpus(repo_root: Path) -> list[CorpusFile]:
    """Resolve the fixed corpus file list against `repo_root`.

    Raises FileNotFoundError early (with the exact missing path) rather than
    silently skipping a file -- ingestion should fail loudly if the corpus
    definition drifts from the repo layout.
    """
    files: list[CorpusFile] = []
    missing: list[str] = []
    for relative in CORPUS_RELATIVE_PATHS:
        absolute = repo_root / relative
        if not absolute.is_file():
            missing.append(relative)
            continue
        files.append(CorpusFile(relative_path=relative, absolute_path=absolute))
    if missing:
        raise FileNotFoundError(f"corpus files missing from repo root {repo_root}: {missing}")
    return files
