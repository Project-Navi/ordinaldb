"""Load paper metadata for PaperScout.

Offline-first: `load_papers()` returns the bundled corpus by default, no
network required. Pass `live=True` to attempt a live fetch from the public
arXiv API (cs.IR, cs.LG, cs.DB) first, falling back to the bundled corpus
if the API is unreachable, rate-limits us, or times out -- the way a real
tool should degrade on a flaky network rather than blocking indefinitely.
"""

from __future__ import annotations

import sys
import time

import defusedxml.ElementTree as ET
import requests

from corpus_data import load_bundled_corpus

ARXIV_API = "https://export.arxiv.org/api/query"
CATEGORIES = ["cs.IR", "cs.LG", "cs.DB"]
PER_CATEGORY = 15
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _log(message: str) -> None:
    print(f"[fetch_papers] {message}", file=sys.stderr)


def _fetch_category(category: str, max_results: int, timeout: float) -> list[dict]:
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": "PaperScout/0.1 (local research tool)"}
    response = requests.get(ARXIV_API, params=params, timeout=timeout, headers=headers)
    response.raise_for_status()
    root = ET.fromstring(response.text)

    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id = entry.findtext(f"{ATOM_NS}id", default="").rsplit("/", 1)[-1]
        title = " ".join(entry.findtext(f"{ATOM_NS}title", default="").split())
        summary = " ".join(entry.findtext(f"{ATOM_NS}summary", default="").split())
        published = entry.findtext(f"{ATOM_NS}published", default="")
        year = int(published[:4]) if published[:4].isdigit() else 0
        authors = entry.findall(f"{ATOM_NS}author")
        papers.append(
            {
                "id": arxiv_id or f"{category}-{len(papers)}",
                "title": title,
                "abstract": summary,
                "category": category,
                "year": year,
                "authors_count": len(authors),
                "source": "arxiv",
            }
        )
    return papers


def try_live_arxiv_fetch(*, attempts: int = 2, timeout: float = 12.0) -> list[dict] | None:
    """Attempt to fetch real abstracts from the arXiv API.

    Returns None (rather than raising) if the API is unreachable or
    rate-limits us, so the caller can fall back to the bundled corpus.
    """
    all_papers: list[dict] = []
    for category in CATEGORIES:
        fetched = False
        for attempt in range(1, attempts + 1):
            try:
                _log(f"fetching {category} (attempt {attempt}/{attempts})...")
                papers = _fetch_category(category, PER_CATEGORY, timeout)
                if not papers:
                    _log(f"{category}: empty response, treating as failure")
                    break
                all_papers.extend(papers)
                fetched = True
                break
            except Exception as exc:  # noqa: BLE001 - any network failure triggers fallback
                _log(f"{category}: {type(exc).__name__}: {exc}")
                time.sleep(2)
        if not fetched:
            _log(f"{category}: giving up on live fetch for this category")
            return None

    # arXiv papers are frequently cross-listed (e.g. a paper can carry both
    # cat:cs.IR and cat:cs.DB). Since each category is fetched with a
    # separate query, a cross-listed paper comes back once per matching
    # category with the SAME arxiv id but a DIFFERENT `category` value.
    # Left unhandled, that produces two documents with the same doc_id/
    # ref_doc_id and inconsistent category metadata. Dedupe by id, keeping
    # the first category encountered.
    deduped: dict[str, dict] = {}
    for paper in all_papers:
        deduped.setdefault(paper["id"], paper)
    duplicate_count = len(all_papers) - len(deduped)
    if duplicate_count:
        _log(f"deduplicated {duplicate_count} cross-listed paper(s) fetched under >1 category")
    return list(deduped.values()) or None


def load_papers(*, live: bool = False) -> tuple[list[dict], str]:
    """Load the paper corpus. Returns (papers, source_label).

    Defaults to the bundled corpus (offline, deterministic, ~40 papers).
    Pass live=True to try the arXiv API first, falling back to the bundled
    corpus automatically if the API isn't reachable.
    """
    if not live:
        return load_bundled_corpus(), "bundled_corpus"

    start = time.time()
    live_papers = try_live_arxiv_fetch()
    elapsed = time.time() - start
    if live_papers:
        _log(f"live arXiv fetch succeeded: {len(live_papers)} papers in {elapsed:.1f}s")
        return live_papers, "arxiv"

    _log(
        f"live arXiv fetch failed/unavailable after {elapsed:.1f}s of attempts; "
        "falling back to the bundled corpus."
    )
    return load_bundled_corpus(), "bundled_corpus"


if __name__ == "__main__":
    live = "--live" in sys.argv[1:]
    papers, source = load_papers(live=live)
    print(f"Loaded {len(papers)} papers from source={source}")
    for paper in papers[:5]:
        print(" -", paper["id"], paper["title"])
