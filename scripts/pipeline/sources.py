"""Source adapters.

Each adapter yields uniform Document records. Fetching is separate
(scripts/fetch_probe.py) so the pipeline can re-run over local raw data
without touching the network.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import clean


@dataclass
class Document:
    doc_id: str
    source: str
    text: str
    pub_year: int | None = None
    title: str | None = None


# --- Gutenberg --------------------------------------------------------------

# Bulk mirror. aleph.gutenberg.org was unreachable when this was written;
# pglaf is the working sanctioned rsync mirror. PG asks that the website itself
# not be crawled, which is why neither is www.gutenberg.org.
GUTENBERG_RSYNC = "gutenberg.pglaf.org::gutenberg"

_YEAR = re.compile(r"\b(1[6-9]\d{2}|20[0-2]\d)\b")


def _guess_year(header: str) -> int | None:
    """Best-effort publication year from the PG header block."""
    m = re.search(r"(?:Release Date|Posting Date)[^\n]*?(\d{4})", header)
    if m:
        return int(m.group(1))
    years = [int(y) for y in _YEAR.findall(header)]
    return min(years) if years else None


GUTENBERG_HF = "sedthh/gutenberg_english"

_AUTHOR_YEAR = re.compile(r"(1[0-9]{3}|20[0-2][0-9])")


def _gutenberg_hf_year(meta: dict) -> int | None:
    """Rough era from the author's dates.

    `issued` is the Project Gutenberg posting date, not publication -- the US
    Constitution is stamped 1975 -- so it is useless for the archaic-language
    ablation this field exists for. The author's death year (or birth, failing
    that) is a far better proxy for when the text was written.
    """
    years = _AUTHOR_YEAR.findall(str(meta.get("authors", "")))
    return int(years[-1]) if years else None


def read_gutenberg_hf(limit: int | None = None) -> Iterator[Document]:
    """Stream Gutenberg from the HuggingFace mirror.

    The rsync mirror (`read_gutenberg`) serves ~0.47 MB/s, which puts a 3B-word
    Gutenberg budget at over ten hours. This copy streams at HF speeds (~25
    MB/s measured) and is already language-filtered to English.

    Crucially it is still HARD-WRAPPED (~51 char lines), so the unwrapping and
    hyphenation-rejoining chain still applies and is still exercised -- that
    was the condition for accepting a pre-packaged copy over raw mirror files.
    PG's own header is already stripped; our stripper is a no-op on it.
    """
    import datasets

    ds = datasets.load_dataset(GUTENBERG_HF, split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        try:
            meta = json.loads(row["METADATA"]) if isinstance(row["METADATA"], str) \
                else (row["METADATA"] or {})
        except Exception:
            meta = {}
        raw = (row["TEXT"] or "").replace("\r\n", "\n").replace("\r", "\n")
        if len(raw) < 2000:
            continue
        body = clean.strip_gutenberg_boilerplate(raw)
        body = clean.drop_gutenberg_frontmatter(body)
        # This copy is double-spaced; without this, unwrapping is a no-op and
        # sentences are truncated at line breaks.
        body = clean.collapse_double_spacing(body)
        body = clean.rejoin_hyphenation(body)
        body = clean.strip_emphasis_markers(body)
        body = clean.unwrap_paragraphs(body)
        yield Document(
            doc_id=f"pg:{meta.get('text_id', i)}",
            source="gutenberg",
            text=body,
            pub_year=_gutenberg_hf_year(meta),
            title=meta.get("title"),
        )


def read_gutenberg(root: Path) -> Iterator[Document]:
    """Yield documents from a local mirror of PG .txt files.

    Applies the boilerplate/frontmatter/unwrapping chain, which is the whole
    reason we take raw mirror files rather than a pre-cleaned redistribution:
    hard-wrap unwrapping is the documented correctness hazard
    and it must actually be exercised.
    """
    for path in sorted(root.rglob("*.txt")):
        # Skip encoding variants and superseded copies.
        if "old" in path.parts or path.stem.endswith(("-8", "-0")):
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(raw) < 2000:
            continue

        year = _guess_year(raw[:3000])
        body = clean.strip_gutenberg_boilerplate(raw)
        body = clean.drop_gutenberg_frontmatter(body)
        # Hyphenation must be rejoined BEFORE unwrapping, while the line
        # breaks that caused the split still exist.
        body = clean.rejoin_hyphenation(body)
        body = clean.strip_emphasis_markers(body)
        body = clean.unwrap_paragraphs(body)
        yield Document(
            doc_id=f"pg:{path.stem}", source="gutenberg", text=body, pub_year=year
        )


# --- Wikipedia --------------------------------------------------------------

WIKIPEDIA_DATASET = ("wikimedia/wikipedia", "20231101.en")


def read_wikipedia(limit: int | None = None, local: Path | None = None) -> Iterator[Document]:
    if local is not None:
        yield from _read_jsonl(local, "wikipedia")
        return
    import datasets

    ds = datasets.load_dataset(*WIKIPEDIA_DATASET, split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield Document(
            doc_id=f"wiki:{row['id']}",
            source="wikipedia",
            text=clean.drop_wikipedia_tail_sections(row["text"]),
            title=row.get("title"),
        )


# --- C4 realnewslike --------------------------------------------------------

C4_DATASET = ("allenai/c4", "realnewslike")


def read_c4news(limit: int | None = None, local: Path | None = None) -> Iterator[Document]:
    if local is not None:
        yield from _read_jsonl(local, "c4news")
        return
    import datasets

    ds = datasets.load_dataset(*C4_DATASET, split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        year = None
        ts = row.get("timestamp")
        if ts:
            m = _YEAR.search(str(ts))
            if m:
                year = int(m.group(1))
        yield Document(
            doc_id=f"c4:{i}", source="c4news", text=row["text"], pub_year=year
        )


# --- Stack Exchange ---------------------------------------------------------

STACKEXCHANGE_BASE = "https://archive.org/download/stackexchange"


def read_stackexchange(root: Path) -> Iterator[Document]:
    """Yield post bodies from extracted per-site Posts.xml files.

    Iterparse because Posts.xml runs to gigabytes for the larger sites.
    """
    for posts in sorted(root.rglob("Posts.xml")):
        site = posts.parent.name
        for _, elem in ET.iterparse(posts, events=("end",)):
            if elem.tag != "row":
                continue
            body = elem.get("Body")
            if body:
                date = elem.get("CreationDate", "")
                year = int(date[:4]) if date[:4].isdigit() else None
                yield Document(
                    doc_id=f"se:{site}:{elem.get('Id')}",
                    source="stackexchange",
                    text=clean.clean_stackexchange_body(body),
                    pub_year=year,
                )
            elem.clear()


# --- Shared -----------------------------------------------------------------


def _read_jsonl(path: Path, source: str) -> Iterator[Document]:
    import json

    paths = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    for p in paths:
        with p.open() as fh:
            for line in fh:
                row = json.loads(line)
                yield Document(
                    doc_id=row["doc_id"],
                    source=source,
                    text=row["text"],
                    pub_year=row.get("pub_year"),
                    title=row.get("title"),
                )


READERS = {
    "gutenberg": read_gutenberg,
    "wikipedia": read_wikipedia,
    "c4news": read_c4news,
    "stackexchange": read_stackexchange,
}
