"""Streaming document iterators for Pass 1.

Pass 0 read a local sample. Pass 1 streams ~60GB and discards it: nothing is
kept that is not an artifact, so peak disk stays around 5GB regardless of
corpus size.

Each iterator takes a word budget and stops once it is met, so the caller
controls volume without knowing anything about shard layout.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Iterator

from . import clean, config, sources
from .sources import Document

# Pass 1 word budgets per source.
#
# Reduced from the original 8.7B. Retention is capped per word, so streaming
# past the point where every band word has its 50 contexts adds nothing. The
# binding case is the rarest band word at 0.05 per million, which needs ~1B
# words to reach 50 occurrences -- so a few billion suffices and 8.7B was
# paying for coverage we already had.
SOURCE_WORDS = {
    "gutenberg": 1.0e9,
    "wikipedia": 1.0e9,
    "c4news": 0.5e9,
    "stackexchange": 0.2e9,
}

GUTENBERG_BATCH = 250  # books per rsync call


def log(msg: str) -> None:
    print(f"[stream] {msg}", flush=True)


# --- Gutenberg --------------------------------------------------------------


def gutenberg_listing(workdir: Path) -> list[str]:
    """Full mirror listing of English-candidate .txt paths, cached.

    Listing the whole tree once is far cheaper than resolving the catalog, and
    non-English books are removed downstream by per-document language ID.
    """
    cache = workdir / "_listing_full.json"
    if cache.exists():
        return json.loads(cache.read_text())

    workdir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for top in "0123456789":
        log(f"gutenberg: listing /{top} ...")
        proc = subprocess.run(
            ["rsync", "--contimeout=30", "--list-only", "-r",
             f"{sources.GUTENBERG_RSYNC}/{top}/"],
            capture_output=True, text=True, timeout=3600,
        )
        if proc.returncode != 0:
            log(f"gutenberg: listing /{top} failed: {proc.stderr[:160]}")
            continue
        for line in proc.stdout.splitlines():
            if line.startswith("d"):
                continue
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            rel = parts[4]
            if not rel.endswith(".txt"):
                continue
            stem = rel.rsplit("/", 1)[-1][:-4]
            if stem.endswith(("-8", "-0")) or "old/" in rel:
                continue
            paths.append(f"{top}/{rel}")
    cache.write_text(json.dumps(paths))
    log(f"gutenberg: {len(paths):,} candidate books")
    return paths


def stream_gutenberg(workdir: Path, word_budget: float) -> Iterator[Document]:
    """Gutenberg via the HuggingFace mirror.

    Measured: the pglaf rsync mirror serves 0.47 MB/s (250 books = 86MB in
    183s), which puts a 3B-word budget past ten hours. The HF copy streams at
    ~25 MB/s -- 53x faster -- and the whole English corpus is 10.7GB.

    `stream_gutenberg_rsync` below is kept as a fallback if the HF copy ever
    disappears; it is correct, just slow.
    """
    words = 0
    for doc in sources.read_gutenberg_hf():
        words += len(clean.WORD.findall(doc.text))
        yield doc
        if words >= word_budget:
            break


def stream_gutenberg_rsync(workdir: Path, word_budget: float, seed: int = 0) -> Iterator[Document]:
    paths = gutenberg_listing(workdir)
    random.Random(seed).shuffle(paths)

    batch_dir = workdir / "_batch"
    words = 0
    for start in range(0, len(paths), GUTENBERG_BATCH):
        if words >= word_budget:
            break
        batch = paths[start : start + GUTENBERG_BATCH]
        shutil.rmtree(batch_dir, ignore_errors=True)
        batch_dir.mkdir(parents=True, exist_ok=True)
        files_from = workdir / "_batch_files.txt"
        files_from.write_text("\n".join(batch) + "\n")

        proc = subprocess.run(
            ["rsync", "-a", "--contimeout=30", f"--files-from={files_from}",
             f"{sources.GUTENBERG_RSYNC}/", str(batch_dir)],
            capture_output=True, text=True, timeout=3600,
        )
        if proc.returncode != 0:
            log(f"gutenberg: rsync batch failed: {proc.stderr[:160]}")
            continue

        for doc in sources.read_gutenberg(batch_dir):
            words += len(clean.WORD.findall(doc.text))
            yield doc
            if words >= word_budget:
                break
    shutil.rmtree(batch_dir, ignore_errors=True)


# --- Stack Exchange ---------------------------------------------------------


def ensure_stackexchange(dest: Path, sites: list[str] | None = None) -> Path:
    """Download and extract every configured site. ~1.7GB, kept: it is small."""
    sites = sites or config.STACKEXCHANGE_SITES
    dest.mkdir(parents=True, exist_ok=True)
    mirrors = _archive_mirrors()
    for site in sites:
        if (dest / site / "Posts.xml").exists():
            continue
        archive = dest / f"{site}.7z"
        if not archive.exists():
            for base in mirrors:
                log(f"stackexchange/{site}: downloading ...")
                proc = subprocess.run(
                    ["curl", "-sSL", "--fail", "-C", "-", "--retry", "3",
                     "--retry-delay", "5", "--connect-timeout", "30",
                     "-A", "complexity-injector research fetch",
                     "-o", str(archive),
                     f"{base}/{site}.stackexchange.com.7z"],
                    capture_output=True, text=True, timeout=7200,
                )
                if proc.returncode == 0 and archive.stat().st_size > 500_000:
                    break
            else:
                log(f"stackexchange/{site}: all mirrors failed")
                continue
        try:
            import py7zr

            (dest / site).mkdir(exist_ok=True)
            with py7zr.SevenZipFile(archive, "r") as z:
                z.extract(path=dest / site, targets=["Posts.xml"])
            archive.unlink()
        except Exception as exc:
            log(f"stackexchange/{site}: extract failed: {exc}")
    return dest


def _archive_mirrors(item: str = "stackexchange") -> list[str]:
    with urllib.request.urlopen(f"https://archive.org/metadata/{item}", timeout=60) as fh:
        meta = json.load(fh)
    dirn = meta.get("dir", f"/items/{item}")
    bases = [f"https://{s}{dirn}" for s in (meta.get("d1"), meta.get("d2")) if s]
    bases += [f"https://{a['server']}{a['dir']}"
              for a in meta.get("alternate_locations", {}).get("servers", [])]
    return bases


def stream_stackexchange(workdir: Path, word_budget: float) -> Iterator[Document]:
    root = ensure_stackexchange(workdir)
    words = 0
    for doc in sources.read_stackexchange(root):
        words += len(clean.WORD.findall(doc.text))
        yield doc
        if words >= word_budget:
            break


# --- HuggingFace-streamed ---------------------------------------------------


def _stream_hf(reader, word_budget: float, label: str) -> Iterator[Document]:
    words = 0
    for doc in reader(limit=None):
        words += len(clean.WORD.findall(doc.text))
        yield doc
        if words >= word_budget:
            break


def stream_wikipedia(workdir: Path, word_budget: float) -> Iterator[Document]:
    yield from _stream_hf(sources.read_wikipedia, word_budget, "wikipedia")


def stream_c4news(workdir: Path, word_budget: float) -> Iterator[Document]:
    yield from _stream_hf(sources.read_c4news, word_budget, "c4news")


STREAMERS = {
    "gutenberg": stream_gutenberg,
    "wikipedia": stream_wikipedia,
    "c4news": stream_c4news,
    "stackexchange": stream_stackexchange,
}


def stream(source: str, workdir: Path, word_budget: float) -> Iterator[Document]:
    return STREAMERS[source](workdir, word_budget)
