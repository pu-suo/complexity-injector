#!/usr/bin/env python3
"""Fetch the Pass 0 probe sample.

Downloads ~1.3GB across four sources into data/raw/. Idempotent: each source
skips work already on disk, so an interrupted run resumes.

    python scripts/fetch_probe.py --source all
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import config, sources  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# Probe sizes. Chosen to give >100k sentences per source -- ample for the 7
# thresholds -- while staying near 1.3GB total.
PROBE_GUTENBERG_BOOKS = 600
PROBE_WIKIPEDIA_DOCS = 100_000
PROBE_C4_DOCS = 200_000
PROBE_SE_SITES = ["literature", "interpersonal", "writers", "skeptics"]

RSYNC_SUBTREES = ["1/2", "3/4", "5/6", "7/8"]  # spread across the ID range


def log(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)


# --- Gutenberg --------------------------------------------------------------


def fetch_gutenberg(dest: Path, n_books: int = PROBE_GUTENBERG_BOOKS) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    existing = list(dest.rglob("*.txt"))
    if len(existing) >= n_books:
        log(f"gutenberg: {len(existing)} books already present, skipping")
        return

    listing_cache = dest / "_listing.json"
    if listing_cache.exists():
        candidates = json.loads(listing_cache.read_text())
    else:
        candidates = []
        for subtree in RSYNC_SUBTREES:
            log(f"gutenberg: listing {subtree}/ ...")
            proc = subprocess.run(
                ["rsync", "--contimeout=30", "--list-only", "-r",
                 f"{sources.GUTENBERG_RSYNC}/{subtree}/"],
                capture_output=True, text=True, timeout=900,
            )
            if proc.returncode != 0:
                log(f"gutenberg: listing {subtree} failed: {proc.stderr[:200]}")
                continue
            for line in proc.stdout.splitlines():
                parts = line.split(None, 4)
                if len(parts) < 5 or line.startswith("d"):
                    continue
                rel = parts[4]
                if not rel.endswith(".txt"):
                    continue
                stem = rel.rsplit("/", 1)[-1][:-4]
                # Skip encoding variants (-8 latin-1, -0 utf-8) and old copies.
                if stem.endswith(("-8", "-0")) or "old/" in rel:
                    continue
                candidates.append(f"{subtree}/{rel}")
        listing_cache.write_text(json.dumps(candidates))
        log(f"gutenberg: {len(candidates)} candidate .txt files")

    random.Random(0).shuffle(candidates)
    picked = candidates[:n_books]
    files_from = dest / "_files.txt"
    files_from.write_text("\n".join(picked) + "\n")

    log(f"gutenberg: rsyncing {len(picked)} books ...")
    proc = subprocess.run(
        ["rsync", "-a", "--contimeout=30", f"--files-from={files_from}",
         f"{sources.GUTENBERG_RSYNC}/", str(dest)],
        capture_output=True, text=True, timeout=5400,
    )
    if proc.returncode != 0:
        log(f"gutenberg: rsync failed: {proc.stderr[:400]}")
    log(f"gutenberg: {len(list(dest.rglob('*.txt')))} books on disk")


# --- HuggingFace-streamed sources ------------------------------------------


def _dump_stream(reader, dest: Path, limit: int, name: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{name}.jsonl"
    if out.exists() and sum(1 for _ in out.open()) >= limit:
        log(f"{name}: already have {limit} docs, skipping")
        return
    log(f"{name}: streaming {limit} docs ...")
    written = 0
    with out.open("w") as fh:
        for doc in reader(limit=limit):
            fh.write(json.dumps({
                "doc_id": doc.doc_id, "text": doc.text,
                "pub_year": doc.pub_year, "title": doc.title,
            }) + "\n")
            written += 1
            if written % 20_000 == 0:
                log(f"{name}: {written:,}")
    log(f"{name}: wrote {written:,} docs ({out.stat().st_size/1e6:.0f} MB)")


def fetch_wikipedia(dest: Path) -> None:
    _dump_stream(sources.read_wikipedia, dest, PROBE_WIKIPEDIA_DOCS, "wikipedia")


def fetch_c4news(dest: Path) -> None:
    _dump_stream(sources.read_c4news, dest, PROBE_C4_DOCS, "c4news")


# --- Stack Exchange ---------------------------------------------------------


def _archive_org_mirrors(item: str = "stackexchange") -> list[str]:
    """Resolve the real storage nodes for an archive.org item.

    The /download/ redirect endpoint load-balances onto nodes that are
    intermittently 500/502; the metadata endpoint names the nodes directly, so
    we bypass the balancer and fail over ourselves.
    """
    with urllib.request.urlopen(f"https://archive.org/metadata/{item}", timeout=60) as fh:
        meta = json.load(fh)
    dirn = meta.get("dir", f"/items/{item}")
    bases = []
    for server in (meta.get("d1"), meta.get("d2")):
        if server:
            bases.append(f"https://{server}{dirn}")
    for alt in meta.get("alternate_locations", {}).get("servers", []):
        bases.append(f"https://{alt['server']}{alt['dir']}")
    return bases


def fetch_stackexchange(dest: Path, sites: list[str] = PROBE_SE_SITES) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    mirrors = _archive_org_mirrors()
    log(f"stackexchange: {len(mirrors)} mirrors resolved")

    for site in sites:
        site_dir = dest / site
        if (site_dir / "Posts.xml").exists():
            log(f"stackexchange/{site}: already extracted, skipping")
            continue
        archive = dest / f"{site}.7z"
        if not archive.exists():
            filename = f"{site}.stackexchange.com.7z"
            for base in mirrors:
                log(f"stackexchange/{site}: downloading from {base.split('/')[2]} ...")
                proc = subprocess.run(
                    ["curl", "-sSL", "--fail", "-C", "-", "--retry", "3",
                     "--retry-delay", "5", "--connect-timeout", "30",
                     "-A", "complexity-injector research fetch",
                     "-o", str(archive), f"{base}/{filename}"],
                    capture_output=True, text=True, timeout=3600,
                )
                if proc.returncode == 0 and archive.exists() and archive.stat().st_size > 1_000_000:
                    break
                log(f"stackexchange/{site}: mirror failed ({proc.stderr.strip()[:120]})")
            else:
                log(f"stackexchange/{site}: all mirrors failed, skipping")
                continue
        log(f"stackexchange/{site}: extracting Posts.xml ...")
        try:
            import py7zr

            site_dir.mkdir(exist_ok=True)
            with py7zr.SevenZipFile(archive, "r") as z:
                z.extract(path=site_dir, targets=["Posts.xml"])
            archive.unlink()  # the .7z is dead weight once extracted
        except Exception as exc:
            log(f"stackexchange/{site}: extract failed: {exc}")


# --- Entry point ------------------------------------------------------------

FETCHERS = {
    "gutenberg": lambda: fetch_gutenberg(RAW / "gutenberg"),
    "wikipedia": lambda: fetch_wikipedia(RAW / "wikipedia"),
    "c4news": lambda: fetch_c4news(RAW / "c4news"),
    "stackexchange": lambda: fetch_stackexchange(RAW / "stackexchange"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all", choices=["all", *FETCHERS])
    args = ap.parse_args()

    targets = list(FETCHERS) if args.source == "all" else [args.source]
    for name in targets:
        FETCHERS[name]()
    log("done")


if __name__ == "__main__":
    main()
