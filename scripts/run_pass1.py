#!/usr/bin/env python3
"""Pass 1 — the full corpus.

Two stages:

  calibrate  Stream a ~1B-word stratified sample, count, discard. Produces the
             pooled frequency table that defines the rarity band. A sample
             suffices because the band floor (0.05/million) has ~50 expected
             occurrences at 1B words; Pass 0's 31M tokens gave 1.5, which is
             why it could not calibrate.

  stream     One full pass over ~8.7B words. Segments, filters, dedups, retains
             by per-word coverage, annotates only what is retained, and mines
             the collocation and name tables on the way through.

    python scripts/run_pass1.py --stage calibrate
    python scripts/run_pass1.py --stage stream
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import clean, config, dedup, segment, stream as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
TABLES = ROOT / "data" / "tables"

CALIBRATION_WORDS = 1.0e9
SOURCES = ["gutenberg", "wikipedia", "c4news", "stackexchange"]

# Lossy counting: prune rare entries when the table grows past this, so bigram
# counting over billions of words stays inside laptop RAM.
COUNTER_MAX = 12_000_000
COUNTER_PRUNE_TO = 6_000_000

PROGRESS_WORDS = 25_000_000


def log(msg: str) -> None:
    print(f"[pass1] {msg}", flush=True)


def prune(counter: Counter, keep: int) -> int:
    """Drop the lowest-count entries. Returns the cutoff applied."""
    if len(counter) <= keep:
        return 0
    cutoff = sorted(counter.values(), reverse=True)[keep]
    for k in [k for k, v in counter.items() if v <= cutoff]:
        del counter[k]
    return cutoff


# --- Stage: calibrate -------------------------------------------------------


def stage_calibrate(total_words: float) -> None:
    share = sum(st.SOURCE_WORDS.values())
    budgets = {s: total_words * st.SOURCE_WORDS[s] / share for s in SOURCES}
    log("calibration budgets: " + ", ".join(
        f"{s} {b/1e6:.0f}M" for s, b in budgets.items()))

    pooled: Counter = Counter()
    per_source: dict[str, int] = {}
    t0 = time.perf_counter()

    for source in SOURCES:
        counts: Counter = Counter()
        words = 0
        next_mark = PROGRESS_WORDS
        for doc in st.stream(source, WORK / source, budgets[source]):
            if not clean.is_acceptable_document(doc.text):
                continue
            toks = clean.words(clean.normalize(doc.text))
            counts.update(toks)
            words += len(toks)
            if words >= next_mark:
                log(f"{source}: {words/1e6:.0f}M words, {len(counts):,} forms")
                next_mark += PROGRESS_WORDS
            if len(counts) > COUNTER_MAX:
                cut = prune(counts, COUNTER_PRUNE_TO)
                log(f"{source}: pruned counter at <= {cut}")
        per_source[source] = words
        pooled.update(counts)
        log(f"{source}: done, {words:,} words")
        # Per-source counts feed the formality score.
        _write_counts(TABLES / f"pass1_freq_{source}.parquet", counts)
        del counts

    total = sum(pooled.values())
    floor = 1 / total * 1e6
    lo, hi = config.RARITY_BAND
    band = {w for w, c in pooled.items() if lo <= c / total * 1e6 <= hi}

    _write_counts(TABLES / "pass1_pooled_frequency.parquet", pooled)
    (TABLES / "pass1_calibration.json").write_text(json.dumps({
        "total_tokens": total,
        "distinct_forms": len(pooled),
        "resolution_floor_per_million": floor,
        "band": [lo, hi],
        "band_vocab_size": len(band),
        "words_per_source": per_source,
        "contexts_per_word_cap": config.CONTEXTS_PER_WORD,
        "elapsed_seconds": time.perf_counter() - t0,
    }, indent=2))

    log(f"pooled: {total:,} tokens, {len(pooled):,} forms, floor {floor:.5f}/M")
    log(f"band [{lo}, {hi}]/M -> {len(band):,} wordforms")
    log(f"expected occurrences at band floor: {lo * total / 1e6:.0f}")
    projected = len(band) * config.CONTEXTS_PER_WORD / 2.5
    log(f"projected retained sentences: ~{projected/1e6:.1f}M "
        f"(~{projected * 59 / 1e9:.2f} GB)")


def _write_counts(path: Path, counts: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = counts.most_common()
    pq.write_table(
        pa.table({"form": pa.array([w for w, _ in items]),
                  "count": pa.array([c for _, c in items], type=pa.int64())}),
        path, compression="zstd",
    )


# --- Stage: stream ----------------------------------------------------------


PRIORITY_LIST = ROOT / "data" / "inventory" / "gregmat_900.tsv"

# The regex segmenter, not pysbd. Measured: pysbd runs at 77k words/sec against
# the regex backend's 3.9M -- 51x slower, and 95% of stage-1b runtime. Exact
# agreement on modern prose is 92.7%, and the disagreements are mostly where an
# opening quote attaches, not sentence content. blingfire would be better still
# but has no arm64 wheel. This is a deliberate quality-for-speed trade.
SEGMENT_BACKEND = "regex-fallback"


def load_priority_words() -> set[str]:
    """The GregMat 900 — the core vocabulary the product is built around."""
    if not PRIORITY_LIST.exists():
        return set()
    import csv

    with PRIORITY_LIST.open() as fh:
        rows = csv.DictReader((l for l in fh if not l.startswith("#")),
                              delimiter="\t")
        return {r["word"].strip().lower() for r in rows if r.get("word")}


def load_band() -> tuple[set[str], set[str], dict]:
    meta = json.loads((TABLES / "pass1_calibration.json").read_text())
    table = pq.read_table(TABLES / "pass1_pooled_frequency.parquet")
    total = meta["total_tokens"]
    lo, hi = meta["band"]
    forms = table["form"].to_pylist()
    counts = table["count"].to_pylist()
    band = {f for f, c in zip(forms, counts) if lo <= c / total * 1e6 <= hi}

    # Priority words join the band regardless of frequency. 64 of the 900 sit
    # ABOVE the band because they are polysemous -- "august", "sound", "base",
    # "prime" are common in a sense other than the one being taught -- and
    # without this they would get no banked contexts at all.
    priority = load_priority_words()
    forced = priority - band
    band |= priority
    if forced:
        log(f"priority list: {len(priority):,} words, {len(forced)} forced into band")
    return band, priority, meta


def interleave_sources(chunk: int = 25):
    """Round-robin documents across sources.

    Processing sources sequentially front-loads the coverage quota onto
    whichever runs first: in the aborted run, Gutenberg had claimed contexts
    for 68,560 band words before a single news or forum sentence was read.
    Since every word stops at `cap` contexts, those slots were gone -- the
    banked contexts for most of the vocabulary would have been 19th-century
    literature, which is exactly the register skew 1.1 warns about.

    Interleaving spends the quota evenly across registers instead.
    """
    iters = {s: iter(st.stream(s, WORK / s, st.SOURCE_WORDS[s])) for s in SOURCES}
    while iters:
        for source in list(iters):
            for _ in range(chunk):
                try:
                    yield source, next(iters[source])
                except StopIteration:
                    log(f"{source}: exhausted")
                    iters.pop(source, None)
                    break


def stage_stream() -> None:
    band, priority, meta = load_band()
    cap = meta["contexts_per_word_cap"]
    log(f"band: {len(band):,} wordforms, cap {cap} contexts each")
    log(f"segmenter: {SEGMENT_BACKEND}")

    coverage: Counter = Counter()
    bigrams: Counter = Counter()
    capitalized: Counter = Counter()
    lowercased: Counter = Counter()
    rng = random.Random(0)
    INTERIM.mkdir(parents=True, exist_ok=True)

    totals = {"words": 0, "sentences": 0, "retained": 0, "filtered": 0,
              "exact_dupes": 0, "dropped_lang": 0, "dropped_quality": 0,
              "baseline": 0}
    t0 = time.perf_counter()

    seen_exact: set[str] = set()
    rows: list[dict] = []
    shard = 0
    next_mark = PROGRESS_WORDS

    for source, doc in interleave_sources():
        if not clean.is_acceptable_document(doc.text):
            totals["dropped_quality"] += 1
            continue
        if not _language_ok(doc.text):
            totals["dropped_lang"] += 1
            continue

        text = clean.normalize(doc.text)
        toks = clean.words(text)
        totals["words"] += len(toks)

        # Collocation and name evidence, mined on the way through.
        bigrams.update(zip(toks, toks[1:]))
        for t in clean.WORD.findall(text):
            (capitalized if t[0].isupper() else lowercased)[t.lower()] += 1

        for sentence in segment.segment_document(
            text, backend=SEGMENT_BACKEND, apply_filters=False
        ):
            totals["sentences"] += 1
            if not clean.is_acceptable_sentence(sentence):
                totals["filtered"] += 1
                continue
            key = dedup.exact_key(sentence)
            if key in seen_exact:
                totals["exact_dupes"] += 1
                continue
            seen_exact.add(key)

            hits = {t for t in clean.words(sentence) if t in band}
            if any(coverage[w] < cap for w in hits):
                coverage.update(hits)
            elif (totals["baseline"] < config.BASELINE_MAX_SENTENCES
                  and rng.random() < config.BASELINE_SAMPLE_RATE):
                totals["baseline"] += 1
            else:
                continue

            rows.append({
                "sid": f"{source}:{totals['retained']}",
                "source": source,
                "doc_id": doc.doc_id,
                "pub_year": doc.pub_year,
                "text": sentence,
                "n_tok": clean.token_count(sentence),
            })
            totals["retained"] += 1
            if len(rows) >= 500_000:
                _flush(rows, "mixed", shard)
                shard += 1
                rows = []

        if totals["words"] >= next_mark:
            covered_priority = sum(1 for w in priority if coverage[w] >= cap)
            log(f"{totals['words']/1e6:.0f}M words | {totals['retained']:,} retained "
                f"| band {len(coverage):,} covered "
                f"| priority {covered_priority}/{len(priority)} at cap")
            next_mark += PROGRESS_WORDS
        if len(bigrams) > COUNTER_MAX:
            prune(bigrams, COUNTER_PRUNE_TO)
        if len(seen_exact) > 40_000_000:
            seen_exact.clear()  # bounded memory; costs a few missed dupes

    if rows:
        _flush(rows, "mixed", shard)

    meta["_priority"] = priority
    _finish_tables(bigrams, capitalized, lowercased, coverage, totals, t0, meta)


def _flush(rows: list[dict], source: str, shard: int) -> None:
    path = INTERIM / f"pass1_{source}_{shard:04d}.parquet"
    pq.write_table(pa.table({
        k: pa.array([r[k] for r in rows],
                    type=pa.int32() if k in ("pub_year", "n_tok") else None)
        for k in ("sid", "source", "doc_id", "pub_year", "text", "n_tok")
    }), path, compression="zstd")
    log(f"wrote {path.name} ({len(rows):,} rows, {path.stat().st_size/1e6:.0f} MB)")


def _language_ok(text: str) -> bool:
    try:
        import py3langid

        return py3langid.classify(text[:2000])[0] == "en"
    except Exception:
        return True


def _finish_tables(bigrams, capitalized, lowercased, coverage, totals, t0, meta) -> None:
    """Collocations, names, and the coverage report."""
    total_bi = sum(bigrams.values())
    keep = [(a, b, c) for (a, b), c in bigrams.items()
            if c >= config.COLLOCATION_MIN_COUNT]
    keep.sort(key=lambda x: -x[2])
    keep = keep[: config.COLLOCATION_TOP_K]
    pq.write_table(pa.table({
        "w1": pa.array([a for a, _, _ in keep]),
        "w2": pa.array([b for _, b, _ in keep]),
        "count": pa.array([c for _, _, c in keep], type=pa.int64()),
    }), TABLES / "pass1_collocations.parquet", compression="zstd")

    names = [w for w, c in capitalized.items()
             if c >= 20 and c / (c + lowercased.get(w, 0)) >= 0.9]
    (TABLES / "pass1_names.txt").write_text("\n".join(sorted(names)) + "\n")

    shippable = sum(1 for c in coverage.values()
                    if c >= config.MIN_CONTEXTS_TO_SHIP)
    priority = meta.get("_priority", set())
    prio_shippable = sum(1 for w in priority
                         if coverage[w] >= config.MIN_CONTEXTS_TO_SHIP)
    prio_at_cap = sum(1 for w in priority
                      if coverage[w] >= meta["contexts_per_word_cap"])
    report = {
        **totals,
        "band_words_covered": len(coverage),
        "band_words_shippable": shippable,
        "band_vocab_size": meta["band_vocab_size"],
        "priority_words": len(priority),
        "priority_shippable": prio_shippable,
        "priority_at_cap": prio_at_cap,
        "priority_uncovered": sorted(w for w in priority
                                     if coverage[w] < config.MIN_CONTEXTS_TO_SHIP),
        "coverage_of_band": round(len(coverage) / max(1, meta["band_vocab_size"]), 4),
        "collocations_kept": len(keep),
        "bigram_tokens": total_bi,
        "names": len(names),
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (TABLES / "pass1_report.json").write_text(json.dumps(report, indent=2))
    log(f"retained {totals['retained']:,} sentences; "
        f"{shippable:,} band words at >= {config.MIN_CONTEXTS_TO_SHIP} contexts; "
        f"{len(keep):,} collocations; {len(names):,} names")



# --- Stage: topup -----------------------------------------------------------


def stage_topup(sources_wanted: list[str], quota: int = 25) -> None:
    """Bank contexts for the priority list from under-represented registers.

    The main stream interleaves by DOCUMENT, so 25 Gutenberg books (~2.5M
    words) alternate with 25 forum posts (~2.5k) -- a 1000x imbalance. The
    per-word cap was full long before Stack Exchange contributed, leaving only
    133 of the 900 priority words with 10+ contexts in the deployment
    register, against 883 from Gutenberg.

    This pass ignores the band and the cap entirely: it streams the named
    sources and keeps any sentence containing a priority word, up to `quota`
    per word per source. Cheap, because it only has to re-read the small
    sources.
    """
    priority = load_priority_words()
    log(f"topup: {len(priority):,} priority words, quota {quota}/word/source")
    INTERIM.mkdir(parents=True, exist_ok=True)

    for source in sources_wanted:
        coverage: Counter = Counter()
        seen: set[str] = set()
        rows: list[dict] = []
        words = kept = 0
        next_mark = PROGRESS_WORDS

        for doc in st.stream(source, WORK / source, st.SOURCE_WORDS[source]):
            if not clean.is_acceptable_document(doc.text) or not _language_ok(doc.text):
                continue
            text = clean.normalize(doc.text)
            words += len(clean.words(text))

            for sentence in segment.segment_document(
                text, backend=SEGMENT_BACKEND, apply_filters=False
            ):
                if not clean.is_acceptable_sentence(sentence):
                    continue
                hits = {t for t in clean.words(sentence) if t in priority}
                if not hits or all(coverage[w] >= quota for w in hits):
                    continue
                key = dedup.exact_key(sentence)
                if key in seen:
                    continue
                seen.add(key)
                coverage.update(hits)
                rows.append({
                    "sid": f"topup:{source}:{kept}", "source": source,
                    "doc_id": doc.doc_id, "pub_year": doc.pub_year,
                    "text": sentence, "n_tok": clean.token_count(sentence),
                })
                kept += 1

            if words >= next_mark:
                at10 = sum(1 for w in priority if coverage[w] >= 10)
                log(f"{source}: {words/1e6:.0f}M words | {kept:,} kept | "
                    f"{at10}/{len(priority)} priority words at 10+ contexts")
                next_mark += PROGRESS_WORDS

        if rows:
            _flush(rows, f"topup_{source}", 0)
        at10 = sum(1 for w in priority if coverage[w] >= 10)
        log(f"{source}: done -- {kept:,} sentences, "
            f"{at10}/{len(priority)} priority words at 10+ contexts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["calibrate", "stream", "topup"])
    ap.add_argument("--sources", default="stackexchange,c4news")
    ap.add_argument("--words", type=float, default=CALIBRATION_WORDS)
    args = ap.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    if args.stage == "calibrate":
        stage_calibrate(args.words)
    elif args.stage == "stream":
        stage_stream()
    else:
        stage_topup([s for s in args.sources.split(",") if s])


if __name__ == "__main__":
    main()
