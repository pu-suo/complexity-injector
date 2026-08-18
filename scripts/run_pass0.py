#!/usr/bin/env python3
"""Pass 0 — the probe.

Runs the full pipeline over the probe sample and emits the 7 acceptance
checklist. Produces no inventory and labels nothing; its only job is to answer
"is Pass 1 worth renting a box for, and with what settings".

    python scripts/run_pass0.py --source all
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import clean, config, dedup, segment, sources  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
TABLES = ROOT / "data" / "tables"

TARGET_SENTENCES = 400_000          # per source
NEAR_DUP_SAMPLE = 100_000           # per source; LSH index is memory-bound
TIER2_SAMPLE = 25_000               # spaCy throughput measurement
_TOKEN = clean.WORD  # canonical tokenizer; see the em-dash note in clean.py


def log(msg: str) -> None:
    print(f"[pass0] {msg}", flush=True)


# --- document iteration -----------------------------------------------------


def iter_documents(source: str):
    if source == "gutenberg":
        return sources.read_gutenberg(RAW / "gutenberg")
    if source == "stackexchange":
        return sources.read_stackexchange(RAW / "stackexchange")
    if source == "wikipedia":
        return sources.read_wikipedia(local=RAW / "wikipedia")
    if source == "c4news":
        return sources.read_c4news(local=RAW / "c4news")
    raise ValueError(source)


def language_ok(text: str) -> bool:
    """English check on every document.

    Matters most for Gutenberg, which carries a real share of non-English
    books; the other three sources are already English-filtered upstream.
    """
    try:
        import py3langid

        lang, _ = py3langid.classify(text[:2000])
        return lang == "en"
    except Exception:
        return True


# --- stage 1: segment -------------------------------------------------------


def collect_sentences(source: str, target: int) -> dict:
    """Normalize, segment, filter, exact-dedup. Returns sentences + timings."""
    backend = segment.segmenter_name()
    log(f"{source}: segmenting with {backend} (target {target:,} sentences)")

    sentences: list[str] = []
    years: list[int | None] = []
    doc_ids: list[str] = []
    seen_exact: set[str] = set()

    n_docs = n_dropped_lang = n_dropped_quality = 0
    n_raw_sentences = n_filtered = n_exact_dupes = 0
    n_words = 0
    t0 = time.perf_counter()

    for doc in iter_documents(source):
        n_docs += 1
        if not clean.is_acceptable_document(doc.text):
            n_dropped_quality += 1
            continue
        if not language_ok(doc.text):
            n_dropped_lang += 1
            continue

        text = clean.normalize(doc.text)
        n_words += len(_TOKEN.findall(text))

        for sentence in segment.segment_document(text, apply_filters=False):
            n_raw_sentences += 1
            if not clean.is_acceptable_sentence(sentence):
                n_filtered += 1
                continue
            key = dedup.exact_key(sentence)
            if key in seen_exact:
                n_exact_dupes += 1
                continue
            seen_exact.add(key)
            sentences.append(sentence)
            years.append(doc.pub_year)
            doc_ids.append(doc.doc_id)

        if len(sentences) >= target:
            break
        # Frequent enough for Gutenberg's 601 long books to show progress.
        if n_docs % 250 == 0:
            log(f"{source}: {n_docs:,} docs -> {len(sentences):,} sentences")

    elapsed = time.perf_counter() - t0
    log(f"{source}: {len(sentences):,} sentences from {n_docs:,} docs in {elapsed:.0f}s")

    return {
        "sentences": sentences,
        "years": years,
        "doc_ids": doc_ids,
        "n_docs": n_docs,
        "n_words": n_words,
        "n_raw_sentences": n_raw_sentences,
        "n_filtered": n_filtered,
        "n_exact_dupes": n_exact_dupes,
        "n_dropped_quality": n_dropped_quality,
        "n_dropped_lang": n_dropped_lang,
        "segment_seconds": elapsed,
        "segment_backend": backend,
    }


# --- stage 2: near-duplicate rate -------------------------------------------


def measure_near_dupes(sentences: list[str], sample: int = NEAR_DUP_SAMPLE) -> dict:
    """Near-dup rate on a sample; the full LSH index does not fit in RAM."""
    from datasketch import MinHash, MinHashLSH

    subset = sentences[:sample]
    lsh = MinHashLSH(threshold=config.JACCARD_THRESHOLD, num_perm=64)
    n_near = 0
    t0 = time.perf_counter()
    for i, sentence in enumerate(subset):
        words = re.findall(r"[a-z0-9']+", sentence.lower())
        shingles = {
            " ".join(words[j : j + config.SHINGLE_SIZE])
            for j in range(max(1, len(words) - config.SHINGLE_SIZE + 1))
        }
        m = MinHash(num_perm=64)
        for s in shingles:
            m.update(s.encode())
        if lsh.query(m):
            n_near += 1
        else:
            lsh.insert(str(i), m)
    return {
        "near_dup_sample": len(subset),
        "near_dupes": n_near,
        "near_dup_rate": n_near / max(1, len(subset)),
        "near_dup_seconds": time.perf_counter() - t0,
    }


# --- stage 3: tier-1 counting ----------------------------------------------


def tier1_count(sentences: list[str]) -> tuple[Counter, dict]:
    t0 = time.perf_counter()
    counts: Counter = Counter()
    n_tokens = 0
    for sentence in sentences:
        tokens = _TOKEN.findall(sentence.lower())
        n_tokens += len(tokens)
        counts.update(tokens)
    elapsed = time.perf_counter() - t0
    return counts, {
        "tier1_tokens": n_tokens,
        "tier1_seconds": elapsed,
        "tier1_words_per_sec": n_tokens / max(elapsed, 1e-9),
        "vocab_size": len(counts),
    }


# --- stage 4: retention calibration ----------------------------------------


def sentence_rate_profiles(
    sentences: list[str], rates: dict[str, float], max_ceiling: float
) -> list[list[float]]:
    """Per sentence, the sorted distinct token rates at or below max_ceiling.

    Common words dominate every sentence and are irrelevant to band membership,
    so dropping them keeps these lists to a handful of entries and makes the
    band sweep cheap.
    """
    profiles = []
    for sentence in sentences:
        low = {r for t in _TOKEN.findall(sentence.lower())
               if (r := rates.get(t, 0.0)) <= max_ceiling}
        profiles.append(sorted(low))
    return profiles


def _band_hit(profile: list[float], floor: float, ceiling: float) -> bool:
    i = bisect.bisect_left(profile, floor)
    return i < len(profile) and profile[i] <= ceiling


def calibrate_band(profiles: list[list[float]], resolution_floor: float) -> dict:
    """Sweep rarity bands for the one landing retention in the 4-8% target."""
    results = []
    for floor in config.RARITY_BAND_FLOORS:
        for ceiling in config.RARITY_BAND_CEILINGS:
            if floor >= ceiling:
                continue
            hits = sum(1 for p in profiles if _band_hit(p, floor, ceiling))
            hit_rate = hits / max(1, len(profiles))
            combined = hit_rate + (1 - hit_rate) * config.BASELINE_SAMPLE_RATE
            results.append({
                "floor_per_million": floor,
                "ceiling_per_million": ceiling,
                # Below the resolution floor the band's lower edge is not
                # meaningful: hapax words sit exactly at it, so the band cannot
                # exclude the noise tail it is designed to exclude.
                "floor_below_resolution": floor < resolution_floor,
                "band_hit_rate": round(hit_rate, 5),
                "combined_retention": round(combined, 5),
                "in_target_band": config.TARGET_RETENTION_RANGE[0]
                <= combined
                <= config.TARGET_RETENTION_RANGE[1],
            })
    return {"band_curve": results}


def retain_by_coverage(
    sentences: list[str],
    band_vocab: set[str],
    cap: int,
    rng: random.Random,
) -> tuple[list[int], Counter]:
    """Keep sentences until every band word has `cap` contexts.

    Replaces rate-based retention. Sentences are visited in shuffled order so
    the contexts kept for a word are spread across the corpus rather than
    taken from whichever documents happened to be read first.

    Returns the retained indices and the per-word context counts -- the latter
    is the number that actually matters, since the design notes gates shipping a
    word on having at least MIN_CONTEXTS_TO_SHIP real occurrences.
    """
    order = list(range(len(sentences)))
    rng.shuffle(order)

    coverage: Counter = Counter()
    retained: list[int] = []
    for i in order:
        hits = {t for t in _TOKEN.findall(sentences[i].lower()) if t in band_vocab}
        if any(coverage[w] < cap for w in hits):
            retained.append(i)
            coverage.update(hits)
        elif rng.random() < config.BASELINE_SAMPLE_RATE:
            retained.append(i)
    retained.sort()
    return retained, coverage


# --- stage 5: tier-2 annotation --------------------------------------------


def tier2_annotate(sentences: list[str], sample: int = TIER2_SAMPLE) -> dict:
    import spacy

    nlp = spacy.load("en_core_web_sm")
    subset = sentences[:sample]
    n_tokens = 0
    t0 = time.perf_counter()
    for doc in nlp.pipe(subset, batch_size=256):
        n_tokens += len(doc)
    elapsed = time.perf_counter() - t0
    return {
        "tier2_sentences": len(subset),
        "tier2_tokens": n_tokens,
        "tier2_seconds": elapsed,
        "tier2_words_per_sec": n_tokens / max(elapsed, 1e-9),
    }


# --- output -----------------------------------------------------------------


def write_parquet(source: str, data: dict, retained: list[int]) -> dict:
    INTERIM.mkdir(parents=True, exist_ok=True)
    idx = retained
    table = pa.table({
        "sid": pa.array([f"{source}:{i}" for i in idx]),
        "source": pa.array([source] * len(idx)),
        "doc_id": pa.array([data["doc_ids"][i] for i in idx]),
        "pub_year": pa.array([data["years"][i] for i in idx], type=pa.int32()),
        "text": pa.array([data["sentences"][i] for i in idx]),
        "n_tok": pa.array([clean.token_count(data["sentences"][i]) for i in idx],
                          type=pa.int32()),
    })
    out = INTERIM / f"{source}.parquet"
    pq.write_table(table, out, compression="zstd")
    size = out.stat().st_size
    return {
        "parquet_bytes": size,
        "parquet_rows": len(idx),
        "bytes_per_sentence": size / max(1, len(idx)),
    }


def dump_samples(source: str, sentences: list[str], n: int = 100) -> None:
    """Sentences for hand-checking the segmentation error rate."""
    TABLES.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1)
    picks = rng.sample(sentences, min(n, len(sentences)))
    path = TABLES / f"pass0_sample_{source}.txt"
    path.write_text("\n".join(picks) + "\n")


def main() -> None:
    """Two phases, mirroring Pass 1.

    Phase A counts every source into one pooled frequency table. Phase B
    derives the rarity band from that pooled table and applies retention. The
    split is not incidental: rarity cannot be judged from a single source, and
    the band cannot be chosen until all counting is done.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all",
                    choices=["all", "gutenberg", "wikipedia", "c4news", "stackexchange"])
    ap.add_argument("--target", type=int, default=TARGET_SENTENCES)
    args = ap.parse_args()

    todo = (["gutenberg", "wikipedia", "c4news", "stackexchange"]
            if args.source == "all" else [args.source])
    TABLES.mkdir(parents=True, exist_ok=True)

    # --- Phase A: collect and count -----------------------------------------
    collected: dict[str, dict] = {}
    per_source_counts: dict[str, Counter] = {}
    pooled: Counter = Counter()

    for source in todo:
        data = collect_sentences(source, args.target)
        if not data["sentences"]:
            log(f"{source}: no sentences produced, skipping")
            continue
        log(f"{source}: tier-1 counting ...")
        counts, t1 = tier1_count(data["sentences"])
        data.update(t1)
        collected[source] = data
        per_source_counts[source] = counts
        pooled.update(counts)

    if not collected:
        log("nothing collected")
        return

    # --- Phase B: pooled band, then per-source retention --------------------
    pooled_total = sum(pooled.values())
    resolution_floor = 1 / pooled_total * 1e6
    rates = {w: c / pooled_total * 1e6 for w, c in pooled.items()}
    max_ceiling = max(config.RARITY_BAND_CEILINGS)
    log(f"pooled: {pooled_total:,} tokens, {len(pooled):,} forms, "
        f"resolution floor {resolution_floor:.4f}/million")

    # Diagnostic only -- retention volume is set by the per-word cap below.
    all_profiles = {
        s: sentence_rate_profiles(d["sentences"], rates, max_ceiling)
        for s, d in collected.items()
    }
    calib = calibrate_band(
        [p for profs in all_profiles.values() for p in profs], resolution_floor
    )

    floor, ceiling = config.RARITY_BAND
    band_vocab = {w for w, r in rates.items() if floor <= r <= ceiling}
    log(f"rarity band [{floor}, {ceiling}]/million -> {len(band_vocab):,} wordforms")

    (TABLES / "pass0_pooled_frequency.json").write_text(json.dumps({
        "total_tokens": pooled_total,
        "distinct_forms": len(pooled),
        "resolution_floor_per_million": resolution_floor,
        "band": [floor, ceiling],
        "band_vocab_size": len(band_vocab),
        "contexts_per_word_cap": config.CONTEXTS_PER_WORD,
        "band_curve": calib["band_curve"],
        "top": pooled.most_common(20_000),
    }))

    rng = random.Random(0)
    for source, data in collected.items():
        sentences = data["sentences"]
        stats: dict = {k: v for k, v in data.items()
                       if k not in ("sentences", "years", "doc_ids")}
        stats["source"] = source
        stats["n_sentences"] = len(sentences)
        stats["pooled_resolution_floor_per_million"] = resolution_floor
        stats["chosen_band_per_million"] = [floor, ceiling]
        stats.update(calib)

        log(f"{source}: measuring near-duplicates ...")
        stats.update(measure_near_dupes(sentences))
        log(f"{source}: tier-2 annotation sample ...")
        stats.update(tier2_annotate(sentences))

        retained, coverage = retain_by_coverage(
            sentences, band_vocab, config.CONTEXTS_PER_WORD, rng
        )
        stats["retained"] = len(retained)
        stats["actual_retention"] = len(retained) / len(sentences)

        # The number Pass 0 should actually be judged on.
        covered = sum(1 for c in coverage.values()
                      if c >= config.MIN_CONTEXTS_TO_SHIP)
        stats["band_vocab_size"] = len(band_vocab)
        stats["band_words_seen"] = len(coverage)
        stats["band_words_shippable"] = covered
        stats["shippable_fraction_of_seen"] = round(covered / max(1, len(coverage)), 4)
        stats["mean_contexts_per_seen_word"] = round(
            sum(coverage.values()) / max(1, len(coverage)), 2)
        stats.update(write_parquet(source, data, retained))
        dump_samples(source, sentences)

        counts = per_source_counts[source]
        (TABLES / f"pass0_freq_{source}.json").write_text(json.dumps({
            "total_tokens": sum(counts.values()),
            "top": counts.most_common(20_000),
        }))

        out = TABLES / f"pass0_report_{source}.json"
        out.write_text(json.dumps(stats, indent=2))
        log(f"{source}: retention {stats['actual_retention']:.2%} -> {out}")


if __name__ == "__main__":
    main()
