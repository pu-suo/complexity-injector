#!/usr/bin/env python3
"""Assemble the Pass 0 acceptance checklist.

Reads the per-source reports and the tagger-agreement report, applies the
thresholds, and prints a pass/fail table plus the Pass 1 projection.

    python scripts/pass0_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"

SOURCES = ["gutenberg", "wikipedia", "c4news", "stackexchange"]

# Pass 1 corpus sizes from the data notes, in words.
PASS1_WORDS = {
    "gutenberg": 3.0e9,
    "wikipedia": 4.0e9,
    "c4news": 1.5e9,
    "stackexchange": 0.2e9,
}


def load(name: str) -> dict | None:
    path = TABLES / name
    return json.loads(path.read_text()) if path.exists() else None


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    reports = {s: load(f"pass0_report_{s}.json") for s in SOURCES}
    reports = {k: v for k, v in reports.items() if v}
    tagger = load("pass0_tagger_agreement.json")

    if not reports:
        print("No Pass 0 reports found. Run scripts/run_pass0.py first.")
        return

    checks: list[tuple[str, str, str, bool | None]] = []

    print("=" * 78)
    print("PASS 0 — PER-SOURCE RESULTS")
    print("=" * 78)
    header = (f"{'source':<15}{'docs':>9}{'sents':>10}{'words':>12}"
              f"{'filt%':>8}{'exact%':>8}{'near%':>8}{'retain%':>9}")
    print(header)
    print("-" * 78)

    total_sentences = 0
    for src, r in reports.items():
        filt = r["n_filtered"] / max(1, r["n_raw_sentences"])
        exact = r["n_exact_dupes"] / max(1, r["n_raw_sentences"])
        near = r["near_dup_rate"]
        retain = r["actual_retention"]
        total_sentences += r["n_sentences"]
        print(f"{src:<15}{r['n_docs']:>9,}{r['n_sentences']:>10,}{r['n_words']:>12,}"
              f"{filt * 100:>7.1f}%{exact * 100:>7.2f}%{near * 100:>7.2f}%"
              f"{retain * 100:>8.2f}%")

    # - checklist -------------------------------------------------------
    print()
    print("=" * 78)
    print("PASS 0 — 7 ACCEPTANCE CHECKLIST")
    print("=" * 78)

    # 1. Segmentation error rate: needs hand-checking; samples were dumped.
    sample_files = sorted(TABLES.glob("pass0_sample_*.txt"))
    checks.append((
        "Segmentation error rate < 2%",
        f"{len(sample_files)} sample files x 100 sentences",
        "HAND-CHECK REQUIRED",
        None,
    ))

    # 2. Gutenberg unwrapping.
    g = reports.get("gutenberg")
    if g:
        checks.append((
            "Gutenberg unwrapping (no false breaks)",
            f"{g['n_sentences']:,} sentences, mean {g['n_words'] / max(1, g['n_sentences']):.1f} words",
            "SPOT-CHECK",
            None,
        ))

    # 3. Dedup ordering: Gutenberg > news > Wikipedia was the prediction.
    order = {s: reports[s]["near_dup_rate"] for s in reports}
    if {"gutenberg", "c4news", "wikipedia"} <= set(order):
        ok = order["gutenberg"] > order["c4news"] > order["wikipedia"]
        detail = " > ".join(f"{s} {fmt_pct(order[s])}"
                            for s in sorted(order, key=order.get, reverse=True))
        checks.append(("Near-dup rate ordering plausible", detail,
                       "PASS" if ok else "REVIEW", ok))

    # 4. Tagger agreement -- the check that decides whether error 1 is
    #    deterministic.
    if tagger:
        pooled = tagger.get("pooled_match_eligible_agreement", 0)
        ok = pooled >= 0.95
        checks.append((
            "spaCy<->compromise agreement (match-eligible) >= 95%",
            f"pooled {fmt_pct(pooled)}",
            "PASS" if ok else "FAIL — escalate error 1 out of 'deterministic'",
            ok,
        ))

    # 5-6. Throughput.
    t1 = min(r["tier1_words_per_sec"] for r in reports.values())
    t2 = min(r["tier2_words_per_sec"] for r in reports.values())
    checks.append(("Tier-1 throughput >= 500k words/sec/core",
                   f"worst source {t1:,.0f}", "PASS" if t1 >= 5e5 else "FAIL",
                   t1 >= 5e5))
    checks.append(("Tier-2 throughput >= 5k words/sec/core",
                   f"worst source {t2:,.0f}", "PASS" if t2 >= 5e3 else "FAIL",
                   t2 >= 5e3))

    # 7. Coverage, which replaced the retention-rate target. The per-word cap
    #    does not bind at probe scale (mean contexts sit well under it), so
    #    retention here is uninformative -- what transfers is contexts-per-word.
    pooled = load("pass0_pooled_frequency.json") or {}
    cap = pooled.get("contexts_per_word_cap", 50)
    mean_ctx = {s: r.get("mean_contexts_per_seen_word", 0) for s, r in reports.items()}
    binds = any(v >= cap for v in mean_ctx.values())
    checks.append((
        f"Per-word cap ({cap}) binds at this scale?",
        ", ".join(f"{s} mean {v}" for s, v in mean_ctx.items()),
        "NO — expected; probe is ~280x too small" if not binds else "YES",
        None,
    ))

    # 8. Projected store, under the per-word cap rather than a retention rate.
    #    Sentences needed = band_vocab x cap / (band words per sentence).
    retained_rows = sum(r["parquet_rows"] for r in reports.values())
    retained_bytes = sum(r["parquet_bytes"] for r in reports.values())
    bytes_per_sentence = retained_bytes / max(1, retained_rows)
    contexts = sum(r.get("mean_contexts_per_seen_word", 0) * r.get("band_words_seen", 0)
                   for r in reports.values())
    band_words_per_sentence = contexts / max(1, retained_rows)

    probe_tokens = pooled.get("total_tokens", 1)
    probe_vocab = pooled.get("distinct_forms", 1)
    scale = sum(PASS1_WORDS.values()) / probe_tokens
    # Heaps' law, beta ~= 0.5, for vocabulary growth from probe to Pass 1.
    vocab_growth = scale ** 0.5
    band_share = pooled.get("band_vocab_size", 0) / max(1, probe_vocab)
    # The band's share of vocabulary shrinks as the hapax tail grows; bracket it.
    band_vocab_lo = probe_vocab * vocab_growth * band_share * 0.15
    band_vocab_hi = probe_vocab * vocab_growth * band_share * 0.35

    def project(band_vocab: float) -> tuple[float, float]:
        sentences = band_vocab * cap / max(0.5, band_words_per_sentence)
        gb = sentences * bytes_per_sentence / 1e9
        return sentences, gb

    s_lo, gb_lo = project(band_vocab_lo)
    s_hi, gb_hi = project(band_vocab_hi)
    checks.append((
        "Projected retained store <= 8 GB",
        f"{gb_lo:.1f}-{gb_hi:.1f} GB "
        f"({s_lo/1e6:.0f}-{s_hi/1e6:.0f}M sentences, {bytes_per_sentence:.0f} B/sentence)",
        "PASS" if gb_hi <= 8 else "REVIEW",
        gb_hi <= 8,
    ))
    projection_sentences = (s_lo, s_hi)

    for name, detail, verdict, _ in checks:
        print(f"\n  {name}")
        print(f"      {detail}")
        print(f"      -> {verdict}")

    # --- Pass 1 projection --------------------------------------------------
    print()
    print("=" * 78)
    print("PASS 1 PROJECTION")
    print("=" * 78)
    total_words = sum(PASS1_WORDS.values())
    words_per_sentence = sum(r["n_words"] for r in reports.values()) / max(
        1, sum(r["n_sentences"] for r in reports.values()))
    t1_hours = total_words / t1 / 3600
    t2_lo = projection_sentences[0] * words_per_sentence / t2 / 3600
    t2_hi = projection_sentences[1] * words_per_sentence / t2 / 3600
    print(f"  corpus                 {total_words / 1e9:.1f}B words")
    print(f"  tier-1 (all words)     {t1_hours:.1f} core-hours")
    print(f"  tier-2 (capped)        {t2_lo:.0f}-{t2_hi:.0f} core-hours")
    print(f"  total                  {t1_hours + t2_lo:.0f}-{t1_hours + t2_hi:.0f} core-hours")
    print(f"  on 16 vCPU             ~{(t1_hours + t2_lo) / 16:.1f}-{(t1_hours + t2_hi) / 16:.1f} hours")
    print(f"  retained store         {gb_lo:.1f}-{gb_hi:.1f} GB")
    print()
    print("  Tier-2 no longer scales with corpus size: the per-word cap bounds")
    print("  it by vocabulary, so a bigger corpus costs tier-1 time only.")

    out = TABLES / "pass0_checklist.json"
    out.write_text(json.dumps({
        "per_source": reports,
        "tagger": tagger,
        "checks": [{"name": n, "detail": d, "verdict": v} for n, d, v, _ in checks],
        "projection": {
            "tier1_core_hours": t1_hours,
            "tier2_core_hours": [t2_lo, t2_hi],
            "total_core_hours": [t1_hours + t2_lo, t1_hours + t2_hi],
            "retained_store_gb": [gb_lo, gb_hi],
            "projected_sentences": list(projection_sentences),
            "band": pooled.get("band"),
            "band_vocab_probe": pooled.get("band_vocab_size"),
            "contexts_per_word_cap": cap,
        },
    }, indent=2))
    print(f"\n  written -> {out}")


if __name__ == "__main__":
    main()
