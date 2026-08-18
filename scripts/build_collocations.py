#!/usr/bin/env python3
"""Turn raw bigram counts into a fixed-phrase table.

Pass 1 saved bigrams ranked by raw count, which is useless as an error-3
blocklist: the top entries are "of the", "in the", "to the". Frequency is not
association. A fixed phrase is one whose parts co-occur far more than their
individual frequencies predict -- that is what PMI and the log-likelihood
ratio measure, and it is what makes "good morning" and "hot dog" separate
cleanly from "of the".

    python scripts/build_collocations.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"

# A pair must clear both to be called a fixed phrase.
MIN_PMI = 3.0
MIN_LLR = 1000.0

# Function words are frequent enough to pass association tests on sheer volume
# ("of the" has a real statistical signal) but a span never keys on them.
STOPWORDS = set("""
a an the of to in and or but for nor so yet at by from with as is are was were
be been being have has had do does did will would shall should can could may
might must this that these those it its he she they them his her their our your
my me you we i not no if then than there here when where which who whom whose
on off up down out over under into onto about after before during while
""".split())


def log(msg: str) -> None:
    print(f"[colloc] {msg}", flush=True)


def main() -> None:
    bigrams = pq.read_table(TABLES / "pass1_collocations.parquet")
    unigrams = pq.read_table(TABLES / "pass1_pooled_frequency.parquet")
    meta = json.loads((TABLES / "pass1_calibration.json").read_text())
    report = json.loads((TABLES / "pass1_report.json").read_text())

    uni = dict(zip(unigrams["form"].to_pylist(), unigrams["count"].to_pylist()))
    uni_total = meta["total_tokens"]
    bi_total = report["bigram_tokens"]

    w1 = bigrams["w1"].to_pylist()
    w2 = bigrams["w2"].to_pylist()
    cnt = bigrams["count"].to_pylist()
    log(f"{len(cnt):,} candidate pairs, {bi_total:,} bigram tokens")

    rows = []
    for a, b, c in zip(w1, w2, cnt):
        if a in STOPWORDS or b in STOPWORDS:
            continue
        ca, cb = uni.get(a, 0), uni.get(b, 0)
        if ca == 0 or cb == 0 or c < config.COLLOCATION_MIN_COUNT:
            continue

        # Unigram counts come from the 0.99B calibration sample and bigrams
        # from the 2.63B stream, so work in probabilities rather than counts.
        p_ab = c / bi_total
        p_a, p_b = ca / uni_total, cb / uni_total
        pmi = math.log2(p_ab / (p_a * p_b))

        # Log-likelihood ratio (Dunning), the standard significance test for
        # collocation: robust where PMI alone over-rewards rare pairs.
        e = p_a * p_b * bi_total
        llr = 2 * (c * math.log(c / e) - (c - e)) if e > 0 and c > 0 else 0.0

        if pmi >= MIN_PMI and llr >= MIN_LLR:
            rows.append((a, b, c, pmi, llr))

    rows.sort(key=lambda r: -r[4])
    log(f"{len(rows):,} pairs clear PMI>={MIN_PMI} and LLR>={MIN_LLR}")

    pq.write_table(pa.table({
        "w1": pa.array([r[0] for r in rows]),
        "w2": pa.array([r[1] for r in rows]),
        "count": pa.array([r[2] for r in rows], type=pa.int64()),
        "pmi": pa.array([r[3] for r in rows], type=pa.float32()),
        "llr": pa.array([r[4] for r in rows], type=pa.float32()),
    }), TABLES / "fixed_phrases.parquet", compression="zstd")

    log("top 20 by association:")
    for a, b, c, pmi, llr in rows[:20]:
        log(f"    {a + ' ' + b:<26}{c:>9,}  pmi {pmi:5.1f}  llr {llr:12,.0f}")


if __name__ == "__main__":
    main()
