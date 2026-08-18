#!/usr/bin/env python3
"""Does context-gating solve the sense problem?

Mechanism under test: before proposing candidate C for a span in sentence S,
compare S against the contexts where C actually occurs in the corpus. Sense IS
context distribution, so a candidate whose attested contexts look nothing like
S is being used in the wrong sense.

Two tests, deliberately ordered:

  A  Signal check, no human labels. Hold out real contexts of each word and ask
     whether the gate ranks them above contexts of a different word. If it
     cannot do this, nothing downstream matters.

  B  The actual job. Score the judging set's surviving proposals against its
     CONSTRUCTED wrong-sense negatives -- which are wrong by construction, so
     the labels owe nothing to my judgement.

    python scripts/sense_gate_experiment.py
"""

from __future__ import annotations

import csv
import glob
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import clean  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
TABLES = ROOT / "data" / "tables"
INVENTORY = ROOT / "data" / "inventory"

CONTEXTS_PER_WORD = 60
STOP = set("""a an the of to in and or but for nor so yet at by from with as is
are was were be been being have has had do does did will would shall should can
could may might must this that these those it its he she they them his her their
our your my me you we i not no if then than there here when where which who whom
whose on off up down out over under into onto about after before during while
said says say more most very much many some any all one two new also just like
than into what how why who them then""".split())


def log(msg: str) -> None:
    print(f"[sense] {msg}", flush=True)


def content(sentence: str, drop: set[str]) -> set[str]:
    return {w for w in clean.words(sentence)
            if w not in STOP and w not in drop and len(w) > 2}


def build_idf() -> dict[str, float]:
    """IDF from the corpus frequency table -- rare shared words mean more."""
    t = pq.read_table(TABLES / "pass1_pooled_frequency.parquet")
    forms, counts = t["form"].to_pylist(), t["count"].to_pylist()
    total = sum(counts)
    return {f: math.log(total / c) for f, c in zip(forms, counts) if c > 0}


def similarity(a: set[str], b: set[str], idf: dict[str, float]) -> float:
    """IDF-weighted overlap (cosine on binary vectors)."""
    if not a or not b:
        return 0.0
    shared = a & b
    if not shared:
        return 0.0
    num = sum(idf.get(w, 12.0) for w in shared)
    na = math.sqrt(sum(idf.get(w, 12.0) ** 2 for w in a))
    nb = math.sqrt(sum(idf.get(w, 12.0) ** 2 for w in b))
    return num / (na * nb) if na and nb else 0.0


def gate_score(sentence: str, word: str, banked: list[set[str]],
               idf: dict[str, float]) -> float:
    """Max similarity between the proposed sentence and any attested context."""
    q = content(sentence, drop={word})
    return max((similarity(q, c, idf) for c in banked), default=0.0)


def collect_contexts(words: set[str]) -> dict[str, list[str]]:
    files = sorted(glob.glob(str(INTERIM / "pass1_*.parquet")))
    dataset = ds.dataset(files, format="parquet")
    out: dict[str, list[str]] = defaultdict(list)
    for batch in dataset.to_batches(columns=["text"], batch_size=200_000):
        for text in batch["text"].to_pylist():
            toks = set(clean.words(text))
            for w in toks & words:
                if len(out[w]) < CONTEXTS_PER_WORD:
                    out[w].append(text)
        if all(len(out[w]) >= CONTEXTS_PER_WORD for w in words):
            break
    return out


def main() -> None:
    inv = list(csv.DictReader(
        (l for l in (INVENTORY / "inversions_v1.tsv").open()
         if not l.startswith("#")), delimiter="\t"))
    words = {r["replacement"].lower() for r in inv}
    log(f"collecting up to {CONTEXTS_PER_WORD} contexts for {len(words)} words ...")
    ctx = collect_contexts(words)
    usable = {w: c for w, c in ctx.items() if len(c) >= 20}
    log(f"{len(usable)} words have >=20 contexts")

    idf = build_idf()
    banked = {w: [content(s, drop={w}) for s in c] for w, c in usable.items()}
    rng = random.Random(3)

    # --- Test A: can the gate tell a word's own contexts from another's? ----
    hits = trials = 0
    for w, sents in usable.items():
        held, rest = sents[:8], [content(s, drop={w}) for s in sents[8:]]
        if len(rest) < 10:
            continue
        others = [x for x in usable if x != w]
        for s in held:
            other = rng.choice(others)
            true_score = gate_score(s, w, rest, idf)
            foil = rng.choice(usable[other])
            foil_score = gate_score(foil, w, rest, idf)
            trials += 1
            hits += true_score > foil_score
    log("")
    log("TEST A -- does the signal exist at all? (no human labels)")
    log(f"  a word's own held-out context outranks a foreign context: "
        f"{hits}/{trials} = {hits/max(1,trials)*100:.1f}%   (chance = 50%)")

    # --- Test B: constructed wrong-sense negatives vs surviving proposals ---
    items = [json.loads(l) for l in (TABLES / "judging_set.jsonl").open()]
    pos_scores, neg_scores = [], []
    for it in items:
        w = it["replacement"].lower()
        if w not in banked:
            continue
        s = gate_score(it["swapped"], w, banked[w], idf)
        (neg_scores if it.get("neg_type") == "wrong sense" else
         pos_scores if it["kind"] == "proposal" else []).append(s)

    def mean(x):
        return sum(x) / len(x) if x else 0.0

    log("")
    log("TEST B -- constructed wrong-sense negatives vs surviving proposals")
    log(f"  proposals            n={len(pos_scores):>3}  mean gate score {mean(pos_scores):.4f}")
    log(f"  wrong-sense negatives n={len(neg_scores):>3}  mean gate score {mean(neg_scores):.4f}")

    best = (0.0, 0.0, 0.0, 0.0)
    for i in range(1, 60):
        thr = i * 0.005
        keep = sum(1 for s in pos_scores if s >= thr) / max(1, len(pos_scores))
        block = sum(1 for s in neg_scores if s < thr) / max(1, len(neg_scores))
        if keep + block > best[1] + best[2]:
            best = (thr, keep, block, keep + block)
    thr, keep, block, _ = best
    log(f"  best threshold {thr:.3f}: keeps {keep*100:.0f}% of proposals, "
        f"blocks {block*100:.0f}% of wrong-sense negatives")

    json.dump({"test_a_accuracy": hits / max(1, trials),
               "proposal_mean": mean(pos_scores),
               "wrong_sense_mean": mean(neg_scores),
               "threshold": thr, "keep_rate": keep, "block_rate": block},
              (TABLES / "sense_gate_result.json").open("w"), indent=2)


if __name__ == "__main__":
    main()
