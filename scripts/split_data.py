#!/usr/bin/env python3
"""Word-disjoint train/dev/test split.

Split by HARD WORD, not randomly. The architectural claim is that passing the
candidate as *input* lets vocabulary grow without retraining -- that claim is
only testable when the eval words were never seen in training. A random split
would score higher and measure nothing.

    python scripts/split_data.py
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"


def log(msg: str) -> None:
    print(f"[split] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="labeled_full.jsonl")
    ap.add_argument("--test-words", type=int, default=100)
    ap.add_argument("--dev-words", type=int, default=60)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (TABLES / args.labeled).open()]
    by_word: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_word[r["hard_word"]].append(r)

    # Shuffle words, but take eval words from the WELL-ATTESTED half. A test set
    # built from words with one item each would be measuring the long tail of
    # the inventory rather than the model.
    rng = random.Random(args.seed)
    words = sorted(by_word, key=lambda w: (-len(by_word[w]), w))
    median = len(by_word[words[len(words) // 2]])
    eligible = [w for w in words if len(by_word[w]) >= median]
    rng.shuffle(eligible)

    test_w = set(eligible[: args.test_words])
    dev_w = set(eligible[args.test_words: args.test_words + args.dev_words])
    train_w = set(by_word) - test_w - dev_w

    splits = {"train": train_w, "dev": dev_w, "test": test_w}
    for name, wordset in splits.items():
        items = [r for w in sorted(wordset) for r in by_word[w]]
        rng.shuffle(items)
        out = TABLES / f"split_{name}.jsonl"
        with out.open("w") as fh:
            for it in items:
                fh.write(json.dumps(it) + "\n")
        labels = sum(len(it["grades"]) for it in items)
        log(f"{name:<5} {len(wordset):>3} words  {len(items):>5} items  "
            f"{labels:>6} labels -> {out.name}")

    overlap = (test_w & train_w) | (dev_w & train_w) | (test_w & dev_w)
    assert not overlap, f"word leakage: {overlap}"
    log("verified: no hard word appears in more than one split")


if __name__ == "__main__":
    main()
