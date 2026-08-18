#!/usr/bin/env python3
"""Build the hand-judging set.

Step 0 is the gate: "If we cannot make it read acceptably by hand, with full
understanding and unlimited time, no model will." It was reframed from
PRODUCING swaps to JUDGING them, because judging needs only receptive
vocabulary.

The guard that makes it honest: proposals are mixed blind with swaps generated
by the 5.3 negative recipes. Without that, the set inherits only the proposer's
distribution and the judging drifts into rubber-stamping.

    python scripts/build_judging_set.py --n 200
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import clean  # noqa: E402

_NLP = None


def pos_of_span(sentence: str, start: int, end: int) -> str | None:
    """UPOS of the span head. The v1 table matched on surface form alone,
    which put verbs into noun slots ("a 50 percent augment")."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    doc = _NLP(sentence)
    toks = [t for t in doc if t.idx < end and t.idx + len(t.text) > start]
    if not toks:
        return None
    return max(toks, key=lambda t: len(t.text)).pos_

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
TABLES = ROOT / "data" / "tables"
INVENTORY = ROOT / "data" / "inventory"


def log(msg: str) -> None:
    print(f"[judge] {msg}", flush=True)


def load_inversions() -> list[dict]:
    rows = []
    with (INVENTORY / "inversions_v1.tsv").open() as fh:
        for r in csv.DictReader(
            (l for l in fh if not l.startswith("#")), delimiter="\t"
        ):
            if r.get("trigger"):
                rows.append(r)
    return rows


def load_fixed_phrases() -> set[tuple[str, str]]:
    """All of them. Loading only the top 60k by LLR let "hate speech",
    "user friendly", "cautious optimism" and "reckless abandon" through --
    they are in the table, just further down the ranking."""
    t = pq.read_table(TABLES / "fixed_phrases.parquet")
    return set(zip(t["w1"].to_pylist(), t["w2"].to_pylist()))


def load_names(limit: int = 400_000) -> set[str]:
    path = TABLES / "pass1_names.txt"
    if not path.exists():
        return set()
    with path.open() as fh:
        return {line.strip() for _, line in zip(range(limit), fh)}


def apply_swap(sentence: str, trigger: str, replacement: str) -> str | None:
    """Replace the first whole-word occurrence, preserving capitalization."""
    m = re.search(rf"\b{re.escape(trigger)}\b", sentence, re.IGNORECASE)
    if not m:
        return None
    found = m.group(0)
    repl = replacement.capitalize() if found[0].isupper() else replacement
    out = sentence[: m.start()] + repl + sentence[m.end():]
    return fix_articles(out)


# Orthographic vowels are a decent proxy, but these are the common exceptions.
_AN_BEFORE_CONSONANT = ("hour", "honest", "honor", "heir")
_A_BEFORE_VOWEL = ("one", "once", "uni", "use", "user", "usual", "euro", "ubiq")


def fix_articles(text: str) -> str:
    """Repair a/an after a substitution."""

    def repl(m):
        art, nxt = m.group(1), m.group(2)
        low = nxt.lower()
        if low.startswith(_AN_BEFORE_CONSONANT):
            want = "an"
        elif low.startswith(_A_BEFORE_VOWEL):
            want = "a"
        else:
            want = "an" if low[0] in "aeiou" else "a"
        if art[0].isupper():
            want = want.capitalize()
        return f"{want} {nxt}"

    return re.sub(r"\b([Aa]n?) ([A-Za-z]+)", repl, text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--neg-fraction", type=float, default=0.4)
    # the design is "substitute by hand in 20 real threads" -- the judging
    # set must look like what the extension meets, not like 19th-century prose.
    # Unweighted, Gutenberg supplied 6 of 10 proposals (the King James Bible
    # alone dominated), which would have told us nothing about the product.
    ap.add_argument("--sources", default="stackexchange,c4news")
    args = ap.parse_args()

    inversions = load_inversions()
    fixed = load_fixed_phrases()
    names = load_names()
    log(f"{len(inversions)} inversion rules, {len(fixed):,} fixed phrases, "
        f"{len(names):,} names")

    by_trigger: dict[str, list[dict]] = {}
    for r in inversions:
        by_trigger.setdefault(r["trigger"].lower(), []).append(r)
    pattern = re.compile(
        r"\b(" + "|".join(sorted((re.escape(t) for t in by_trigger),
                                 key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )

    files = sorted(glob.glob(str(INTERIM / "pass1_*.parquet")))
    dataset = ds.dataset(files, format="parquet")
    rng = random.Random(11)

    n_neg = int(args.n * args.neg_fraction)
    n_pos = args.n - n_neg
    positives: list[dict] = []
    negatives: list[dict] = []
    seen_words: dict[str, int] = {}

    wanted = {x for x in args.sources.split(',') if x}
    for batch in dataset.to_batches(columns=["source", "text"], batch_size=100_000):
        srcs, txts = batch["source"].to_pylist(), batch["text"].to_pylist()
        for source, text in zip(srcs, txts):
            if source not in wanted:
                continue
            if not clean.is_acceptable_sentence(text):
                continue  # re-apply filters; the store predates the verse-ref rule
            if len(positives) >= n_pos * 3 and len(negatives) >= n_neg * 3:
                break
            m = pattern.search(text)
            if not m:
                continue
            trig = m.group(0).lower()
            rule = rng.choice(by_trigger[trig])
            words = clean.words(text)

            # --- negative recipes --------------------------
            neg_type = None
            idx = [i for i, w in enumerate(words) if w == trig.split()[0]]
            if idx:
                i = idx[0]
                if i + 1 < len(words) and (words[i], words[i + 1]) in fixed:
                    neg_type = "collocation break"
                elif i > 0 and (words[i - 1], words[i]) in fixed:
                    neg_type = "collocation break"
            if neg_type is None and m.group(0)[0].isupper() and m.start() > 0:
                if trig in names:
                    neg_type = "proper noun"

            if neg_type:
                swapped = apply_swap(text, m.group(0), rule["replacement"])
                if swapped:
                    negatives.append({
                        "kind": "negative", "neg_type": neg_type, "source": source,
                        "original": text, "swapped": swapped,
                        "span": m.group(0), "replacement": rule["replacement"],
                        "hard_word": rule["hard_word"],
                    })
                continue

            # POS gate.
            actual = pos_of_span(text, m.start(), m.end())
            if actual and rule["pos"] and actual != rule["pos"]:
                continue
            # Cap per word so a few common triggers do not dominate the set.
            if seen_words.get(rule["hard_word"], 0) >= 4:
                continue
            swapped = apply_swap(text, m.group(0), rule["replacement"])
            if not swapped:
                continue
            seen_words[rule["hard_word"]] = seen_words.get(rule["hard_word"], 0) + 1
            positives.append({
                "kind": "proposal", "neg_type": None, "source": source,
                "original": text, "swapped": swapped,
                "span": m.group(0), "replacement": rule["replacement"],
                "hard_word": rule["hard_word"], "sense_note": rule["sense_note"],
            })

    # Wrong-sense negatives: a real synonym applied to the wrong trigger.
    extra = []
    for p in positives[: n_neg]:
        other = rng.choice([r for r in inversions
                            if r["hard_word"] != p["replacement"]
                            and r["pos"] != "VERB"])
        swapped = apply_swap(p["original"], p["span"], other["replacement"])
        if swapped and swapped != p["swapped"]:
            extra.append({
                "kind": "negative", "neg_type": "wrong sense", "source": p["source"],
                "original": p["original"], "swapped": swapped,
                "span": p["span"], "replacement": other["replacement"],
                "hard_word": other["hard_word"],
            })
    negatives.extend(extra)

    # Stratify negatives by failure mode. Pooling them lets whichever recipe
    # fires most often (collocation break) crowd out the rest, and the whole
    # point of the blind mix is to span the 3 error table.
    by_type: dict[str, list[dict]] = {}
    for n in negatives:
        by_type.setdefault(n["neg_type"], []).append(n)
    for bucket in by_type.values():
        rng.shuffle(bucket)
    chosen: list[dict] = []
    while len(chosen) < n_neg and any(by_type.values()):
        for t in sorted(by_type):
            if by_type[t] and len(chosen) < n_neg:
                chosen.append(by_type[t].pop())
    log("negative mix: " + ", ".join(
        f"{t} {sum(1 for c in chosen if c['neg_type'] == t)}" for t in sorted(by_type)))

    rng.shuffle(positives)
    items = positives[:n_pos] + chosen
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["id"] = i

    out = TABLES / "judging_set.jsonl"
    with out.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")

    glosses = {}
    with (INVENTORY / "gregmat_900.tsv").open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            glosses[r["word"]] = r["gloss"]

    # The judge must not be asked to recall 900 definitions. Show the original,
    # the swap, and the gloss: judging then needs comprehension, not recall.
    sheet = TABLES / "judging_sheet.txt"
    with sheet.open("w") as fh:
        for it in items:
            g = glosses.get(it.get("hard_word", ""), "?")
            fh.write(f"[{it['id']:>3}] {it['replacement'].upper()} = {g}\n")
            fh.write(f"      BEFORE: {it['original']}\n")
            fh.write(f"      AFTER:  {it['swapped']}\n\n")

    kinds = {}
    for it in items:
        k = it["neg_type"] or "proposal"
        kinds[k] = kinds.get(k, 0) + 1
    log(f"{len(items)} items: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))
    log(f"blind sheet -> {sheet}")
    log(f"answer key  -> {out}")


if __name__ == "__main__":
    main()
