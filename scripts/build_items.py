#!/usr/bin/env python3
"""Assemble labelling items.

Rewritten after the first label pilot, which showed the teacher was being paid
to adjudicate cases that were never in doubt: 33% clearly_good / 65%
clearly_bad and only 3% across the middle of the scale. Three causes, three
fixes:

  1. Items came from every source, so 88% were Gutenberg. Now restricted to the
     deployment register, matching build_judging_set.py.

  2. Negatives were random other synonyms -- trivially separable, exactly what
     5.3 warns teaches nothing. Now the hard negatives are the NEAREST
     NEIGHBOURS of the correct candidate in embedding space: same register,
     same rough meaning, wrong in the specific context.

  3. Identity almost never won, so the judge would learn never to abstain
. The sense constraints veto 29.4% of
     trigger matches, and those vetoes are near-misses BY CONSTRUCTION -- the
     trigger and POS matched and only the sense was wrong. Instead of
     discarding them they are kept as ABSTENTION items, where the right answer
     is that no substitution belongs.

    python scripts/build_items.py --n 20000
"""
from __future__ import annotations
import argparse, csv, glob, json, random, re, sys
from pathlib import Path
import pyarrow.dataset as ds, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import clean, senses  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INTERIM, TABLES, INVENTORY = ROOT/"data"/"interim", ROOT/"data"/"tables", ROOT/"data"/"inventory"
_NLP = None

def log(m): print(f"[items] {m}", flush=True)

def pos_of(sentence, start, end):
    global _NLP
    if _NLP is None:
        import spacy; _NLP = spacy.load("en_core_web_sm")
    doc = _NLP(sentence)
    toks = [t for t in doc if t.idx < end and t.idx+len(t.text) > start]
    return max(toks, key=lambda t: len(t.text)).pos_ if toks else None

def nearest_neighbours(words):
    """Embedding neighbours of each candidate -- the source of hard negatives."""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    m = SentenceTransformer("all-MiniLM-L6-v2")
    V = m.encode(sorted(words), normalize_embeddings=True, show_progress_bar=False)
    order = sorted(words)
    sim = V @ V.T
    np.fill_diagonal(sim, -1)
    return {order[i]: [order[j] for j in sim[i].argsort()[::-1][:6]] for i in range(len(order))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--per-word", type=int, default=25)
    ap.add_argument("--sources", default="c4news,stackexchange")
    ap.add_argument("--abstain-fraction", type=float, default=0.35)
    ap.add_argument("--out", default="items.jsonl")
    args = ap.parse_args()

    rules = senses.load_inversions()
    glosses = {r["word"]: r["gloss"] for r in
               csv.DictReader((INVENTORY/"gregmat_900.tsv").open(), delimiter="\t")}
    by_trigger = {}
    for r in rules: by_trigger.setdefault(r["trigger"].lower(), []).append(r)
    pattern = re.compile(r"\b("+"|".join(sorted((re.escape(t) for t in by_trigger),
                         key=len, reverse=True))+r")\b", re.I)

    ft = pq.read_table(TABLES/"fixed_phrases.parquet")
    fixed = set(zip(ft["w1"].to_pylist(), ft["w2"].to_pylist()))
    surface = {r["replacement"] for r in rules}
    nn = nearest_neighbours(surface)
    log(f"{len(rules)} rules, {len(surface)} candidate surfaces, neighbours computed")

    rng = random.Random(17)
    wanted = {s for s in args.sources.split(",") if s}
    n_abstain_target = int(args.n * args.abstain_fraction)

    dset = ds.dataset(sorted(glob.glob(str(INTERIM/"pass1_*.parquet"))), format="parquet")
    subs, abstains, per_word = [], [], {}
    stats = dict(matched=0, veto_pos=0, veto_phrase=0, veto_sense=0)

    for batch in dset.to_batches(columns=["source","text"], batch_size=200_000):
        for source, text in zip(batch["source"].to_pylist(), batch["text"].to_pylist()):
            if len(subs) >= args.n - n_abstain_target and len(abstains) >= n_abstain_target:
                break
            if source not in wanted or not clean.is_acceptable_sentence(text): continue
            m = pattern.search(text)
            if not m: continue
            stats["matched"] += 1
            rule = by_trigger[m.group(0).lower()][0]
            hw, repl = rule["hard_word"], rule["replacement"]

            words = clean.words(text); head = rule["trigger"].split()[0]
            if head in words:
                i = words.index(head)
                if (i+1 < len(words) and (words[i],words[i+1]) in fixed) or \
                   (i > 0 and (words[i-1],words[i]) in fixed):
                    stats["veto_phrase"] += 1; continue
            actual = pos_of(text, m.start(), m.end())
            if actual and rule["pos"] and actual != rule["pos"]:
                stats["veto_pos"] += 1; continue

            ok, why = senses.check_constraints(text, rule)
            kind = "substitution" if ok else "abstention"
            if not ok:
                stats["veto_sense"] += 1
                if len(abstains) >= n_abstain_target: continue
            elif len(subs) >= args.n - n_abstain_target:
                continue
            if per_word.get((hw, kind), 0) >= args.per_word: continue

            # Hard negatives: nearest neighbours of the candidate in meaning
            # space -- plausible, same register, wrong here.
            hard = [w for w in nn.get(repl, []) if w != repl][:2]
            easy = rng.choice([r["replacement"] for r in rules
                               if r["pos"] != rule["pos"]] or [repl])
            cands = [{"text": repl, "origin": "inventory" if ok else "vetoed_sense"},
                     {"text": m.group(0).lower(), "origin": "original"}]
            cands += [{"text": w, "origin": "hard_neg"} for w in hard]
            if easy not in {c["text"] for c in cands}:
                cands.append({"text": easy, "origin": "easy_neg"})
            rng.shuffle(cands)

            per_word[(hw, kind)] = per_word.get((hw, kind), 0) + 1
            rec = {"item_id": f"it{len(subs)+len(abstains):06d}", "kind": kind,
                   "source": source, "sentence": text, "span": m.group(0),
                   "blanked": text[:m.start()] + "____" + text[m.end():],
                   "hard_word": hw, "gloss": glosses.get(hw, ""),
                   "sense": rule["trigger_sense"], "pos": rule["pos"],
                   "veto_reason": None if ok else why, "candidates": cands}
            (subs if ok else abstains).append(rec)
        if len(subs) >= args.n - n_abstain_target and len(abstains) >= n_abstain_target:
            break

    items = subs + abstains
    rng.shuffle(items)
    out = TABLES/args.out
    with out.open("w") as fh:
        for it in items: fh.write(json.dumps(it)+"\n")
    log(f"matched {stats['matched']:,} | vetoed pos {stats['veto_pos']:,} "
        f"phrase {stats['veto_phrase']:,} sense {stats['veto_sense']:,}")
    log(f"items {len(items):,}: {len(subs):,} substitution + {len(abstains):,} abstention -> {out}")

if __name__ == "__main__":
    main()
