#!/usr/bin/env python3
"""What does the extension ACTUALLY deliver, and is it worth delivering?

Coverage was measured on held-out training items. This measures the product on
raw text: how often a swap lands, which words land, and whether each landed swap
is genuinely harder than what it replaced.

"Harder" = rarer in the Pass 1 corpus. A swap whose replacement is no rarer than
the trigger teaches nothing, however grammatical it is.

    python scripts/diagnose_delivery.py --source stackexchange --n 3000
"""
from __future__ import annotations
import argparse, collections, glob, math, re, sys
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline import clean, senses           # noqa: E402
from train_judge import KEEP, Judge, MODELS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="stackexchange")
    ap.add_argument("--files", default="", help="parsed text files instead of the corpus")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="drop table rows whose replacement is not this much rarer")
    args = ap.parse_args()

    t = pq.read_table(ROOT / "data/tables/pass1_pooled_frequency.parquet")
    freq = dict(zip(t["form"].to_pylist(), t["count"].to_pylist()))
    total = sum(freq.values())

    def pm(w: str) -> float:
        ws = clean.words(w.lower())
        return (min(freq.get(x, 0) for x in ws) / total * 1e6) if ws else 0.0

    inv = senses.load_inversions()
    by_trigger: dict[str, list] = {}
    for r in inv:
        if r["trigger"].lower() == r["replacement"].lower():
            continue
        if args.min_gain > 0:
            a, b = pm(r["trigger"]), pm(r["replacement"])
            if b <= 0 or a / b < args.min_gain:
                continue
        by_trigger.setdefault(r["trigger"].lower(), []).append(r)
    pat = re.compile(r"\b(" + "|".join(sorted((re.escape(x) for x in by_trigger),
                     key=len, reverse=True)) + r")\b", re.IGNORECASE)

    dev = ("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(MODELS / "judge_base.pt", map_location="cpu", weights_only=False)
    tok = AutoTokenizer.from_pretrained(MODELS / "tok_base")
    model = Judge(ck["model_name"], len(tok))
    model.load_state_dict(ck["state_dict"]); model.to(dev).eval()

    sents, words = [], 0
    if args.files:
        sys.path.insert(0, str(ROOT / "scripts"))
        from parse_reddit import comments as parse_comments
        from pipeline.segment import split_sentences
        for f in sorted(glob.glob(args.files)):
            for c in parse_comments(Path(f).read_text(errors="replace")):
                for snt in split_sentences(c):
                    sents.append(snt); words += len(clean.words(snt))
    files = sorted(glob.glob(str(ROOT / "data/interim/pass1_*.parquet")))
    for batch in ([] if args.files else ds.dataset(files, format="parquet").to_batches(
            columns=["source", "text"], batch_size=50_000)):
        for src, txt in zip(batch["source"].to_pylist(), batch["text"].to_pylist()):
            if src != args.source or len(sents) >= args.n:
                continue
            sents.append(txt); words += len(clean.words(txt))
        if len(sents) >= args.n:
            break

    # The extension skips fragments under five words -- most Reddit replies.
    short = sum(1 for s in sents if len(clean.words(s)) < 5)
    kept = [s for s in sents if len(clean.words(s)) >= 5]
    print(f"[sim] {len(sents)} sentences, {short} below the 5-word floor "
          f"({short/max(1,len(sents))*100:.0f}%), {len(kept)} eligible")
    spans = []
    for s in kept:
        for m in pat.finditer(s):
            trig = m.group(0).lower()
            keep = [r for r in by_trigger[trig] if senses.check_constraints(s, r)[0]]
            if keep:
                spans.append((s, m, keep))

    delivered = collections.Counter()
    gains, accepted, rejected = [], 0, 0
    B = 64
    pairs, meta = [], []

    def flush():
        nonlocal accepted, rejected, pairs, meta
        if not pairs:
            return
        enc = tok([l for l, _ in pairs], [r for _, r in pairs], padding=True,
                  truncation=True, max_length=128, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.sigmoid(model(**enc)).tolist()
        i = 0
        for cands, trig in meta:
            sc = p[i:i + len(cands) + 1]; i += len(cands) + 1
            best = max(range(len(cands)), key=lambda k: sc[k])
            if sc[best] >= args.threshold and sc[best] > sc[-1]:
                accepted += 1
                rep = cands[best]
                delivered[f"{trig} → {rep}"] += 1
                a, b = pm(trig), pm(rep)
                if a > 0 and b > 0:
                    gains.append(math.log2(a / b))
            else:
                rejected += 1
        pairs, meta = [], []

    for s, m, keep in spans:
        marked = s[:m.start()] + "<t> " + m.group(0) + " </t>" + s[m.end():]
        cands = [r["replacement"] for r in keep]
        for c in cands:
            pairs.append((marked, c))
        pairs.append((marked, KEEP))
        meta.append((cands, m.group(0).lower()))
        if len(meta) >= B:
            flush()
    flush()

    n = accepted + rejected
    print(f"\nsource={args.source}  {len(sents)} sentences, {words:,} words")
    print(f"  proposer spans      {n:>6}   ({n/words*1000:.2f} per 1k words)")
    print(f"  judge accepted      {accepted:>6}   ({accepted/max(1,n)*100:.1f}%)")
    print(f"  DELIVERED RATE      {accepted/words*1000:>6.2f} swaps per 1k words"
          f"   (~{accepted/words*500:.1f} per 500-word comment)")
    if gains:
        gains.sort()
        back = sum(1 for g in gains if g < 1)
        print(f"\n  difficulty of delivered swaps:")
        print(f"    median {2**gains[len(gains)//2]:.1f}x rarer")
        print(f"    no real gain (<2x): {back}/{len(gains)} = {back/len(gains)*100:.0f}%")
    print(f"\n  top delivered swaps ({len(delivered)} distinct):")
    for k, v in delivered.most_common(15):
        trig, rep = k.split(" → ")
        g = math.log2(pm(trig)/pm(rep)) if pm(trig) > 0 and pm(rep) > 0 else 0
        flag = "  <-- no gain" if g < 1 else ""
        print(f"    {v:>4}x  {k:<34} {2**g:6.1f}x rarer{flag}")


if __name__ == "__main__":
    main()
