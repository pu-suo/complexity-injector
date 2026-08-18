#!/usr/bin/env python3
"""Simulate the actual runtime decision, per span.

Every metric so far has been per-CANDIDATE, pooled over a training item's
artificial 5-candidate mix. The product does something different and simpler:

  for each span:
      score the proposer's real candidates + the identity option
      if best candidate > threshold AND beats identity -> substitute
      else -> leave it alone

So the numbers that matter are:
  coverage  -- share of spans where we substitute at all
  precision -- of the substitutions we make, share the teacher calls good
  errors    -- visible wrong words per 100 spans, which is what users notice

Candidates are restricted to what the proposer would really offer: the
inventory row(s) for that trigger. Hard/easy negatives are training scaffolding
and never appear at runtime.

    python scripts/simulate_runtime.py --tags base,mini_kd2,medium
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_judge import KEEP, Judge, MODELS, TABLES, marked  # noqa: E402

MS = {"base": 16.640, "medium": 6.287, "deep12_kd2": 3.818,
      "mini_kd2": 1.314, "mini": 1.381, "distil": 8.365}


def load(path: Path, grade_key: str):
    """One row per span, carrying only production-realistic candidates."""
    out = []
    for line in path.open():
        it = json.loads(line)
        grades = it.get(grade_key) or {}
        cands = [c["text"] for c in it["candidates"]
                 if c["origin"] == "inventory" and c["text"] in grades]
        if not cands:
            continue
        out.append({"text": marked(it), "cands": cands,
                    "targets": [grades[c] for c in cands]})
    return out


@torch.no_grad()
def score_all(model, tok, rows, device, bs=64):
    left, right, owner = [], [], []
    for i, r in enumerate(rows):
        for c in r["cands"]:
            left.append(r["text"]); right.append(c); owner.append(i)
        left.append(r["text"]); right.append(KEEP); owner.append(i)
    probs = []
    for s in range(0, len(left), bs):
        enc = tok(left[s:s+bs], right[s:s+bs], padding=True, truncation=True,
                  max_length=128, return_tensors="pt").to(device)
        probs += torch.sigmoid(model(**enc)).tolist()
    per: list[dict] = [{"c": [], "ident": None} for _ in rows]
    k = 0
    for i, r in enumerate(rows):
        for _ in r["cands"]:
            per[i]["c"].append(probs[k]); k += 1
        per[i]["ident"] = probs[k]; k += 1
    return per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="base,medium,deep12_kd2,mini_kd2")
    ap.add_argument("--split", default="gold_test.jsonl")
    ap.add_argument("--grade-key", default="gold_values")
    args = ap.parse_args()
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    rows = load(TABLES / args.split, args.grade_key)
    print(f"[sim] {len(rows)} spans with a real proposer candidate "
          f"({args.split})")
    results = {}
    for tag in args.tags.split(","):
        p = MODELS / f"judge_{tag}.pt"
        if not p.exists():
            print(f"[sim] missing {tag}"); continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        tok = AutoTokenizer.from_pretrained(MODELS / f"tok_{tag}")
        m = Judge(ck["model_name"], len(tok))
        m.load_state_dict(ck["state_dict"]); m.to(device).eval()
        per = score_all(m, tok, rows, device)

        print(f"\n[sim] === {tag} ===")
        print(f"{'thr':>6}{'coverage':>10}{'precision':>11}"
              f"{'errors/100 spans':>18}")
        best = None
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            made = good = 0
            for r, s in zip(rows, per):
                j = int(np.argmax(s["c"]))
                if s["c"][j] >= thr and s["c"][j] > s["ident"]:
                    made += 1
                    good += r["targets"][j] >= 0.8
            cov = made / len(rows)
            prec = good / made if made else 0.0
            err = (made - good) / len(rows) * 100
            print(f"{thr:>6.1f}{cov*100:>9.1f}%{prec*100:>10.1f}%{err:>17.1f}")
            # Pick the operating point 6.7 implies: maximise coverage subject
            # to a low visible-error rate, because a wrong word on the page
            # costs far more than a missed opportunity.
            if prec >= 0.90 and (best is None or cov > best["coverage"]):
                best = {"threshold": thr, "coverage": round(cov, 4),
                        "precision": round(prec, 4), "errors_per_100": round(err, 2)}
        results[tag] = {"best_at_prec90": best, "ms_per_score": MS.get(tag),
                        "scores_per_span": 2.30}
        if best:
            ms = MS.get(tag)
            print(f"  best @ precision>=90%: thr {best['threshold']}  "
                  f"coverage {best['coverage']*100:.1f}%  "
                  f"{best['errors_per_100']:.1f} errors/100 spans")
            if ms:
                for spans in (20, 100):
                    print(f"    {spans:>3} spans -> {spans*2.30*ms:7.0f} ms "
                          f"(2.30 scores/span, 1-thread CPU)")
        else:
            print("  never reaches 90% precision at any threshold")
    json.dump(results, (MODELS / "runtime_sim.json").open("w"), indent=2)


if __name__ == "__main__":
    main()
