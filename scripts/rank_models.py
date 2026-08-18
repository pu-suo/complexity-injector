#!/usr/bin/env python3
"""Rank every trained judge on the DEV split and pick the two finalists.

Dev, not test: the test split has to stay clean for the final number. Ranking
on it and then reporting it is the same mistake as tuning on the test set.

Large winner  = best dev AUC among models too slow for CPU-only.
Small pool    = the CPU-affordable shapes worth distilling into.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_judge import Groups, Judge, MODELS, TABLES, collate  # noqa: E402

# Measured ms/candidate, int8, single-thread CPU (export_and_benchmark.py).
MS = {"tiny": 0.31, "mini": 1.38, "mini_kd": 1.38, "pw_2.0": 1.38,
      "pw_4.0": 1.38, "deep8": 2.63, "small": 3.24, "deep12": 3.92,
      "medium": 6.29, "distil": 8.37}
CPU_BUDGET_MS = 4.0     # ~2s for a 100-span page at 5 candidates


def main() -> None:
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for ck_path in sorted(MODELS.glob("judge_*.pt")):
        tag = ck_path.stem.replace("judge_", "")
        try:
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            tok = AutoTokenizer.from_pretrained(MODELS / f"tok_{tag}")
            m = Judge(ck["model_name"], len(tok))
            m.load_state_dict(ck["state_dict"]); m.to(dev).eval()
        except Exception as e:                       # noqa: BLE001
            print(f"[rank] skip {tag}: {e}")
            continue
        dl = DataLoader(Groups(TABLES / "split_dev.jsonl"), batch_size=16,
                        collate_fn=lambda b: collate(b, tok))
        P, N = [], []
        with torch.no_grad():
            for b in dl:
                b.pop("n_groups"); org = b.pop("origin")
                b = {k: v.to(dev) for k, v in b.items()}
                p = torch.sigmoid(m(**b)).tolist(); t = b["target"].tolist()
                for o, pi, ti in zip(org, p, t):
                    if o in ("original", "identity"):
                        continue
                    (P if ti >= 0.8 else N if ti <= 0.2 else []).append(pi)
        P, N = np.array(P), np.array(N)
        rk = np.concatenate([P, N]).argsort().argsort() + 1
        auc = float((rk[:len(P)].sum() - len(P)*(len(P)+1)/2) / (len(P)*len(N)))
        r2 = float((P >= np.quantile(N, 0.98)).mean())
        n = sum(x.numel() for x in m.parameters()) / 1e6
        rows.append({"tag": tag, "params_m": round(n, 1), "auc": round(auc, 4),
                     "r_at_fp2": round(r2, 4), "ms": MS.get(tag)})
        print(f"[rank] {tag:<12} {n:>6.1f}M  AUC {auc:.3f}  R@FP2% {r2*100:5.1f}%")

    rows.sort(key=lambda r: -r["auc"])
    best_large = rows[0]["tag"] if rows else None
    cheap = [r["tag"] for r in rows
             if r["ms"] is not None and r["ms"] <= CPU_BUDGET_MS]
    small_candidates = [t for t in ("mini", "deep12") if t in cheap] or ["mini"]
    out = {"ranking": rows, "best_large": best_large,
           "small_candidates": small_candidates}
    json.dump(out, (MODELS / "ranking.json").open("w"), indent=2)
    print(f"[rank] best large = {best_large}; small pool = {small_candidates}")


if __name__ == "__main__":
    main()
