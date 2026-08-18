#!/usr/bin/env python3
"""Choose each finalist's operating threshold on DEV, then report GOLD numbers.

6.7: "abstention is free ... be confidently right on the clear cases and
decline on the rest." That makes the false-positive rate the constraint and
recall the thing being maximised under it -- not accuracy.

Thresholds come from dev. Gold is touched once, for reporting.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_judge import Groups, Judge, MODELS, TABLES, collate  # noqa: E402

TARGET_FP = 0.02


def scores(model, path, tok, device, grade_key="grade_values",
           ident_key="identity_target"):
    ds = Groups(TABLES / path, grade_key=grade_key, identity_key=ident_key)
    dl = DataLoader(ds, batch_size=16, collate_fn=lambda b: collate(b, tok))
    P, N = [], []
    with torch.no_grad():
        for b in dl:
            b.pop("n_groups"); org = b.pop("origin")
            b = {k: v.to(device) for k, v in b.items()}
            p = torch.sigmoid(model(**b)).tolist(); t = b["target"].tolist()
            for o, pi, ti in zip(org, p, t):
                if o in ("original", "identity"):
                    continue
                (P if ti >= 0.8 else N if ti <= 0.2 else []).append(pi)
    return np.array(P), np.array(N)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--target-fp", type=float, default=TARGET_FP)
    args = ap.parse_args()
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    out = {}
    for tag in args.tags.split(","):
        p = MODELS / f"judge_{tag}.pt"
        if not p.exists():
            print(f"[thr] missing {tag}"); continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        tok = AutoTokenizer.from_pretrained(MODELS / f"tok_{tag}")
        m = Judge(ck["model_name"], len(tok))
        m.load_state_dict(ck["state_dict"]); m.to(device).eval()

        dP, dN = scores(m, "split_dev.jsonl", tok, device)
        thr = float(np.quantile(dN, 1 - args.target_fp))
        gP, gN = scores(m, "gold_test.jsonl", tok, device,
                        "gold_values", "gold_identity_target")
        rec = float((gP >= thr).mean()) if len(gP) else 0.0
        fp = float((gN >= thr).mean()) if len(gN) else 0.0
        out[tag] = {"threshold": round(thr, 4), "gold_recall": round(rec, 4),
                    "gold_fp": round(fp, 4), "n_pos": int(len(gP)),
                    "n_neg": int(len(gN))}
        print(f"[thr] {tag:<12} thr {thr:.3f}  GOLD recall {rec*100:5.1f}%  "
              f"FP {fp*100:4.1f}%  (dev-calibrated for {args.target_fp*100:.0f}%)")
    json.dump(out, (MODELS / "thresholds.json").open("w"), indent=2)


if __name__ == "__main__":
    main()
