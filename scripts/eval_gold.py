#!/usr/bin/env python3
"""Score trained judges against the gold test slice.

Every metric reported so far is measured against teacher labels drawn from the
same distribution the models trained on -- so it measures IMITATION of Opus,
not whether the substitutions are good. This scores the same models against the
three-vote consensus slice instead, which is the closest thing to an
independent standard we have.

It also reports the ceiling that consensus implies: on candidates where the
three votes disagreed, no model can be scored at all, and that disagreement
rate is itself the most honest statement of how well-defined the task is.

    python scripts/eval_gold.py
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_judge import (Groups, Judge, MODELS, TABLES, collate,  # noqa: E402
                         evaluate)


def log(m: str) -> None:
    print(f"[gold-eval] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="mini,distil")
    ap.add_argument("--gold", default="gold_test.jsonl")
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    path = TABLES / args.gold
    rows = [json.loads(l) for l in path.open()]
    n_unstable = sum(len(r.get("unstable", {})) for r in rows)
    n_gold = sum(len(r["gold"]) for r in rows)
    log(f"{len(rows)} items | {n_gold} unanimous candidates | "
        f"{n_unstable} contested (excluded)")
    log(f"three-vote agreement rate: "
        f"{n_gold/max(1,n_gold+n_unstable)*100:.1f}%")

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for tag in args.tags.split(","):
        ckpt = torch.load(MODELS / f"judge_{tag}.pt", map_location="cpu",
                          weights_only=False)
        tok = AutoTokenizer.from_pretrained(MODELS / f"tok_{tag}")
        model = Judge(ckpt["model_name"], len(tok))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        ds = Groups(path, grade_key="gold_values",
                    identity_key="gold_identity_target")
        dl = DataLoader(ds, batch_size=args.bs, shuffle=False,
                        collate_fn=lambda b: collate(b, tok))
        log(f"--- {tag} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M) "
            f"on {len(ds)} gold groups ---")
        out[tag] = evaluate(model, dl, device, tag=f"{tag} vs gold")
        log("  mean score by origin: " + ", ".join(
            f"{k} {v:.3f}" for k, v in sorted(out[tag]["mean_by_origin"].items())))

    # How far does teacher-label accuracy overstate gold accuracy? That gap is
    # the price of having trained and evaluated on the same annotator.
    for tag in out:
        tl = json.load((MODELS / f"result_{tag}.json").open())["test"]
        log(f"{tag}: teacher-label acc {tl['acc']*100:.1f}%  "
            f"-> gold acc {out[tag]['acc']*100:.1f}%  "
            f"(delta {(out[tag]['acc']-tl['acc'])*100:+.1f})")
    json.dump(out, (MODELS / "gold_eval.json").open("w"), indent=2)


if __name__ == "__main__":
    main()
