#!/usr/bin/env python3
"""Distil the mid-size judge into BERT-Mini.

Why this is no longer optional: measured on SUBSTITUTION recall -- the only
metric the product cares about -- DistilBERT accepts 41.0% of teacher-endorsed
proposals against BERT-Mini's 15.6%, at a lower false-positive rate. But at 66M
parameters it is ~6x the latency, which puts a 100-span page near 4 seconds.
Distillation is the only route to that accuracy at shippable latency.

Two transfers, matching the two losses in 6.4:
  * soft per-candidate targets  -> calibration
  * the teacher's softmax over each group -> RANKING, which is where the
    measured gap is widest (top-1 50.2% vs 74.2%)

    python scripts/distill.py --teacher distil --student google/bert_uncased_L-4_H-256_A-4
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_judge import (Groups, Judge, MODELS, TABLES, collate,  # noqa: E402
                         evaluate, group_ranking_loss)


def log(m: str) -> None:
    print(f"[distil] {m}", flush=True)


@torch.no_grad()
def teacher_scores(tag: str, split: str, device: str) -> dict:
    """Cache the teacher's probability for every (item, candidate) pair."""
    cache = MODELS / f"teacher_{tag}_{split.replace('.jsonl','')}.json"
    if cache.exists():
        return json.load(cache.open())
    ck = torch.load(MODELS / f"judge_{tag}.pt", map_location="cpu",
                    weights_only=False)
    tok = AutoTokenizer.from_pretrained(MODELS / f"tok_{tag}")
    m = Judge(ck["model_name"], len(tok))
    m.load_state_dict(ck["state_dict"]); m.to(device).eval()
    ds = Groups(TABLES / split)
    out: dict[str, dict[str, float]] = {}
    dl = DataLoader(ds, batch_size=16, shuffle=False,
                    collate_fn=lambda b: (b, collate(b, tok)))
    for rows, b in dl:
        b.pop("n_groups"); origins = b.pop("origin")
        b = {k: v.to(device) for k, v in b.items()}
        p = torch.sigmoid(m(**b)).tolist()
        i = 0
        for row in rows:
            d = out.setdefault(row["item_id"], {})
            for c in row["cands"]:
                d[c] = p[i]; i += 1
            d["__identity__"] = p[i]; i += 1
    json.dump(out, cache.open("w"))
    log(f"cached teacher scores for {len(out)} items -> {cache.name}")
    return out


def soft_targets(rows, scores, device):
    """Teacher probability per pair, in the same flattened order collate uses."""
    vals = []
    for row in rows:
        d = scores.get(row["item_id"], {})
        for c in row["cands"]:
            vals.append(d.get(c, 0.0))
        vals.append(d.get("__identity__", 0.0))
    return torch.tensor(vals, dtype=torch.float, device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="distil")
    ap.add_argument("--student", default="google/bert_uncased_L-4_H-256_A-4")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="weight on the teacher's soft targets vs hard labels")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--tag", default="mini_kd")
    # The first distillation run raised AUC 0.670 -> 0.735 but LOST recall at
    # 3% false positives (23.5% -> 20.2%). Cause: the KD objective weights all
    # pairs equally, while the product only operates at the top of the ranking.
    # `--kd-focus` upweights pairs the teacher scored highly, so the student
    # spends capacity where the decisions actually get made.
    ap.add_argument("--kd-focus", type=float, default=2.0)
    # Sharpening the teacher's group distribution (T<1) transfers a more
    # decisive ranking, which is the measured weak spot at small sizes.
    ap.add_argument("--kd-temp", type=float, default=0.7)
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    log(f"teacher={args.teacher} student={args.student} alpha={args.alpha}")
    tr_scores = teacher_scores(args.teacher, "split_train.jsonl", device)

    tok = AutoTokenizer.from_pretrained(MODELS / f"tok_{args.teacher}")
    model = Judge(args.student, len(tok)).to(device)
    log(f"student {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    def loader(name, shuffle):
        ds = Groups(TABLES / name)
        return DataLoader(ds, batch_size=args.bs, shuffle=shuffle,
                          collate_fn=lambda b: (b, collate(b, tok))), ds

    tr, tr_ds = loader("split_train.jsonl", True)
    dv, _ = loader("split_dev.jsonl", False)
    te, _ = loader("split_test.jsonl", False)
    plain = lambda dl: ((b for _, b in dl))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=len(tr) * args.epochs, pct_start=0.1)

    t0 = time.time(); best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        for rows, b in tr:
            n_groups = b.pop("n_groups"); b.pop("origin")
            soft = soft_targets(rows, tr_scores, device)
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(**b)
            real = b["is_identity"] < 0.5
            hard_t, rank_m = b["target"], b["in_rank"]

            bce_hard = F.binary_cross_entropy_with_logits(logits[real], hard_t[real])
            w = 1.0 + args.kd_focus * soft[real]
            bce_soft = F.binary_cross_entropy_with_logits(
                logits[real], soft[real], weight=w)
            rank_hard = group_ranking_loss(logits, hard_t, b["group"], n_groups, rank_m)
            sharp = soft.clamp(min=1e-6) ** (1.0 / args.kd_temp)
            rank_soft = group_ranking_loss(logits, sharp, b["group"], n_groups, rank_m)

            a = args.alpha
            loss = (a * (bce_soft + rank_soft) + (1 - a) * (bce_hard + rank_hard))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()

        d = evaluate(model, plain(dv), device, tag=f"epoch {ep} dev")
        score = d["sub_recall"] - d["sub_fp"]      # select on the product metric
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone()
                                       for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test = evaluate(model, plain(te), device, tag="test")
    log("mean by origin: " + ", ".join(
        f"{k} {v:.3f}" for k, v in sorted(test["mean_by_origin"].items())))
    torch.save({"state_dict": model.state_dict(), "model_name": args.student},
               MODELS / f"judge_{args.tag}.pt")
    tok.save_pretrained(MODELS / f"tok_{args.tag}")
    json.dump({"student": args.student, "teacher": args.teacher,
               "alpha": args.alpha, "test": test,
               "minutes": (time.time() - t0) / 60},
              (MODELS / f"result_{args.tag}.json").open("w"), indent=2)
    log(f"saved -> judge_{args.tag}.pt  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
