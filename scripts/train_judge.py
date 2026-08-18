#!/usr/bin/env python3
"""Fine-tune the judge as a cross-encoder.

Input is one sequence, two segments:

    [CLS] it was a <t> very big </t> problem . [SEP] enormous [SEP]

Output is the [CLS] vector through a linear head to one logit.

Two losses on the same scalar:
  * per-candidate BCE for CALIBRATION, so 0.9 means 90%
  * softmax across all candidates of one span for RANKING

The identity candidate ("leave it alone", segment B = [KEEP]) takes part in the
ranking loss ONLY, never BCE -- its target is derived, not observed, so training
it as a calibrated probability would be asserting a number nobody measured.

Run both sizes to get the 6.5 diagnostic:
    python scripts/train_judge.py --model google/bert_uncased_L-4_H-256_A-4
    python scripts/train_judge.py --model distilbert-base-uncased
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "tables"
MODELS = ROOT / "data" / "models"
KEEP = "[KEEP]"
SPAN_OPEN, SPAN_CLOSE = "<t>", "</t>"


def log(m: str) -> None:
    print(f"[train] {m}", flush=True)


def marked(item: dict) -> str:
    """Sentence with the span wrapped in markers. Built from `blanked` rather
    than by searching `sentence` for `span`: the blank is already exactly where
    the substitution goes, so there is no ambiguity when the span text also
    occurs elsewhere in the sentence."""
    return item["blanked"].replace(
        "____", f"{SPAN_OPEN} {item['span']} {SPAN_CLOSE}")


class Groups(Dataset):
    """One example = one span with ALL its candidates. Ranking is defined over
    the group, so the group must survive batching intact."""

    def __init__(self, path: Path, grade_key="grade_values",
                 identity_key="identity_target"):
        self.rows = []
        for line in path.open():
            it = json.loads(line)
            grades = it.get(grade_key) or {}
            cands = [c for c in it["candidates"] if c["text"] in grades]
            if not cands:
                continue
            self.rows.append({
                "item_id": it["item_id"],
                "text": marked(it),
                "cands": [c["text"] for c in cands],
                "origins": [c["origin"] for c in cands],
                "targets": [grades[c["text"]] for c in cands],
                "identity": it.get(identity_key, 0.0),
                "hard_word": it["hard_word"],
                "kind": it["kind"],
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, tok, max_len=128):
    """Flatten groups into pairs, remember group boundaries.

    `in_rank` excludes the ORIGINAL candidate from the ranking softmax. Keeping
    it in was a design error in the first run: "score the word already there"
    and "abstain" are the SAME action at runtime, so the original absorbed all
    the probability mass that should have taught abstention, and the model
    ranked identity first on 0 of 828 items where the teacher said it should.
    The original stays in the BCE term, where it is a useful calibration anchor.
    """
    left, right, gid, tgt, is_ident, origin, rank = [], [], [], [], [], [], []
    for g, row in enumerate(batch):
        for c, t, o in zip(row["cands"], row["targets"], row["origins"]):
            left.append(row["text"]); right.append(c); gid.append(g)
            tgt.append(t); is_ident.append(0.0); origin.append(o)
            rank.append(0.0 if o == "original" else 1.0)
        left.append(row["text"]); right.append(KEEP); gid.append(g)
        tgt.append(row["identity"]); is_ident.append(1.0); origin.append("identity")
        rank.append(1.0)
    enc = tok(left, right, padding=True, truncation=True,
              max_length=max_len, return_tensors="pt")
    enc["group"] = torch.tensor(gid)
    enc["target"] = torch.tensor(tgt, dtype=torch.float)
    enc["is_identity"] = torch.tensor(is_ident)
    enc["in_rank"] = torch.tensor(rank)
    enc["n_groups"] = len(batch)
    enc["origin"] = origin
    return enc


class Judge(nn.Module):
    def __init__(self, name: str, n_tokens: int):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        self.enc.resize_token_embeddings(n_tokens)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h, 1)

    def forward(self, **kw):
        kw = {k: v for k, v in kw.items()
              if k in ("input_ids", "attention_mask", "token_type_ids")}
        out = self.enc(**kw).last_hidden_state[:, 0]     # [CLS]
        return self.head(self.drop(out)).squeeze(-1)     # logits


def group_ranking_loss(logits, target, group, n_groups, in_rank):
    """Soft cross-entropy: the teacher's graded scores, normalized within the
    group, become the target distribution. Using argmax instead would throw
    away the distinction between 'clearly good' and 'arguable'."""
    loss = logits.new_zeros(())
    used = 0
    for g in range(n_groups):
        m = (group == g) & (in_rank > 0.5)
        if m.sum() < 2:
            continue
        t = target[m]
        if t.sum() <= 0:
            continue
        p = t / t.sum()
        loss = loss - (p * F.log_softmax(logits[m], dim=0)).sum()
        used += 1
    return loss / max(1, used)


@torch.no_grad()
def evaluate(model, loader, device, tag=""):
    model.eval()
    tp = fp = tn = fn = 0
    sub_stats = {"hit": 0, "n": 0, "fp": 0, "neg": 0}
    pos_scores, neg_scores = [], []
    top1 = top1_n = 0
    abst_hit = abst_n = 0
    sq = n = 0
    scores_by_origin = {}
    for b in loader:
        n_groups, origins = b.pop("n_groups"), b.pop("origin")
        b = {k: v.to(device) for k, v in b.items()}
        logits = model(**b)
        prob = torch.sigmoid(logits)
        t, ident, grp = b["target"], b["is_identity"], b["group"]
        rk = b["in_rank"]

        real = ident < 0.5
        # Calibration/accuracy on decisive labels only; 'borderline' (0.5) is
        # genuinely arguable, so counting it would add noise, not signal.
        good, bad = real & (t >= 0.8), real & (t <= 0.2)
        pred = prob >= 0.5
        tp += (good & pred).sum().item(); fn += (good & ~pred).sum().item()
        fp += (bad & pred).sum().item();  tn += (bad & ~pred).sum().item()
        sq += ((prob[real] - t[real]) ** 2).sum().item(); n += real.sum().item()
        for o, p, tv in zip(origins, prob.tolist(), t.tolist()):
            scores_by_origin.setdefault(o, []).append(p)
            if o in ("original", "identity"):
                continue
            if tv >= 0.8:
                sub_stats["n"] += 1; sub_stats["hit"] += p >= 0.5
                pos_scores.append(p)
            elif tv <= 0.2:
                sub_stats["neg"] += 1; sub_stats["fp"] += p >= 0.5
                neg_scores.append(p)

        for g in range(n_groups):
            m = (grp == g) & (rk > 0.5)
            if m.sum() < 2:
                continue
            top1_n += 1
            # Credit the pick if it TIES the teacher's best, rather than
            # requiring the same index. Groups routinely hold several
            # candidates at 0.95, and index equality would punish the model
            # for a choice the teacher rates identically.
            top1 += int(t[m][logits[m].argmax()].item() >= t[m].max().item())
            ti = t[m][ident[m] > 0.5]
            if ti.numel() and ti.item() >= 0.5:
                abst_n += 1
                abst_hit += int(ident[m][logits[m].argmax()].item() > 0.5)

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    # Pooled recall is NOT the product number. 62% of teacher-good candidates
    # are the `original` word -- the one already in the sentence -- and every
    # model scores it ~1.0. Pooling hid a 2.6x difference between models behind
    # a 6-point one. sub_recall counts only PROPOSED substitutions, which is
    # what the extension actually has to accept to do anything at all.
    sub_recall = (sub_stats["hit"] / sub_stats["n"]) if sub_stats["n"] else 0.0
    sub_fp = (sub_stats["fp"] / sub_stats["neg"]) if sub_stats["neg"] else 0.0
    # AUC is threshold-free, so it compares models that sit at different
    # operating points -- the mistake that reversed two earlier conclusions.
    auc, r_at = 0.0, {}
    if pos_scores and neg_scores:
        import numpy as _np
        P, N = _np.array(pos_scores), _np.array(neg_scores)
        rk = _np.concatenate([P, N]).argsort().argsort() + 1
        auc = float((rk[:len(P)].sum() - len(P) * (len(P) + 1) / 2) / (len(P) * len(N)))
        for fp_t in (0.01, 0.02, 0.03, 0.05):
            r_at[f"r_at_fp{int(fp_t*100)}"] = float((P >= _np.quantile(N, 1 - fp_t)).mean())
    res = {"acc": acc, "precision": prec, "recall": rec,
           "sub_recall": sub_recall, "sub_fp": sub_fp, "auc": auc, **r_at,
           "brier": sq / max(1, n),
           "top1": top1 / max(1, top1_n),
           "abstain_recall": abst_hit / max(1, abst_n), "abstain_n": abst_n,
           "mean_by_origin": {k: sum(v) / len(v) for k, v in scores_by_origin.items()}}
    if tag:
        log(f"{tag}: AUC {auc:.3f}  R@FP2% {r_at.get('r_at_fp2',0)*100:.1f}%  "
            f"acc {acc*100:.1f}%  SUB-RECALL {sub_recall*100:.1f}% "
            f"@ FP {sub_fp*100:.1f}%  brier {res['brier']:.4f}  "
            f"top1 {res['top1']*100:.1f}%  "
            f"abstain-recall {res['abstain_recall']*100:.1f}% (n={abst_n})")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/bert_uncased_L-4_H-256_A-4")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)          # groups, not pairs
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--train", default="split_train.jsonl")
    ap.add_argument("--dev", default="split_dev.jsonl")
    ap.add_argument("--test", default="split_test.jsonl")
    ap.add_argument("--tag", default="")
    # 5.6: "Plot accuracy against training size. If still climbing at 20K, get
    # more." A flat curve says the ceiling is the recipe, not the label count --
    # which is a very different (and free) thing to fix.
    ap.add_argument("--train-frac", type=float, default=1.0)
    # 60% of labels are clearly_bad, and both first models came out badly
    # pessimistic on exactly the candidates they exist to accept. Weighting the
    # positive class in the BCE term is the cheapest test of whether that
    # imbalance is the cause.
    ap.add_argument("--pos-weight", type=float, default=1.0)
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.add_special_tokens({"additional_special_tokens":
                            [SPAN_OPEN, SPAN_CLOSE, KEEP]})
    model = Judge(args.model, len(tok)).to(device)
    params = sum(p.numel() for p in model.parameters())
    log(f"{args.model}  {params/1e6:.1f}M params  device={device}")

    def loader(name, shuffle):
        ds = Groups(TABLES / name)
        if shuffle and args.train_frac < 1.0:
            keep = int(len(ds.rows) * args.train_frac)
            random.Random(7).shuffle(ds.rows)
            ds.rows = ds.rows[:keep]
        return DataLoader(ds, batch_size=args.bs, shuffle=shuffle,
                          collate_fn=lambda b: collate(b, tok)), ds

    tr, tr_ds = loader(args.train, True)
    dv, dv_ds = loader(args.dev, False)
    te, te_ds = loader(args.test, False)
    log(f"train {len(tr_ds)} groups | dev {len(dv_ds)} | test {len(te_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(tr) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    t0 = time.time()
    best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for i, b in enumerate(tr, 1):
            n_groups, _ = b.pop("n_groups"), b.pop("origin")
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(**b)
            real = b["is_identity"] < 0.5
            t_real = b["target"][real]
            w = torch.ones_like(t_real)
            if args.pos_weight != 1.0:
                w = w + (args.pos_weight - 1.0) * (t_real >= 0.8).float()
            bce = F.binary_cross_entropy_with_logits(
                logits[real], t_real, weight=w)
            rank = group_ranking_loss(logits, b["target"], b["group"], n_groups,
                                      b["in_rank"])
            loss = bce + args.rank_weight * rank
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            run += loss.item()
            if i % 100 == 0:
                log(f"  epoch {ep} step {i}/{len(tr)} loss {run/i:.4f} "
                    f"({time.time()-t0:.0f}s)")
        d = evaluate(model, dv, device, tag=f"epoch {ep} dev")
        score = d["acc"] + d["top1"]
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    log("--- best-dev checkpoint on held-out WORDS ---")
    test = evaluate(model, te, device, tag="test")
    log("mean score by candidate origin: " + ", ".join(
        f"{k} {v:.3f}" for k, v in sorted(test["mean_by_origin"].items())))

    tag = args.tag or args.model.split("/")[-1]
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_name": args.model,
                "tokens": tok.get_added_vocab()}, MODELS / f"judge_{tag}.pt")
    tok.save_pretrained(MODELS / f"tok_{tag}")
    json.dump({"model": args.model, "params": params, "args": vars(args),
               "dev_best_score": best, "test": test,
               "minutes": (time.time() - t0) / 60},
              (MODELS / f"result_{tag}.json").open("w"), indent=2)
    log(f"saved -> {MODELS / f'judge_{tag}.pt'}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
