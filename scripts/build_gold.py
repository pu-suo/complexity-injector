#!/usr/bin/env python3
"""Gold test slice.

5.6 specifies a test set "hand-verified by us". Human judging was ruled out,
so this substitutes the strongest available proxy: three INDEPENDENT high-effort
Opus votes per item, keeping only candidates where all three agree.

This is weaker than the plan's standard and must be recorded as such. Three
votes from one model can be systematically wrong in a way three people would
not be -- consensus measures confidence, not correctness.

Two things make the votes less correlated than a plain re-run:
  * candidate ORDER is shuffled per vote, so position bias cannot align
  * extended thinking runs at temperature 1, so sampling genuinely differs

    python scripts/build_gold.py --n 400 --votes 3
"""
from __future__ import annotations
import argparse, collections, json, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_items import GRADES, SCHEMA, SYSTEM, render, TABLES, ROOT  # noqa: E402

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import anthropic  # noqa: E402
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming  # noqa: E402
from anthropic.types.messages.batch_create_params import Request  # noqa: E402


def log(m: str) -> None:
    print(f"[gold] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--votes", type=int, default=3)
    ap.add_argument("--per-request", type=int, default=10)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--split", default="split_test.jsonl")
    ap.add_argument("--out", default="gold_test.jsonl")
    args = ap.parse_args()

    items = [json.loads(l) for l in (TABLES / args.split).open()]
    rng = random.Random(23)
    # Stratify by kind so the gold slice is not accidentally all substitutions.
    by_kind = collections.defaultdict(list)
    for it in items:
        by_kind[it["kind"]].append(it)
    for v in by_kind.values():
        rng.shuffle(v)
    chosen: list[dict] = []
    while len(chosen) < args.n and any(by_kind.values()):
        for k in sorted(by_kind):
            if by_kind[k] and len(chosen) < args.n:
                chosen.append(by_kind[k].pop())
    log(f"{len(chosen)} items from {args.split}, {args.votes} votes, "
        f"{args.model} effort={args.effort}")

    reqs = []
    for v in range(args.votes):
        vote_rng = random.Random(100 + v)
        for i in range(0, len(chosen), args.per_request):
            chunk = []
            for it in chosen[i: i + args.per_request]:
                shuffled = dict(it)
                cands = list(it["candidates"])
                vote_rng.shuffle(cands)
                shuffled["candidates"] = cands
                chunk.append(shuffled)
            body = "\n".join(render(it) for it in chunk)
            reqs.append(Request(
                custom_id=f"v{v}_b{i // args.per_request:04d}",
                params=MessageCreateParamsNonStreaming(
                    model=args.model, max_tokens=16000,
                    system=[{"type": "text", "text": SYSTEM,
                             "cache_control": {"type": "ephemeral"}}],
                    thinking={"type": "adaptive"},
                    output_config={"effort": args.effort,
                                   "format": {"type": "json_schema", "schema": SCHEMA}},
                    messages=[{"role": "user", "content":
                               f"Grade every candidate for each item.\n\n{body}"}],
                )))

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=reqs)
    log(f"batch {batch.id} submitted ({len(reqs)} requests)")
    t0 = time.time()
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        log(f"{b.processing_status} proc={b.request_counts.processing} "
            f"ok={b.request_counts.succeeded} err={b.request_counts.errored} "
            f"{time.time()-t0:.0f}s")
        time.sleep(20)

    votes: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    usage = {"in": 0, "out": 0, "errors": 0}
    for res in client.messages.batches.results(batch.id):
        if res.result.type != "succeeded":
            usage["errors"] += 1
            continue
        msg = res.result.message
        usage["in"] += msg.usage.input_tokens
        usage["out"] += msg.usage.output_tokens
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if not text:
            usage["errors"] += 1
            continue
        for row in json.loads(text)["items"]:
            for g in row["grades"]:
                votes[row["item_id"]][g["candidate"]].append(g["grade"])

    by_id = {it["item_id"]: it for it in chosen}
    kept_c = total_c = 0
    kept_items = 0
    agree_hist = collections.Counter()
    out = TABLES / args.out
    with out.open("w") as fh:
        for iid, cand_votes in votes.items():
            it = dict(by_id[iid])
            gold, unstable = {}, {}
            for cand, gs in cand_votes.items():
                total_c += 1
                if len(gs) < args.votes:
                    unstable[cand] = gs
                    continue
                agree_hist[len(set(gs))] += 1
                if len(set(gs)) == 1:
                    gold[cand] = gs[0]
                    kept_c += 1
                else:
                    unstable[cand] = gs
            if not gold:
                continue
            kept_items += 1
            it["gold"] = gold
            it["gold_values"] = {c: GRADES[g] for c, g in gold.items()}
            it["unstable"] = unstable
            orig = {c["text"] for c in it["candidates"] if c["origin"] == "original"}
            subs = [v for t, v in it["gold_values"].items() if t not in orig]
            it["gold_identity_target"] = round(1 - max(subs, default=0.0), 3)
            fh.write(json.dumps(it) + "\n")

    RATES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0)}
    rin, rout = RATES.get(args.model, (5.0, 25.0))
    cost = usage["in"] / 1e6 * rin / 2 + usage["out"] / 1e6 * rout / 2
    log(f"unanimous {kept_c}/{total_c} candidates = {kept_c/max(1,total_c)*100:.1f}%"
        f"  across {kept_items} items -> {out}")
    log("distinct grades per candidate: " + ", ".join(
        f"{k}:{v}" for k, v in sorted(agree_hist.items())))
    log(f"tokens in={usage['in']:,} out={usage['out']:,} errors={usage['errors']}")
    log(f"cost ~${cost:.2f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
