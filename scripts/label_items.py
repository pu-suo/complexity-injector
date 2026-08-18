#!/usr/bin/env python3
"""Teacher labelling via the Batch API.

Poses each item in GRE Sentence Equivalence form -- sentence with the span
blanked, plus candidate fillers -- and asks for GRADED labels per candidate
rather than a binary flag (5.5: "costs a few extra output tokens and gives the
student far more to learn from").

Two design decisions carried in from earlier work:
  * The intended SENSE is stated in the prompt. Sense is the dominant residual
    failure mode, so the teacher must judge against a stated gloss rather than
    guessing which meaning we meant.
  * Identity is NOT asked about. Its training target is derived as
    1 - max(candidate grade), so asking would buy nothing.

    python scripts/label_items.py --n 500 --model claude-sonnet-5 --effort medium
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

TABLES = ROOT / "data" / "tables"
GRADES = {"clearly_good": 0.95, "probably_good": 0.8, "borderline": 0.5,
          "probably_bad": 0.2, "clearly_bad": 0.05}

SYSTEM = """You are grading vocabulary substitutions for a language-learning tool.

Each question gives a real sentence with one span blanked out, the everyday
words that were removed, and several candidate replacements. For each candidate
you judge whether putting it in the blank yields a sentence that is CORRECT.

The bar is correctness, not elegance:
  - Grammatical, right part of speech, right inflection.
  - The RIGHT SENSE. Each item names the sense we intend; a candidate that is a
    real synonym of some OTHER sense of the removed words is wrong.
  - Meaning preserved. No drift in degree, polarity, or scope.
  - Does not break a fixed phrase or a proper name.

Deliberately NOT part of the bar: whether the result reads smoothly, or whether
the register matches. A correct sentence that sounds slightly stiff or unusually
formal is still correct - grade it on correctness alone.

Grades:
  clearly_good  - correct; a careful editor would accept it
  probably_good - correct, with a minor infelicity
  borderline    - genuinely arguable
  probably_bad  - likely wrong (odd collocation, slight sense drift)
  clearly_bad   - plainly wrong (ungrammatical, wrong sense, breaks a phrase)

Grade every candidate independently. One of the candidates is usually the
original wording, which should grade clearly_good; use it to calibrate."""

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "item_id": {"type": "string"},
            "grades": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "candidate": {"type": "string"},
                    "grade": {"type": "string", "enum": list(GRADES)},
                },
                "required": ["candidate", "grade"], "additionalProperties": False}},
        },
        "required": ["item_id", "grades"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False,
}


def render(it: dict) -> str:
    cands = "\n".join(f"  - {c['text']}" for c in it["candidates"])
    return (f"[{it['item_id']}]\n"
            f"Sentence: {it['blanked']}\n"
            f"Removed:  \"{it['span']}\"  (intended sense: {it['sense']})\n"
            f"Target word gloss: {it['hard_word']} = {it['gloss']}\n"
            f"Candidates:\n{cands}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--per-request", type=int, default=20)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--items", default="items.jsonl")
    ap.add_argument("--out", default="labeled_pilot.jsonl")
    args = ap.parse_args()

    items = [json.loads(l) for l in (TABLES / args.items).open()]
    random.Random(5).shuffle(items)
    items = items[: args.n]
    print(f"[label] {len(items)} items, model={args.model}, effort={args.effort}")

    client = anthropic.Anthropic()
    reqs, batches = [], []
    for i in range(0, len(items), args.per_request):
        chunk = items[i : i + args.per_request]
        batches.append(chunk)
        body = "\n".join(render(it) for it in chunk)
        reqs.append(Request(
            custom_id=f"b{i // args.per_request:04d}",
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

    batch = client.messages.batches.create(requests=reqs)
    print(f"[label] batch {batch.id} submitted ({len(reqs)} requests)")
    t0 = time.time()
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        print(f"[label] {b.processing_status} "
              f"proc={b.request_counts.processing} ok={b.request_counts.succeeded} "
              f"err={b.request_counts.errored}  {time.time()-t0:.0f}s", flush=True)
        time.sleep(20)

    graded, usage = {}, {"in": 0, "out": 0, "refusals": 0, "errors": 0}
    for res in client.messages.batches.results(batch.id):
        if res.result.type != "succeeded":
            usage["errors"] += 1
            continue
        msg = res.result.message
        usage["in"] += msg.usage.input_tokens
        usage["out"] += msg.usage.output_tokens
        if msg.stop_reason == "refusal":      # no `fallbacks` on the Batch API
            usage["refusals"] += 1
            continue
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if not text:
            usage["errors"] += 1
            continue
        for row in json.loads(text)["items"]:
            graded[row["item_id"]] = {g["candidate"]: g["grade"] for g in row["grades"]}

    by_id = {it["item_id"]: it for it in items}
    out = TABLES / args.out
    with out.open("w") as fh:
        for iid, g in graded.items():
            it = dict(by_id[iid])
            it["grades"] = g
            it["grade_values"] = {c: GRADES[v] for c, v in g.items()}
            # Identity's target excludes the ORIGINAL candidate. Including it
            # pinned identity at ~0.05 on every item in the first pilot, since
            # the original always grades clearly_good -- so the judge would have
            # learned never to abstain.
            orig = {c["text"] for c in it["candidates"] if c["origin"] == "original"}
            subs = [v for t, v in it["grade_values"].items() if t not in orig]
            it["identity_target"] = round(1 - max(subs, default=0.0), 3)
            fh.write(json.dumps(it) + "\n")

    # Batch API = 50% off. Rates must follow the model actually used -- this
    # line was hardcoded to Sonnet and mis-reported the Opus cross-check by 2.6x.
    RATES = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0),
             "claude-haiku-4-5": (1.0, 5.0)}
    rin, rout = RATES.get(args.model, (5.0, 25.0))
    cost = usage["in"] / 1e6 * rin / 2 + usage["out"] / 1e6 * rout / 2
    print(f"[label] graded {len(graded)}/{len(items)} items -> {out}")
    print(f"[label] tokens in={usage['in']:,} out={usage['out']:,} "
          f"| refusals={usage['refusals']} errors={usage['errors']}")
    print(f"[label] batch cost ~${cost:.2f}  ({time.time()-t0:.0f}s)")
    print(f"[label] thinking+output per request: {usage['out']//max(1,len(reqs)):,} tokens")


if __name__ == "__main__":
    main()
