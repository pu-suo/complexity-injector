#!/usr/bin/env python3
"""Generate the inversion table for the full priority vocabulary.

The hand-written v2 table covers 76 hard words; 865 are active. This asks the
teacher, for each priority word, what everyday span it can replace -- keyed to
ONE sense, with the block/require constraints that keep it in that sense.

Sense-keying is the point. the design records that v1's root cause was a data
model asserting "one part of speech and one sense per word", so every row here
carries a trigger_sense and the context lists that pin it.

    python scripts/generate_inversions.py --model claude-opus-5
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

INVENTORY, TABLES = ROOT/"data"/"inventory", ROOT/"data"/"tables"

SYSTEM = """You build a substitution table for a vocabulary-learning browser extension.

The extension scans ordinary web prose, finds an everyday word, and swaps in a
harder near-synonym so the reader meets advanced vocabulary in context. You are
given the hard word and its intended meaning. Your job is the INVERSE mapping:
which everyday word or short phrase, appearing in real text, could this hard
word correctly replace?

Rules that make an entry usable:

1. ONE SENSE PER ROW. Key each row to a single sense of the everyday trigger. If
   the trigger is ambiguous ("brief" = short vs. a legal brief), the row applies
   to one of those and the constraints must exclude the other.

2. CONTIGUOUS SPANS ONLY. The trigger must be one word or a short fixed phrase
   that appears verbatim and unbroken.

3. SURFACE FORMS MUST AGREE. `replacement` is the exact inflected form that
   substitutes for `trigger`. "hated" -> "abhorred", not "abhor". Give separate
   rows for the inflections that actually occur in prose.

4. CONSTRAINTS ARE THE VALUABLE PART.
   block_context   - content words that, if present anywhere in the sentence,
                     mean this is the WRONG sense. Derive them from the
                     trigger's other senses and from fixed phrases it appears
                     in. Example: for "bitter" -> "acrimonious", block taste,
                     flavour, chocolate, almond.
   require_context - use ONLY when the swap is safe in a narrow context and
                     unsafe by default. Leave empty if the swap is generally
                     safe; an unnecessary require list silently kills good
                     substitutions.

5. SKIP RATHER THAN STRETCH. If the hard word has no everyday equivalent that a
   swap could target, or is too polysemous to constrain, return an empty list.
   An entry that fires wrongly is far worse than a missing entry.

Aim for 1-3 rows per hard word. Prefer common triggers -- a trigger that never
appears in real prose produces no substitutions."""

SCHEMA = {
    "type":"object","properties":{"words":{"type":"array","items":{
        "type":"object","properties":{
            "hard_word":{"type":"string"},
            "rows":{"type":"array","items":{
                "type":"object","properties":{
                    "trigger":{"type":"string"},
                    "replacement":{"type":"string"},
                    "pos":{"type":"string","enum":["ADJ","ADV","VERB","NOUN"]},
                    "trigger_sense":{"type":"string"},
                    "block_context":{"type":"array","items":{"type":"string"}},
                    "require_context":{"type":"array","items":{"type":"string"}},
                },
                "required":["trigger","replacement","pos","trigger_sense",
                            "block_context","require_context"],
                "additionalProperties":False}},
        },
        "required":["hard_word","rows"],"additionalProperties":False}}},
    "required":["words"],"additionalProperties":False,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--per-request", type=int, default=15)
    ap.add_argument("--out", default="inversions_v3.tsv")
    args = ap.parse_args()

    deferred = {r["word"] for r in csv.DictReader(
        (INVENTORY/"deferred_polysemous.tsv").open(), delimiter="\t")}
    words = [r for r in csv.DictReader(
        (l for l in (INVENTORY/"gregmat_900.tsv").open()
         if not l.startswith("#")), delimiter="\t")
             if r["word"] not in deferred]
    print(f"[inv] {len(words)} active words ({len(deferred)} deferred), model={args.model}")

    client = anthropic.Anthropic()
    reqs = []
    for i in range(0, len(words), args.per_request):
        chunk = words[i:i+args.per_request]
        body = "\n".join(f"  {w['word']} = {w['gloss']}" for w in chunk)
        reqs.append(Request(custom_id=f"w{i//args.per_request:04d}",
            params=MessageCreateParamsNonStreaming(
                model=args.model, max_tokens=16000,
                system=[{"type":"text","text":SYSTEM,
                         "cache_control":{"type":"ephemeral"}}],
                thinking={"type":"adaptive"},
                output_config={"effort":args.effort,
                               "format":{"type":"json_schema","schema":SCHEMA}},
                messages=[{"role":"user","content":
                    f"Produce inversion rows for each hard word:\n\n{body}"}],
            )))

    batch = client.messages.batches.create(requests=reqs)
    print(f"[inv] batch {batch.id} ({len(reqs)} requests)")
    t0=time.time()
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended": break
        print(f"[inv] {b.processing_status} ok={b.request_counts.succeeded} "
              f"err={b.request_counts.errored} {time.time()-t0:.0f}s", flush=True)
        time.sleep(20)

    rows, usage = [], {"in":0,"out":0,"skipped":0,"errors":0}
    for res in client.messages.batches.results(batch.id):
        if res.result.type != "succeeded":
            usage["errors"] += 1; continue
        msg = res.result.message
        usage["in"] += msg.usage.input_tokens; usage["out"] += msg.usage.output_tokens
        if msg.stop_reason == "refusal": usage["errors"] += 1; continue
        text = next((b.text for b in msg.content if b.type=="text"), None)
        if not text: usage["errors"] += 1; continue
        for w in json.loads(text)["words"]:
            if not w["rows"]: usage["skipped"] += 1
            for r in w["rows"]:
                rows.append({**r, "hard_word": w["hard_word"]})

    # Deduplicate on (trigger, replacement); first row wins.
    seen, keep = set(), []
    for r in rows:
        k = (r["trigger"].lower(), r["replacement"].lower())
        if k in seen: continue
        seen.add(k); keep.append(r)

    out = INVENTORY/args.out
    with out.open("w") as fh:
        fh.write("# Generated by scripts/generate_inversions.py.\n")
        fh.write("# Sense-keyed: one row per (trigger, sense). See senses.py.\n")
        fh.write("trigger\treplacement\thard_word\tpos\ttrigger_sense\t"
                 "block_context\trequire_context\tnote\n")
        for r in keep:
            fh.write("\t".join([r["trigger"], r["replacement"], r["hard_word"], r["pos"],
                                r["trigger_sense"], "|".join(r["block_context"]),
                                "|".join(r["require_context"]), ""]) + "\n")
    i,o = (5,25) if "opus" in args.model else (2,10)
    print(f"[inv] {len(rows)} rows -> {len(keep)} after dedup; "
          f"{usage['skipped']} words skipped by the model, {usage['errors']} errors")
    print(f"[inv] -> {out}")
    print(f"[inv] tokens in={usage['in']:,} out={usage['out']:,} "
          f"cost ~${usage['in']/1e6*i/2 + usage['out']/1e6*o/2:.2f} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
