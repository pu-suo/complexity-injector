#!/usr/bin/env python3
"""Export everything the extension needs at runtime.

One model ships to every user: BERT-base, int8. The measured alternative was a
small CPU-only model, rejected because it reached 10.6% coverage against
BERT-base's 41.4% at the same 90% precision -- a 4x gap, which is not "on par".
CPU users run the same model more slowly instead of a worse model quickly.

    python scripts/export_runtime_assets.py
"""
from __future__ import annotations
import csv, json, math, shutil
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import clean  # noqa: E402

# A swap only teaches something if the replacement is genuinely rarer than what
# it replaced. Measured on live text, 9% of delivered swaps had no real gain,
# and the offenders were exactly what users notice: "totally -> utterly" is
# 0.93x (utterly is MORE common), "odd -> peculiar" 0.94x, "exhausted -> weary"
# 0.77x. A 2x floor removes 31% of rows but only 10% of hard words, so
# vocabulary breadth survives.
MIN_DIFFICULTY_GAIN = 2.0

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data" / "models"
LIB = ROOT / "extension" / "lib"
TAG = "base"
THRESHOLD = 0.6          # simulate_runtime.py: 41.4% coverage @ 90.4% precision


def main() -> None:
    LIB.mkdir(parents=True, exist_ok=True)

    # fp32, NOT int8. Verified against PyTorch on 120 real inputs:
    #   ort-web fp32  -> max difference 0.0000  (exact)
    #   ort-web int8  -> max difference 0.2667, mean 0.1060
    # ort-web's WASM int8 kernels disagree with ONNX Runtime's own int8 by as
    # much as either disagrees with fp32, so the measured coverage/precision
    # would not have transferred to the browser. fp16 halves the size but fails
    # to load on the WASM backend entirely (no fp16 kernels), so it cannot serve
    # CPU-only users. fp32 is also FASTER than int8 here (14.33 vs 16.64
    # ms/candidate) -- dynamic quantization only ever bought size.
    src = MODELS / f"judge_{TAG}.onnx"
    shutil.copy(src, LIB / "judge.onnx")
    mb = (LIB / "judge.onnx").stat().st_size / 1e6
    print(f"[assets] model -> extension/lib/judge.onnx  {mb:.1f} MB")

    tok = AutoTokenizer.from_pretrained(MODELS / f"tok_{TAG}")
    vocab = tok.get_vocab()
    ordered = [None] * (max(vocab.values()) + 1)
    for piece, idx in vocab.items():
        ordered[idx] = piece
    (LIB / "vocab.txt").write_text("\n".join(p or "[unused]" for p in ordered))
    print(f"[assets] vocab -> extension/lib/vocab.txt  {len(ordered)} pieces")

    # Proposer table, keyed by trigger. block/require lists are kept because
    # 68% of rows carry them and they are the main defence against 3 error 2
    # (wrong sense) before the judge ever runs.
    rows = [r for r in csv.DictReader(
        (l for l in (ROOT / "data" / "inventory" / "inversions_v3.tsv").open()
         if not l.startswith("#")), delimiter="\t") if r.get("trigger")]
    ft = pq.read_table(ROOT / "data" / "tables" / "pass1_pooled_frequency.parquet")
    freq = dict(zip(ft["form"].to_pylist(), ft["count"].to_pylist()))
    corpus_total = sum(freq.values())

    def per_million(text: str) -> float:
        ws = clean.words(text.lower())
        if not ws:
            return 0.0
        # A phrase is only as common as its rarest content word.
        return min(freq.get(w, 0) for w in ws) / corpus_total * 1e6

    table: dict[str, list[dict]] = {}
    dropped_easy = 0
    for r in rows:
        if r["trigger"].lower() == r["replacement"].lower():
            continue
        a, b = per_million(r["trigger"]), per_million(r["replacement"])
        if b <= 0 or a / b < MIN_DIFFICULTY_GAIN:
            dropped_easy += 1
            continue
        entry = {"r": r["replacement"], "w": r["hard_word"], "p": r["pos"]}
        if r.get("block_context"):
            entry["b"] = [t.strip().lower() for t in r["block_context"].split("|") if t.strip()]
        if r.get("require_context"):
            entry["q"] = [t.strip().lower() for t in r["require_context"].split("|") if t.strip()]
        table.setdefault(r["trigger"].lower(), []).append(entry)
    json.dump(table, (LIB / "inversions.json").open("w"),
              separators=(",", ":"), ensure_ascii=False)
    kb = (LIB / "inversions.json").stat().st_size / 1024
    print(f"[assets] table -> extension/lib/inversions.json  "
          f"{len(table)} triggers, {kb:.0f} KB")
    print(f"[assets]   dropped {dropped_easy} rows below the "
          f"{MIN_DIFFICULTY_GAIN}x difficulty floor")

    glosses = {}
    with (ROOT / "data" / "inventory" / "gregmat_900.tsv").open() as fh:
        for r in csv.DictReader((l for l in fh if not l.startswith("#")),
                                delimiter="\t"):
            glosses[r["word"]] = r["gloss"]
    json.dump(glosses, (LIB / "glosses.json").open("w"),
              separators=(",", ":"), ensure_ascii=False)
    print(f"[assets] glosses -> extension/lib/glosses.json  {len(glosses)} words")

    cfg = {
        "threshold": THRESHOLD,
        "maxLen": 128,
        "spanOpen": tok.convert_tokens_to_ids("<t>"),
        "spanClose": tok.convert_tokens_to_ids("</t>"),
        "keep": tok.convert_tokens_to_ids("[KEEP]"),
        "cls": tok.cls_token_id, "sep": tok.sep_token_id,
        "pad": tok.pad_token_id, "unk": tok.unk_token_id,
        "model": "bert-base-uncased fp32",
        # Measured: 41.4% coverage at 90.4% precision on 906 held-out spans,
        # 4.0 visibly wrong words per 100 spans.
        "expectedCoverage": 0.414, "expectedPrecision": 0.904,
        "minDifficultyGain": MIN_DIFFICULTY_GAIN,
    }
    json.dump(cfg, (LIB / "config.json").open("w"), indent=2)
    print(f"[assets] config -> extension/lib/config.json")
    print(f"[assets] special ids: <t>={cfg['spanOpen']} </t>={cfg['spanClose']} "
          f"[KEEP]={cfg['keep']}")


if __name__ == "__main__":
    main()
