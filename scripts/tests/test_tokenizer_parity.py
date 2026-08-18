#!/usr/bin/env python3
"""The JS tokenizer must match HuggingFace exactly.

The judge was trained on Python-side tokenization. Any divergence at inference
changes the input distribution -- silently, with no error and no obvious symptom
in the output. So this is a hard equality test, not a similarity check.

It caught one real bug: BERT does not split currency symbols (Unicode category
Sc), so "£2" is a single wordpiece. The JS split on every non-alphanumeric
character and produced "£" + "2".

    python scripts/tests/test_tokenizer_parity.py --n 2000
"""
from __future__ import annotations
import argparse, json, random, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

ADVERSARIAL = [
    ("It was a <t> very big </t> problem .", "enormous"),
    ("She gave a <t> long-winded </t> reply about the café's naïve policy.", "verbose"),
    ("Dr. Smith vs. the U.S. said e.g. 'yes'!", "[KEEP]"),
    ("A 100% unbeatable — truly unprecedented — antidisestablishmentarianism.", "[KEEP]"),
    ("Costs rose £2.7bn, $3.10, €5, ¥900 and ₹40 last year.", "[KEEP]"),
    ("emoji 🙂 zero-width​join and\ttabs\nnewlines", "[KEEP]"),
    ("MiXeD CaSe ALLCAPS niño Ünïcödé", "[KEEP]"),
]

CHECK_JS = r"""
import { readFileSync } from "fs";
import { WordPiece } from "%(tokenizer)s";
const cfg = JSON.parse(readFileSync("%(config)s"));
const wp = new WordPiece(readFileSync("%(vocab)s", "utf8"), cfg);
const cases = JSON.parse(readFileSync("%(cases)s"));
let bad = 0;
for (const c of cases) {
  const { ids, types } = wp.encodePair(c.left, c.right);
  if (JSON.stringify(ids) !== JSON.stringify(c.ids) ||
      JSON.stringify(types) !== JSON.stringify(c.types)) {
    if (bad < 5) {
      console.log("MISMATCH " + JSON.stringify(c.left.slice(0, 60)));
      console.log("  py " + c.ids.join(","));
      console.log("  js " + ids.join(","));
    }
    bad++;
  }
}
console.log(`RESULT ${cases.length - bad}/${cases.length}`);
process.exit(bad ? 1 : 0);
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from train_judge import marked

    tok = AutoTokenizer.from_pretrained(ROOT / "data" / "models" / "tok_base")
    cases = [{"left": l, "right": r} for l, r in ADVERSARIAL]
    rows = [json.loads(l) for l in
            (ROOT / "data" / "tables" / "split_test.jsonl").open()]
    random.Random(4).shuffle(rows)
    for it in rows:
        if len(cases) >= args.n:
            break
        for c in it["candidates"]:
            cases.append({"left": marked(it), "right": c["text"]})
    for c in cases:
        e = tok(c["left"], c["right"], add_special_tokens=True)
        c["ids"], c["types"] = e["input_ids"], e["token_type_ids"]

    lib = ROOT / "extension" / "lib"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "cases.json").write_text(json.dumps(cases))
        # Node treats .js as CommonJS without a package.json; copy to .mjs.
        (d / "tokenizer.mjs").write_text((lib / "tokenizer.js").read_text())
        (d / "check.mjs").write_text(CHECK_JS % {
            "tokenizer": d / "tokenizer.mjs", "config": lib / "config.json",
            "vocab": lib / "vocab.txt", "cases": d / "cases.json"})
        r = subprocess.run(["node", str(d / "check.mjs")],
                           capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        sys.exit(f"tokenizer parity FAILED over {len(cases)} cases")
    print(f"tokenizer parity OK over {len(cases)} cases")


if __name__ == "__main__":
    main()
