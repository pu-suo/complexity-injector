#!/usr/bin/env python3
"""The JS proposer must find the same spans as the Python one.

If the runtime proposes spans the training pipeline never would, the judge sees
inputs from a distribution it was not trained on -- and the measured 41.4%
coverage / 90.4% precision stops meaning anything.

Checked: segmentation, trigger matching, longest-match preference, and the
block/require context constraints.

    python scripts/tests/test_proposer_parity.py --n 400
"""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

CHECK = r"""
import { readFileSync } from "fs";
import { sentences } from "%(seg)s";
import { Proposer } from "%(prop)s";
const table = JSON.parse(readFileSync("%(table)s"));
const p = new Proposer(table);
const cases = JSON.parse(readFileSync("%(cases)s"));
const out = cases.map(c => {
  const spans = [];
  for (const s of sentences(c.text)) {
    for (const pr of p.proposals(s.text)) {
      spans.push([pr.surface.toLowerCase(),
                  pr.candidates.map(e => e.r).sort().join("|")]);
    }
  }
  return spans;
});
console.log(JSON.stringify(out));
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    from pipeline import senses
    from pipeline.segment import split_sentences

    # Both sides must run on the SAME rule set, otherwise this measures which
    # rows shipped rather than whether the matching logic agrees. The shipped
    # JSON is the source of truth: export_runtime_assets.py applies a difficulty
    # floor, so the live TSV holds rows the extension deliberately does not.
    shipped = json.loads(
        (ROOT / "extension" / "lib" / "inversions.json").read_text())
    by_trigger: dict[str, list] = {
        trig: [{"trigger": trig, "replacement": e["r"], "hard_word": e["w"],
                "pos": e["p"], "block_context": "|".join(e.get("b", [])),
                "require_context": "|".join(e.get("q", []))} for e in entries]
        for trig, entries in shipped.items()}

    rows = [json.loads(l) for l in
            (ROOT / "data" / "tables" / "split_test.jsonl").open()][: args.n]
    cases = [{"text": r["sentence"]} for r in rows]

    import re
    pattern = re.compile(
        r"\b(" + "|".join(sorted((re.escape(t) for t in by_trigger),
                                 key=len, reverse=True)) + r")\b", re.IGNORECASE)
    expected = []
    for c in cases:
        spans = []
        for sent in split_sentences(c["text"]):
            for m in pattern.finditer(sent):
                trig = m.group(0).lower()
                keep = [r for r in by_trigger[trig]
                        if senses.check_constraints(sent, r)[0]]
                if keep:
                    spans.append([trig, "|".join(sorted(r["replacement"] for r in keep))])
        expected.append(spans)

    lib = ROOT / "extension" / "lib"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name in ("segment", "proposer"):
            (d / f"{name}.mjs").write_text((lib / f"{name}.js").read_text())
        (d / "cases.json").write_text(json.dumps(cases))
        (d / "check.mjs").write_text(CHECK % {
            "seg": d / "segment.mjs", "prop": d / "proposer.mjs",
            "table": lib / "inversions.json", "cases": d / "cases.json"})
        r = subprocess.run(["node", str(d / "check.mjs")],
                           capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"node failed: {r.stderr[-800:]}")
    got = json.loads(r.stdout)

    same = diff = 0
    for i, (e, g) in enumerate(zip(expected, got)):
        if sorted(map(tuple, e)) == sorted(map(tuple, g)):
            same += 1
        else:
            diff += 1
            if diff <= 5:
                print(f"MISMATCH case {i}: {cases[i]['text'][:70]!r}")
                print(f"  py: {sorted(map(tuple, e))}")
                print(f"  js: {sorted(map(tuple, g))}")
    print(f"RESULT {same}/{same+diff} sentences identical")
    if diff:
        sys.exit("proposer parity FAILED")
    print("proposer parity OK")


if __name__ == "__main__":
    main()
