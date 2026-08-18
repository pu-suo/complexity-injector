#!/usr/bin/env python3
"""The shipped ONNX file must score exactly like the PyTorch model (step 5).

Every accuracy number in this project was measured with PyTorch. If the file the
extension loads computes anything else, those numbers are fiction.

This test found the most dangerous defect in the project: onnxruntime-web's WASM
int8 kernels disagreed with ONNX Runtime's own int8 by mean 0.104 / max 0.267 on
a 0-1 score, which at a 0.6 threshold would have silently changed which words
got substituted. fp32 is exact; int8 is not usable via ort-web; fp16 will not
load on WASM at all.

    python scripts/tests/test_inference_parity.py --n 40
"""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
TOL = 1e-4

CHECK = r"""
import { readFileSync } from "fs";
import * as ort from "%(ort)s";
import { WordPiece } from "%(tok)s";
ort.env.wasm.wasmPaths = "%(wasmdir)s";
ort.env.wasm.numThreads = 1;
const cfg = JSON.parse(readFileSync("%(config)s"));
const wp = new WordPiece(readFileSync("%(vocab)s", "utf8"), cfg);
const cases = JSON.parse(readFileSync("%(cases)s"));
// Node has no URL loader for ort-web; the extension fetches by URL instead.
const bytes = new Uint8Array(readFileSync("%(model)s"));
const sess = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
const { ids, mask, types, dims } = wp.batch(cases.map(c => [c.left, c.right]));
const T = ort.Tensor;
const out = await sess.run({
  input_ids: new T("int64", ids, dims),
  attention_mask: new T("int64", mask, dims),
  token_type_ids: new T("int64", types, dims),
});
const got = Array.from(out[sess.outputNames[0]].data, Number);
let worst = 0;
cases.forEach((c, i) => { worst = Math.max(worst, Math.abs(got[i] - c.pt)); });
console.log("MAXDIFF " + worst.toExponential(3));
process.exit(worst < %(tol)s ? 0 : 1);
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from train_judge import KEEP, Judge, MODELS, marked

    tok = AutoTokenizer.from_pretrained(MODELS / "tok_base")
    rows = [json.loads(l) for l in
            (ROOT / "data" / "tables" / "split_test.jsonl").open()][: args.n]
    cases = []
    for it in rows:
        left = marked(it)
        for c in it["candidates"][:2] + [{"text": KEEP}]:
            cases.append({"left": left, "right": c["text"]})

    enc = tok([c["left"] for c in cases], [c["right"] for c in cases],
              padding=True, truncation=True, max_length=128, return_tensors="np")
    feed = {k: torch.tensor(enc[k].astype(np.int64))
            for k in ("input_ids", "attention_mask", "token_type_ids")}
    ck = torch.load(MODELS / "judge_base.pt", map_location="cpu", weights_only=False)
    m = Judge(ck["model_name"], len(tok))
    m.load_state_dict(ck["state_dict"]); m.eval()
    with torch.no_grad():
        pt = torch.sigmoid(m(**feed)).numpy()
    for c, v in zip(cases, pt):
        c["pt"] = float(v)

    lib = ROOT / "extension" / "lib"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "tokenizer.mjs").write_text((lib / "tokenizer.js").read_text())
        (d / "cases.json").write_text(json.dumps(cases))
        (d / "check.mjs").write_text(CHECK % {
            "ort": lib / "ort" / "ort.mjs", "tok": d / "tokenizer.mjs",
            "wasmdir": str(lib / "ort") + "/", "config": lib / "config.json",
            "vocab": lib / "vocab.txt", "cases": d / "cases.json",
            "model": lib / "judge.onnx", "tol": TOL})
        r = subprocess.run(["node", str(d / "check.mjs")],
                           capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("MAXDIFF")), "")
    print(line or r.stderr[-500:])
    if r.returncode != 0:
        sys.exit(f"inference parity FAILED (tolerance {TOL}) over {len(cases)} scores")
    print(f"inference parity OK over {len(cases)} scores")


if __name__ == "__main__":
    main()
