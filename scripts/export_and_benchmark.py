#!/usr/bin/env python3
"""ONNX export + latency benchmark.

6.6 estimates ~250 MFLOPs per scored candidate and says "measure before
committing". This measures it, on CPU, single-threaded, which is the honest
proxy for a browser tab: WASM has no GPU and competes with page JavaScript.

What this CANNOT tell us: real WASM throughput. ONNX Runtime native CPU is
faster than ort-web's WASM backend -- treat these numbers as a FLOOR, and
confirm in-browser once the step-1 skeleton exists.

    python scripts/export_and_benchmark.py --tag mini
"""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_judge import KEEP, SPAN_CLOSE, SPAN_OPEN, Judge, MODELS, TABLES, marked  # noqa: E402


def log(m: str) -> None:
    print(f"[bench] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mini")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8, help="candidates per span")
    ap.add_argument("--runs", type=int, default=200)
    args = ap.parse_args()

    ckpt = torch.load(MODELS / f"judge_{args.tag}.pt", map_location="cpu",
                      weights_only=False)
    tok = AutoTokenizer.from_pretrained(MODELS / f"tok_{args.tag}")
    model = Judge(ckpt["model_name"], len(tok))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    log(f"{ckpt['model_name']}  {params/1e6:.2f}M params")

    onnx_path = MODELS / f"judge_{args.tag}.onnx"
    ids = torch.ones(1, args.seq, dtype=torch.long)
    mask = torch.ones(1, args.seq, dtype=torch.long)
    inputs, names = (ids, mask), ["input_ids", "attention_mask"]
    uses_tt = "token_type_ids" in tok.model_input_names
    if uses_tt:
        inputs = (ids, mask, torch.zeros(1, args.seq, dtype=torch.long))
        names.append("token_type_ids")

    class Wrap(torch.nn.Module):
        def __init__(self, m, tt):
            super().__init__(); self.m = m; self.tt = tt

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kw = {"input_ids": input_ids, "attention_mask": attention_mask}
            if self.tt:
                kw["token_type_ids"] = token_type_ids
            return torch.sigmoid(self.m(**kw))

    torch.onnx.export(
        Wrap(model, uses_tt), inputs, str(onnx_path), input_names=names,
        output_names=["score"], opset_version=17,
        dynamic_axes={n: {0: "batch", 1: "seq"} for n in names} |
                     {"score": {0: "batch"}})
    fp32_mb = onnx_path.stat().st_size / 1e6
    log(f"exported fp32 -> {onnx_path.name}  {fp32_mb:.1f} MB")

    from onnxruntime.quantization import QuantType, quantize_dynamic
    q_path = MODELS / f"judge_{args.tag}.int8.onnx"
    quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QInt8)
    int8_mb = q_path.stat().st_size / 1e6
    log(f"quantized int8 -> {q_path.name}  {int8_mb:.1f} MB")

    import onnxruntime as ort
    # One thread: a browser tab does not get the whole machine, and ort-web's
    # default WASM build is single-threaded unless COOP/COEP headers are set.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1

    row = [json.loads(l) for _, l in zip(range(1), (TABLES / "split_test.jsonl").open())][0]
    left = [marked(row)] * args.batch
    right = ([c["text"] for c in row["candidates"]] + [KEEP])
    right = (right * args.batch)[: args.batch]
    enc = tok(left, right, padding="max_length", truncation=True,
              max_length=args.seq, return_tensors="np")
    feed = {"input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)}
    if uses_tt:
        feed["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

    results = {}
    for label, path, size in (("fp32", onnx_path, fp32_mb),
                              ("int8", q_path, int8_mb)):
        sess = ort.InferenceSession(str(path), opts,
                                    providers=["CPUExecutionProvider"])
        for _ in range(20):
            sess.run(None, feed)
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            sess.run(None, feed)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        med = statistics.median(times)
        results[label] = {
            "mb": round(size, 1), "median_ms": round(med, 2),
            "p95_ms": round(times[int(len(times) * 0.95)], 2),
            "per_candidate_ms": round(med / args.batch, 3),
        }
        log(f"{label}: {size:5.1f} MB | batch of {args.batch} "
            f"median {med:6.2f} ms  p95 {results[label]['p95_ms']:6.2f} ms  "
            f"-> {results[label]['per_candidate_ms']:.3f} ms/candidate")

    per = results["int8"]["per_candidate_ms"]
    log("")
    log("projected page cost (int8, 1 thread, native CPU = optimistic floor):")
    for spans, cands in ((20, 5), (50, 5), (100, 5)):
        log(f"  {spans:>3} spans x {cands} candidates = {spans*cands:>4} scores "
            f"-> {spans*cands*per:7.1f} ms")
    json.dump({"params": params, "seq": args.seq, "batch": args.batch,
               "results": results}, (MODELS / f"latency_{args.tag}.json").open("w"),
              indent=2)


if __name__ == "__main__":
    main()
