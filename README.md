# Complexity Injector

A Chrome extension that swaps everyday words for harder equivalents as you
read. Hover a changed word to see what it replaced; click to put it back.

The substitution is judged on-device by a fine-tuned cross-encoder. Nothing
leaves the browser, and there is no API key or server.

![toolbar icon](extension/icons/preview.png)

## How it works

```
proposer  ──▶  gate  ──▶  judge  ──▶  swap
lookup table   sense &     BERT-base   in-place, revertible
1,271 triggers difficulty  on WebGPU
               filters     or WASM
```

1. **Proposer** — a lookup table maps everyday spans to harder words. Each row
   is keyed to a single sense and carries optional context constraints.
2. **Gate** — cheap deterministic filters: part of speech, fixed phrases and
   proper names, sense constraints, and a minimum difficulty floor.
3. **Judge** — a cross-encoder scores every surviving candidate against the
   sentence, plus a "leave it alone" option. A swap must beat both a threshold
   and that option, so acceptance is a comparison on one calibrated scale
   rather than an absolute cutoff.

Passing the candidate as *input* rather than as a class means new vocabulary
can be added to the table without retraining the model.

## Results

Measured on 906 held-out spans whose vocabulary never appeared in training:

| | |
|---|---|
| Coverage | 41.4% of proposals accepted |
| Precision | 90.4% |
| Visible errors | 4.0 per 100 spans |
| Latency | 14.3 ms per candidate, single CPU thread |
| Page cost | ~660 ms for a 20-span viewport (much less on WebGPU) |

Production spans need 2.30 model calls each: 80% of triggers map to exactly one
hard word, plus one call for the leave-it-alone option.

Delivery on real pages, per 1,000 words of prose:

| source | swaps | median difficulty gain |
|---|---|---|
| Reddit (long-form) | 8.6 | 10.6× rarer |
| Stack Exchange | 4.6 | 16.0× rarer |

Delivery scales with passage length, not with register — 30% of short forum
replies contain no eligible span at all, while every 100+ word passage does.

## Install

```bash
python scripts/vendor_ort.py          # stage onnxruntime-web
python scripts/export_runtime_assets.py   # model, vocab, tables
python scripts/build_extension.py     # bundle the content script
```

Then load `extension/` unpacked at `chrome://extensions` with Developer mode
on. Requires Chrome 116+.

## Tests

```bash
pytest scripts/tests -q                        # 60 unit tests
python scripts/tests/test_tokenizer_parity.py  # JS tokenizer vs HuggingFace
python scripts/tests/test_proposer_parity.py   # JS proposer vs Python pipeline
python scripts/tests/test_inference_parity.py  # shipped ONNX vs PyTorch
node scripts/tests/test_content_smoke.mjs      # content script against a DOM
```

The three parity suites exist because the runtime is a JavaScript
reimplementation of a Python pipeline, and a divergence there is silent: the
extension keeps working and simply feeds the model inputs it was not trained
on. All three assert exact equality, not similarity.

## Layout

```
extension/          the extension: manifest, content script, offscreen host
  lib/              tokenizer, segmenter, proposer, judge, generated tables
scripts/            corpus pipeline, training, evaluation, build
  pipeline/         cleaning, segmentation, dedup, sense constraints
  tests/            unit and parity suites
data/inventory/     vocabulary and the sense-keyed substitution table
docs/               engineering notes
```

Model checkpoints, the ONNX export, the corpus and its derived tables are not
in version control; they are produced by the pipeline. See `docs/model.md`.
