# Model notes

## Selection

Nine encoder configurations were fine-tuned on identical data and compared on a
word-disjoint split, ranked by AUC and by recall at a matched false-positive
rate. Fixed-threshold recall is not comparable across models that sit at
different operating points.

| model | params | dev AUC | ms/candidate |
|---|---|---|---|
| BERT-Tiny L-2 H-128 | 4.4M | 0.514 | 0.31 |
| BERT-Mini L-4 H-256 | 11.2M | 0.625 | 1.38 |
| L-12 H-256 | 17.5M | 0.766 | 3.92 |
| BERT-Small L-4 H-512 | 28.8M | 0.616 | 3.24 |
| BERT-Medium L-8 H-512 | 41.4M | 0.800 | 6.29 |
| DistilBERT L-6 H-768 | 66.4M | 0.828 | 8.37 |
| **BERT-base L-12 H-768** | **109.5M** | **0.907** | 14.33 |

Everything between 11M and 29M is indistinguishable at matched false-positive
rate. The step change is between 29M and 41M, and accuracy had not saturated at
110M. Depth is worth roughly three times width by AUC.

A smaller CPU-only companion model was built, distilled and tuned, then
dropped: at matched precision it reached 10.6% coverage against BERT-base's
41.4%. One model ships to every user; machines without WebGPU run the same
model more slowly rather than a worse model quickly.

## Format: fp32, not int8

`onnxruntime-web`'s WASM int8 kernels do not compute what ONNX Runtime's int8
kernels compute. Measured on 120 real inputs against PyTorch:

| build | max difference | mean |
|---|---|---|
| ONNX Runtime fp32 | 0.0000 | 0.0000 |
| **onnxruntime-web fp32** | **0.0000** | **0.0000** |
| ONNX Runtime int8 | 0.1459 | 0.0109 |
| onnxruntime-web int8 | 0.2667 | 0.1060 |
| onnxruntime-web fp16 | fails to load — no WASM fp16 kernels | |

On a 0–1 score with a 0.60 threshold, a mean error of 0.106 changes which words
get substituted. Every accuracy figure here was measured in PyTorch, so int8
would have invalidated all of them. fp32 is also faster (14.33 vs 16.64
ms/candidate); dynamic quantization only bought size.

The cost is a 436 MB artifact. The embedding table is 94 MB of it, so pruning
the vocabulary to the tokens the corpus actually uses would recover most of the
difference without retraining.

## Training data

10,000 items, 49,900 graded candidates, labelled by Claude Opus 5 through the
Batch API. Grades are five-way rather than binary; the identity option's target
is derived as `1 − max(candidate grade)` rather than observed.

Two losses on one scalar: per-candidate binary cross-entropy for calibration,
and a softmax across each span's candidates for ranking. The identity option
takes part in ranking only — its target is derived, so training it as a
calibrated probability would assert a number nobody measured. The original word
is excluded from the ranking softmax, since "keep the existing word" and
"abstain" are the same action at runtime.

Splits are disjoint by hard word, not random. The architecture's claim is that
vocabulary can grow without retraining, and that is only testable on words the
model never saw.

## Evaluation

The test set is 400 items graded by three independent high-effort passes with
candidate order shuffled per pass; only unanimous candidates are kept. Three
votes agreed on 92.0% of candidates, which is the practical ceiling for this
task — roughly one candidate in twelve is genuinely arguable.

This is weaker than human verification and should be read as such. Consensus
measures confidence, not correctness, and one model can be systematically wrong
in a way three people would not be.
