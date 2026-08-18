// Judge inference over ONNX Runtime Web.
//
// WebGPU first, WASM CPU as fallback. 6.6 said "measure before committing";
// measured at 16.6 ms/candidate on one CPU thread natively, and production
// spans need only 2.30 scores each (80% of triggers map to a single hard word),
// so a 20-span viewport is ~765 ms on CPU and far less on WebGPU.

export class Judge {
  constructor(ort, tokenizer, config) {
    this.ort = ort;
    this.tok = tokenizer;
    this.cfg = config;
    this.session = null;
    this.backend = null;
  }

  async load(modelUrl) {
    const tryOrder = ["webgpu", "wasm"];
    let lastError = null;
    for (const ep of tryOrder) {
      try {
        this.session = await this.ort.InferenceSession.create(modelUrl, {
          executionProviders: [ep],
          graphOptimizationLevel: "all",
        });
        this.backend = ep;
        return ep;
      } catch (e) {
        lastError = e;
      }
    }
    throw new Error(`no usable backend: ${lastError}`);
  }

  // One span -> its candidates plus the identity option, scored together.
  async scoreSpan(markedSentence, candidates) {
    const pairs = candidates.map(c => [markedSentence, c]);
    pairs.push([markedSentence, "[KEEP]"]);
    const { ids, mask, types, dims } = this.tok.batch(pairs);
    const T = this.ort.Tensor;
    const feeds = {
      input_ids: new T("int64", ids, dims),
      attention_mask: new T("int64", mask, dims),
      token_type_ids: new T("int64", types, dims),
    };
    const out = await this.session.run(feeds);
    const scores = Array.from(out[Object.keys(out)[0]].data, Number);
    return { candidates: scores.slice(0, -1), identity: scores[scores.length - 1] };
  }

  // 6.4: acceptance is a comparison inside one calibrated scale. The identity
  // candidate is scored like any other, so we never need an absolute floor --
  // a substitution must beat BOTH the threshold and leaving the text alone.
  decide(scores, threshold) {
    let best = -1, bestScore = -Infinity;
    scores.candidates.forEach((s, i) => {
      if (s > bestScore) { bestScore = s; best = i; }
    });
    if (best === -1) return null;
    if (bestScore < threshold) return null;
    if (bestScore <= scores.identity) return null;
    return { index: best, score: bestScore };
  }
}
