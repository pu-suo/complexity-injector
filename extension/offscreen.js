// Inference host. Owns the model and the ORT session for the browser session.

import * as ort from "./lib/ort/ort.mjs";
import { WordPiece } from "./lib/tokenizer.js";
import { Judge } from "./lib/judge.js";

ort.env.wasm.wasmPaths = chrome.runtime.getURL("lib/ort/");
// A browser tab does not get the whole machine. Cap threads so scoring cannot
// starve the page it is annotating.
// Threaded WASM needs SharedArrayBuffer, which needs cross-origin isolation.
// An offscreen document is not isolated by default, so asking for more than one
// thread can fail outright at session creation. Only opt in when the browser
// says the guarantee actually holds.
ort.env.wasm.numThreads = self.crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 1)
  : 1;

let ready = null;
let judge = null;
let config = null;

async function init() {
  const url = p => chrome.runtime.getURL(p);
  const [cfg, vocab] = await Promise.all([
    fetch(url("lib/config.json")).then(r => r.json()),
    fetch(url("lib/vocab.txt")).then(r => r.text()),
  ]);
  config = cfg;
  const tok = new WordPiece(vocab, cfg);
  judge = new Judge(ort, tok, cfg);
  const backend = await judge.load(url("lib/judge.onnx"));
  console.log(`[injector] model ready on ${backend}`);
  return backend;
}

chrome.runtime.onMessage.addListener((msg, _sender, respond) => {
  if (msg.target !== "offscreen") return false;

  if (msg.type === "init") {
    ready = ready || init();
    ready.then(b => respond({ ok: true, backend: b }))
         .catch(e => respond({ ok: false, error: String(e) }));
    return true;
  }

  if (msg.type === "score") {
    ready = ready || init();
    ready.then(async () => {
      const out = [];
      for (const span of msg.spans) {
        const cands = span.candidates.map(c => c.r);
        const scores = await judge.scoreSpan(span.marked, cands);
        const decision = judge.decide(scores, config.threshold);
        out.push({ id: span.id, decision, scores });
      }
      respond({ ok: true, results: out, backend: judge.backend });
    }).catch(e => respond({ ok: false, error: String(e) }));
    return true;
  }
  return false;
});
