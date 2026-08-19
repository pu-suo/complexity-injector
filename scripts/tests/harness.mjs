// Reusable jsdom harness: loads the real content bundle against a real DOM,
// with a scriptable stand-in for the browser APIs it uses.

import { readFileSync } from "fs";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const { JSDOM } = require("../../extension/node_modules/jsdom");
const EXT = new URL("../../extension/", import.meta.url).pathname;

export function boot(bodyHtml, opts = {}) {
  const dom = new JSDOM(`<!doctype html><body>${bodyHtml}`,
                        { pretendToBeVisual: true });
  const { window } = dom;
  const { document } = window;

  // jsdom performs no layout, so everything measures 0x0 and the viewport
  // filter would reject the page. Report a plausible on-screen box.
  window.Element.prototype.getBoundingClientRect = () =>
    ({ top: 10, bottom: 200, left: 0, right: 600, width: 600, height: 190 });

  const state = {
    dom, window, document,
    scored: [],          // every span sent for scoring
    calls: 0,            // number of score messages
    mutationCallbacks: [],
    messageListeners: [],
    store: { enabled: opts.enabled ?? true },
    changeListeners: [],
    accept: opts.accept ?? (() => ({ index: 0, score: 0.99 })),
    beforeReply: opts.beforeReply ?? (async () => {}),
  };

  const asset = p => readFileSync(EXT + p, "utf8");
  const storage = {
    sync: {
      get: async d => Object.fromEntries(Object.entries(d)
        .map(([k, v]) => [k, k in state.store ? state.store[k] : v])),
      set: async patch => {
        const changes = {};
        for (const [k, v] of Object.entries(patch)) {
          changes[k] = { oldValue: state.store[k], newValue: v };
          state.store[k] = v;
        }
        for (const fn of state.changeListeners) fn(changes, "sync");
      },
    },
    onChanged: { addListener: fn => state.changeListeners.push(fn) },
  };

  const chrome = {
    storage,
    runtime: {
      id: "harness-extension-id",
      onMessage: { addListener: fn => state.messageListeners.push(fn) },
      getURL: p => p,
      sendMessage: async msg => {
        if (msg.type !== "score") return { ok: true, backend: "wasm" };
        state.calls++;
        state.scored.push(...msg.spans);
        await state.beforeReply(state, msg);
        return {
          ok: true,
          results: msg.spans.map(s => ({ id: s.id, decision: state.accept(s) })),
        };
      },
    },
  };

  Object.assign(globalThis, {
    window, document, chrome, console,
    NodeFilter: window.NodeFilter,
    MutationObserver: class {
      constructor(fn) { this.fn = fn; state.mutationCallbacks.push(fn); }
      observe() {}
      disconnect() {
        const i = state.mutationCallbacks.indexOf(this.fn);
        if (i >= 0) state.mutationCallbacks.splice(i, 1);
      }
    },
    fetch: async url => ({
      json: async () => JSON.parse(asset(url)),
      text: async () => asset(url),
    }),
  });

  new window.Function(asset("content.bundle.js"))();
  return state;
}

export const wait = ms => new Promise(r => setTimeout(r, ms));
export const swaps = (root = globalThis.document) =>
  root.querySelectorAll(".ci-word");
export const fireMutation = s => { for (const fn of [...s.mutationCallbacks]) fn([], null); };
export const askCount = s => {
  let out = null;
  for (const fn of s.messageListeners) fn({ type: "count" }, null, r => { out = r; });
  return out;
};
