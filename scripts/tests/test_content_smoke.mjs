// End-to-end smoke test for the content script, against a real DOM (jsdom).
//
// Written after a silent failure that shipped: a patch renamed a helper, so a
// later edit's search string no longer matched and `replace()` did nothing.
// The bundle still PARSED -- `node --check` passes on a call to a function that
// was never defined -- and the extension loaded, did nothing, and reported only
// "ReferenceError: applyAll is not defined" in the offscreen console.
//
// Parsing is not evidence. This runs a real pass over a real DOM and asserts
// that words actually change, and that the protected regions do not.
//
//     node scripts/tests/test_content_smoke.mjs

import { readFileSync } from "fs";
import { createRequire } from "module";
const { JSDOM } = createRequire(import.meta.url)("../../extension/node_modules/jsdom");

const EXT = new URL("../../extension/", import.meta.url).pathname;

const dom = new JSDOM(`<!doctype html><body>
  <p id="prose">The talkative neighbour gave a very common and long-winded
     explanation, and it was completely pointless to argue with them at all.</p>
  <p id="linked"><a href="#">a very common and harmful mistake</a></p>
  <pre id="code">const talkative = true; // a chatty comment</pre>
  <textarea id="ta">a stubborn and boring draft</textarea>
  <p id="short">Very common.</p>
</body>`, { pretendToBeVisual: true });

const { window } = dom;
const { document } = window;

// jsdom gives no layout, so every element measures 0x0 and the viewport filter
// would reject the whole page. Report a plausible on-screen box instead.
window.Element.prototype.getBoundingClientRect = function () {
  return { top: 10, bottom: 200, left: 0, right: 600, width: 600, height: 190 };
};

const asset = p => readFileSync(EXT + p, "utf8");
const scored = [];
const mutationCallbacks = [];
const messageListeners = [];

// Minimal chrome.storage.sync: enough to drive the on/off switch and to fire
// the change listener the content script subscribes to.
const store = { enabled: true };
const changeListeners = [];
const storage = {
  sync: {
    get: async defaults => {
      const out = {};
      for (const [k, v] of Object.entries(defaults)) out[k] = k in store ? store[k] : v;
      return out;
    },
    set: async patch => {
      const changes = {};
      for (const [k, v] of Object.entries(patch)) {
        changes[k] = { oldValue: store[k], newValue: v };
        store[k] = v;
      }
      for (const fn of changeListeners) fn(changes, "sync");
    },
  },
  onChanged: { addListener: fn => changeListeners.push(fn) },
};

Object.assign(globalThis, {
  window, document,
  NodeFilter: window.NodeFilter,
  MutationObserver: class {
    constructor(fn) { this.fn = fn; mutationCallbacks.push(fn); }
    observe() {}
    disconnect() {
      const i = mutationCallbacks.indexOf(this.fn);
      if (i >= 0) mutationCallbacks.splice(i, 1);
    }
  },
  console,
  chrome: {
    storage,
    runtime: {
      // Present while the extension is live; cleared when it is reloaded, which
      // is how a content script detects that it has been orphaned.
      id: "smoke-test-extension-id",
      onMessage: { addListener: fn => messageListeners.push(fn) },
      getURL: p => p,
      sendMessage: async msg => {
        if (msg.type !== "score") return { ok: true };
        // Accept every proposal: this test is about the DOM path, not the model.
        scored.push(...msg.spans);
        return {
          ok: true,
          results: msg.spans.map(s => ({
            id: s.id, decision: { index: 0, score: 0.99 },
          })),
        };
      },
    },
  },
  fetch: async url => ({
    json: async () => JSON.parse(asset(url)),
    text: async () => asset(url),
  }),
});

function askCount() {
  let out = null;
  for (const fn of messageListeners) fn({ type: "count" }, null, r => { out = r; });
  return out;
}

const before = document.getElementById("prose").textContent;
new window.Function(asset("content.bundle.js"))();   // no ReferenceErrors allowed

await new Promise(r => setTimeout(r, 400));

const fail = [];
const countOn = askCount();
const swaps = document.querySelectorAll(".ci-word");
if (!countOn || !countOn.ok) fail.push("content script did not answer a count query");
else if (countOn.count !== swaps.length) {
  fail.push(`count says ${countOn.count} but ${swaps.length} words are swapped`);
}
const after = document.getElementById("prose").textContent;

if (!scored.length) fail.push("proposer produced no spans at all");
if (!swaps.length) fail.push("no .ci-word elements were inserted");
if (after === before) fail.push("prose text is unchanged");

// The whole point of the design's bar: never corrupt the page.
for (const id of ["linked", "code", "ta", "short"]) {
  const el = document.getElementById(id);
  if (el.querySelector(".ci-word")) fail.push(`PROTECTED REGION MODIFIED: #${id}`);
}
// Multi-swap: one text node must be able to take more than one replacement.
const inProse = document.getElementById("prose").querySelectorAll(".ci-word");

console.log(`spans proposed : ${scored.length}`);
console.log(`swaps applied  : ${swaps.length} (${inProse.length} in one node)`);
console.log(`before: ${before.replace(/\s+/g, " ").trim()}`);
console.log(`after : ${after.replace(/\s+/g, " ").trim()}`);
if (inProse.length < 2) {
  fail.push(`only ${inProse.length} swap(s) in a multi-span node -- the `
          + `one-per-node regression is back`);
}
// ---- the on/off switch -------------------------------------------------
await storage.sync.set({ enabled: false });
await new Promise(r => setTimeout(r, 50));
const afterOff = document.getElementById("prose").textContent;
if (document.querySelectorAll(".ci-word").length) {
  fail.push("switching off left swapped words on the page");
}
if (afterOff !== before) {
  fail.push("switching off did not restore the original text");
}

// While off, a page mutation must not trigger any scoring. Fire the observer
// callback the content script registered -- a no-op stub would let this pass
// without ever exercising the path.
const countOff = askCount();
if (countOff && countOff.count !== 0) {
  fail.push(`count says ${countOff.count} while switched off`);
}

const seenBefore = scored.length;
document.getElementById("prose").textContent = before;
for (const fn of mutationCallbacks) fn([], null);
await new Promise(r => setTimeout(r, 600));   // longer than the 400ms debounce
const inertOk = scored.length === seenBefore;
if (!inertOk) fail.push("scored spans while switched off");

// Back on: substitution resumes without a reload.
await storage.sync.set({ enabled: true });
await new Promise(r => setTimeout(r, 300));
const backOn = document.querySelectorAll(".ci-word").length;
if (!backOn) fail.push("switching back on did not resume substitution");

console.log(`count (on/off)   : ${countOn ? countOn.count : "?"} / `
          + `${countOff ? countOff.count : "?"}`);
console.log(`off -> reverted  : ${afterOff === before ? "yes" : "NO"}`);
console.log(`off -> inert     : ${inertOk ? "yes" : "NO"} (${mutationCallbacks.length} observer(s) fired)`);
console.log(`on  -> resumed   : ${backOn} swaps`);

// ---- surviving an extension reload -------------------------------------
// Reloading orphans this content script: chrome.runtime.id disappears and
// every API call throws. It must retire quietly, not log on every scroll.
const beforeReload = scored.length;
chrome.runtime.id = undefined;
chrome.runtime.sendMessage = async () => {
  throw new Error("Extension context invalidated.");
};
let threw = null;
try {
  for (const fn of [...mutationCallbacks]) fn([], null);
  await new Promise(r => setTimeout(r, 600));
} catch (e) {
  threw = e;
}
if (threw) fail.push(`threw after extension reload: ${threw.message}`);
if (scored.length !== beforeReload) fail.push("kept scoring after reload");
if (mutationCallbacks.length) {
  fail.push("did not disconnect its observer after reload");
}
console.log(`reload -> quiet   : ${!threw ? "yes" : "NO"}, `
          + `observers left ${mutationCallbacks.length}`);

if (fail.length) {
  console.error("\nFAILED:\n  " + fail.join("\n  "));
  process.exit(1);
}
console.log("\ncontent script smoke test OK");
