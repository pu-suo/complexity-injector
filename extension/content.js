// Content script: find spans on the page, get them judged, swap them in place.
//
// Scope rules exist because a wrong swap inside code, an input, or a legal
// notice is far worse than a missed one -- the design bar is "never corrupt the
// page", and 6.7 makes abstention free.

const SKIP_TAGS = new Set([
  "SCRIPT", "STYLE", "NOSCRIPT", "CODE", "PRE", "KBD", "SAMP", "VAR",
  "TEXTAREA", "INPUT", "SELECT", "OPTION", "BUTTON", "SVG", "MATH",
  "TITLE", "HEAD", "IFRAME", "OBJECT", "CANVAS",
]);
const SKIP_ROLES = new Set(["textbox", "searchbox", "code", "math"]);
const MIN_SENTENCE_WORDS = 5;
const BATCH = 12;

let proposer = null;
let glosses = {};
let config = null;
let counter = 0;
let observing = false;
let enabled = true;
let running = false;
let rerun = false;
let dead = false;

async function boot() {
  const url = p => chrome.runtime.getURL(p);
  const [table, gl, cfg] = await Promise.all([
    fetch(url("lib/inversions.json")).then(r => r.json()),
    fetch(url("lib/glosses.json")).then(r => r.json()),
    fetch(url("lib/config.json")).then(r => r.json()),
  ]);
  proposer = new Proposer(table);
  glosses = gl;
  config = cfg;
  ({ enabled } = await chrome.storage.sync.get({ enabled: true }));
  // Only warm the model when we are actually going to use it: the session is
  // 435 MB, and holding it for a user who has switched off is pure waste.
  if (enabled) chrome.runtime.sendMessage({ type: "init" }).catch(() => {});
}

// Put every swapped word back. The original is kept on the element itself, so
// this needs no bookkeeping and works even on nodes we no longer have refs to.
function revertAll(root = document) {
  hideTip();
  let n = 0;
  const touched = new Set();
  for (const el of root.querySelectorAll(".ci-word")) {
    touched.add(el.parentNode);
    el.replaceWith(document.createTextNode(el.dataset.ciOriginal));
    n++;
  }
  for (const el of root.querySelectorAll("[data-ci-kept]")) {
    touched.add(el.parentNode);
    el.replaceWith(document.createTextNode(el.textContent));
  }
  // Splicing elements out leaves a run of adjacent text nodes where there was
  // one. Without merging them the paragraph stays shattered -- a sentence
  // spanning several short nodes fails the eligibility check, so switching off
  // and on again left the page permanently unprocessable.
  for (const parent of touched) parent?.normalize();
  for (const host of root.querySelectorAll("*")) {
    if (host.shadowRoot) n += revertAll(host.shadowRoot);
  }
  return n;
}

function isEditable(node) {
  return node.isContentEditable
      || (node.closest && node.closest("[contenteditable='true']"));
}

function eligible(textNode) {
  const parent = textNode.parentElement;
  if (!parent) return false;
  if (SKIP_TAGS.has(parent.tagName)) return false;
  if (isEditable(parent)) return false;
  const role = parent.getAttribute && parent.getAttribute("role");
  if (role && SKIP_ROLES.has(role)) return false;
  if (parent.closest("a")) return false;   // never rewrite link text
  // A word the user chose to keep stays kept.
  if (parent.closest("[data-ci-kept]")) return false;
  if (parent.closest("[data-ci-ui]")) return false;
  const t = textNode.nodeValue;
  if (!t || !t.trim()) return false;
  if (t.trim().length > 20) return true;
  // Short node: accept it only if it sits inside a block with real prose in
  // it. This keeps nav labels and buttons out while letting through the
  // fragments left behind when an earlier swap split a paragraph.
  const block = parent.closest("p, li, td, blockquote, article, section, div");
  return !!block && block.textContent.trim().length > 80;
}

// Only what the reader can actually see. Scoring the whole DOM up front wastes
// the latency budget on text nobody reaches.
function inViewport(node) {
  const el = node.parentElement;
  if (!el) return false;
  const r = el.getBoundingClientRect();
  return r.bottom > -200 && r.top < window.innerHeight + 600
      && r.width > 0 && r.height > 0;
}

// A TreeWalker stops at shadow boundaries. Reddit, YouTube and most modern
// component frameworks put their content inside shadow roots, so walking only
// the light DOM finds almost nothing on exactly the sites we care about.
function collect(root = document.body, out = []) {
  const walker = document.createTreeWalker(
    root, NodeFilter.SHOW_TEXT,
    { acceptNode: n => (eligible(n) && inViewport(n))
        ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT });
  let n;
  while ((n = walker.nextNode())) out.push(n);
  for (const el of root.querySelectorAll("*")) {
    if (el.shadowRoot) collect(el.shadowRoot, out);
  }
  return out;
}

function buildJobs(textNodes) {
  const jobs = [];
  for (const node of textNodes) {
    const text = node.nodeValue;
    for (const sent of sentences(text)) {
      if ((sent.text.match(/\S+/g) || []).length < MIN_SENTENCE_WORDS) continue;
      for (const p of proposer.proposals(sent.text)) {
        jobs.push({
          id: `s${counter++}`,
          node, nodeOffset: sent.start + p.start,
          length: p.end - p.start, surface: p.surface,
          candidates: p.candidates,
          // The judge was trained on span markers, not on a blanked slot.
          marked: sent.text.slice(0, p.start) + "<t> " + p.surface
                + " </t>" + sent.text.slice(p.end),
        });
      }
    }
  }
  return jobs;
}

let tip = null;
let tipFor = null;

function hideTip() {
  tip?.remove();
  tip = null;
  tipFor = null;
}

function showTip(span) {
  if (tipFor === span) return;
  hideTip();
  tip = document.createElement("div");
  tip.className = "ci-tip";
  tip.dataset.ciUi = "1";          // never treat our own UI as page content
  const head = document.createElement("div");
  head.className = "ci-tip-word";
  head.textContent = span.dataset.ciWord || span.textContent;
  tip.append(head);
  if (span.dataset.ciGloss) {
    const g = document.createElement("div");
    g.className = "ci-tip-gloss";
    g.textContent = span.dataset.ciGloss;
    tip.append(g);
  }
  const was = document.createElement("div");
  was.className = "ci-tip-was";
  was.textContent = `replaced "${span.dataset.ciOriginal}" \u2014 click to undo`;
  tip.append(was);

  document.body.append(tip);
  const r = span.getBoundingClientRect();
  const t = tip.getBoundingClientRect();
  // Prefer above the word; flip below when there is no room, and keep the
  // card inside the viewport horizontally.
  const above = r.top - t.height - 8;
  tip.style.top = `${(above > 8 ? above : r.bottom + 8) + window.scrollY}px`;
  const left = Math.min(
    Math.max(8, r.left + r.width / 2 - t.width / 2),
    window.innerWidth - t.width - 8);
  tip.style.left = `${left + window.scrollX}px`;
  tipFor = span;
}

function makeSwap(job, decision) {
  const entry = job.candidates[decision.index];
  const span = document.createElement("span");
  span.className = "ci-word";
  span.textContent = matchCase(job.surface, entry.r);
  span.dataset.ciOriginal = job.surface;
  span.dataset.ciScore = decision.score.toFixed(3);
  span.dataset.ciGloss = glosses[entry.w] || "";
  span.dataset.ciWord = entry.w;
  // Screen readers get the same information the tooltip shows.
  span.setAttribute("aria-label",
    `${entry.w}, replacing ${job.surface}` + (span.dataset.ciGloss
      ? `. ${span.dataset.ciGloss}` : ""));
  span.addEventListener("mouseenter", () => showTip(span));
  span.addEventListener("mouseleave", hideTip);
  span.addEventListener("focus", () => showTip(span));
  span.addEventListener("blur", hideTip);
  span.tabIndex = 0;
  span.addEventListener("click", ev => {
    ev.preventDefault(); ev.stopPropagation();
    hideTip();
    const kept = document.createElement("span");
    kept.dataset.ciKept = "1";
    kept.textContent = span.dataset.ciOriginal;
    kept.title = "kept as written";
    span.replaceWith(kept);
  });
  return span;
}

// Apply EVERY accepted swap in a text node at once. Doing one per node per pass
// discarded 43% of available spans on paragraph-sized text -- measured on
// six-sentence blocks, which is what a long forum comment looks like.
// Rebuilding the node in a single pass also keeps every offset valid, which
// applying swaps one at a time does not.
function applyAll(node, jobs) {
  if (!node.parentNode) return 0;
  const text = node.nodeValue;
  const ok = jobs
    .filter(({ job }) => text.substr(job.nodeOffset, job.length) === job.surface)
    .sort((a, b) => a.job.nodeOffset - b.job.nodeOffset);
  if (!ok.length) return 0;

  const frag = document.createDocumentFragment();
  let cursor = 0;
  for (const { job, decision } of ok) {
    if (job.nodeOffset < cursor) continue;          // overlapping span
    if (job.nodeOffset > cursor) {
      frag.append(document.createTextNode(text.slice(cursor, job.nodeOffset)));
    }
    frag.append(makeSwap(job, decision));
    cursor = job.nodeOffset + job.length;
  }
  if (cursor < text.length) frag.append(document.createTextNode(text.slice(cursor)));
  node.replaceWith(frag);
  return ok.length;
}

// Reloading the extension orphans the content scripts already in the page:
// their chrome.runtime handle stops working and every call throws "Extension
// context invalidated". That is normal during development and after an update,
// so retire quietly instead of logging on every scroll.
function contextGone() {
  return dead || !chrome.runtime?.id;
}

function retire(why) {
  if (dead) return;
  dead = true;
  disconnect();
  console.debug(`[injector] stopped: ${why}`);
}

async function pass() {
  if (!proposer || !enabled || contextGone()) return;
  // A scroll and a mutation can both fire while a pass is still awaiting
  // scores. Overlapping passes score the same spans twice and race on the
  // same text nodes -- but simply dropping the second request loses work:
  // toggling off and back on quickly left the page unprocessed until the
  // next scroll. Coalesce instead, so at most one follow-up is queued.
  if (running) {
    rerun = true;
    return;
  }
  running = true;
  try {
    do {
      rerun = false;
      await runPass();
    } while (rerun && enabled && !contextGone());
  } finally {
    running = false;
  }
}

async function runPass() {
  const jobs = buildJobs(collect());
  if (!jobs.length) return;
  const pending = new Map();      // node -> [{job, decision}]

  for (let i = 0; i < jobs.length; i += BATCH) {
    const batch = jobs.slice(i, i + BATCH);
    let reply;
    try {
      reply = await chrome.runtime.sendMessage({
        type: "score",
        spans: batch.map(({ id, marked, candidates }) => ({ id, marked, candidates })),
      });
    } catch (e) {
      retire(String(e));
      return;
    }
    if (!reply || !reply.ok) {
      console.warn("[injector] scoring failed:", reply && reply.error);
      return;
    }
    const byId = new Map(batch.map(j => [j.id, j]));
    for (const r of reply.results) {
      if (!r.decision) continue;
      const job = byId.get(r.id);
      if (!job) continue;
      if (!pending.has(job.node)) pending.set(job.node, []);
      pending.get(job.node).push({ job, decision: r.decision });
    }
  }
  // Mutate only after all scoring is done: every offset was computed against
  // the pre-edit text, and editing mid-flight invalidates the rest. Re-check
  // the switch first -- scoring is slow, and the user may have turned it off
  // while this pass was waiting.
  if (!enabled || contextGone()) return;
  let applied = 0;
  for (const [node, list] of pending) applied += applyAll(node, list);
  if (applied) console.debug(`[injector] ${applied} swaps`);
}

let observer = null;
let nudge = null;
let nudgeTimer = null;

function disconnect() {
  observer?.disconnect();
  if (nudge) window.removeEventListener("scroll", nudge);
  clearTimeout(nudgeTimer);
  observer = null;
  nudge = null;
}

function observe() {
  if (observing) return;
  observing = true;
  nudge = () => {
    if (contextGone()) return retire("extension reloaded");
    clearTimeout(nudgeTimer);
    nudgeTimer = setTimeout(
      () => pass().catch(e => console.warn("[injector]", e)), 400);
  };
  observer = new MutationObserver(nudge);
  observer.observe(document.body, { childList: true, subtree: true });
  window.addEventListener("scroll", nudge, { passive: true });
}

// The popup asks the page how many words it changed. Counting the DOM rather
// than tracking a running total means the answer stays right after the user
// clicks words to revert them.
function countSwaps(root = document) {
  let n = root.querySelectorAll(".ci-word").length;
  for (const host of root.querySelectorAll("*")) {
    if (host.shadowRoot) n += countSwaps(host.shadowRoot);
  }
  return n;
}

chrome.runtime.onMessage.addListener((msg, _sender, respond) => {
  if (msg.type !== "count") return false;
  respond({ ok: true, count: countSwaps() });
  return true;
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "sync" || !changes.enabled) return;
  enabled = changes.enabled.newValue;
  if (enabled) {
    chrome.runtime.sendMessage({ type: "init" }).catch(() => {});
    pass().catch(e => console.warn("[injector]", e));
  } else {
    // Switching off restores the page immediately -- leaving substitutions
    // behind would make "off" mean "off for new text only", which is not what
    // the switch says.
    const n = revertAll();
    if (n) console.debug(`[injector] reverted ${n} swaps`);
  }
});

boot()
  .then(() => pass())
  .then(observe)
  .catch(e => console.warn("[injector] boot failed:", e));
