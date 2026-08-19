// Realistic failure modes, run against the real bundle and a real DOM.
// Each case is something a user does within the first minute.

import { boot, wait, swaps, fireMutation, askCount } from "./harness.mjs";

const PROSE = `<p id="a">The talkative neighbour gave a very common and
  long-winded explanation, and it was completely pointless to argue with them.</p>`;
const results = [];

function check(name, cond, detail = "") {
  results.push({ name, ok: !!cond, detail });
  console.log(`  ${cond ? "pass" : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
}

// 1. Rapid scrolling fires the observer many times in quick succession.
{
  const s = boot(PROSE);
  await wait(300);
  const callsAfterFirst = s.calls;
  for (let i = 0; i < 40; i++) fireMutation(s);
  await wait(900);
  check("rapid scroll does not re-score the same spans repeatedly",
        s.calls <= callsAfterFirst + 1, `${s.calls} score calls total`);
}

// 2. Flipping the switch quickly, as a user testing what it does.
{
  const s = boot(PROSE);
  await wait(300);
  for (let i = 0; i < 6; i++) {
    await chrome.storage.sync.set({ enabled: i % 2 === 1 });
    await wait(30);
  }
  await wait(600);
  const on = s.store.enabled;
  check("toggle spam leaves a consistent state",
        on ? swaps().length > 0 : swaps().length === 0,
        `ended ${on ? "on" : "off"} with ${swaps().length} swaps`);
}

// 3. A single-page app replaces the article body (Reddit, Twitter, YouTube).
{
  const s = boot(PROSE);
  await wait(300);
  const first = swaps().length;
  document.getElementById("a").remove();
  const p = document.createElement("p");
  p.textContent = "A stubborn and boring argument about a harmful idea, honestly.";
  document.body.append(p);
  fireMutation(s);
  await wait(900);
  check("new SPA content gets processed", swaps().length > 0,
        `${first} before navigation, ${swaps().length} after`);
}

// 4. The user clicks a word to revert it. It must stay reverted.
{
  const s = boot(PROSE);
  await wait(300);
  const first = swaps()[0];
  const original = first.dataset.ciOriginal;
  first.click();
  const afterClick = swaps().length;
  fireMutation(s);
  await wait(900);
  const stillThere = document.body.textContent.includes(original);
  const reSwapped = swaps().length > afterClick;
  check("a reverted word is not swapped again", !reSwapped && stillThere,
        `reverted "${original}", ${afterClick} -> ${swaps().length} swaps`);
}

// 5. Pages with nothing to work on must not throw.
{
  let threw = null;
  try {
    const s = boot(`<div></div><img src="x.png"><p>Too short.</p>`);
    await wait(400);
    fireMutation(s);
    await wait(500);
  } catch (e) { threw = e; }
  check("a page with no eligible text is harmless", !threw,
        threw ? threw.message : "no swaps, no errors");
}

// 6. The page mutates while scoring is in flight.
{
  const s = boot(PROSE, {
    beforeReply: async () => { document.getElementById("a")?.remove(); },
  });
  await wait(600);
  check("a node removed mid-score does not corrupt the page",
        document.body.querySelectorAll(".ci-word").length === 0,
        "node vanished before the swap landed");
}

// 7. An editor appears after load (comment box, inline edit).
{
  const s = boot(PROSE);
  await wait(300);
  const ed = document.createElement("div");
  ed.setAttribute("contenteditable", "true");
  ed.textContent = "a talkative and stubborn draft that is completely pointless";
  document.body.append(ed);
  fireMutation(s);
  await wait(900);
  check("a contenteditable added later is never touched",
        ed.querySelectorAll(".ci-word").length === 0);
}

// 8. A long page: every eligible span should be scored, in batches.
{
  const many = Array.from({ length: 40 }, (_, i) =>
    `<p>Paragraph ${i}: the talkative neighbour gave a long-winded and
     completely pointless explanation to argue the case.</p>`).join("");
  const s = boot(many);
  await wait(1500);
  check("a long page scores every span in batches",
        s.scored.length > 40 && s.calls > 1,
        `${s.scored.length} spans over ${s.calls} batches`);
}

// 9. Content inside a shadow root (Reddit, YouTube).
{
  const s = boot(`<div id="host"></div>`);
  const host = document.getElementById("host");
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `<p>The talkative neighbour gave a completely pointless
    and long-winded explanation to argue about it.</p>`;
  fireMutation(s);
  await wait(900);
  const inShadow = root.querySelectorAll(".ci-word").length;
  check("text inside a shadow root is processed", inShadow > 0,
        `${inShadow} swaps in the shadow root`);
  const counted = askCount(s);
  check("the popup count includes shadow-root swaps",
        counted && counted.count === inShadow,
        `count reported ${counted && counted.count}`);
}

const failed = results.filter(r => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} scenarios passed`);
if (failed.length) {
  console.error("FAILED:\n  " + failed.map(f => f.name).join("\n  "));
  process.exit(1);
}
