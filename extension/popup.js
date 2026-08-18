const toggle = document.getElementById("toggle");
const state = document.getElementById("state");
const backend = document.getElementById("backend");
const count = document.getElementById("count");

const render = on => { toggle.checked = on; state.textContent = on ? "On" : "Off"; };

async function refreshCount() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) return (count.textContent = "\u2014");
    const reply = await chrome.tabs.sendMessage(tab.id, { type: "count" });
    count.textContent = reply && reply.ok ? String(reply.count) : "\u2014";
  } catch {
    // No content script on chrome:// pages, the web store, or a tab that has
    // not finished loading. Not an error worth showing.
    count.textContent = "\u2014";
  }
}

async function refreshBackend(on) {
  if (!on) return (backend.textContent = "idle");
  backend.textContent = "starting\u2026";
  const reply = await chrome.runtime.sendMessage({ type: "init" });
  backend.textContent = reply && reply.ok ? reply.backend : "unavailable";
}

const { enabled } = await chrome.storage.sync.get({ enabled: true });
render(enabled);
refreshCount();
refreshBackend(enabled);

toggle.addEventListener("change", async () => {
  const on = toggle.checked;
  render(on);
  await chrome.storage.sync.set({ enabled: on });
  // Previously this set "starting..." and never asked again, so the popup sat
  // on that word until the page was reloaded.
  await refreshBackend(on);
  setTimeout(refreshCount, on ? 900 : 100);
});
