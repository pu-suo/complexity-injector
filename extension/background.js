// Service worker: owns the offscreen document's lifecycle and relays scoring.

const PATH = "offscreen.html";
let creating = null;

async function ensureOffscreen() {
  const existing = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  if (existing.length) return;
  if (creating) return creating;
  creating = chrome.offscreen.createDocument({
    url: PATH,
    reasons: ["WORKERS"],
    justification: "Runs the local judge model; WebGPU and a long-lived "
                 + "110MB session are not available in a service worker.",
  });
  await creating;
  creating = null;
}

chrome.runtime.onMessage.addListener((msg, sender, respond) => {
  if (msg.target === "offscreen") return false;   // not ours

  if (msg.type === "score" || msg.type === "init") {
    (async () => {
      try {
        await ensureOffscreen();
        const reply = await chrome.runtime.sendMessage({
          ...msg, target: "offscreen",
        });
        respond(reply);
      } catch (e) {
        respond({ ok: false, error: String(e) });
      }
    })();
    return true;
  }
  return false;
});

// A badge is the only always-visible signal that the extension is off; without
// it "why is nothing happening" is indistinguishable from a bug.
async function paintBadge() {
  const { enabled } = await chrome.storage.sync.get({ enabled: true });
  await chrome.action.setBadgeText({ text: enabled ? "" : "off" });
  await chrome.action.setBadgeBackgroundColor({ color: "#888888" });
}

chrome.runtime.onInstalled.addListener(paintBadge);
chrome.runtime.onStartup.addListener(paintBadge);
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "sync" && changes.enabled) paintBadge();
});
