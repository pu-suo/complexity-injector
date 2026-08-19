// The inference queue must never let two jobs overlap.
//
// Written after a live failure: ONNX Runtime Web threw "Session already
// started" once more than one tab asked for scores at the same time. The
// offscreen document is shared by every tab, so concurrency is the norm.

import { createRequire } from "module";
import { writeFileSync, mkdtempSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { readFileSync } from "fs";

const src = readFileSync(new URL("../../extension/lib/serialize.js", import.meta.url));
const dir = mkdtempSync(join(tmpdir(), "queue-"));
const mod = join(dir, "serialize.mjs");
writeFileSync(mod, src);
const { makeQueue } = await import(mod);

const fail = [];
const run = makeQueue();

let active = 0;
let maxActive = 0;
const order = [];

function job(id, ms, shouldThrow = false) {
  return run(async () => {
    active++;
    maxActive = Math.max(maxActive, active);
    await new Promise(r => setTimeout(r, ms));
    order.push(id);
    active--;
    if (shouldThrow) throw new Error(`job ${id} failed`);
    return id;
  });
}

// Fire everything at once, the way several tabs would. Deliberately give the
// first job the longest duration: without a queue it would finish last.
const results = await Promise.allSettled([
  job("a", 60), job("b", 10), job("c", 30, true), job("d", 5),
]);

if (maxActive !== 1) fail.push(`${maxActive} jobs ran concurrently, expected 1`);
if (order.join("") !== "abcd") {
  fail.push(`ran out of order: ${order.join("") || "(none)"}`);
}
if (results[2].status !== "rejected") fail.push("a throwing job was not rejected");
if (results[3].status !== "fulfilled") {
  fail.push("a failure upstream poisoned the queue");
}

// The queue must still work after a rejection.
const after = await job("e", 1);
if (after !== "e") fail.push("queue stopped accepting work after a rejection");

console.log(`max concurrent   : ${maxActive} (must be 1)`);
console.log(`completion order : ${order.join(" ")}`);
console.log(`survives failure : ${after === "e" ? "yes" : "NO"}`);

if (fail.length) {
  console.error("\nFAILED:\n  " + fail.join("\n  "));
  process.exit(1);
}
console.log("\nserialize queue test OK");
