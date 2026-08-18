// Sentence segmentation, ported from scripts/pipeline/segment.py so the runtime
// splits text the same way the training data was split.

const ABBREV = new Set(`Mr Mrs Ms Dr Prof Sr Jr St Rev Hon Gen Col Capt Lt Sgt
cf vs etc al Inc Ltd Co Corp No vol pp Fig approx
Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec`.split(/\s+/));

const BOUNDARY = /[.!?]['")\]]?\s+(?=["'(\[]?[A-Z0-9])/g;
const PRECEDING = /([A-Za-z][A-Za-z.]*)$/;

function isFalseBoundary(text, idx) {
  const head = text.slice(0, idx);
  // An ellipsis is a real terminator. Without this the dotted-abbreviation
  // rule below reads "by.." out of 'Standing by..." The' and welds two
  // sentences together.
  if (head.endsWith("..")) return false;
  const m = PRECEDING.exec(head);
  if (!m) return false;
  const word = m[1].replace(/\.+$/, "");
  if (ABBREV.has(word)) return true;
  if (word.length === 1 && word === word.toUpperCase()) return true;  // initial
  if (word.includes(".")) return true;                                // e.g, U.S
  return false;
}

export function sentences(text) {
  const out = [];
  let start = 0;
  BOUNDARY.lastIndex = 0;
  let m;
  while ((m = BOUNDARY.exec(text)) !== null) {
    if (isFalseBoundary(text, m.index)) continue;
    const end = m.index + m[0].trimEnd().length;
    const chunk = text.slice(start, end).trim();
    if (chunk) out.push({ text: chunk, start, end });
    start = m.index + m[0].length;
  }
  const tail = text.slice(start).trim();
  if (tail) out.push({ text: tail, start, end: text.length });
  return out;
}
