// BERT WordPiece tokenizer, ported to match HuggingFace BertTokenizer exactly.
//
// It must match: the judge was trained on Python-side tokenization, and any
// divergence changes the input distribution at inference. The three added
// tokens (<t>, </t>, [KEEP]) are split out BEFORE normalization, the way
// HF handles added tokens -- lowercasing them first would destroy them.

const ADDED = ["<t>", "</t>", "[KEEP]"];

export class WordPiece {
  constructor(vocabText, config) {
    this.vocab = new Map();
    vocabText.split("\n").forEach((piece, i) => this.vocab.set(piece, i));
    this.cfg = config;
    this.unk = config.unk;
    this.maxChars = 100;
  }

  // BERT's exact punctuation rule. NOT "everything non-alphanumeric": the ASCII
  // ranges below are treated as punctuation, and beyond ASCII only Unicode
  // category P is. Currency symbols (category Sc) are NOT split, so "£2" is a
  // single wordpiece -- splitting it produced the only tokenizer divergence
  // found in a 600-case parity test against HuggingFace.
  static isPunct(ch) {
    const cp = ch.codePointAt(0);
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
        (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
    return /\p{P}/u.test(ch);
  }

  // BERT drops control and format characters entirely before tokenizing, and
  // turns tab/newline/CR into spaces. Without this a zero-width joiner stays
  // inside the word and WordPiece falls back to [UNK] -- the second divergence
  // the parity test caught.
  static clean(text) {
    let out = "";
    for (const ch of text) {
      const cp = ch.codePointAt(0);
      if (cp === 0 || cp === 0xfffd) continue;
      if (ch === "\t" || ch === "\n" || ch === "\r") { out += " "; continue; }
      if (/\p{C}/u.test(ch)) continue;
      out += ch;
    }
    return out;
  }

  // Strip accents and lowercase, then isolate punctuation as its own tokens.
  static basicTokenize(text) {
    const clean = WordPiece.clean(text)
      .normalize("NFD")
      .replace(/[̀-ͯ]/g, "")
      .toLowerCase();
    const out = [];
    let cur = "";
    for (const ch of clean) {
      if (/\s/.test(ch)) {
        if (cur) { out.push(cur); cur = ""; }
      } else if (WordPiece.isPunct(ch)) {
        if (cur) { out.push(cur); cur = ""; }
        out.push(ch);
      } else {
        cur += ch;
      }
    }
    if (cur) out.push(cur);
    return out;
  }

  wordToPieces(word) {
    if (word.length > this.maxChars) return [this.unk];
    const ids = [];
    let start = 0;
    while (start < word.length) {
      let end = word.length;
      let found = -1;
      while (start < end) {
        const sub = (start === 0 ? "" : "##") + word.slice(start, end);
        if (this.vocab.has(sub)) { found = this.vocab.get(sub); break; }
        end -= 1;
      }
      if (found === -1) return [this.unk];   // whole word is unknown
      ids.push(found);
      start = end;
    }
    return ids;
  }

  // Added tokens survive normalization; everything between them is normalized.
  tokenizeWithAdded(text) {
    const pattern = new RegExp(`(${ADDED.map(t =>
      t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "g");
    const ids = [];
    for (const part of text.split(pattern)) {
      if (!part) continue;
      if (ADDED.includes(part)) {
        ids.push(this.vocab.get(part));
        continue;
      }
      for (const w of WordPiece.basicTokenize(part)) {
        ids.push(...this.wordToPieces(w));
      }
    }
    return ids;
  }

  // [CLS] left [SEP] right [SEP], truncating the LEFT segment first: the
  // candidate on the right is one or two words and must never be cut.
  encodePair(left, right) {
    const L = this.tokenizeWithAdded(left);
    const R = this.tokenizeWithAdded(right);
    const budget = this.cfg.maxLen - 3 - R.length;
    const l = L.length > budget ? L.slice(0, budget) : L;
    const ids = [this.cfg.cls, ...l, this.cfg.sep, ...R, this.cfg.sep];
    const types = [
      ...new Array(l.length + 2).fill(0),
      ...new Array(R.length + 1).fill(1),
    ];
    return { ids, types };
  }

  // Right-pad a batch to one length; ort needs a rectangular tensor.
  batch(pairs) {
    const encoded = pairs.map(([l, r]) => this.encodePair(l, r));
    const len = Math.max(...encoded.map(e => e.ids.length));
    const n = encoded.length;
    const ids = new BigInt64Array(n * len);
    const mask = new BigInt64Array(n * len);
    const types = new BigInt64Array(n * len);
    encoded.forEach((e, i) => {
      for (let j = 0; j < e.ids.length; j++) {
        ids[i * len + j] = BigInt(e.ids[j]);
        mask[i * len + j] = 1n;
        types[i * len + j] = BigInt(e.types[j]);
      }
      for (let j = e.ids.length; j < len; j++) {
        ids[i * len + j] = BigInt(this.cfg.pad);
      }
    });
    return { ids, mask, types, dims: [n, len] };
  }
}
