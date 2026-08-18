// The proposer: find spans in a sentence that the inversion table covers.
// This is the lookup half of the design -- no model involved. It must be
// conservative, because everything it emits costs a judge call.

const WORD = /[A-Za-z]+(?:['-][A-Za-z]+)*/g;

export class Proposer {
  constructor(table) {
    this.table = table;
    // Longest trigger first so "warm and friendly" wins over "friendly".
    const keys = Object.keys(table).sort((a, b) => b.length - a.length);
    this.pattern = new RegExp(
      "\\b(" + keys.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")\\b",
      "gi");
  }

  // Wrong-sense defence, applied before the judge: a trigger sense can be ruled
  // out by nearby words (block) or require them (require). 68% of table rows
  // carry a block list. Multi-word terms are matched as phrases -- checking
  // them against a token set silently never fires, which made most of these
  // constraints inert in an earlier Python version.
  static contextOk(entry, lowered, tokens) {
    const present = t => t.includes(" ")
      ? lowered.includes(" " + t + " ")
      : tokens.has(t);
    if (entry.b && entry.b.some(present)) return false;
    if (entry.q && !entry.q.some(present)) return false;
    return true;
  }

  proposals(sentence) {
    // Phrase matching runs against the sentence rebuilt from its word tokens,
    // NOT the raw text. Punctuation would otherwise defeat it: "(lazy eye)"
    // and "lie ahead." never match " lazy eye " or " lie ahead ". This mirrors
    // senses.py exactly -- a parity test caught both cases.
    const allWords = sentence.toLowerCase().match(WORD) || [];
    const lowered = " " + allWords.join(" ") + " ";
    const out = [];
    this.pattern.lastIndex = 0;
    let m;
    while ((m = this.pattern.exec(sentence)) !== null) {
      const trigger = m[0].toLowerCase();
      // A rule listing one of its own trigger tokens in block_context would
      // veto every sentence it matched, so the trigger's tokens are removed
      // before single-word block/require checks (senses.py does the same).
      const trigWords = new Set(trigger.match(WORD) || []);
      const tokens = new Set(allWords.filter(w => !trigWords.has(w)));
      const entries = (this.table[trigger] || [])
        .filter(e => Proposer.contextOk(e, lowered, tokens));
      if (!entries.length) continue;
      out.push({ start: m.index, end: m.index + m[0].length,
                 surface: m[0], candidates: entries });
    }
    return out;
  }
}

// Repair a/an after a substitution. Orthographic vowels are a
// good proxy; these are the words where the proxy is wrong.
const AN_BEFORE_CONSONANT = ["hour", "honest", "honor", "honour", "heir"];
const A_BEFORE_VOWEL = ["one", "once", "uni", "use", "user", "usual", "euro",
                        "ubiq", "eulog", "euph"];

export function fixArticles(text) {
  return text.replace(/\b([Aa]n?) ([A-Za-z]+)/g, (_, art, next) => {
    const low = next.toLowerCase();
    let want;
    if (AN_BEFORE_CONSONANT.some(w => low.startsWith(w))) want = "an";
    else if (A_BEFORE_VOWEL.some(w => low.startsWith(w))) want = "a";
    else want = "aeiou".includes(low[0]) ? "an" : "a";
    if (art[0] === art[0].toUpperCase()) want = want[0].toUpperCase() + want.slice(1);
    return `${want} ${next}`;
  });
}

// Match the replacement's surface form to the span it replaces.
export function matchCase(found, replacement) {
  if (found[0] === found[0].toUpperCase() && found[0] !== found[0].toLowerCase()) {
    return replacement[0].toUpperCase() + replacement.slice(1);
  }
  return replacement;
}
