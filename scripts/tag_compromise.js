// Tag sentences with compromise.js — the RUNTIME tagger.
// Reads {"sentences": [...]} on stdin, writes [[["token","POS"],...],...] on stdout.
//
// Exists because offline tagging is spaCy and runtime tagging is compromise.
// If they disagree on the tokens a span pattern keys on, then error 1 (wrong
// part of speech) is not "deterministic, blocking" as the design notes claims.

const nlp = require('compromise');

// compromise tag -> Universal POS, checked in priority order because a term
// carries several tags at once (e.g. Verb + PastTense + PhrasalVerb).
const PRIORITY = [
  ['Verb', 'VERB'],
  ['Adjective', 'ADJ'],
  ['Adverb', 'ADV'],
  ['Noun', 'NOUN'],
  ['Pronoun', 'PRON'],
  ['Determiner', 'DET'],
  ['Preposition', 'ADP'],
  ['Conjunction', 'CCONJ'],
  ['Value', 'NUM'],
];

function toUpos(tags) {
  const set = new Set(tags);
  // Auxiliaries and copulas are tagged Verb by compromise but AUX by spaCy.
  if (set.has('Auxiliary') || set.has('Copula')) return 'AUX';
  if (set.has('ProperNoun')) return 'PROPN';
  for (const [tag, upos] of PRIORITY) if (set.has(tag)) return upos;
  return 'X';
}

let input = '';
process.stdin.on('data', (d) => (input += d));
process.stdin.on('end', () => {
  const { sentences } = JSON.parse(input);
  const out = sentences.map((s) => {
    const doc = nlp(s);
    const terms = doc.json({ terms: { tags: true } }).flatMap((x) => x.terms);
    return terms.map((t) => [
      (t.text || '').replace(/^[^\w]+|[^\w]+$/g, ''),
      toUpos(t.tags || []),
    ]).filter(([text]) => text.length > 0);
  });
  process.stdout.write(JSON.stringify(out));
});
