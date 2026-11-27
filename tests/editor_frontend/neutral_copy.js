// @ts-check

/**
 * The words the editor never shows: it works for any data, so no text it draws may speak
 * the language of one trade. Matched whole, case-insensitively, with a plural ending.
 */
export const BANNED_WORDS = Object.freeze([
  "claim", "exposure", "premium", "policy", "frequency", "severity", "loss", "losses",
  "pricing", "priced", "book", "burning cost", "underpriced", "insurance", "actuarial",
]);

/** @param {string} word @returns {RegExp} the word whole, with `s`, `es` or (for `-y`) `ies` */
function wordPattern(word) {
  const stem = word.replace(/\s+/g, "\\s+");
  const plural = word.endsWith("y") ? `(?:${stem}|${stem.slice(0, -1)}ies)` : `${stem}(?:s|es)?`;
  return new RegExp(`\\b${plural}\\b`, "i");
}

const PATTERNS = BANNED_WORDS.map((word) => /** @type {const} */ ([word, wordPattern(word)]));

/**
 * The banned words `text` contains, in list order; empty when the text is neutral.
 * @param {string} text
 * @returns {string[]}
 */
export function neutralCopy(text) {
  return PATTERNS.filter(([, pattern]) => pattern.test(text)).map(([word]) => word);
}
