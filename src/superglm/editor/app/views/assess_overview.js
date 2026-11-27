// @ts-check

import { isReportName } from "../api/contracts.js";
import { escapeHTML } from "../format.js";
import { ASSESS_GROUPS } from "../state/assess.js";

/** @typedef {import('../api/contracts.js').ReportName} ReportName */
/** @typedef {import('../api/contracts.js').EvidenceStatus} EvidenceStatus */
/** @typedef {import('../api/contracts.js').EvidenceState} EvidenceState */
/**
 * One overview card: its report, the report's name, the headline number, the
 * line under it, and the state of the evidence the number comes from.
 * @typedef {{id: ReportName, title: string, value: string, note: string, freshness: EvidenceStatus}} CardModel
 */
/** @typedef {{name: string, cards: CardModel[]}} CardGroup */

/** @param {CardModel} card */
function cardMarkup(card) {
  return `<button class="assess-card" type="button" data-card="${card.id}" data-freshness="${card.freshness}">` +
    `<span class="assess-card-title">${escapeHTML(card.title)}</span>` +
    `<span class="assess-card-value">${escapeHTML(card.value)}</span>` +
    `<span class="assess-card-note">${escapeHTML(card.note)}</span></button>`;
}

/**
 * The overview's cards, a section per group under its heading.
 * @param {CardGroup[]} groups
 * @returns {string}
 */
export function overviewMarkup(groups) {
  return groups.map((group) =>
    '<section class="assess-card-group">' +
    `<h3 class="assess-section-title">${escapeHTML(group.name)}</h3>` +
    `<div class="assess-card-grid">${group.cards.map(cardMarkup).join("")}</div>` +
    "</section>").join("");
}

/**
 * A card per Assess entry, in list order. Each names its report and says what
 * the report holds; its value stays empty here.
 * @param {Record<ReportName, EvidenceState>} evidence
 * @returns {CardGroup[]}
 */
export function overviewGroups(evidence) {
  return ASSESS_GROUPS.map((group) => ({
    name: group.name,
    cards: group.entries.map((entry) => ({
      id: entry.id,
      title: entry.label,
      value: "",
      note: entry.hint,
      freshness: evidence[entry.id].status,
    })),
  }));
}

/**
 * Wire the cards once: a click opens the card's report.
 * @param {HTMLElement} root
 * @param {{onOpen: (id: ReportName) => unknown}} handlers
 * @returns {{destroy(): void}}
 */
export function bindOverview(root, { onOpen }) {
  /** @param {MouseEvent} event */
  function onClick(event) {
    const card = event.target instanceof Element ? event.target.closest("[data-card]") : null;
    const id = card instanceof HTMLElement ? card.dataset.card : undefined;
    if (isReportName(id)) onOpen(id);
  }

  root.addEventListener("click", onClick);
  return Object.freeze({
    destroy() {
      root.removeEventListener("click", onClick);
    },
  });
}
