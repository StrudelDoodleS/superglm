// @ts-check

import { isCheckReport, isReportName, isReportView } from "../api/contracts.js";
import { escapeHTML } from "../format.js";
import { countText, deltaText, metricColumns, metricText } from "../reports.js";
import { ASSESS_GROUPS } from "../state/assess.js";
import { checkFace, checkPayload, checkThumbnail } from "./assess_check.js";
import { cvSourceLine, meanSpreadText } from "./cv_tab.js";
import { verdictParts } from "./diagnose_tab.js";

/** @typedef {import('../api/contracts.js').ReportName} ReportName */
/** @typedef {import('../api/contracts.js').ReportView} ReportView */
/** @typedef {import('../api/contracts.js').CheckReport} CheckReport */
/** @typedef {import('../api/contracts.js').EvidenceStatus} EvidenceStatus */
/** @typedef {import('../api/contracts.js').EvidenceState} EvidenceState */
/**
 * One overview card: its report, the report's name, the headline number, the
 * line under it, and the state of the evidence the number comes from. A number
 * from an earlier report carries a `marker` saying so, in words: shown where the
 * update failed, and read only by a screen reader while it is asked again. A check's
 * card adds its thumbnail's markup (empty for a report) and whether it is flagged.
 * @typedef {{id: ReportName, title: string, value: string, note: string, freshness: EvidenceStatus,
 *   marker: string, markerShown: boolean, thumbnail: string, flagged: boolean}} CardModel
 */
/** @typedef {{name: string, cards: CardModel[]}} CardGroup */
/** @typedef {import('../api/contracts.js').CVReportPayload} CVReportPayload */
/** @typedef {import('./diagnose_tab.js').DiagnosePayload} DiagnosePayload */
/** @typedef {import('../state/assess.js').AssessEntry} AssessEntry */
/** What the overview draws: the checks the model offers, and whether to draw the fit beside an edit. @typedef {{offered: readonly CheckReport[], compare: boolean}} OverviewOptions */
/**
 * The parts of a validation report a card reads.
 * @typedef {{name?: string, n_obs?: number, available?: boolean, error?: string,
 *   metrics?: {edited?: Record<string, number|null>, delta?: Record<string, number|null>}}} SplitScores
 */
/**
 * @typedef {{available: boolean, note?: string, metric_labels?: Record<string, string>,
 *   compares_edit?: boolean, splits?: SplitScores[]}} ValidationPayload
 */
/** A card's headline number, the line under it and its flag. @typedef {{value: string, note: string, flagged?: boolean}} CardFace */

const MISSING = "—";
const EARLIER = "Earlier result";
/** @type {OverviewOptions} */
const NO_CHECKS = Object.freeze({ offered: Object.freeze([]), compare: false });
/** The thumbnail each check card's plot holds now, so a redraw writes only one that moved. */
/** @type {WeakMap<Element, string>} */
const DRAWN_PLOTS = new WeakMap();

/**
 * The report's first score on the validation split, else on the first split it scored.
 * A validation split the model could not score shows the report's reason instead.
 * @param {ValidationPayload} payload
 * @returns {CardFace}
 */
export function validationCard(payload) {
  const note = payload.note ?? "";
  const labels = payload.metric_labels ?? {};
  const splits = payload.splits ?? [];
  const metric = metricColumns(labels, splits)[0];
  const split = splits.find((candidate) => candidate.name === "validation")
    ?? splits.find((candidate) => candidate.available !== false && candidate.metrics);
  // A refused split says why before anything asks which score to show: with no other split
  // scored, the report may name no score column at all.
  if (payload.available && split?.available === false) return { value: MISSING, note: split.error || note };
  if (!payload.available || !split || metric === undefined) return { value: MISSING, note };
  const value = `${labels[metric] ?? metric} ${metricText(metric, split.metrics?.edited?.[metric])}`;
  if (split.name !== "validation") return { value, note };
  const delta = payload.compares_edit ? ` · Δ ${deltaText(metric, split.metrics?.delta?.[metric])}` : "";
  return { value, note: `Validation, ${countText(split.n_obs)} rows${delta}` };
}

/**
 * The newest cross-validation result's first score, its mean and spread across the folds.
 * @param {CVReportPayload} payload
 * @returns {CardFace}
 */
export function cvCard(payload) {
  const run = payload.results.at(-1);
  const metric = payload.metrics[0];
  if (!run || !metric) return { value: MISSING, note: cvSourceLine(payload) };
  return {
    value: `${metric.label} ${meanSpreadText(run.mean[metric.name], run.std[metric.name])}`,
    note: `${run.stale ? `${run.label}, earlier model` : run.label} · ${run.folds.length} folds`,
  };
}

/**
 * The rows a final fit refitted on, or that none has run: never a verdict on the refit.
 * The note is the report's own `final_fit.note`, which names the rows a book without
 * validation data refits on; a refusal's reason, and a final fit the model has moved
 * past, say that instead.
 * @param {CVReportPayload} cvPayload
 * @returns {CardFace}
 */
export function finalCard(cvPayload) {
  const final = cvPayload.final_fit;
  if (final.done) {
    return {
      value: `${countText(final.n_rows)} rows`,
      note: final.stale
        ? "The model changed after the final fit."
        : final.note || "Refitted on train and validation rows.",
    };
  }
  return { value: "Not run", note: final.reason ?? (final.note || "Run Final fit on all rows from Cross-validation.") };
}

/**
 * How many fits' smoothing searches converged; a lone fit, or fits with no search, say
 * their own status.
 * @param {DiagnosePayload} payload
 * @returns {CardFace}
 */
export function smoothingCard(payload) {
  const { fits } = payload;
  const n = fits.length;
  const failed = fits.filter((fit) => fit.status === "not_converged").length;
  const converged = fits.filter((fit) => fit.status === "converged" || fit.status === "loose").length;
  let value = MISSING;
  if (n === 1 || (n > 0 && failed + converged === 0)) value = verdictParts(fits[0].verdict).status;
  else if (failed > 0) value = `${failed} of ${n} fits did not converge`;
  else if (n > 0) value = `${converged} of ${n} fits converged`;
  return { value, note: payload.note };
}

/**
 * The face of card `id` from the payload of the report it reads.
 * @param {ReportView} id @param {unknown} payload
 * @returns {CardFace}
 */
function cardFace(id, payload) {
  switch (id) {
    case "validation":
      return validationCard(/** @type {ValidationPayload} */ (payload));
    case "cv":
      return cvCard(/** @type {CVReportPayload} */ (payload));
    case "final":
      return finalCard(/** @type {CVReportPayload} */ (payload));
    case "diagnose":
      return smoothingCard(/** @type {DiagnosePayload} */ (payload));
  }
}

/**
 * An entry's card. Final fit reads the cv report, whose `final_fit` says whether one ran; a
 * check reads its own payload and draws its thumbnail.
 * @param {Readonly<AssessEntry>} entry @param {Record<ReportName, EvidenceState>} evidence
 * @param {boolean} compare
 * @returns {CardModel}
 */
function overviewCard(entry, evidence, compare) {
  const { id } = entry;
  const { status, payload, error } = evidence[id === "final" ? "cv" : id];
  const card = { id, title: entry.label, freshness: status, marker: "", markerShown: false, thumbnail: "", flagged: false };
  /** @type {CardFace} */
  let face;
  if (isCheckReport(id)) {
    const check = checkPayload(evidence[id]);
    card.thumbnail = checkThumbnail(id, check, compare);
    face = checkFace(id, check, status);
    if (check === null) return { ...card, ...face };
  } else if (payload === null) {
    return { ...card, value: MISSING, note: status === "error" ? "Report unavailable." : "Loading report..." };
  } else {
    face = cardFace(/** @type {ReportView} */ (id), payload);
  }
  // A failed update keeps the earlier report on screen, as the report does, and says why.
  if (error) {
    return { ...card, ...face, note: `Update failed. ${error}`, marker: EARLIER, markerShown: true };
  }
  if (status === "updating") return { ...card, ...face, marker: `${EARLIER}, updating` };
  if (status === "stale") return { ...card, ...face, marker: `${EARLIER}, may be out of date` };
  return { ...card, ...face };
}

/**
 * The entries of `group` with a card: every report view, and each check the model offers.
 * @param {Readonly<{entries: readonly AssessEntry[]}>} group @param {readonly CheckReport[]} offered
 * @returns {AssessEntry[]}
 */
function carded(group, offered) {
  return group.entries.filter((entry) => isReportView(entry.id) || (isCheckReport(entry.id) && offered.includes(entry.id)));
}

/**
 * A card per Assess report, in list order; a check's when the model offers it.
 * @param {Record<ReportName, EvidenceState>} evidence
 * @param {OverviewOptions} [options]
 * @returns {CardModel[]}
 */
export function overviewCards(evidence, { offered, compare } = NO_CHECKS) {
  return ASSESS_GROUPS.flatMap((group) => carded(group, offered).map((entry) => overviewCard(entry, evidence, compare)));
}

/** @param {CardModel} card */
function cardMarkup(card) {
  const plot = card.thumbnail ? `<span class="assess-card-plot" aria-hidden="true">${card.thumbnail}</span>` : "";
  return `<button class="assess-card" type="button" data-card="${card.id}" data-freshness="${card.freshness}"` +
    ` data-flagged="${card.flagged}">` +
    `<span class="assess-card-title"><span class="assess-flag" aria-hidden="true"></span>${escapeHTML(card.title)}</span>` +
    plot +
    `<span class="assess-card-value">${escapeHTML(card.value)}</span>` +
    `<span class="assess-card-marker" data-shown="${card.markerShown}">${escapeHTML(card.marker)}</span>` +
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
 * Draw the cards into `root`: built when the set of cards changes, else each card's value,
 * marker, note, freshness, flag and thumbnail written in place, so a card being pressed or
 * holding focus stays the same element while another report lands.
 * @param {HTMLElement} root
 * @param {CardGroup[]} groups
 */
export function drawOverview(root, groups) {
  const cards = groups.flatMap((group) => group.cards);
  const drawn = Array.from(root.querySelectorAll("[data-card]"), (node) => node instanceof HTMLElement ? node.dataset.card : "");
  if (drawn.join() !== cards.map((card) => card.id).join()) {
    root.innerHTML = overviewMarkup(groups);
    for (const plot of root.querySelectorAll(".assess-card-plot")) DRAWN_PLOTS.set(plot, plot.innerHTML);
    return;
  }
  for (const card of cards) {
    const node = root.querySelector(`[data-card="${card.id}"]`);
    if (!(node instanceof HTMLElement)) continue;
    node.dataset.freshness = card.freshness;
    node.dataset.flagged = String(card.flagged);
    const plot = node.querySelector(".assess-card-plot");
    if (plot && DRAWN_PLOTS.get(plot) !== card.thumbnail) {
      plot.innerHTML = card.thumbnail;
      DRAWN_PLOTS.set(plot, card.thumbnail);
    }
    const value = node.querySelector(".assess-card-value");
    const marker = node.querySelector(".assess-card-marker");
    const note = node.querySelector(".assess-card-note");
    if (value) value.textContent = card.value;
    if (marker instanceof HTMLElement) {
      marker.textContent = card.marker;
      marker.dataset.shown = String(card.markerShown);
    }
    if (note) note.textContent = card.note;
  }
}

/**
 * The cards under their group headings; a group with no card the model offers is left out.
 * @param {Record<ReportName, EvidenceState>} evidence
 * @param {OverviewOptions} [options]
 * @returns {CardGroup[]}
 */
export function overviewGroups(evidence, { offered, compare } = NO_CHECKS) {
  return ASSESS_GROUPS.map((group) => ({
    name: group.name,
    cards: carded(group, offered).map((entry) => overviewCard(entry, evidence, compare)),
  })).filter((group) => group.cards.length > 0);
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
