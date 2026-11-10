// @ts-check

import { escapeHTML, fmt } from "../format.js";
import { defineMachine } from "../state/machine.js";
import { clockText, foldColour } from "./cv_tab.js";

/**
 * @typedef {object} DiagnoseCell
 * @property {number} edf
 * @property {number|null} lambda
 * @property {"curve"|"straight"|"simplest"|"flat"|"unsmoothed"|"unrecorded"} shape
 * @property {boolean} at_limit
 */
/**
 * @typedef {object} DiagnoseRow
 * @property {number} row
 * @property {number|null} cost
 * @property {number} response
 * @property {number} predicted
 * @property {number|null} weight
 * @property {(string|number|boolean|null)[]} values
 */
/**
 * @typedef {object} DiagnoseFit
 * @property {string} id
 * @property {string} label
 * @property {number|null} rows
 * @property {number} rounds
 * @property {string|null} stop
 * @property {string} stop_text
 * @property {"converged"|"loose"|"not_converged"|"fixed"|"edited"} status
 * @property {string} verdict one plain sentence on the status; its first sentence is the status itself
 * @property {string|null} explanation the full statement of a fit that did not converge, with what to try
 * @property {string|null} search which smoothing search produced the fit, with its option; null where not recorded
 * @property {boolean} converged
 * @property {number|null} seconds
 * @property {number[]} path
 * @property {number|null} late_gain
 * @property {number|null} settled_round
 * @property {Record<string, DiagnoseCell>} terms
 * @property {{name:string, seconds:number, share:number}[]} phases
 * @property {{severity:string, headline:string, observed:string, question:string|null, affects_numbers:boolean}[]} findings
 * @property {{columns: string[], rows: DiagnoseRow[]}} [worst_rows]
 */
/**
 * @typedef {object} DiagnosePayload
 * @property {"diagnose"} report
 * @property {string} title
 * @property {string} note
 * @property {DiagnoseFit[]} fits
 * @property {{name:string, label:string, short:string, group:string|null, role:string|null}[]} terms
 * @property {string} cost_label
 */

// A round that gains less REML than this reads as settled on the chart.
const PATH_FLOOR = 1e-3;
const CHART = Object.freeze({ width: 1100, left: 62, right: 16, top: 12, bottom: 38 });
// A phase that took less than this is not worth a row.
const MIN_PHASE_SECONDS = 0.005;
const SHAPE_WORDS = Object.freeze({
  curve: "shaped by the data",
  straight: "smoothed to a straight line",
  simplest: "flattened to the simplest shape its penalty allows",
  flat: "smoothed away",
  unsmoothed: "no smoothing",
  unrecorded: "smoothing not kept (this fit used retain_fit_state=False)",
});
const STATUS_ICON = Object.freeze({ converged: "✓", loose: "✓", not_converged: "✕", fixed: "–", edited: "✎" });

/**
 * A folded section of the page: shut until asked, and shut again when asked. The tab keeps one
 * of these per foldable section, and `moveSection` is the only writer.
 * @typedef {"folded"|"open"} SectionState
 * @typedef {Record<"search"|"phases", SectionState>} Sections
 */
export const SECTION = defineMachine({
  name: "diagnose section",
  initial: /** @type {SectionState} */ ("folded"),
  moves: { folded: { open: "open" }, open: { fold: "folded" } },
});
/** @type {Readonly<Sections>} */
export const FOLDED = Object.freeze({ search: SECTION.initial, phases: SECTION.initial });

/**
 * Open a folded section or fold an open one.
 * @param {Sections} sections @param {keyof Sections} name
 * @returns {Sections}
 */
export function moveSection(sections, name) {
  const next = SECTION.next(sections[name], sections[name] === "folded" ? "open" : "fold");
  return next === null ? sections : { ...sections, [name]: next };
}

/** @param {number|null|undefined} value @param {number} digits */
function fixed(value, digits) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "--";
}

/** @param {string} title @param {string} [hintHTML] */
function head(title, hintHTML = "") {
  return `<div class="cv-section-head diagnose-head"><h3>${escapeHTML(title)}</h3>${hintHTML ? `<span class="cv-hint">${hintHTML}</span>` : ""}</div>`;
}

/** @param {string} title @param {string} hint */
function textHead(title, hint) {
  return head(title, escapeHTML(hint));
}

/** The fold's own colour, as on the Cross-validation tab, whichever folds failed. @param {DiagnoseFit} fit */
function fitColour(fit) {
  return fit.id === "model" ? "var(--text)" : foldColour(Number(fit.id.slice("fold-".length)) - 1);
}

/**
 * The fits whose search took at least twice the middle fold's rounds. A short
 * search is not worth a mark, so it takes ten rounds as well.
 * @param {DiagnoseFit[]} fits
 * @returns {Set<string>}
 */
export function slowFits(fits) {
  const folds = fits.filter((fit) => fit.id !== "model");
  const rounds = folds.map((fit) => fit.rounds).sort((a, b) => a - b);
  if (rounds.length < 3) return new Set();
  const middle = rounds[Math.floor(rounds.length / 2)];
  return new Set(folds.filter((fit) => fit.rounds >= 10 && fit.rounds >= 2 * middle).map((fit) => fit.id));
}

/** @param {DiagnoseFit} fit @param {string} selected */
function pressed(fit, selected) {
  return fit.id === selected ? "true" : "false";
}

/** A fit's button, which shows that fit below. @param {DiagnoseFit} fit @param {string} selected */
function fitButton(fit, selected) {
  return `<button type="button" class="diagnose-fit" data-diagnose-fit="${escapeHTML(fit.id)}" aria-pressed="${pressed(fit, selected)}">
        <span class="diagnose-swatch" style="background: ${fitColour(fit)}"></span>${escapeHTML(fit.label)}</button>`;
}

/** A fit named inside a sentence, as a link that shows it. @param {DiagnoseFit} fit */
function fitLink(fit) {
  return `<button type="button" class="diagnose-link" data-diagnose-fit="${escapeHTML(fit.id)}">${escapeHTML(fit.label)}</button>`;
}

/**
 * The verdict sentence split into its status (the first sentence) and what follows.
 * @param {string} verdict
 */
function verdictParts(verdict) {
  const end = verdict.indexOf(". ");
  return end < 0 ? { status: verdict, rest: "" } : { status: verdict.slice(0, end + 1), rest: verdict.slice(end + 2) };
}

/**
 * What several converged fits say beyond the word "Converged": the sentence
 * that follows it, when at least two fits say the same one. A row then shows
 * only the word, and the sentence is said once above the table.
 * @param {DiagnoseFit[]} fits
 * @returns {string} empty when no sentence is shared
 */
export function sharedConvergedSentence(fits) {
  /** @type {Map<string, number>} */
  const counts = new Map();
  for (const fit of fits) {
    if (fit.status !== "converged") continue;
    const { rest } = verdictParts(fit.verdict);
    if (rest) counts.set(rest, (counts.get(rest) ?? 0) + 1);
  }
  const [best] = [...counts].sort((a, b) => b[1] - a[1]);
  return best && best[1] >= 2 ? best[0] : "";
}

/** @param {DiagnoseFit[]} fits */
function allFixed(fits) {
  return fits.every((fit) => fit.status === "fixed");
}

/**
 * One line across the fits: how many did not converge, and which.
 * @param {DiagnoseFit[]} fits
 * @returns {string} HTML
 */
export function summaryMarkup(fits) {
  if (allFixed(fits)) return `${escapeHTML(fits[0].verdict)} Each term's flexibility is below.`;
  if (fits.length < 2) return "";
  const failed = fits.filter((fit) => fit.status === "not_converged");
  const loose = fits.filter((fit) => fit.status === "loose");
  const names = (/** @type {DiagnoseFit[]} */ list) => list.map(fitLink).join(", ");
  const looseText = loose.length ? ` ${loose.length} converged with a looser stop: ${names(loose)}.` : "";
  const shared = sharedConvergedSentence(fits);
  if (failed.length) {
    const rest = shared ? ` For the fits that converged: ${escapeHTML(shared.charAt(0).toLowerCase() + shared.slice(1))}` : "";
    return `<strong>${failed.length} of ${fits.length} fits did not converge:</strong> ${names(failed)}.${rest}${looseText}`;
  }
  return `All ${fits.length} fits converged.${shared ? ` ${escapeHTML(shared)}` : ""}${looseText}`;
}

/**
 * The fits, one row each, the verdict first; a row's button shows that fit below.
 * @param {DiagnosePayload} payload @param {string} selected
 */
export function fitsTableMarkup(payload, selected) {
  const fits = payload.fits;
  const slow = slowFits(fits);
  const fixedOnly = allFixed(fits);
  const searches = new Set(fits.map((fit) => fit.search ?? ""));
  const searchColumn = !fixedOnly && searches.size > 1;
  const timed = fits.some((fit) => fit.seconds !== null);
  const shared = sharedConvergedSentence(fits);
  const rows = fits.map((fit) => {
    const parts = verdictParts(fit.verdict);
    const { status } = parts;
    // A sentence the summary above already gives for the converged fits is not repeated on each row.
    const rest = fit.status === "converged" && parts.rest === shared ? "" : parts.rest;
    const reason = fit.status === "not_converged" || fit.status === "loose"
      ? `<span class="diagnose-reason">Stopped because ${escapeHTML(fit.stop_text.replace(/\.$/, ""))}.</span>` : "";
    const result = fixedOnly ? "" : `<td class="diagnose-result"><span class="diagnose-icon" aria-hidden="true">${STATUS_ICON[fit.status]}</span>
        <strong>${escapeHTML(status)}</strong> ${escapeHTML(rest)}${reason}</td>`;
    const search = searchColumn ? `<td class="diagnose-text diagnose-search">${escapeHTML(fit.search ?? "")}</td>` : "";
    const rounds = fixedOnly ? "" : `<td${fit.settled_round === null ? "" : ` title="Settled after round ${fit.settled_round}"`}>${fit.rounds}${slow.has(fit.id) ? ' <span class="diagnose-slow">slow</span>' : ""}</td>`;
    return `<tr data-selected="${pressed(fit, selected)}" data-status="${fit.status}">
      <td>${fitButton(fit, selected)}</td>
      ${result}${search}${rounds}
      <td>${fit.rows === null ? "--" : fit.rows.toLocaleString("en")}</td>
      ${timed ? `<td>${fit.seconds === null ? "--" : `${fit.seconds.toFixed(1)} s`}</td>` : ""}
    </tr>`;
  }).join("");
  const columns = [
    "<th>Fit</th>",
    fixedOnly ? "" : '<th class="diagnose-text diagnose-result">Result</th>',
    searchColumn ? '<th class="diagnose-text diagnose-search">Search</th>' : "",
    fixedOnly ? "" : "<th>Smoothing rounds</th>",
    "<th>Rows</th>",
    timed ? "<th>Fit time</th>" : "",
  ].join("");
  return `<section class="report-section">
    ${head("Fits", summaryMarkup(fits))}
    <table class="report-table diagnose-table diagnose-fits" data-plain="${fixedOnly}" aria-label="Fits">
      <thead><tr>${columns}</tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </section>`;
}

/** @param {DiagnoseCell|undefined} cell */
function cellMarkup(cell) {
  if (!cell) return '<td class="diagnose-cell">--</td>';
  /** @type {string[]} */
  const words = [SHAPE_WORDS[cell.shape]];
  if (cell.lambda !== null) words.push(`smoothing ${cell.lambda.toPrecision(3)}`);
  if (cell.at_limit) words.push("at the largest smoothing allowed");
  const mark = cell.at_limit ? '<span class="diagnose-limit" aria-label="at the largest smoothing allowed" role="img"></span>' : "";
  return `<td class="diagnose-cell" data-shape="${cell.shape}" title="${escapeHTML(words.join(", "))}">${cell.edf.toFixed(2)}${mark}</td>`;
}

/**
 * Every term across the fits, as EDF: a fold that left a term straight when
 * the others curve it stands out in its own column.
 * @param {DiagnosePayload} payload @param {string} selected
 */
export function termMatrixMarkup(payload, selected) {
  const columns = payload.fits.map((fit) => `<th data-selected="${pressed(fit, selected)}">${escapeHTML(fit.label)}</th>`).join("");
  const span = payload.fits.length + 1;
  /** @type {string|null} */
  let group = null;
  const rows = payload.terms.map((term) => {
    const heading = term.group && term.group !== group
      ? `<tr class="diagnose-group"><th colspan="${span}" data-role="${term.role ?? ""}">${escapeHTML(term.group)}</th></tr>`
      : "";
    group = term.group;
    const cells = payload.fits.map((fit) => cellMarkup(fit.terms[term.name])).join("");
    return `${heading}<tr><th scope="row" class="diagnose-term">${escapeHTML(term.short)}</th>${cells}</tr>`;
  }).join("");
  return `<section class="report-section">
    ${head("Smoothing by term")}
    <div class="diagnose-scroll"><table class="report-table diagnose-table diagnose-matrix" aria-label="Smoothing by term">
      <thead><tr><th class="diagnose-term">Term</th>${columns}</tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <p class="cv-hint diagnose-keyline">Numbers are EDF (effective degrees of freedom, how flexible a term is). Shaded: a straight line or smoothed away. Dot: at the largest smoothing allowed.</p>
  </section>`;
}

/**
 * The REML score a search gained in each round, from the round before it.
 * @param {number[]} path
 */
function roundGains(path) {
  return path.slice(1).map((value, round) => Math.max(path[round] - value, 0));
}

/**
 * How much REML score each fit gained in each smoothing round, on one log axis. A
 * search that settled ends on the floor; one that stopped while still gaining ends
 * high, and its last point is crossed.
 * @param {DiagnosePayload} payload @param {string} selected @param {number} [width]
 */
export function searchPathMarkup(payload, selected, width = CHART.width) {
  const traced = payload.fits.filter((fit) => fit.path.length >= 2);
  const unrecorded = payload.fits.filter((fit) => fit.rounds > 0 && fit.path.length < 2).map((fit) => fit.label);
  const shown = payload.fits.find((fit) => fit.id === selected);
  const searches = new Set(payload.fits.map((fit) => fit.search ?? ""));
  const notes = [
    searches.size === 1 && shown?.search ? `${shown.search}.` : "",
    shown && shown.rounds > 0 ? `${shown.label} stopped because ${shown.stop_text.replace(/\.$/, "")}.` : "",
    unrecorded.length ? `Not recorded by this fit's search: ${unrecorded.join(", ")}.` : "",
  ].filter(Boolean).join(" ");
  const heading = textHead("Search path", notes);
  if (!traced.length) return unrecorded.length ? `<section class="report-section">${heading}</section>` : "";
  const { left, right, top, bottom } = CHART;
  const height = Math.round(Math.min(300, Math.max(180, width * 0.18)));
  const gains = new Map(traced.map((fit) => [fit.id, roundGains(fit.path)]));
  const rounds = Math.max(...[...gains.values()].map((list) => list.length));
  const largest = Math.max(...[...gains.values()].flat());
  const decades = Math.max(1, Math.ceil(Math.log10(Math.max(largest, PATH_FLOOR * 10))));
  const lowest = Math.log10(PATH_FLOOR);
  const inset = 8;
  const x = (/** @type {number} */ round) => left + inset + ((round - 1) / Math.max(rounds - 1, 1)) * (width - left - right - 2 * inset);
  const y = (/** @type {number} */ gain) => {
    const place = (Math.log10(Math.max(gain, PATH_FLOOR)) - lowest) / (decades - lowest);
    return top + (1 - place) * (height - top - bottom);
  };
  const ticks = [];
  for (let power = lowest; power <= decades; power += 1) ticks.push(10 ** power);
  const grid = ticks.map((tick) => `<line class="diagnose-grid" x1="${left}" x2="${width - right}" y1="${y(tick).toFixed(1)}" y2="${y(tick).toFixed(1)}"></line>
    <text class="cv-tick" x="${left - 6}" y="${(y(tick) + 4).toFixed(1)}" text-anchor="end">${escapeHTML(fmt(tick))}</text>`).join("");
  const steps = Array.from({ length: Math.min(rounds, 5) }, (_, part) => 1 + Math.round((part * (rounds - 1)) / Math.max(Math.min(rounds, 5) - 1, 1)));
  const xTicks = [...new Set(steps)].map((round) => `<text class="cv-tick" x="${x(round).toFixed(1)}" y="${height - bottom + 16}" text-anchor="middle">${round}</text>`).join("");
  // The fit in view draws last, so nothing crosses over it.
  const ordered = [...traced].sort((a, b) => Number(a.id === selected) - Number(b.id === selected));
  const lines = ordered.map((fit) => {
    const list = /** @type {number[]} */ (gains.get(fit.id));
    const points = list.map((gain, index) => [x(index + 1), y(gain)]);
    const d = points.map(([px, py], index) => `${index === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`).join(" ");
    const [ex, ey] = points[points.length - 1];
    const unfinished = fit.status === "not_converged";
    const cross = 4;
    const end = unfinished
      ? `<path class="diagnose-end" data-selected="${pressed(fit, selected)}" d="M${(ex - cross).toFixed(1)},${(ey - cross).toFixed(1)}L${(ex + cross).toFixed(1)},${(ey + cross).toFixed(1)}M${(ex - cross).toFixed(1)},${(ey + cross).toFixed(1)}L${(ex + cross).toFixed(1)},${(ey - cross).toFixed(1)}" style="stroke: ${fitColour(fit)}"></path>`
      : list.length === 1 ? `<circle class="diagnose-end" data-selected="${pressed(fit, selected)}" cx="${ex.toFixed(1)}" cy="${ey.toFixed(1)}" r="3" style="fill: ${fitColour(fit)}"></circle>` : "";
    const title = unfinished
      ? `${fit.label}: stopped before it converged (its last round gained ${fmt(list[list.length - 1])})`
      : `${fit.label}: ${fit.rounds} smoothing rounds`;
    return `<path class="diagnose-line" data-selected="${pressed(fit, selected)}" d="${d}" style="stroke: ${fitColour(fit)}"><title>${escapeHTML(title)}</title></path>${end}`;
  }).join("");
  const axes = `<text class="cv-axis-title" x="${left + (width - left - right) / 2}" y="${height - 4}" text-anchor="middle">Smoothing round</text>
    <text class="cv-axis-title" transform="translate(12 ${top + (height - top - bottom) / 2}) rotate(-90)" text-anchor="middle">REML score gained per round</text>`;
  return `<section class="report-section">
    ${heading}
    <svg class="diagnose-chart" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="REML score gained in each smoothing round, by fit. A search that settled ends at the bottom; a crossed end is a search that stopped while still gaining.">${grid}${xTicks}${lines}${axes}</svg>
  </section>`;
}

/** @param {number} seconds */
function clockSeconds(seconds) {
  return seconds >= 60 ? clockText(seconds) : `${seconds.toFixed(2)} s`;
}

/**
 * A button that opens or folds a section, with its state said to assistive technology.
 * @param {keyof Sections} name @param {string} text @param {Sections} sections
 */
function foldButton(name, text, sections) {
  const open = sections[name] === "open";
  return `<button type="button" class="diagnose-fold" data-diagnose-fold="${name}" aria-expanded="${open}">${escapeHTML(text)}</button>`;
}

/** @param {DiagnoseFit} fit @param {Sections} [sections] */
export function phasesMarkup(fit, sections = FOLDED) {
  const phases = fit.phases.filter((phase) => phase.seconds >= MIN_PHASE_SECONDS);
  if (!phases.length) return "";
  const button = `<div class="cv-section-head diagnose-head">${foldButton("phases", "Where the time went", sections)}<span class="cv-hint">${escapeHTML(fit.label)}</span></div>`;
  if (sections.phases === "folded") return `<section class="report-section">${button}</section>`;
  const widest = Math.max(...phases.map((phase) => phase.share));
  const rows = phases.map((phase) => `<div class="diagnose-phase">
      <span>${escapeHTML(phase.name)}</span>
      <span class="diagnose-bar"><i style="width: ${((phase.share / widest) * 100).toFixed(1)}%"></i></span>
      <span>${clockSeconds(phase.seconds)}</span><span>${(phase.share * 100).toFixed(0)}%</span>
    </div>`).join("");
  return `<section class="report-section">${button}<div class="diagnose-phases">${rows}</div></section>`;
}

/** @param {DiagnoseFit["findings"]} found @param {boolean} withQuestion */
function findingItems(found, withQuestion) {
  const items = found.map((item) => `<li data-severity="${escapeHTML(item.severity)}">
      <span class="diagnose-severity">${escapeHTML(item.severity)}</span>
      <div><strong>${escapeHTML(item.headline)}</strong>
        <p>${escapeHTML(item.observed)}</p>
        ${withQuestion && item.question ? `<p class="cv-hint">${escapeHTML(item.question)}</p>` : ""}</div>
    </li>`).join("");
  return `<ul class="diagnose-findings">${items}</ul>`;
}

/**
 * What the fit reports: the findings that bear on the numbers a reader uses, then,
 * folded, those about how the search went.
 * @param {DiagnoseFit} fit @param {Sections} [sections]
 */
export function findingsMarkup(fit, sections = FOLDED) {
  if (!fit.findings.length) return "";
  const numbers = fit.findings.filter((found) => found.affects_numbers);
  const search = fit.findings.filter((found) => !found.affects_numbers);
  const listed = numbers.length ? `<h4 class="diagnose-subhead">Affects your numbers</h4>${findingItems(numbers, true)}` : "";
  const searchBlock = search.length
    ? `<div class="diagnose-fold-row">${foldButton("search", `About the search (${search.length})`, sections)}</div>${sections.search === "open" ? findingItems(search, false) : ""}`
    : "";
  return `<section class="report-section">${textHead("What the fit reports", fit.label)}${listed}${searchBlock}</section>`;
}

/** @param {DiagnosePayload} payload @param {DiagnoseFit} fit */
export function worstRowsMarkup(payload, fit) {
  const whole = fit.id === "model";
  if (whole && !fit.worst_rows) return "";
  if (!fit.worst_rows) {
    return `<section class="report-section">${textHead("Rows that cost the most", "These folds' rows are not available here, so this fold's worst rows cannot be listed.")}</section>`;
  }
  const columns = fit.worst_rows.columns.map((name) => `<th>${escapeHTML(name)}</th>`).join("");
  const rows = fit.worst_rows.rows.map((row) => `<tr><td>${row.row}</td><td>${fixed(row.cost, 3)}</td><td>${escapeHTML(fmt(row.response))}</td>
      <td>${escapeHTML(fmt(row.predicted))}</td><td>${row.weight === null ? "--" : escapeHTML(fmt(row.weight))}</td>
      ${row.values.map((value) => `<td>${escapeHTML(typeof value === "number" ? fmt(value) : String(value))}</td>`).join("")}</tr>`).join("");
  return `<section class="report-section">
    ${textHead("Rows that cost the most", whole
      ? `The training rows this fit predicted worst, by ${payload.cost_label.toLowerCase()}. It was fitted on them, so a fold's held-out rows are the fairer test.`
      : `${fit.label}'s held-out rows that its fit predicted worst, by ${payload.cost_label.toLowerCase()}. Its fit never saw these.`)}
    <div class="diagnose-scroll"><table class="report-table diagnose-table" aria-label="Rows that cost the most">
      <thead><tr><th>Row</th><th>${escapeHTML(payload.cost_label)}</th><th>Response</th><th>Predicted</th><th>Weight</th>${columns}</tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
  </section>`;
}

/**
 * The fit in view: the one asked for while it exists, else the first fit that did
 * not converge, else the fold whose search took the most rounds, which is why the
 * tab is usually open, else this model when no fold searched.
 * @param {DiagnosePayload} payload @param {string} wanted
 */
export function chosenFit(payload, wanted) {
  const asked = payload.fits.find((fit) => fit.id === wanted);
  if (asked) return asked;
  const failed = payload.fits.find((fit) => fit.status === "not_converged");
  if (failed) return failed;
  const folds = payload.fits.filter((fit) => fit.id !== "model" && fit.rounds > 0);
  return folds.reduce((slowest, fit) => (fit.rounds > slowest.rounds ? fit : slowest), folds[0] ?? payload.fits[0]);
}

/**
 * What to try, for the fit in view when it did not converge.
 * @param {DiagnoseFit} fit
 */
function explanationMarkup(fit) {
  if (fit.status !== "not_converged" || !fit.explanation) return "";
  return `<section class="report-section">${head("What to try")}<p class="diagnose-explain"><strong>${escapeHTML(fit.label)}:</strong> ${escapeHTML(fit.explanation)}</p></section>`;
}

/**
 * @param {DiagnosePayload} payload @param {string} wanted
 * @param {Sections} [sections] @param {number} [width] the chart's width in pixels
 */
export function diagnoseTabMarkup(payload, wanted, sections = FOLDED, width = CHART.width) {
  const fit = chosenFit(payload, wanted);
  return [
    fitsTableMarkup(payload, fit.id),
    searchPathMarkup(payload, fit.id, width),
    explanationMarkup(fit),
    findingsMarkup(fit, sections),
    termMatrixMarkup(payload, fit.id),
    worstRowsMarkup(payload, fit),
    phasesMarkup(fit, sections),
  ].join("");
}

/**
 * @param {object} options
 * @param {HTMLElement} options.frame
 * @param {()=>boolean} [options.isShown]
 */
export function createDiagnoseTab({ frame, isShown = () => true }) {
  /** @type {DiagnosePayload|null} */
  let payload = null;
  let wanted = "";
  let sections = /** @type {Sections} */ (FOLDED);
  let drawnWidth = 0;

  function draw() {
    if (!payload || !isShown()) return;
    drawnWidth = frame.clientWidth;
    frame.innerHTML = diagnoseTabMarkup(payload, wanted, sections, drawnWidth || CHART.width);
  }

  /** @param {Event} event */
  function onClick(event) {
    const target = event.target instanceof Element ? event.target : null;
    const fold = target?.closest("[data-diagnose-fold]");
    if (fold instanceof HTMLElement) {
      sections = moveSection(sections, /** @type {keyof Sections} */ (fold.dataset.diagnoseFold));
      draw();
      return;
    }
    const button = target?.closest("[data-diagnose-fit]");
    if (!(button instanceof HTMLElement)) return;
    wanted = button.dataset.diagnoseFit ?? "";
    draw();
  }

  frame.addEventListener("click", onClick);
  // The chart takes the page's width, so it is drawn again when the page's width changes.
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => {
      if (frame.clientWidth && frame.clientWidth !== drawnWidth) draw();
    }).observe(frame);
  }

  return Object.freeze({
    /** @param {DiagnosePayload} next */
    render(next) {
      payload = next;
      draw();
    },
    /** Show this fit the next time the tab draws. @param {string} fitId */
    select(fitId) {
      wanted = fitId;
    },
  });
}
