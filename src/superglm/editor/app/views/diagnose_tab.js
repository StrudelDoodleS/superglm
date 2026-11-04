// @ts-check

import { escapeHTML, fmt } from "../format.js";
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
 * @property {boolean} converged
 * @property {number|null} seconds
 * @property {number[]} path
 * @property {number|null} late_gain
 * @property {number|null} settled_round
 * @property {Record<string, DiagnoseCell>} terms
 * @property {{name:string, seconds:number, share:number}[]} phases
 * @property {{severity:string, headline:string, observed:string, question:string|null}[]} findings
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

// REML this close to its final value reads as settled on the chart.
const PATH_FLOOR = 1e-3;
const CHART = Object.freeze({ width: 1100, height: 260, left: 56, right: 14, top: 14, bottom: 36 });
const SHAPE_WORDS = Object.freeze({
  curve: "shaped by the data",
  straight: "smoothed to a straight line",
  simplest: "smoothed to the simplest shape its penalty allows",
  flat: "smoothed away",
  unsmoothed: "no smoothing",
  unrecorded: "smoothing not kept: this fit used retain_fit_state=False",
});

/** @param {number|null|undefined} value @param {number} digits */
function fixed(value, digits) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "--";
}

/** @param {string} title @param {string} hint */
function head(title, hint) {
  return `<div class="cv-section-head"><h3>${escapeHTML(title)}</h3><span class="cv-hint">${escapeHTML(hint)}</span></div>`;
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

/**
 * The fits, one row each; a row's button shows that fit below.
 * @param {DiagnosePayload} payload @param {string} selected
 */
export function fitsTableMarkup(payload, selected) {
  const slow = slowFits(payload.fits);
  const rows = payload.fits.map((fit) => `<tr data-selected="${pressed(fit, selected)}">
      <td><button type="button" class="diagnose-fit" data-diagnose-fit="${escapeHTML(fit.id)}" aria-pressed="${pressed(fit, selected)}">
        <span class="diagnose-swatch" style="background: ${fitColour(fit)}"></span>${escapeHTML(fit.label)}</button></td>
      <td>${fit.rows === null ? "--" : fit.rows.toLocaleString("en")}</td>
      <td>${fit.rounds}${slow.has(fit.id) ? ' <span class="diagnose-slow">slow</span>' : ""}</td>
      <td>${fit.settled_round === null ? "--" : fit.settled_round}</td>
      <td>${fit.seconds === null ? "--" : `${fit.seconds.toFixed(1)} s`}</td>
      <td class="diagnose-text">${escapeHTML(fit.stop_text)}</td>
      <td>${fixed(fit.late_gain, 4)}</td>
      <td data-converged="${fit.converged}">${fit.converged ? "yes" : "no"}</td>
    </tr>`).join("");
  return `<section class="report-section">
    ${head("Fits", "Pick one to see its search, time and findings below.")}
    <table class="report-table diagnose-table" aria-label="Fits">
      <thead><tr><th>Fit</th><th>Rows</th><th>Rounds</th><th>Settled by round</th><th>Fit time</th>
        <th class="diagnose-text">Stopped because</th><th>REML gained in the last half</th><th>Converged</th></tr></thead>
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
  if (cell.at_limit) words.push("at the smoothing limit");
  const mark = cell.at_limit ? '<span class="diagnose-limit" aria-label="at the smoothing limit" role="img"></span>' : "";
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
    ${head("Smoothing by term", "Effective degrees of freedom. Tinted: smoothed to its simplest shape or away. Dot: at the smoothing limit. Hover for the smoothing.")}
    <div class="diagnose-scroll"><table class="report-table diagnose-table diagnose-matrix" aria-label="Smoothing by term">
      <thead><tr><th class="diagnose-term">Term</th>${columns}</tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
  </section>`;
}

/**
 * How much REML each fit had left to gain, round by round, on one log axis.
 * @param {DiagnosePayload} payload @param {string} selected
 */
export function searchPathMarkup(payload, selected) {
  const traced = payload.fits.filter((fit) => fit.path.length >= 2);
  const unrecorded = payload.fits.filter((fit) => fit.rounds > 0 && fit.path.length < 2).map((fit) => fit.label);
  const hint = unrecorded.length ? ` Not recorded by this fit's search: ${unrecorded.join(", ")}.` : "";
  const heading = head("Search path", `REML still to gain at each round. A long tail near the bottom is rounds that bought almost nothing.${hint}`);
  if (!traced.length) return unrecorded.length ? `<section class="report-section">${heading}</section>` : "";
  const { width, height, left, right, top, bottom } = CHART;
  const rounds = Math.max(...traced.map((fit) => fit.path.length - 1));
  const largest = Math.max(...traced.map((fit) => Math.max(...fit.path.map((value) => value - fit.path[fit.path.length - 1]))));
  const decades = Math.max(1, Math.ceil(Math.log10(Math.max(largest, PATH_FLOOR * 10))));
  const lowest = Math.log10(PATH_FLOOR);
  const x = (/** @type {number} */ round) => left + (round / Math.max(rounds, 1)) * (width - left - right);
  const y = (/** @type {number} */ gain) => {
    const place = (Math.log10(Math.max(gain, PATH_FLOOR)) - lowest) / (decades - lowest);
    return top + (1 - place) * (height - top - bottom);
  };
  const ticks = [];
  for (let power = lowest; power <= decades; power += 1) ticks.push(10 ** power);
  const grid = ticks.map((tick) => `<line class="diagnose-grid" x1="${left}" x2="${width - right}" y1="${y(tick).toFixed(1)}" y2="${y(tick).toFixed(1)}"></line>
    <text class="cv-tick" x="${left - 6}" y="${(y(tick) + 4).toFixed(1)}" text-anchor="end">${escapeHTML(fmt(tick))}</text>`).join("");
  const steps = Array.from({ length: 5 }, (_, part) => Math.round((part * rounds) / 4));
  const xTicks = [...new Set(steps)].map((round) => `<text class="cv-tick" x="${x(round).toFixed(1)}" y="${height - bottom + 16}" text-anchor="middle">${round}</text>`).join("");
  // The fit in view draws last, so nothing crosses over it.
  const ordered = [...traced].sort((a, b) => Number(a.id === selected) - Number(b.id === selected));
  const lines = ordered.map((fit) => {
    const final = fit.path[fit.path.length - 1];
    const d = fit.path.map((value, round) => `${round === 0 ? "M" : "L"}${x(round).toFixed(1)},${y(value - final).toFixed(1)}`).join(" ");
    return `<path class="diagnose-line" data-selected="${pressed(fit, selected)}" d="${d}" style="stroke: ${fitColour(fit)}"><title>${escapeHTML(`${fit.label}: ${fit.rounds} rounds`)}</title></path>`;
  }).join("");
  const legend = traced.map((fit) => `<span class="diagnose-key"><span class="diagnose-swatch" style="background: ${fitColour(fit)}"></span>${escapeHTML(fit.label)}</span>`).join("");
  const axis = `<text class="cv-axis-title" x="${left}" y="${height - 4}">Round</text>`;
  return `<section class="report-section">
    ${heading}
    <div class="diagnose-legend">${legend}</div>
    <svg class="diagnose-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="REML still to gain at each round, by fit">${grid}${xTicks}${lines}${axis}</svg>
  </section>`;
}

/** @param {DiagnoseFit} fit */
export function phasesMarkup(fit) {
  if (!fit.phases.length) return "";
  const widest = Math.max(...fit.phases.map((phase) => phase.share));
  const rows = fit.phases.map((phase) => `<div class="diagnose-phase">
      <span>${escapeHTML(phase.name)}</span>
      <span class="diagnose-bar"><i style="width: ${((phase.share / widest) * 100).toFixed(1)}%"></i></span>
      <span>${clockSeconds(phase.seconds)}</span><span>${(phase.share * 100).toFixed(0)}%</span>
    </div>`).join("");
  return `<section class="report-section">${head("Time by phase", `${fit.label}, while it fitted.`)}<div class="diagnose-phases">${rows}</div></section>`;
}

/** @param {number} seconds */
function clockSeconds(seconds) {
  return seconds >= 60 ? clockText(seconds) : `${seconds.toFixed(2)} s`;
}

/** @param {DiagnoseFit} fit */
export function findingsMarkup(fit) {
  if (!fit.findings.length) return "";
  const items = fit.findings.map((found) => `<li data-severity="${escapeHTML(found.severity)}">
      <span class="diagnose-severity">${escapeHTML(found.severity)}</span>
      <div><strong>${escapeHTML(found.headline)}</strong>
        <p>${escapeHTML(found.observed)}</p>
        ${found.question ? `<p class="cv-hint">${escapeHTML(found.question)}</p>` : ""}</div>
    </li>`).join("");
  return `<section class="report-section">${head("What the fit reports", `${fit.label}, from diagnose().`)}<ul class="diagnose-findings">${items}</ul></section>`;
}

/** @param {DiagnosePayload} payload @param {DiagnoseFit} fit */
export function worstRowsMarkup(payload, fit) {
  if (fit.id === "model") {
    return `<section class="report-section">${head("Rows that cost the most", "A fold's held-out rows: pick a fold to see its worst ten.")}</section>`;
  }
  if (!fit.worst_rows) {
    return `<section class="report-section">${head("Rows that cost the most", "These folds' rows are not available here, so this fold's worst rows cannot be listed.")}</section>`;
  }
  const columns = fit.worst_rows.columns.map((name) => `<th>${escapeHTML(name)}</th>`).join("");
  const rows = fit.worst_rows.rows.map((row) => `<tr><td>${row.row}</td><td>${fixed(row.cost, 3)}</td><td>${escapeHTML(fmt(row.response))}</td>
      <td>${escapeHTML(fmt(row.predicted))}</td><td>${row.weight === null ? "--" : escapeHTML(fmt(row.weight))}</td>
      ${row.values.map((value) => `<td>${escapeHTML(typeof value === "number" ? fmt(value) : String(value))}</td>`).join("")}</tr>`).join("");
  return `<section class="report-section">
    ${head("Rows that cost the most", `${fit.label}'s held-out rows that its fit predicted worst, by ${payload.cost_label.toLowerCase()}. Its fit never saw these.`)}
    <div class="diagnose-scroll"><table class="report-table diagnose-table" aria-label="Rows that cost the most">
      <thead><tr><th>Row</th><th>${escapeHTML(payload.cost_label)}</th><th>Response</th><th>Predicted</th><th>Weight</th>${columns}</tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
  </section>`;
}

/**
 * The fit in view: the one asked for while it exists, else the fold whose
 * search took the most rounds, which is why the tab is usually open, else
 * this model when no fold searched.
 * @param {DiagnosePayload} payload @param {string} wanted
 */
export function chosenFit(payload, wanted) {
  const asked = payload.fits.find((fit) => fit.id === wanted);
  if (asked) return asked;
  const folds = payload.fits.filter((fit) => fit.id !== "model" && fit.rounds > 0);
  return folds.reduce((slowest, fit) => (fit.rounds > slowest.rounds ? fit : slowest), folds[0] ?? payload.fits[0]);
}

/** @param {DiagnosePayload} payload @param {string} wanted */
export function diagnoseTabMarkup(payload, wanted) {
  const fit = chosenFit(payload, wanted);
  return [
    fitsTableMarkup(payload, fit.id),
    termMatrixMarkup(payload, fit.id),
    searchPathMarkup(payload, fit.id),
    findingsMarkup(fit),
    phasesMarkup(fit),
    worstRowsMarkup(payload, fit),
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

  function draw() {
    if (payload && isShown()) frame.innerHTML = diagnoseTabMarkup(payload, wanted);
  }

  /** @param {Event} event */
  function onClick(event) {
    const target = event.target instanceof Element ? event.target : null;
    const button = target?.closest("[data-diagnose-fit]");
    if (!(button instanceof HTMLElement)) return;
    wanted = button.dataset.diagnoseFit ?? "";
    draw();
  }

  frame.addEventListener("click", onClick);

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
