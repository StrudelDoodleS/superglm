// @ts-check

import { escapeHTML, fmt } from "../format.js";

/** @typedef {import('../api/contracts.js').CVReportPayload} CVReportPayload */
/** @typedef {import('../api/contracts.js').CVResultPayload} CVResultPayload */
/** @typedef {import('../api/contracts.js').CVTermItem} CVTermItem */
/** @typedef {import('../api/contracts.js').CVFoldCurve} CVFoldCurve */
/** @typedef {import('../api/contracts.js').JobKind} JobKind */
/** @typedef {import('../api/contracts.js').JobStatus} JobStatus */
/**
 * @typedef {object} JobClient
 * @property {(kind:JobKind)=>Promise<unknown>} jobStart
 * @property {(jobId:string, wait?:boolean)=>Promise<unknown>} jobStatus
 * @property {(jobId:string)=>Promise<unknown>} jobCancel
 */
/**
 * @typedef {object} CVTabState
 * @property {string} term the term whose chart is shown
 * @property {string} query the term search
 * @property {Record<JobKind, JobStatus|null>} jobs the newest status of each job
 * @property {Record<JobKind, string>} errors a refused job request's message
 */

/** @type {readonly JobKind[]} */
export const JOB_KINDS = Object.freeze(["cv", "final_fit"]);
const POLL_MS = 250;
const CHART = Object.freeze({ width: 760, height: 320, left: 56, right: 18, top: 30, bottom: 66 });
const JOB_NAMES = Object.freeze({ cv: "Run CV", final_fit: "Final fit" });
const SHORT_METRICS = Object.freeze({ deviance: "Deviance", gini: "Gini", nll: "NLL" });
const RUN_CV_HELP = "Refit the current structure on each stored fold, put your hand edits back "
  + "exactly as set, and score the held-out rows.";
const FINAL_FIT_HELP = "Refit the current structure on train and validation rows and put your "
  + "hand edits back. The test split stays held out. Export offers the result as Final fit model.";
const ICONS = Object.freeze({
  cv: '<svg class="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">'
    + '<path d="m12 3 9 5-9 5-9-5z"></path><path d="m3 13 9 5 9-5"></path></svg>',
  final_fit: '<svg class="toolbar-icon" viewBox="0 0 24 24" aria-hidden="true">'
    + '<path d="M20 11a8 8 0 1 0-2.3 5.7"></path><path d="M20 4v7h-7"></path></svg>'
});

/** @param {unknown} value @returns {value is number} */
function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/** @param {unknown} value */
function metricText(value) {
  return isNumber(value) ? value.toFixed(4) : "--";
}

/** @param {unknown} value */
function countText(value) {
  return isNumber(value) ? value.toLocaleString("en-US") : "--";
}

/** @param {unknown} value @param {number} digits */
function fixedText(value, digits) {
  return isNumber(value) ? value.toFixed(digits) : "--";
}

/** @param {number} value */
function px(value) {
  return value.toFixed(1);
}

/** The folds with a palette of their own; later folds take the trace colours. */
const FOLD_SLOTS = 5;

/** @param {number} fold the fold's 0-based number */
function foldColour(fold) {
  return fold < FOLD_SLOTS ? `var(--fold-${fold})` : `var(--trace-${fold % 10})`;
}

/**
 * The fold's edge, a darker shade of its colour that keeps a pastel mark
 * readable on the chart's ground.
 * @param {number} fold the fold's 0-based number
 */
function foldEdge(fold) {
  return fold < FOLD_SLOTS ? `var(--fold-edge-${fold})` : foldColour(fold);
}

/** @param {number} fold the fold's 0-based number */
function foldMarkStyle(fold) {
  return `fill: ${foldColour(fold)}; stroke: ${foldEdge(fold)}`;
}

/** @param {number} fold the fold's 0-based number */
function foldSwatchStyle(fold) {
  return `background: ${foldColour(fold)}; box-shadow: inset 0 0 0 1px ${foldEdge(fold)}`;
}

/**
 * A fold curve's own number, 0-based. It picks the fold's colour and its
 * place within a level, so both stay fixed when another fold is missing.
 * @param {{fold?:number}} fold @param {number} index its position in the list
 */
function foldNumber(fold, index) {
  return Number.isInteger(fold.fold) ? /** @type {number} */ (fold.fold) : index;
}

/** @param {string} value @returns {JobKind} */
function jobKind(value) {
  return value === "final_fit" ? "final_fit" : "cv";
}

/**
 * The header's source line: folds, splitter, rows and where the result came from.
 * @param {CVReportPayload} payload
 */
export function cvSourceLine(payload) {
  const { header } = payload;
  if (!header.supplied) return "No cross-validation result supplied";
  const parts = [`${header.n_folds} folds`];
  if (header.splitter) parts.push(header.splitter);
  if (header.n_rows !== null) parts.push(`${countText(header.n_rows)} rows`);
  parts.push("supplied with edit(model, cv=result)");
  return parts.join(" · ");
}

/**
 * Terms whose name contains ``query``, ignoring case, in the server's
 * order: least stable first.
 * @param {CVTermItem[]} terms @param {string} query
 */
export function filterTerms(terms, query) {
  const needle = query.trim().toLowerCase();
  return needle ? terms.filter((term) => term.name.toLowerCase().includes(needle)) : terms;
}

/**
 * The lowest and highest fold value at each point; null where no fold has one.
 * @param {{fit:number[], folds:Array<{values:Array<number|null>}>}} term
 * @returns {{lo:Array<number|null>, hi:Array<number|null>}}
 */
export function foldEnvelope(term) {
  /** @param {number} index */
  const column = (index) => term.folds.map((fold) => fold.values[index]).filter(isNumber);
  const columns = term.fit.map((_value, index) => column(index));
  return {
    lo: columns.map((values) => (values.length ? Math.min(...values) : null)),
    hi: columns.map((values) => (values.length ? Math.max(...values) : null))
  };
}

/**
 * Round tick values covering [lo, hi], about ``target`` of them.
 * @param {number} lo @param {number} hi @param {number} [target]
 * @returns {number[]}
 */
export function niceTicks(lo, hi, target = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / target;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((multiple) => multiple * power)
    .find((candidate) => candidate >= raw) ?? 10 * power;
  const first = Math.ceil(lo / step) * step;
  const count = Math.floor((hi - first) / step + 1e-9) + 1;
  return Array.from({ length: count }, (_unused, index) =>
    Number((first + index * step).toPrecision(12)));
}

/**
 * The newer of two statuses for one job kind: a later job wins, and for the
 * same job a finished status or one with more progress.
 * @param {JobStatus|null} known @param {JobStatus|null} reported
 * @returns {JobStatus|null}
 */
export function newerJob(known, reported) {
  if (!reported) return known;
  if (!known) return reported;
  const order = (/** @type {JobStatus} */ job) => Number(job.job_id.split("-").pop());
  if (order(reported) !== order(known)) return order(reported) > order(known) ? reported : known;
  if (known.status !== "running") return known;
  if (reported.status !== "running") return reported;
  return reported.progress.length >= known.progress.length ? reported : known;
}

/**
 * One line on what a job is doing, or how it ended.
 * @param {JobKind} kind @param {JobStatus|null} job
 */
export function jobLine(kind, job) {
  if (!job) return "";
  const name = JOB_NAMES[kind];
  if (job.status === "cancelled") return `${name} was cancelled. Nothing was kept.`;
  if (job.status === "failed") return `${name} failed: ${job.error || "internal editor error"}`;
  if (job.status === "done") {
    return kind === "final_fit"
      ? `Final fit finished on ${countText(job.result?.n_rows)} rows. Export offers it as Final fit model.`
      : "Run CV finished. Its folds are shown beside the supplied ones.";
  }
  if (job.cancel_requested) return `${name}: cancelling after this step…`;
  const last = job.progress[job.progress.length - 1];
  if (last?.phase === "fold") return `${name}: fold ${last.fold} of ${last.n_folds}…`;
  if (last?.phase === "fitting") return `${name}: fitting ${countText(last.n_rows)} rows…`;
  if (last?.phase === "carrying") return `${name}: putting the hand edits back…`;
  if (last?.phase === "curves") return `${name}: reading the fold curves…`;
  return `${name}: starting…`;
}

/** @param {string} text @param {string} tone */
function chip(text, tone) {
  return `<span class="cv-chip" data-tone="${tone}">${escapeHTML(text)}</span>`;
}

/**
 * @param {JobKind} kind @param {string} label @param {boolean} enabled
 * @param {string} help @param {boolean} primary
 */
function actionButton(kind, label, enabled, help, primary) {
  return `<button type="button" class="cv-action${primary ? " is-primary" : ""}"
    data-cv-start="${kind}" ${enabled ? "" : "disabled"}
    data-popover-title="${escapeHTML(label)}" data-popover-body="${escapeHTML(help)}">
    ${ICONS[kind]}<span>${escapeHTML(label)}</span></button>`;
}

/** @param {JobKind} kind @param {CVTabState} state */
function jobRow(kind, state) {
  const job = state.jobs[kind];
  const error = state.errors[kind];
  if (error) return `<p class="cv-job" data-cv-job="${kind}" data-status="failed">${escapeHTML(error)}</p>`;
  if (!job) return "";
  const cancel = job.status === "running" && !job.cancel_requested
    ? ` <button type="button" class="cv-cancel" data-cv-cancel="${kind}">Cancel</button>`
    : "";
  return `<p class="cv-job" data-cv-job="${kind}" data-status="${job.status}">`
    + `${escapeHTML(jobLine(kind, job))}${cancel}</p>`;
}

/**
 * The header row: what is waiting or stale, the two job buttons, why a
 * button is disabled, and each job's progress.
 * @param {CVReportPayload} payload @param {CVTabState} state
 */
export function toolbarMarkup(payload, state) {
  const chips = [];
  if (payload.pending) {
    chips.push(chip(`${payload.pending} ${payload.pending === 1 ? "change" : "changes"} waiting for refit`, "waiting"));
  }
  if (payload.results.some((result) => result.stale)) {
    chips.push(chip("The model has changed since these folds were fitted", "stale"));
  }
  if (payload.final_fit.done && payload.final_fit.stale) {
    chips.push(chip("The final fit is from an earlier version of the model", "stale"));
  }
  if (payload.run_cv.note) chips.push(chip(payload.run_cv.note, "note"));
  const runEnabled = payload.run_cv.available && state.jobs.cv?.status !== "running";
  const finalEnabled = payload.final_fit.available && state.jobs.final_fit?.status !== "running";
  const reasons = [
    payload.header.supplied ? payload.run_cv.reason : null,
    payload.final_fit.reason,
    payload.final_fit.note
  ].filter((reason) => reason);
  return `
    <section class="cv-toolbar" data-cv-toolbar>
      <div class="cv-chips">${chips.join("")}</div>
      <div class="cv-actions">
        ${actionButton("final_fit", "Final fit on all rows", finalEnabled, payload.final_fit.reason || FINAL_FIT_HELP, false)}
        ${actionButton("cv", "Run CV on current model", runEnabled, payload.run_cv.reason || RUN_CV_HELP, true)}
      </div>
      ${reasons.map((reason) => `<p class="cv-reason">${escapeHTML(reason)}</p>`).join("")}
      <div class="cv-jobs" aria-live="polite">${JOB_KINDS.map((kind) => jobRow(kind, state)).join("")}</div>
    </section>`;
}

/** @param {string} title @param {string} hint */
function sectionHead(title, hint) {
  return `<div class="cv-section-head"><h3>${escapeHTML(title)}</h3>`
    + `<span class="cv-hint">${escapeHTML(hint)}</span></div>`;
}

// Folds crowd into a few pixels of their strip when the card's common scale
// is over 1/READABLE_SHARE times wider than they spread: one supplied fold's
// deviance a million times the rest does that to every other row.
const READABLE_SHARE = 0.05;

// The unit roundoff of a float64: the relative error of one rounding.
const UNIT_ROUNDOFF = 2 ** -53;

/**
 * The strip positions of each row's values: one scale for every row, so the
 * rows compare at a glance, unless far-off folds alone stretch it. Each row's
 * core is the half of its folds nearest the median of every row's folds; when
 * the cores together span under READABLE_SHARE of the common scale, each row
 * gets its own scale.
 * A steadier row beside a spread one keeps the common scale, which shows it
 * is steadier. A row whose values are equal, or differ by no more than their
 * rounding, sits mid-strip, and a value a rounding outside its row's folds
 * (their mean, say) stays on the strip.
 * @param {number[][]} rows each row's finite values
 * @returns {{own: boolean, x: ((value:number) => number)[]}}
 */
export function stripScales(rows) {
  const all = rows.flat();
  const shared = range(all);
  const centre = median(all);
  // One row, or a common scale that is rounding alone, has nothing to switch.
  const own = rows.filter((values) => values.length).length > 1
    && shared > all.length * UNIT_ROUNDOFF * Math.max(0, ...all.map(Math.abs))
    && range(rows.flatMap((values) => core(values, centre))) < READABLE_SHARE * shared;
  if (own) return { own, x: rows.map(stripScale) };
  const common = stripScale(all);
  return { own, x: rows.map(() => common) };
}

/** @param {number[]} values */
function range(values) {
  return values.length ? Math.max(...values) - Math.min(...values) : 0;
}

/**
 * ``values`` onto the strip, 10 to 210. A span within the values' rounding,
 * ``n`` roundings of the largest magnitude, is no spread at all.
 * @param {number[]} values
 * @returns {(value:number) => number}
 */
function stripScale(values) {
  const lo = values.length ? Math.min(...values) : 0;
  const span = range(values);
  const magnitude = Math.max(0, ...values.map(Math.abs));
  if (!(span > values.length * UNIT_ROUNDOFF * magnitude)) return () => 110;
  return (value) => Math.min(210, Math.max(10, 10 + ((value - lo) / span) * 200));
}

/**
 * The half of ``values`` nearest ``centre`` (the larger half of an odd
 * count): what is left when up to half of them are far off. The centre is
 * every row's median, so a row of two folds keeps the one nearer the rest.
 * @param {number[]} values @param {number} centre
 * @returns {number[]}
 */
function core(values, centre) {
  return values
    .map((value) => ({ value, distance: Math.abs(value - centre) }))
    // Ties go to the smaller value, so the core is the same whatever the fold order.
    .sort((a, b) => a.distance - b.distance || a.value - b.value)
    .slice(0, Math.ceil(values.length / 2))
    .map(({ value }) => value);
}

/** @param {number[]} values */
function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length / 2;
  if (!sorted.length) return 0;
  return sorted.length % 2 ? sorted[Math.floor(middle)] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/**
 * @param {{name:string, label:string, lower_is_better:boolean}} metric
 * @param {CVResultPayload[]} results
 */
function metricCard(metric, results) {
  const scales = stripScales(results.map((result) =>
    result.folds.map((fold) => fold.scores[metric.name]).filter(isNumber)));
  const rows = results.map((result, row) => {
    const x = scales.x[row];
    const mean = result.mean[metric.name];
    const pooled = result.pooled[metric.name];
    const dots = result.folds.map((fold, index) => {
      const value = fold.scores[metric.name];
      const number = foldNumber(fold, index);
      return isNumber(value)
        ? `<circle cx="${px(x(value))}" cy="12" r="4.5" class="cv-card-dot" style="${foldMarkStyle(number)}">`
          + `<title>Fold ${number + 1}: ${metricText(value)}</title></circle>`
        : "";
    }).join("");
    const meanTick = isNumber(mean) ? `<path class="cv-mean" d="M${px(x(mean))},3 V21"></path>` : "";
    const pooledText = isNumber(pooled) ? ` · pooled ${metricText(pooled)}` : "";
    return `<div class="cv-card-row" data-origin="${result.origin}">
      <div class="cv-card-value"><span class="cv-card-label">${escapeHTML(result.label)}</span>
        <strong>${metricText(mean)}</strong>
        <span class="cv-card-spread">± ${metricText(result.std[metric.name])}${pooledText}</span></div>
      <svg class="cv-strip" viewBox="0 0 220 24" role="img"
        aria-label="${escapeHTML(`${metric.label} by fold, ${result.label}`)}">
        <path class="cv-strip-axis" d="M10,12 H210"></path>${meanTick}${dots}</svg>
    </div>`;
  }).join("");
  const ownScales = scales.own ? " · each row on its own scale" : "";
  return `<div class="cv-card"><div class="cv-card-title">${escapeHTML(metric.label)}
    <span>· ${metric.lower_is_better ? "lower" : "higher"} is better${ownScales}</span></div>${rows}</div>`;
}

/**
 * One card per metric: mean ± sd and pooled for each run, one dot per fold.
 * @param {CVReportPayload} payload
 */
export function performanceMarkup(payload) {
  const head = sectionHead("Performance across folds", "each dot is a fold; the bar is the mean");
  if (!payload.results.length) {
    return `<section class="report-section">${head}<div class="report-note">${escapeHTML(payload.note)}</div></section>`;
  }
  const cards = payload.metrics.map((metric) => metricCard(metric, payload.results)).join("");
  return `<section class="report-section cv-performance">${head}<div class="cv-cards">${cards}</div></section>`;
}

/**
 * The latest run's folds: rows, scores, EDF, fit time and convergence. A
 * score a fold could not compute reads as --.
 * @param {CVReportPayload} payload
 */
export function foldTableMarkup(payload) {
  const result = payload.results[payload.results.length - 1];
  if (!result) return "";
  const names = payload.metrics.map((metric) => metric.name);
  const label = (/** @type {string} */ name) =>
    SHORT_METRICS[/** @type {keyof typeof SHORT_METRICS} */ (name)] ?? name;
  const rows = result.folds.map((fold, index) => `<tr>
      <td><span class="cv-fold-swatch" style="${foldSwatchStyle(foldNumber(fold, index))}"></span>Fold ${fold.fold + 1}</td>
      <td>${countText(fold.n_train)}</td><td>${countText(fold.n_test)}</td>
      ${names.map((name) => `<td>${metricText(fold.scores[name])}</td>`).join("")}
      <td>${fixedText(fold.effective_df, 1)}</td>
      <td>${isNumber(fold.fit_time_s) ? `${fold.fit_time_s.toFixed(2)} s` : "--"}</td>
      <td data-converged="${fold.converged}">${fold.converged ? "yes" : "no"}</td>
    </tr>`).join("");
  const summary = names.map((name) =>
    `<td>${metricText(result.mean[name])} ± ${metricText(result.std[name])}</td>`).join("");
  return `<section class="report-section">
    ${sectionHead("Folds", result.label)}
    <table class="report-table cv-fold-table" aria-label="Fold scores">
      <thead><tr><th>Fold</th><th>Train rows</th><th>Test rows</th>
        ${names.map((name) => `<th>${escapeHTML(label(name))}</th>`).join("")}
        <th>EDF</th><th>Fit time</th><th>Converged</th></tr></thead>
      <tbody>${rows}<tr class="cv-mean-row"><td>Mean ± sd</td><td></td><td></td>${summary}<td></td><td></td><td></td></tr></tbody>
    </table></section>`;
}

/** @param {string} name @param {string} query */
function highlighted(name, query) {
  const needle = query.trim().toLowerCase();
  const at = needle ? name.toLowerCase().indexOf(needle) : -1;
  if (at < 0) return escapeHTML(name);
  const end = at + needle.length;
  return `${escapeHTML(name.slice(0, at))}<mark>${escapeHTML(name.slice(at, end))}</mark>${escapeHTML(name.slice(end))}`;
}

// A hand edit Run CV put back on every fold has no spread to show.
const HELD = `<span class="cv-term-held" data-popover-title="Hand-edited"
      data-popover-body="The same curve on every fold.">held</span>`;

/**
 * The term list: name, spread and min r, each term a button that shows its chart.
 * @param {CVTermItem[]} terms @param {string} current @param {string} query
 */
export function termListMarkup(terms, current, query) {
  if (!terms.length) return '<div class="report-note">No terms match.</div>';
  return terms.map((term) => `<button type="button" class="cv-term" data-cv-term="${escapeHTML(term.name)}"
      aria-current="${term.name === current}">
      <span class="cv-term-name">${highlighted(term.name, query)}</span>
      ${term.held ? HELD : `<span>${fixedText(term.spread, 3)}</span><span>${fixedText(term.min_correlation, 2)}</span>`}</button>`).join("");
}

/**
 * @param {CVTermItem} term
 * @param {(value:number)=>number} y
 * @param {number[]} ticks
 */
function frameMarkup(term, y, ticks) {
  const { width, left, right } = CHART;
  return ticks.map((tick) => `<line class="cv-grid" x1="${left}" x2="${width - right}"
      y1="${px(y(tick))}" y2="${px(y(tick))}"></line>
    <text class="cv-tick" x="${left - 6}" y="${px(y(tick) + 4)}" text-anchor="end">${escapeHTML(fmt(tick))}</text>`).join("")
    + `<line class="zero" x1="${left}" x2="${width - right}" y1="${px(y(1))}" y2="${px(y(1))}"></line>`
    + `<text class="cv-axis-title" x="${left}" y="16">${escapeHTML(term.name)} · relativity</text>`;
}

/**
 * The chart's legend. Each fold has its own entry, named, in its colour:
 * hovering or focusing it picks that fold out on the chart (highlightFold),
 * the second channel for fold colours too close to tell apart.
 * @param {CVTermItem} term
 */
function legendMarkup(term) {
  const folds = term.folds.map((fold, index) => {
    const number = foldNumber(fold, index);
    return `<button type="button" class="cv-fold-key" data-fold="${number}" data-cv-fold-key="${number}">`
      + `<span class="cv-key-swatch" style="${foldSwatchStyle(number)}"></span>${escapeHTML(fold.label)}</button>`;
  }).join("");
  const kinds = [
    ["cv-legend-fit", "all-rows fit"],
    ...(term.edited ? [["cv-legend-edited", "edited"]] : []),
    ["cv-legend-range", term.kind === "levels" ? "fold range" : "fold min–max"],
    ["cv-legend-exposure", "exposure"]
  ].map(([kind, text]) => `<span class="cv-key"><span class="cv-key-swatch ${kind}"></span>${escapeHTML(text)}</span>`)
    .join("");
  return `<div class="cv-legend" role="group" aria-label="Legend: hover or focus a fold to pick it out">`
    + `${folds}${kinds}</div>`;
}

/** @param {CVTermItem} term */
function yScale(term) {
  const { height, top, bottom } = CHART;
  const values = [...term.folds.flatMap((fold) => fold.values), ...term.fit, ...(term.edited ?? [])]
    .filter(isNumber);
  const lo = Math.min(...values, 1);
  const hi = Math.max(...values, 1);
  const pad = (hi - lo) * 0.1 || 0.05;
  const domain = [lo - pad, hi + pad];
  const y = (/** @type {number} */ value) =>
    height - bottom - ((value - domain[0]) / (domain[1] - domain[0])) * (height - top - bottom);
  return { y, ticks: niceTicks(domain[0], domain[1], 5) };
}

/**
 * How many fold places a level holds: one per fold number, so a fold
 * missing from this term leaves its place empty.
 * @param {CVTermItem} term
 */
function foldSlots(term) {
  return Math.max(term.folds.length, ...term.folds.map((fold, index) => foldNumber(fold, index) + 1));
}

/**
 * A level term: fold dots, the fold range as a whisker and the all-rows fit
 * as a bar at each level, in the model's level order, over exposure. Each
 * fold has a fixed place within every level, fold 1 leftmost; a fold with
 * no value at a level leaves its place there empty.
 * @param {CVTermItem} term
 */
export function levelChartMarkup(term) {
  const { width, height, left, right, top, bottom } = CHART;
  const levels = term.levels ?? [];
  const band = (width - left - right) / Math.max(levels.length, 1);
  const x = (/** @type {number} */ index) => left + (index + 0.5) * band;
  const { y, ticks } = yScale(term);
  const envelope = foldEnvelope(term);
  const base = height - bottom;
  const maxWeight = Math.max(...term.weights, 0) || 1;
  const barWidth = Math.min(band * 0.7, 28);
  const slots = foldSlots(term);
  const step = Math.min(6, band / (slots + 2));
  const bars = term.weights.map((weight, index) => {
    const tall = (weight / maxWeight) * (base - top) / 3;
    return `<rect class="exposure" x="${px(x(index) - barWidth / 2)}" y="${px(base - tall)}"
      width="${px(barWidth)}" height="${px(tall)}"></rect>`;
  }).join("");
  const whiskers = levels.map((_level, index) => {
    const lo = envelope.lo[index];
    const hi = envelope.hi[index];
    return isNumber(lo) && isNumber(hi)
      ? `<line class="cv-range" x1="${px(x(index))}" x2="${px(x(index))}"
      y1="${px(y(lo))}" y2="${px(y(hi))}"></line>`
      : "";
  }).join("");
  const dots = term.folds.flatMap((fold, foldIndex) => {
    const number = foldNumber(fold, foldIndex);
    const offset = (number - (slots - 1) / 2) * step;
    return fold.values.map((value, index) => (isNumber(value)
      ? `<circle class="cv-fold-dot" data-fold="${number}" data-level="${index}"
      cx="${px(x(index) + offset)}" cy="${px(y(value))}" r="3.6" style="${foldMarkStyle(number)}">
      <title>${escapeHTML(`${fold.label} · ${levels[index]}: ${fmt(value)}`)}</title></circle>`
      : ""));
  }).join("");
  /** @param {Array<number|null>} values @param {string} kind */
  const levelTicks = (values, kind) => values.map((value, index) => (isNumber(value)
    ? `<line class="${kind}"
      x1="${px(x(index) - 16)}" x2="${px(x(index) + 16)}" y1="${px(y(value))}" y2="${px(y(value))}"></line>`
    : "")).join("");
  // Labels turn only when the longest no longer fits its level's width.
  const longest = Math.max(0, ...levels.map((level) => Math.min(level.length, 14)));
  const rotate = longest * 6.5 > band - 4;
  const labels = levels.map((level, index) => `<text class="cv-level" x="${px(x(index))}" y="${base + 16}"
      text-anchor="${rotate ? "end" : "middle"}"${rotate ? ` transform="rotate(-40 ${px(x(index))} ${base + 16})"` : ""}>
      ${escapeHTML(level.length > 14 ? `${level.slice(0, 13)}…` : level)}<title>${escapeHTML(level)}</title></text>`).join("");
  return `${legendMarkup(term)}<svg class="cv-chart-svg" viewBox="0 0 ${width} ${height}" role="img"
      aria-label="${escapeHTML(`${term.name} relativities by fold`)}">
    ${frameMarkup(term, y, ticks)}${bars}${whiskers}${dots}${levelTicks(term.fit, "cv-fit")}
    ${term.edited ? levelTicks(term.edited, "cv-edited") : ""}${labels}</svg>`;
}

/**
 * The runs of consecutive points that have a value; a gap ends a run.
 * @param {Array<number|null>} values @returns {number[][]} indices
 */
function valueRuns(values) {
  /** @type {number[][]} */
  const runs = [];
  /** @type {number[]|null} */
  let run = null;
  values.forEach((value, index) => {
    if (!isNumber(value)) {
      run = null;
      return;
    }
    if (!run) {
      run = [];
      runs.push(run);
    }
    run.push(index);
  });
  return runs;
}

/**
 * The exposure behind a curve, smoothed along x with a Gaussian kernel a
 * thirtieth of the range wide. The grid's weights come from the rows
 * nearest each point, so a variable recorded in whole units, such as an
 * age, leaves most points empty; drawn raw, the strip is a comb.
 * @param {number[]} xs @param {Array<number|null>} weights
 * @returns {number[]}
 */
export function exposureProfile(xs, weights) {
  const width = ((xs[xs.length - 1] ?? 0) - (xs[0] ?? 0)) / 30;
  if (!(width > 0)) return weights.map((weight) => (isNumber(weight) ? weight : 0));
  return xs.map((at) => xs.reduce((total, from, index) => {
    const weight = weights[index];
    const z = (at - from) / width;
    return isNumber(weight) ? total + weight * Math.exp(-0.5 * z * z) : total;
  }, 0));
}

/** @param {Array<[number, number]>} points */
function polyline(points) {
  return points.map(([px0, py0], index) => `${index ? "L" : "M"}${px(px0)},${px(py0)}`).join("");
}

/**
 * A numeric term: one line per fold, the folds' min–max envelope and the
 * all-rows fit, over exposure. A fold's line breaks where it has no value.
 * @param {CVTermItem} term
 */
export function curveChartMarkup(term) {
  const { width, height, left, right, top, bottom } = CHART;
  const xs = term.x ?? [];
  const first = xs[0] ?? 0;
  const last = xs[xs.length - 1] ?? 1;
  const x = (/** @type {number} */ value) =>
    left + ((value - first) / (last - first || 1)) * (width - left - right);
  const { y, ticks } = yScale(term);
  const envelope = foldEnvelope(term);
  const base = height - bottom;
  const profile = exposureProfile(xs, term.weights);
  const maxWeight = Math.max(...profile, 0) || 1;
  /** @param {Array<number|null>} values @param {number} index @returns {[number, number]} */
  const point = (values, index) => [x(xs[index]), y(/** @type {number} */ (values[index]))];
  /** @param {Array<number|null>} values */
  const line = (values) => valueRuns(values)
    .map((run) => polyline(run.map((index) => point(values, index)))).join("");
  const exposure = polyline([
    [x(first), base],
    ...xs.map((value, index) => /** @type {[number, number]} */ (
      [x(value), base - (profile[index] / maxWeight) * (base - top) / 4])),
    [x(last), base]
  ]);
  const banded = envelope.lo.map((value, index) => (isNumber(envelope.hi[index]) ? value : null));
  const band = valueRuns(banded).map((run) => `${polyline([
    ...run.map((index) => point(envelope.hi, index)),
    ...[...run].reverse().map((index) => point(envelope.lo, index))
  ])}Z`).join("");
  const lines = term.folds.map((fold, index) => {
    const number = foldNumber(fold, index);
    const d = line(fold.values);
    return `<path class="cv-fold-casing" data-fold="${number}" style="stroke: ${foldEdge(number)}"
      d="${d}"></path><path class="cv-fold-line" data-fold="${number}" style="stroke: ${foldColour(number)}"
      d="${d}"><title>${escapeHTML(fold.label)}</title></path>`;
  }).join("");
  const xTicks = niceTicks(first, last, 6).map((tick) => `<text class="cv-tick" x="${px(x(tick))}"
      y="${base + 16}" text-anchor="middle">${escapeHTML(fmt(tick))}</text>`).join("");
  return `${legendMarkup(term)}<svg class="cv-chart-svg" viewBox="0 0 ${width} ${height}" role="img"
      aria-label="${escapeHTML(`${term.name} relativities by fold`)}">
    ${frameMarkup(term, y, ticks)}<path class="exposure-density" d="${exposure}Z"></path>
    <path class="cv-envelope" d="${band}"></path>${lines}
    <path class="cv-fit-line" d="${line(term.fit)}"></path>
    ${term.edited ? `<path class="cv-edited-line" d="${line(term.edited)}"></path>` : ""}
    ${xTicks}</svg>`;
}

/**
 * Pick out one fold on a chart: its marks and legend entry lit, the other
 * folds' dimmed. ``null`` shows every fold alike again.
 * @param {ParentNode} root @param {string|null} fold
 */
export function highlightFold(root, fold) {
  for (const node of root.querySelectorAll("[data-fold]")) {
    const lit = fold !== null && node.getAttribute("data-fold") === fold;
    node.classList.toggle("is-lit", lit);
    node.classList.toggle("is-dimmed", fold !== null && !lit);
  }
}

/** @param {CVTermItem} term */
function chartMarkup(term) {
  return term.kind === "levels" ? levelChartMarkup(term) : curveChartMarkup(term);
}

/** @param {CVTermItem[]} terms @param {string} name */
function currentTerm(terms, name) {
  return terms.find((term) => term.name === name) ?? terms[0] ?? null;
}

/**
 * The searchable term list ranked by spread, and the chosen term's chart.
 * @param {CVReportPayload} payload @param {CVTabState} state
 */
export function relativitiesMarkup(payload, state) {
  const relativities = payload.relativities;
  const head = sectionHead(
    "Relativities across folds",
    "levels in the model's order · each curve re-centred on its exposure-weighted mean"
  );
  if (!relativities.available) {
    return `<section class="report-section cv-relativities">${head}
      <div class="report-note">${escapeHTML(relativities.note || "")}</div></section>`;
  }
  const shown = filterTerms(relativities.terms, state.query);
  const current = currentTerm(shown, state.term);
  const origin = relativities.origin === "run"
    ? "From Run CV on the current model, with the hand edits put back on every fold."
    : "From the supplied result's fold models.";
  return `<section class="report-section cv-relativities">${head}
    <p class="cv-hint">${escapeHTML(origin)}${relativities.stale ? " The model has changed since." : ""}</p>
    <div class="cv-rel-layout">
      <div class="cv-term-panel">
        <input id="cvTermSearch" class="cv-search" type="search" placeholder="Search terms"
          aria-label="Search terms" value="${escapeHTML(state.query)}">
        <div class="cv-term-head" aria-hidden="true"><span>Least stable first</span><span>spread</span><span>min r</span></div>
        <div class="cv-term-list" data-cv-term-list>${termListMarkup(shown, current?.name ?? "", state.query)}</div>
        <p class="cv-hint">spread: mean distance of a fold's curve from the fold average. min r: lowest fold correlation with it, -- when a fold's curve is flat.</p>
      </div>
      <div class="cv-chart" data-cv-chart>${current ? chartMarkup(current) : ""}</div>
    </div></section>`;
}

/**
 * The whole tab below the report header.
 * @param {CVReportPayload} payload @param {CVTabState} state
 */
export function cvTabMarkup(payload, state) {
  return toolbarMarkup(payload, state) + performanceMarkup(payload) + foldTableMarkup(payload)
    + relativitiesMarkup(payload, state);
}

/**
 * Bind the Cross-validation tab to the report frame: render each ``cv``
 * payload, and start, poll and cancel its two jobs. The frame is shared
 * with the other reports, so a job's progress is drawn only while
 * ``isShown()`` says the Cross-validation tab is the one open.
 *
 * @param {object} options
 * @param {HTMLElement} options.frame
 * @param {JobClient} options.client
 * @param {(kind:JobKind, job:JobStatus)=>unknown} options.onJobSettled
 * @param {(ms:number)=>Promise<void>} [options.pause]
 * @param {()=>boolean} [options.isShown]
 */
export function createCVTab({
  frame,
  client,
  onJobSettled,
  pause = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); }),
  isShown = () => true
}) {
  /** @type {CVReportPayload|null} */
  let payload = null;
  /** @type {CVTabState} */
  const state = { term: "", query: "", jobs: { cv: null, final_fit: null }, errors: { cv: "", final_fit: "" } };
  /** @type {Set<string>} */
  const polling = new Set();
  /** @type {Set<JobKind>} The kinds whose start is sent but not yet answered. */
  const starting = new Set();

  function renderAll() {
    if (!payload || !isShown()) return;
    const search = frame.querySelector("#cvTermSearch");
    const caret = search === frame.ownerDocument?.activeElement && search instanceof HTMLInputElement
      ? search.selectionStart
      : null;
    frame.innerHTML = cvTabMarkup(payload, state);
    const next = frame.querySelector("#cvTermSearch");
    if (caret !== null && next instanceof HTMLInputElement) {
      next.focus();
      next.setSelectionRange(caret, caret);
    }
  }

  function renderToolbar() {
    if (!payload || !isShown()) return;
    const toolbar = frame.querySelector("[data-cv-toolbar]");
    if (!toolbar) return renderAll();
    toolbar.outerHTML = toolbarMarkup(payload, state);
  }

  function renderTerms() {
    const list = frame.querySelector("[data-cv-term-list]");
    const chart = frame.querySelector("[data-cv-chart]");
    if (!payload || !list || !chart) return renderAll();
    const shown = filterTerms(payload.relativities.terms, state.query);
    const current = currentTerm(shown, state.term);
    list.innerHTML = termListMarkup(shown, current?.name ?? "", state.query);
    chart.innerHTML = current ? chartMarkup(current) : "";
  }

  /** @param {JobKind} kind @param {unknown} error */
  function refuse(kind, error) {
    state.errors[kind] = error instanceof Error ? error.message : String(error);
    renderToolbar();
  }

  /** @param {JobKind} kind @param {string} jobId */
  async function poll(kind, jobId) {
    if (polling.has(jobId)) return;
    polling.add(jobId);
    try {
      let job = state.jobs[kind];
      while (job?.job_id === jobId && job.status === "running") {
        await pause(POLL_MS);
        state.jobs[kind] = newerJob(state.jobs[kind], /** @type {JobStatus} */ (await client.jobStatus(jobId)));
        job = state.jobs[kind];
        renderToolbar();
      }
      if (job?.job_id === jobId) await onJobSettled(kind, job);
    } catch (error) {
      refuse(kind, error);
    } finally {
      polling.delete(jobId);
    }
  }

  /**
   * Start a job of ``kind`` unless one is running or being started: the job
   * is marked running only once the server answers, so a second click in
   * between would otherwise send a second start the server refuses.
   * @param {JobKind} kind
   */
  async function start(kind) {
    if (state.jobs[kind]?.status === "running" || starting.has(kind)) return;
    starting.add(kind);
    state.errors[kind] = "";
    let job;
    try {
      job = /** @type {JobStatus} */ (await client.jobStart(kind));
    } catch (error) {
      refuse(kind, error);
      return;
    } finally {
      starting.delete(kind);
    }
    state.jobs[kind] = job;
    renderToolbar();
    try {
      await poll(kind, job.job_id);
    } catch (error) {
      refuse(kind, error);
    }
  }

  /** @param {JobKind} kind */
  async function cancel(kind) {
    const job = state.jobs[kind];
    if (job?.status !== "running") return;
    try {
      await client.jobCancel(job.job_id);
      state.jobs[kind] = { ...job, cancel_requested: true };
      renderToolbar();
    } catch (error) {
      refuse(kind, error);
    }
  }

  /** @param {Event} event */
  function onClick(event) {
    const target = event.target instanceof Element ? event.target : null;
    const starter = target?.closest("[data-cv-start]");
    const canceller = target?.closest("[data-cv-cancel]");
    const termButton = target?.closest("[data-cv-term]");
    if (starter instanceof HTMLButtonElement) void start(jobKind(starter.dataset.cvStart || ""));
    else if (canceller instanceof HTMLElement) void cancel(jobKind(canceller.dataset.cvCancel || ""));
    else if (termButton instanceof HTMLElement) {
      state.term = termButton.dataset.cvTerm || "";
      renderTerms();
    }
  }

  /** @param {Event} event */
  function onInput(event) {
    if (!(event.target instanceof HTMLInputElement) || event.target.id !== "cvTermSearch") return;
    state.query = event.target.value;
    renderTerms();
  }

  /** @param {KeyboardEvent} event */
  function onKeyDown(event) {
    const input = event.target;
    if (event.key !== "Escape" || !(input instanceof HTMLInputElement) || input.id !== "cvTermSearch") return;
    input.value = "";
    state.query = "";
    renderTerms();
  }

  /** @param {EventTarget|null} target @returns {HTMLElement|null} */
  function foldKey(target) {
    const key = target instanceof Element ? target.closest("[data-cv-fold-key]") : null;
    return key instanceof HTMLElement ? key : null;
  }

  // A legend entry picks its fold out while hovered or focused.
  /** @param {Event} event */
  function onFoldKeyEnter(event) {
    const key = foldKey(event.target);
    const chart = key?.closest("[data-cv-chart]");
    if (key && chart) highlightFold(chart, key.dataset.cvFoldKey ?? null);
  }

  /** @param {Event} event */
  function onFoldKeyLeave(event) {
    const key = foldKey(event.target);
    const chart = key?.closest("[data-cv-chart]");
    if (!key || !chart) return;
    const next = /** @type {MouseEvent|FocusEvent} */ (event).relatedTarget;
    if (next instanceof Node && key.contains(next)) return;
    // The mouse leaving gives the highlight back to a focused entry.
    const focused = event.type === "mouseout" ? foldKey(frame.ownerDocument.activeElement) : null;
    highlightFold(chart, focused && chart.contains(focused) ? focused.dataset.cvFoldKey ?? null : null);
  }

  frame.addEventListener("click", onClick);
  frame.addEventListener("input", onInput);
  frame.addEventListener("keydown", onKeyDown);
  frame.addEventListener("mouseover", onFoldKeyEnter);
  frame.addEventListener("focusin", onFoldKeyEnter);
  frame.addEventListener("mouseout", onFoldKeyLeave);
  frame.addEventListener("focusout", onFoldKeyLeave);

  return Object.freeze({
    /** @param {CVReportPayload} next */
    render(next) {
      payload = next;
      for (const kind of JOB_KINDS) {
        state.jobs[kind] = newerJob(state.jobs[kind], next.jobs[kind]);
        const job = state.jobs[kind];
        if (job?.status === "running") void poll(kind, job.job_id);
      }
      renderAll();
    },
    start,
    cancel,
    destroy() {
      frame.removeEventListener("click", onClick);
      frame.removeEventListener("input", onInput);
      frame.removeEventListener("keydown", onKeyDown);
      frame.removeEventListener("mouseover", onFoldKeyEnter);
      frame.removeEventListener("focusin", onFoldKeyEnter);
      frame.removeEventListener("mouseout", onFoldKeyLeave);
      frame.removeEventListener("focusout", onFoldKeyLeave);
    }
  });
}
