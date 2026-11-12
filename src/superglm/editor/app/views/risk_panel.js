// @ts-check
/**
 * The Explain tab of a SuperLSS session: the whole predicted distribution of up to eight risks, a
 * value for every feature each. Two plots share the amount axis: how likely each amount is, and
 * the chance of an amount above each one. A table reads the same laws at fixed points and at the
 * line, and `risk_drivers.js` walks to the selected risk a feature at a time. A model of counts
 * draws `risk_counts.js`'s plot instead.
 *
 * `riskModel` is pure and holds every number the view shows. `renderRiskPanel` draws it and reports
 * what the analyst changes through its handlers; the caller owns the risks and the line.
 */

import { meanAboveZero, medianAboveZero } from "../distribution/families.js";
import { formatAmount, formatChance, parseLine, parseOneInN } from "../distribution/line.js";
import { fmtSignificant } from "../format.js";
import { el, patchChildren } from "../chart/svg.js";
import { decadeLabel, decadeTicks } from "../chart/ridges.js";
import { RISK_IDS } from "../state/risk.js";
import { clampedLaw, lawInputs, lineModel, segmented } from "./distribution_panel.js";
import { CHANCE_TICKS, countColumns, countLaw, drawCounts } from "./risk_counts.js";
import { driversModel, parameterValue, renderDrivers, shownValue as shown } from "./risk_drivers.js";

/** @typedef {import('../api/contracts.js').RiskResponse} RiskResponse */
/** @typedef {import('../api/contracts.js').RiskFeature} RiskFeature */
/** @typedef {import('../api/contracts.js').RiskProfiles} RiskProfiles */
/** @typedef {import('../api/contracts.js').RiskId} RiskId */
/** @typedef {import('../distribution/families.js').Law} Law */
/** @typedef {import('./distribution_panel.js').LineState} LineState */
/** @typedef {import('./distribution_panel.js').LineModel} LineModel */
/** @typedef {{min_log10: number, max_log10: number, clipped_below: boolean}} RiskAxis */
/** @typedef {import('./risk_counts.js').CountCurve} CountCurve */
/** @typedef {{label: string, title: string, read: (value: {law: Law, cv: number}) => number, show: (value: number) => string}} AmountColumn */

/**
 * @typedef {object} RiskCurve
 * @property {RiskId} id
 * @property {string} name
 * @property {Law} law  the model in force
 * @property {Law | null} original  the fitted model's law, where the edits moved this risk
 * @property {number} cv  the coefficient of variation in force
 * @property {number | null} originalCv
 * @property {boolean} spreadPinned
 */
/**
 * @typedef {object} RiskTable
 * @property {string | null} unit  the amounts' unit, shown once
 * @property {Array<{label: string, title: string}>} columns  the figures read from each law
 * @property {Array<{swatch: RiskId | 'original' | null, label: string, cells: string[]}>} rows  a
 *   risk each, then the ratio of the two rows where there are exactly two
 */
/**
 * @typedef {object} RiskModel
 * @property {string | null} empty  the reason the view draws nothing, or null
 * @property {RiskCurve[]} curves
 * @property {CountCurve[] | null} counts  the risks of a model of counts, which has no curves
 * @property {RiskId} selected  the risk the plots fill and the walk ends at
 * @property {LineModel} line
 * @property {RiskAxis | null} axis
 * @property {string | null} unit
 * @property {RiskTable} table
 * @property {string | null} spreadNote
 */
/**
 * @typedef {object} RiskHandlers
 * @property {(id: RiskId, column: string, value: number | string) => void} onValue
 * @property {() => void} onAdd  one more risk, a training row drawn at random
 * @property {(id: RiskId) => void} onCopy  one more risk, a copy of this one
 * @property {(id: RiskId) => void} onRemove
 * @property {(id: RiskId) => void} onSelect
 * @property {(id: RiskId) => void} onReset
 * @property {(id: RiskId) => void} onBookRow
 * @property {(next: LineState) => void} onLine
 * @property {() => void} onRetry
 */
/**
 * @typedef {object} RiskView
 * @property {RiskResponse | null} reply  the last answer, drawn while a newer one loads
 * @property {boolean} loading
 * @property {string | null} message  what to say instead of the plots
 * @property {boolean} retry  whether `message` offers Retry
 * @property {RiskProfiles} profiles
 * @property {LineState} line
 * @property {number} width  the plots' room in pixels
 * @property {number} height
 */

export const RISK_LOADING = "Loading this risk's distribution...";
export const RISK_FAILED =
  "This risk's distribution could not be loaded: the editor's server did not answer. Retry to load it again.";
export const RISK_SUPERSEDED =
  "The request was overtaken by a newer change. Retry to refresh this view.";
export const RISK_UNDRAWABLE =
  "This risk's distribution cannot be drawn: a predicted amount leaves the range the browser computes. Change the risk's values, or undo the last edit.";
const SPREAD_NOTE = "A CV above the largest the view computes is drawn at that largest.";

/** @param {RiskId} id */
const nameOf = (id) => `Risk ${id.toUpperCase()}`;
// The chance axis runs from certain down to 1 in 10,000, as the panel's dot column does.
const CHANCE_FLOOR = -4;
const DEFAULT_PERCENTILE = 0.99;
const SAMPLES = 160;
const AXIS_STEP = 0.25;

/** @param {readonly number[]} a @param {readonly number[]} b */
const sameEta = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

/**
 * The coefficient of variation of a risk with these predictors: what the gamma and the log-normal
 * show for their second parameter, and sqrt(dispersion x mean^(power - 2)) for a Tweedie.
 * @param {RiskResponse} reply @param {ReturnType<typeof lawInputs>} inputs @param {readonly number[]} eta
 */
function cvOf(reply, inputs, eta) {
  const [mean, second, power] = inputs.order.map((at) => parameterValue(reply, at, eta[at]));
  return reply.family === "tweedie" ? Math.sqrt(second * mean ** (power - 2)) : second;
}

/** @param {number} second @param {number} first */
function ratioText(second, first) {
  const ratio = second / first;
  return Number.isFinite(ratio) && first > 0 ? `×${fmtSignificant(ratio, 3)}` : "";
}

/**
 * The table: a row per risk and a column per figure. One risk the session's edits moved is read
 * under the original model too, and two rows are followed by their ratio.
 * @template T
 * @param {Array<{id: RiskId, inForce: T, original: T | null}>} risks
 * @param {Array<{label: string, title: string, read: (value: T) => number, show: (value: number) => string}>} columns
 * @param {string | null} unit
 * @returns {RiskTable}
 */
function tableOf(risks, columns, unit) {
  const [only] = risks;
  /** @type {Array<{swatch: RiskId | 'original', label: string, value: T}>} */
  const sides = risks.length === 1 && only.original
    ? [
      { swatch: "original", label: `${only.id.toUpperCase()}, original`, value: only.original },
      { swatch: only.id, label: only.id.toUpperCase(), value: only.inForce },
    ]
    : risks.map((risk) => ({ swatch: risk.id, label: risk.id.toUpperCase(), value: risk.inForce }));
  const values = sides.map((side) => columns.map((column) => column.read(side.value)));
  /** @type {RiskTable["rows"]} */
  const rows = sides.map((side, i) => ({
    swatch: side.swatch, label: side.label, cells: values[i].map((value, c) => columns[c].show(value)),
  }));
  if (sides.length === 2) {
    rows.push({
      swatch: null,
      label: sides[0].swatch === "original" ? "Edited ÷ original" : `${sides[1].label} ÷ ${sides[0].label}`,
      cells: columns.map((_, c) => ratioText(values[1][c], values[0][c])),
    });
  }
  return { unit, columns: columns.map(({ label, title }) => ({ label, title })), rows };
}

// The percentiles the table always reads. A line at one of them would repeat its column.
const FIXED_PERCENTILES = [0.5, 0.9, 0.99, 0.995];
const SAME_PERCENTILE = 1e-9;

/**
 * The column the line adds: the chance above an amount; or, in percentile mode, the amount at its
 * percentile, unless a column already reads that percentile. A line written "1 in N" heads its
 * column "1 in N".
 * @param {LineModel} line @param {LineState} state
 * @param {(label: string, title: string, p: number) => AmountColumn} at
 * @returns {AmountColumn[]}
 */
export function lineColumn(line, state, at) {
  if (line.mode === "amount") {
    return [{
      label: `Above ${line.text}`,
      title: "The chance of an amount above the line.",
      read: ({ law }) => law.sf(line.value),
      show: (p) => formatChance(p, state.style),
    }];
  }
  if (FIXED_PERCENTILES.some((p) => Math.abs(p - line.p) < SAME_PERCENTILE)) return [];
  const oneIn = state.style === "one_in" && 1 - state.value < 0.5;
  return [at(oneIn ? formatChance(1 - state.value, "one_in") : `At ${line.label}`, "The amount at the line's percentile.", line.p)];
}

/**
 * The table's columns for a law of amounts: fixed points of the law, the line, and the CV; and
 * the chance of no claim where the family has one (`atom`).
 * @param {LineModel} line @param {LineState} state @param {boolean} atom
 * @returns {AmountColumn[]}
 */
function amountColumns(line, state, atom) {
  /** @param {number} v */
  const amount = (v) => formatAmount(v, null);
  /** @param {string} label @param {string} title @param {number} p */
  const at = (label, title, p) => ({ label, title, read: (/** @type {{law: Law}} */ { law }) => law.quantile(p), show: amount });
  return [
    { label: "Mean", title: "The expected amount.", read: ({ law }) => law.mean, show: amount },
    ...(atom
      ? [{
        label: "No claim",
        title: "The chance of an amount of exactly zero.",
        read: (/** @type {{law: Law}} */ { law }) => law.zero,
        show: (/** @type {number} */ p) => formatChance(p, state.style),
      }]
      : []),
    at("Median", "Half of amounts fall below it.", 0.5),
    at("90th", "The 90th percentile: 9 amounts in 10 fall below it.", 0.9),
    at("99th", "The 99th percentile: 99 amounts in 100 fall below it.", 0.99),
    at("99.5th", "The 99.5th percentile: 1 amount in 200 is above it.", 0.995),
    ...lineColumn(line, state, at),
    {
      label: "CV",
      title: "The coefficient of variation: the standard deviation divided by the mean.",
      read: ({ cv }) => cv,
      show: (v) => fmtSignificant(v, 3),
    },
  ];
}

/**
 * The amounts the plots span: from the smallest 1st percentile drawn to the largest amount
 * exceeded once in 10,000, where the chance plot ends. The ends are widened to quarter decades,
 * so a small change to a risk leaves the axis where it is, and a quantile float64 cannot hold
 * falls back to the server's axis.
 * @param {RiskCurve[]} curves @param {RiskAxis} axis the server's axis
 * @returns {RiskAxis}
 */
export function plotAxis(curves, axis) {
  const laws = curves.flatMap((curve) => (curve.original ? [curve.law, curve.original] : [curve.law]));
  // Where a law has a chance of exactly zero, the low end is read from the amounts above zero.
  const low = Math.min(...laws.map((law) => Math.log10(law.quantile(law.zero + 0.01 * (1 - law.zero)))));
  const high = Math.max(...laws.map((law) => Math.log10(law.quantile(1 - 10 ** CHANCE_FLOOR))));
  const min = Number.isFinite(low)
    ? Math.max(axis.min_log10, Math.floor(low / AXIS_STEP) * AXIS_STEP)
    : axis.min_log10;
  const max = Number.isFinite(high) ? Math.ceil(high / AXIS_STEP) * AXIS_STEP : axis.max_log10;
  return max > min ? { min_log10: min, max_log10: max, clipped_below: false } : axis;
}

/**
 * The amounts to label: whole decades, with 2 and 5 between them where a decade is wide enough
 * to hold them, or every few decades where it is narrow.
 * @param {RiskAxis} axis @param {number} plotWidth @param {string | null} unit
 * @returns {Array<{l: number, label: string}>}
 */
export function amountTicks(axis, plotWidth, unit) {
  const perDecade = plotWidth / (axis.max_log10 - axis.min_log10);
  if (perDecade < 150) {
    return decadeTicks(axis, plotWidth, 56).map((decade) => ({ l: decade, label: decadeLabel(decade, unit) }));
  }
  const ticks = [];
  for (let decade = Math.floor(axis.min_log10); decade <= Math.ceil(axis.max_log10); decade += 1) {
    for (const mantissa of [1, 2, 5]) {
      const l = decade + Math.log10(mantissa);
      if (l < axis.min_log10 || l > axis.max_log10) continue;
      ticks.push({ l, label: mantissa === 1 ? decadeLabel(decade, unit) : compactAmount(mantissa * 10 ** decade, unit) });
    }
  }
  return ticks;
}

/** @param {number} v @param {string | null} unit */
function compactAmount(v, unit) {
  const [scaled, suffix] = v >= 1e6 ? [v / 1e6, "m"] : v >= 1e3 ? [v / 1e3, "k"] : [v, ""];
  return `${unit ?? ""}${Number(scaled.toPrecision(3))}${suffix}`;
}

/**
 * Every number the view shows.
 * @param {RiskResponse} reply @param {LineState} state
 * @param {RiskId} [selected] the first risk where the reply holds no such risk
 * @returns {RiskModel}
 */
export function riskModel(reply, state, selected) {
  const unit = reply.amount_unit;
  const risks = reply.risks.slice(0, RISK_IDS.length);
  const ids = risks.map((risk, i) => risk.id ?? RISK_IDS[i]);
  const chosen = selected !== undefined && ids.includes(selected) ? selected : ids[0] ?? "a";
  if (reply.kind === "counts") {
    const counts = risks.map((risk, i) => ({
      id: ids[i],
      name: nameOf(ids[i]),
      inForce: countLaw(reply, risk.eta.edited),
      original: sameEta(risk.eta.edited, risk.eta.fitted) ? null : countLaw(reply, risk.eta.fitted),
    }));
    return {
      empty: null, curves: [], counts, selected: chosen, line: lineModel(reply, state), axis: null, unit,
      table: tableOf(counts, countColumns(reply, state.style), null), spreadNote: null,
    };
  }
  if (reply.kind === "unavailable" || !reply.amount_axis || !risks.length) {
    return {
      empty: reply.reason ?? "This model has no distribution to draw.",
      curves: [], counts: null, selected: chosen, line: lineModel(reply, state), axis: null, unit,
      table: { unit, columns: [], rows: [] }, spreadNote: null,
    };
  }
  const inputs = lawInputs(reply);
  const curves = risks.map((risk, i) => {
    const edited = risk.eta.edited;
    const inForce = clampedLaw(reply, inputs, edited);
    const moved = !sameEta(edited, risk.eta.fitted);
    const fitted = moved ? clampedLaw(reply, inputs, risk.eta.fitted) : null;
    return /** @type {RiskCurve} */ ({
      id: ids[i],
      name: nameOf(ids[i]),
      law: inForce.law,
      original: fitted ? fitted.law : null,
      cv: cvOf(reply, inputs, edited),
      originalCv: fitted ? cvOf(reply, inputs, risk.eta.fitted) : null,
      spreadPinned: inForce.spreadPinned || Boolean(fitted?.spreadPinned),
    });
  });
  const axis = plotAxis(curves, reply.amount_axis);
  const line = lineModel({ amount_unit: unit, amount_axis: axis }, state);
  const sides = curves.map((curve) => ({
    id: curve.id,
    inForce: { law: curve.law, cv: curve.cv },
    original: curve.original ? { law: curve.original, cv: curve.originalCv ?? curve.cv } : null,
  }));
  return {
    empty: null,
    curves,
    counts: null,
    selected: chosen,
    line,
    axis,
    unit,
    table: tableOf(sides, amountColumns(line, state, reply.family === "tweedie"), unit),
    spreadNote: curves.some((curve) => curve.spreadPinned) ? SPREAD_NOTE : null,
  };
}

/**
 * The model, or the message to show instead when a law refuses its predictors.
 * @param {RiskResponse} reply @param {LineState} state @param {RiskId} [selected]
 * @returns {{ok: true, model: RiskModel} | {ok: false, message: string}}
 */
export function guardedRiskModel(reply, state, selected) {
  try {
    return { ok: true, model: riskModel(reply, state, selected) };
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    return { ok: false, message: RISK_UNDRAWABLE };
  }
}

/**
 * The density of log10 amount at `l`, given that the amount is above zero: a law with a chance
 * of exactly zero is drawn over its positive part, so every curve has an area of one.
 * @param {Law} law @param {number} l
 */
export const densityAbove = (law, l) => 10 ** law.log10Density(l) / (1 - law.zero);

/**
 * A law sampled across the axis: its density given an amount above zero, and log10 of the chance
 * of an amount above each point, the chance of zero included.
 * @param {Law} law @param {RiskAxis} axis @param {number} [samples]
 * @returns {{l: number[], density: number[], chance: number[]}}
 */
export function sampleLaw(law, axis, samples = SAMPLES) {
  const span = axis.max_log10 - axis.min_log10;
  const l = Array.from({ length: samples + 1 }, (_, j) => axis.min_log10 + (span * j) / samples);
  return {
    l,
    density: l.map((v) => densityAbove(law, v)),
    chance: l.map((v) => Math.log10(law.sf(10 ** v))),
  };
}

/**
 * The density axis's ticks: round steps of 1, 2 or 5, two to six of them up to the peak.
 * @param {number} peak the largest density drawn
 * @returns {number[]}
 */
export function densityTicks(peak) {
  if (!(peak > 0) || !Number.isFinite(peak)) return [];
  const decade = 10 ** Math.floor(Math.log10(peak / 3));
  const step = [1, 2, 5, 10].map((m) => m * decade).find((candidate) => peak / candidate <= 6) ?? decade;
  return Array.from({ length: Math.floor(peak / step) }, (_, i) => Number(((i + 1) * step).toPrecision(12)));
}

/**
 * Where the plots sit in an SVG of `width` by `height`: the density above, the chance below, one
 * amount axis. The two share what the title rows and the axis leave of the height.
 * @param {number} width @param {number} [height]
 */
export function riskLayout(width, height = 406) {
  const left = 58;
  const right = Math.max(left + 120, width - 18);
  const each = Math.max(110, (height - 120) / 2);
  const density = { top: 30, bottom: 30 + each };
  const chance = { top: density.bottom + 46, bottom: density.bottom + 46 + each };
  return { width, left, right, density, chance, axisY: chance.bottom + 16, total: chance.bottom + 44 };
}

/**
 * The scales of the two plots.
 * @param {RiskAxis} axis @param {ReturnType<typeof riskLayout>} layout @param {number} peak the
 *   largest density drawn
 */
export function riskScales(axis, layout, peak) {
  const span = axis.max_log10 - axis.min_log10;
  const px = layout.right - layout.left;
  const { density, chance } = layout;
  return {
    /** @param {number} l */
    x: (l) => layout.left + ((l - axis.min_log10) / span) * px,
    /** @param {number} x */
    l: (x) => axis.min_log10 + ((x - layout.left) / px) * span,
    /** @param {number} d */
    density: (d) => density.bottom - (d / peak) * (density.bottom - density.top) * 0.94,
    /** @param {number} c log10 chance */
    chance: (c) => chance.top + (Math.max(c, CHANCE_FLOOR) / CHANCE_FLOOR) * (chance.bottom - chance.top),
    /** @param {number} y */
    chanceAt: (y) => (CHANCE_FLOOR * (y - chance.top)) / (chance.bottom - chance.top),
  };
}

/**
 * The line a press on the plots asks for: the amount under the pointer, or, in percentile mode on
 * the chance plot, the percentile whose chance above is the one under the pointer.
 * @param {RiskAxis} axis @param {ReturnType<typeof riskLayout>} layout @param {LineState} state
 * @param {number} x @param {number} y
 * @returns {LineState | null}
 */
export function lineAt(axis, layout, state, x, y) {
  if (x < layout.left || x > layout.right) return null;
  const scales = riskScales(axis, layout, 1);
  if (state.mode === "amount") {
    if (y < layout.density.top || y > layout.chance.bottom) return null;
    return { ...state, value: Number((10 ** scales.l(x)).toPrecision(3)) };
  }
  if (y < layout.chance.top || y > layout.chance.bottom) return null;
  const chance = 10 ** scales.chanceAt(y);
  return { ...state, value: 1 - Number(chance.toPrecision(3)) };
}

// ---------------------------------------------------------------------------------------------
// The view

/**
 * @typedef {object} RiskNodes
 * @property {HTMLElement} note
 * @property {HTMLElement} noteText
 * @property {HTMLButtonElement} retry
 * @property {HTMLElement} rowsHost
 * @property {HTMLTableElement} rows  a row of controls per risk
 * @property {HTMLButtonElement} addButton
 * @property {string} rowsKey  the features and risks the rows were built for
 * @property {HTMLElement} main  the plots beside the table and the walk
 * @property {HTMLButtonElement[]} modeButtons
 * @property {HTMLButtonElement[]} styleButtons
 * @property {HTMLInputElement} box  the line's value, as a number alone
 * @property {HTMLElement} boxPrefix  the fixed "1 in" before a box that takes only N
 * @property {HTMLElement} boxUnit  the amount's unit, after the box
 * @property {HTMLElement} hint
 * @property {string | null} invalid  why the box's text was not read, while it stands
 * @property {{amount: number | null, percentile: number | null}} remembered  the last line of each kind
 * @property {{state: LineState, reply: RiskResponse} | null} current
 * @property {HTMLElement} plotsHost  the plots are drawn its width and height
 * @property {SVGSVGElement} plots
 * @property {HTMLTableElement} table
 * @property {HTMLElement} drivers  the walk to the selected risk
 * @property {RiskResponse | null} driversOf  the reply the walk was drawn from
 * @property {RiskId | null} driversTo  the risk it ends at
 * @property {RiskHandlers | null} handlers
 * @property {(x: number, y: number) => LineState | null} drag
 */

/** @type {WeakMap<HTMLElement, RiskNodes>} */
const MOUNTED = new WeakMap();
const DIAMOND = "M0,-4.5 L4.5,0 L0,4.5 L-4.5,0 Z";
// Letters over the curves keep at least this many pixels between them.
const LETTER_GAP = 12;
const ICONS = /** @type {const} */ ({
  add: "M5 12h14M12 5v14",
  book: "M16 3h5v5M4 20 21 3M21 16v5h-5M15 15l6 6M4 4l5 5",
  copy: "M9 9h11v11H9zM5 15H4V4h11v1",
  reset: "M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5",
  remove: "M18 6 6 18M6 6l12 12",
});
const ADD_LABEL = "Add a risk: a training row drawn at random";
const COPY_LABEL = "Add a copy of this risk, to change one value and compare";
const FULL_LABEL = `The tab holds up to ${RISK_IDS.length} risks`;

/**
 * @template {keyof HTMLElementTagNameMap} K
 * @param {K} tag @param {Record<string, string>} attrs @param {string} [content]
 * @returns {HTMLElementTagNameMap[K]}
 */
function html(tag, attrs, content) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (content !== undefined) node.textContent = content;
  return node;
}

/**
 * @param {Element} parent @param {string} tag @param {Record<string, string | number>} attrs
 * @param {string} [content]
 */
function add(parent, tag, attrs, content) {
  const node = el(tag, attrs);
  if (content !== undefined) node.textContent = content;
  parent.appendChild(node);
  return node;
}

/** A button that is an icon alone, named on hover. @param {keyof typeof ICONS} name @param {string} label @param {RiskId} [id] */
function iconButton(name, label, id) {
  const button = html("button", {
    type: "button", class: "risk-icon-button", "data-risk-action": name, title: label, "aria-label": label,
    ...(id ? { "data-risk": id } : {}),
  });
  const svg = el("svg", { class: "risk-icon", viewBox: "0 0 24 24", "aria-hidden": "true" });
  svg.appendChild(el("path", { d: ICONS[name] }));
  button.append(svg);
  return button;
}

/** @param {HTMLElement} root @returns {RiskNodes} */
function mount(root) {
  const known = MOUNTED.get(root);
  if (known) return known;
  root.textContent = "";
  const note = html("div", { class: "distribution-message risk-note", hidden: "" });
  const noteText = html("p", {});
  const retry = html("button", { type: "button", class: "evidence-retry", hidden: "" }, "Retry");
  note.append(noteText, retry);
  const rowsHost = html("div", { class: "risk-rows-host", hidden: "" });
  const rows = html("table", { class: "risk-rows" });
  const addButton = iconButton("add", ADD_LABEL);
  addButton.classList.add("risk-add");
  rowsHost.append(rows, addButton);

  const main = html("div", { class: "risk-main" });
  const pane = html("div", { class: "risk-plots-pane" });
  const bar = html("div", { class: "distribution-bar risk-linebar" });
  const mode = segmented("What the line marks", [["amount", "Amount"], ["percentile", "Percentile"]], "mode");
  const style = segmented("Show chances as", [["percent", "%"], ["one_in", "1 in N"]], "style");
  const boxLabel = html("label", { class: "distribution-line-control" });
  const boxPrefix = html("span", { class: "distribution-line-prefix", hidden: "" }, "1 in");
  const box = html("input", {
    type: "text", class: "distribution-line-input", inputmode: "decimal", spellcheck: "false",
    autocomplete: "off", "aria-label": "Line value: an amount like 25k, or a percentile like 99.5 or 1 in 200",
  });
  const boxUnit = html("span", { class: "distribution-line-unit", hidden: "" });
  boxLabel.append(boxPrefix, box, boxUnit);
  bar.append(mode.group, boxLabel, style.group);
  const hint = html("p", { class: "distribution-hint", "aria-live": "polite", hidden: "" });
  const plotsHost = html("div", { class: "risk-plots-host" });
  const plots = /** @type {SVGSVGElement} */ (el("svg", { class: "risk-plots", role: "img" }));
  plotsHost.append(plots);
  pane.append(bar, hint, plotsHost);
  const side = html("div", { class: "risk-side" });
  const table = html("table", { class: "risk-table" });
  const drivers = html("section", { class: "risk-drivers" });
  side.append(table, drivers);
  main.append(pane, side);
  root.append(rowsHost, note, main);

  /** @type {RiskNodes} */
  const nodes = {
    note, noteText, retry, rowsHost, rows, addButton, rowsKey: "", main,
    modeButtons: mode.buttons, styleButtons: style.buttons, box, boxPrefix, boxUnit, hint,
    invalid: null, remembered: { amount: null, percentile: null }, current: null,
    plotsHost, plots, table, drivers, driversOf: null, driversTo: null, handlers: null, drag: () => null,
  };
  bindLineControls(nodes);
  retry.addEventListener("click", () => nodes.handlers?.onRetry());
  rowsHost.addEventListener("change", (event) => onRowChange(nodes, event));
  rowsHost.addEventListener("click", (event) => onRowClick(nodes, event));
  // Working in a risk's row selects that risk.
  rowsHost.addEventListener("focusin", (event) => {
    const row = event.target instanceof Element ? event.target.closest("tr[data-risk-row]") : null;
    if (row instanceof HTMLElement) nodes.handlers?.onSelect(/** @type {RiskId} */ (row.dataset.riskRow));
  });
  bindLineDrag(nodes);
  MOUNTED.set(root, nodes);
  return nodes;
}

/** The line's controls: what it marks, its value, and how chances are written. @param {RiskNodes} nodes */
function bindLineControls(nodes) {
  for (const button of nodes.modeButtons) {
    button.addEventListener("click", () => {
      const current = nodes.current;
      const mode = /** @type {"amount" | "percentile"} */ (button.dataset.mode);
      if (!current || current.state.mode === mode) return;
      nodes.invalid = null;
      const value = nodes.remembered[mode] ??
        (mode === "amount" ? defaultRiskLine(current.reply).value : DEFAULT_PERCENTILE);
      nodes.handlers?.onLine({ ...current.state, mode, value });
    });
  }
  for (const button of nodes.styleButtons) {
    button.addEventListener("click", () => {
      const current = nodes.current;
      const style = /** @type {"percent" | "one_in"} */ (button.dataset.style);
      if (current && current.state.style !== style) nodes.handlers?.onLine({ ...current.state, style });
    });
  }
  nodes.box.addEventListener("keydown", (event) => {
    if (event.key === "Enter") nodes.box.blur();
  });
  nodes.box.addEventListener("change", () => {
    const current = nodes.current;
    if (!current) return;
    const oneInN = current.state.mode === "percentile" && current.state.style === "one_in";
    const parsed = oneInN
      ? parseOneInN(nodes.box.value)
      : parseLine(nodes.box.value, current.state.mode, current.reply.amount_unit);
    nodes.invalid = parsed.ok ? null : parsed.hint;
    if (parsed.ok) nodes.handlers?.onLine({ ...current.state, mode: parsed.mode, value: parsed.value });
    else syncLine(nodes, lineModel(current.reply, current.state), current.state, current.reply, null);
  });
}

/**
 * Keeps the line's controls in step with the line. The box holds a number alone, and is left as
 * typed while the analyst is in it or its text was not read. The line under it says why a text
 * was not read, where the line could not be drawn as asked, or `also`.
 * @param {RiskNodes} nodes @param {LineModel} line @param {LineState} state @param {RiskResponse} reply
 * @param {string | null} also
 */
function syncLine(nodes, line, state, reply, also) {
  nodes.current = { state, reply };
  nodes.remembered[state.mode] = state.value;
  for (const button of nodes.modeButtons) button.setAttribute("aria-pressed", String(button.dataset.mode === state.mode));
  for (const button of nodes.styleButtons) button.setAttribute("aria-pressed", String(button.dataset.style === state.style));
  const oneInN = state.mode === "percentile" && state.style === "one_in";
  nodes.boxPrefix.hidden = !oneInN;
  nodes.boxUnit.textContent = reply.amount_unit ?? "";
  nodes.boxUnit.hidden = state.mode !== "amount" || !reply.amount_unit;
  nodes.box.classList.toggle("is-one-in-n", oneInN);
  if (nodes.invalid !== null) {
    nodes.box.setAttribute("aria-invalid", "true");
  } else {
    nodes.box.removeAttribute("aria-invalid");
    if (nodes.box.ownerDocument.activeElement !== nodes.box) nodes.box.value = line.text;
  }
  const hint = nodes.invalid ?? line.hint ?? also ?? "";
  nodes.hint.textContent = hint;
  nodes.hint.hidden = hint === "";
  nodes.hint.classList.toggle("is-invalid", nodes.invalid !== null);
}

/** @param {RiskNodes} nodes @param {Event} event */
function onRowChange(nodes, event) {
  const target = event.target;
  if (!(target instanceof HTMLInputElement || target instanceof HTMLSelectElement)) return;
  const id = /** @type {RiskId | undefined} */ (target.dataset.risk);
  const column = target.dataset.column;
  if (!id || column === undefined) return;
  if (target instanceof HTMLSelectElement) {
    nodes.handlers?.onValue(id, column, target.value);
    return;
  }
  const value = Number(target.value);
  const valid = target.value.trim() !== "" && Number.isFinite(value);
  target.setAttribute("aria-invalid", String(!valid));
  if (valid) nodes.handlers?.onValue(id, column, value);
}

/** @param {RiskNodes} nodes @param {Event} event */
function onRowClick(nodes, event) {
  const button = event.target instanceof Element ? event.target.closest("button[data-risk-action]") : null;
  const handlers = nodes.handlers;
  if (!(button instanceof HTMLButtonElement) || !handlers) return;
  const id = /** @type {RiskId} */ (button.dataset.risk);
  const action = button.dataset.riskAction;
  if (action === "add") handlers.onAdd();
  else if (action === "select") handlers.onSelect(id);
  else if (action === "book") handlers.onBookRow(id);
  else if (action === "copy") handlers.onCopy(id);
  else if (action === "reset") handlers.onReset(id);
  else if (action === "remove") handlers.onRemove(id);
}

/** A press or a drag on the plots moves the line. @param {RiskNodes} nodes */
function bindLineDrag(nodes) {
  const svg = nodes.plots;
  let dragging = false;
  /** @param {PointerEvent} event */
  const move = (event) => {
    const box = svg.getBoundingClientRect();
    const next = nodes.drag(event.clientX - box.left, event.clientY - box.top);
    if (next) nodes.handlers?.onLine(next);
    return next !== null;
  };
  svg.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !move(event)) return;
    dragging = true;
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (dragging) move(event);
  });
  for (const type of ["pointerup", "pointercancel"]) {
    svg.addEventListener(type, () => {
      dragging = false;
    });
  }
}

/**
 * What a numeric box says on hover: its training rows' range.
 * @param {RiskFeature} feature
 */
function rangeTitle(feature) {
  return feature.min === null || feature.max === null
    ? feature.column
    : `${feature.column}: ${shown(feature.min)} to ${shown(feature.max)} in the training rows`;
}

/**
 * The sentence for a numeric value outside the training rows' range, where the model has no data
 * and extrapolates; null inside the range, or where the range is unknown.
 * @param {RiskFeature} feature @param {number | string} value
 * @returns {string | null}
 */
export function outsideRange(feature, value) {
  if (feature.kind !== "numeric" || feature.min === null || feature.max === null) return null;
  const number = Number(value);
  if (typeof value === "string" && value.trim() === "") return null;
  if (!Number.isFinite(number) || (number >= feature.min && number <= feature.max)) return null;
  return `Outside the training rows (${shown(feature.min)} to ${shown(feature.max)}): the model extrapolates here`;
}

/** A step that moves a numeric feature about a hundredth of its range. @param {RiskFeature} feature */
function stepOf(feature) {
  const span = (feature.max ?? 1) - (feature.min ?? 0);
  if (!(span > 0)) return "any";
  return String(10 ** Math.floor(Math.log10(span / 50)));
}

/**
 * One risk's row: its letter, which selects it; a number box for a numeric feature and a list
 * for a column of levels; and what can be done to it.
 * @param {RiskId} id @param {RiskFeature[]} features @param {boolean} removable
 */
function riskRow(id, features, removable) {
  const row = html("tr", { "data-risk-row": id });
  const head = html("th", { scope: "row" });
  const pick = html("button", {
    type: "button", class: "risk-pick", "data-risk-action": "select", "data-risk": id,
    "aria-pressed": "false", "aria-label": nameOf(id), title: `${nameOf(id)}: fill its curve and walk to it`,
  });
  pick.append(html("i", { class: "risk-swatch", "data-risk": id }), id.toUpperCase());
  head.append(pick);
  row.append(head);
  for (const feature of features) {
    const cell = html("td", {});
    const shared = { "data-risk": id, "data-column": feature.column, "aria-label": `${nameOf(id)}: ${feature.column}` };
    if (feature.kind === "numeric") {
      cell.append(html("input", {
        ...shared,
        type: "number",
        class: "risk-input",
        step: stepOf(feature),
        ...(feature.min === null ? {} : { min: String(feature.min) }),
        ...(feature.max === null ? {} : { max: String(feature.max) }),
        title: rangeTitle(feature),
      }));
    } else {
      const select = html("select", { ...shared, class: "risk-select" });
      for (const level of feature.levels ?? []) select.append(html("option", { value: level }, level));
      cell.append(select);
    }
    row.append(cell);
  }
  const actions = html("td", { class: "risk-actions" });
  actions.append(
    iconButton("book", "Take the values of a training row drawn at random", id),
    iconButton("reset", "Back to the reference risk", id),
    iconButton("copy", COPY_LABEL, id)
  );
  if (removable) actions.append(iconButton("remove", `Remove risk ${id.toUpperCase()}`, id));
  row.append(actions);
  return row;
}

/**
 * Builds the rows once per set of features and risks, then keeps their values and the selection
 * in step. A control the analyst is typing in is left alone.
 * @param {RiskNodes} nodes @param {RiskResponse} reply @param {RiskProfiles} profiles
 */
function syncRows(nodes, reply, profiles) {
  const ids = profiles.risks.map((risk) => risk.id);
  const key = JSON.stringify([reply.features, ids]);
  if (key !== nodes.rowsKey) {
    nodes.rowsKey = key;
    const head = html("tr", {});
    head.append(html("th", {}), ...reply.features.map((feature) => html("th", { scope: "col" }, feature.column)), html("th", {}));
    const thead = html("thead", {});
    thead.append(head);
    const tbody = html("tbody", {});
    tbody.append(...ids.map((id) => riskRow(id, reply.features, ids.length > 1)));
    nodes.rows.replaceChildren(thead, tbody);
  }
  const full = ids.length >= RISK_IDS.length;
  nodes.addButton.disabled = full;
  for (const name of ["title", "aria-label"]) nodes.addButton.setAttribute(name, full ? FULL_LABEL : ADD_LABEL);
  for (const copy of nodes.rows.querySelectorAll('button[data-risk-action="copy"]')) {
    /** @type {HTMLButtonElement} */ (copy).disabled = full;
    for (const name of ["title", "aria-label"]) copy.setAttribute(name, full ? FULL_LABEL : COPY_LABEL);
  }
  const active = nodes.rows.ownerDocument.activeElement;
  for (const risk of profiles.risks) {
    const row = nodes.rows.querySelector(`tr[data-risk-row="${risk.id}"]`);
    if (!row) continue;
    const selected = risk.id === profiles.selected;
    row.classList.toggle("is-selected", selected);
    row.querySelector(".risk-pick")?.setAttribute("aria-pressed", String(selected));
    const answered = reply.risks.find((candidate) => candidate.id === risk.id)?.values;
    for (const feature of reply.features) {
      const control = row.querySelector(`[data-column="${CSS.escape(feature.column)}"]`);
      if (!(control instanceof HTMLInputElement || control instanceof HTMLSelectElement)) continue;
      const held = risk.values[feature.column] ?? answered?.[feature.column] ?? feature.reference;
      const value = shown(held);
      if (control !== active && control.value !== value) control.value = value;
      if (control instanceof HTMLInputElement) {
        const outside = outsideRange(feature, held);
        control.toggleAttribute("data-outside", outside !== null);
        control.title = outside ?? rangeTitle(feature);
      }
    }
  }
}

/** @param {RiskNodes} nodes @param {RiskTable} table @param {RiskId} selected */
function renderTable(nodes, table, selected) {
  const head = html("tr", {});
  head.append(html("th", { scope: "col", class: "risk-unit" }, table.unit ?? ""));
  for (const column of table.columns) head.append(html("th", { scope: "col", title: column.title }, column.label));
  const rows = table.rows.map((row) => {
    const tr = html("tr", { class: row.swatch === null ? "risk-ratio" : row.swatch === selected ? "is-selected" : "" });
    const th = html("th", { scope: "row" });
    if (row.swatch) th.append(html("i", { class: "risk-swatch", "data-risk": row.swatch }));
    th.append(row.label);
    tr.append(th, ...row.cells.map((cell) => html("td", {}, cell)));
    return tr;
  });
  const thead = html("thead", {});
  thead.append(head);
  const tbody = html("tbody", {});
  tbody.append(...rows);
  nodes.table.replaceChildren(thead, tbody);
}

/** @param {number[]} ls @param {number[]} ys @param {(l: number) => number} x */
const pathOf = (ls, ys, x) =>
  ls.map((l, j) => `${j ? "L" : "M"}${x(l).toFixed(1)},${ys[j].toFixed(1)}`).join(" ");

/**
 * Draws the two plots and returns the rule for a press on them. The selected risk is filled,
 * marked at its median and mean, and drawn over the others; where the session's edits moved it,
 * the original model's curve for it is drawn in grey.
 * @param {RiskNodes} nodes @param {RiskModel} model @param {LineState} state
 * @param {number} width @param {number} height
 * @returns {(x: number, y: number) => LineState | null}
 */
function drawPlots(nodes, model, state, width, height) {
  const axis = /** @type {RiskAxis} */ (model.axis);
  const layout = riskLayout(width, height);
  const sampled = model.curves
    .map((curve) => {
      const chosen = curve.id === model.selected;
      return {
        curve,
        chosen,
        inForce: sampleLaw(curve.law, axis),
        original: chosen && curve.original ? sampleLaw(curve.original, axis) : null,
      };
    })
    .sort((a, b) => Number(a.chosen) - Number(b.chosen));
  const peak = Math.max(
    ...sampled.flatMap(({ inForce, original }) => [...inForce.density, ...(original?.density ?? [])])
  );
  const scales = riskScales(axis, layout, peak);
  const { density, chance } = layout;
  const next = el("svg", {});
  const oneIn = state.style === "one_in";

  // A law with a chance of exactly zero is drawn over the amounts above it, and says so.
  const atom = model.curves.some((curve) => curve.law.zero > 0);
  add(next, "text", { x: 2, y: 16, class: "dist-title" }, `How likely each amount is${atom ? ", given a claim" : ""}`);
  for (const tick of densityTicks(peak)) {
    const y = scales.density(tick);
    add(next, "line", { x1: layout.left, y1: y, x2: layout.right, y2: y, class: "risk-grid" });
    add(next, "text", { x: layout.left - 6, y: y + 3.5, class: "dist-muted", "text-anchor": "end" }, String(tick));
  }
  const middle = (density.top + density.bottom) / 2;
  add(next, "text", {
    x: 12, y: middle, class: "dist-muted", "text-anchor": "middle", transform: `rotate(-90 12 ${middle})`,
  }, "density");
  add(next, "text", { x: 2, y: chance.top - 14, class: "dist-title" }, "Chance of an amount above");

  // The shared amount axis: hairlines through both plots, labelled once at the foot.
  for (const tick of amountTicks(axis, layout.right - layout.left, model.unit)) {
    const x = scales.x(tick.l);
    add(next, "line", { x1: x, y1: density.top, x2: x, y2: density.bottom, class: "risk-grid" });
    add(next, "line", { x1: x, y1: chance.top, x2: x, y2: chance.bottom, class: "risk-grid" });
    add(next, "text", { x, y: layout.axisY, class: "dist-muted", "text-anchor": "middle" }, tick.label);
  }
  add(next, "line", { x1: layout.left, y1: density.bottom, x2: layout.right, y2: density.bottom, class: "risk-axis" });
  for (const [exponent, percent, one] of CHANCE_TICKS) {
    const y = scales.chance(exponent);
    add(next, "line", { x1: layout.left, y1: y, x2: layout.right, y2: y, class: exponent === CHANCE_FLOOR ? "risk-axis" : "risk-grid" });
    add(next, "text", { x: layout.left - 6, y: y + 3.5, class: "dist-muted", "text-anchor": "end" }, oneIn ? one : percent);
  }

  const clip = `risk-clip-${Math.round(width)}-${Math.round(height)}`;
  const clipPath = add(add(next, "defs", {}), "clipPath", { id: clip });
  add(clipPath, "rect", { x: layout.left, y: chance.top, width: layout.right - layout.left, height: chance.bottom - chance.top });

  /** @type {Array<{x: number, y: number}>} */
  const letters = [];
  for (const { curve, chosen, inForce, original } of sampled) {
    if (original) {
      add(next, "path", { d: pathOf(original.l, original.density.map(scales.density), scales.x), class: "risk-line risk-line-original" });
      add(next, "path", {
        d: pathOf(original.l, original.chance.map(scales.chance), scales.x),
        class: "risk-line risk-line-original", "clip-path": `url(#${clip})`,
      });
    }
    const top = pathOf(inForce.l, inForce.density.map(scales.density), scales.x);
    if (chosen) {
      const first = scales.x(inForce.l[0]).toFixed(1);
      const last = scales.x(inForce.l[inForce.l.length - 1]).toFixed(1);
      add(next, "path", { d: `${top} L${last},${density.bottom} L${first},${density.bottom} Z`, class: "risk-fill", "data-risk": curve.id });
    }
    const line = `risk-line${chosen ? " is-selected" : ""}`;
    add(next, "path", { d: top, class: line, "data-risk": curve.id });
    add(next, "path", {
      d: pathOf(inForce.l, inForce.chance.map(scales.chance), scales.x),
      class: line, "data-risk": curve.id, "clip-path": `url(#${clip})`,
    });
    if (sampled.length > 1) {
      const at = inForce.density.indexOf(Math.max(...inForce.density));
      const y = scales.density(inForce.density[at]) - 7;
      // Risks whose peaks coincide have their letters side by side.
      let x = scales.x(inForce.l[at]);
      while (letters.some((placed) => Math.abs(placed.x - x) < LETTER_GAP && Math.abs(placed.y - y) < LETTER_GAP)) x += LETTER_GAP;
      letters.push({ x, y });
      add(next, "text", { x, y, class: "dist-readout", "text-anchor": "middle" }, curve.id.toUpperCase());
    }
  }

  // The selected risk's median, a dot, and mean, a diamond, on its density: those of its amounts
  // above zero, as the curve is.
  const chosen = sampled[sampled.length - 1].curve;
  const claim = chosen.law.zero > 0 ? " claim" : "";
  for (const [value, mark] of /** @type {const} */ ([
    [medianAboveZero(chosen.law), "median"], [meanAboveZero(chosen.law), "mean"],
  ])) {
    const l = Math.log10(value);
    if (!(l >= axis.min_log10 && l <= axis.max_log10)) continue;
    const x = scales.x(l);
    const y = scales.density(densityAbove(chosen.law, l));
    const node = mark === "median"
      ? add(next, "circle", { cx: x, cy: y, r: 4, class: "risk-point", "data-risk": chosen.id })
      : add(next, "path", { d: DIAMOND, transform: `translate(${x.toFixed(1)},${y.toFixed(1)})`, class: "risk-point", "data-risk": chosen.id });
    add(node, "title", {}, `${chosen.name}: ${mark}${claim} ${formatAmount(value, model.unit)}`);
  }

  // The line, and where each risk meets it on the chance plot.
  if (model.line.mode === "amount" && Number.isFinite(model.line.log10)) {
    const x = scales.x(model.line.log10);
    add(next, "line", { x1: x, y1: density.top, x2: x, y2: chance.bottom, class: "risk-rule" });
    const label = model.line.label;
    const pill = Math.max(40, label.length * 6.4 + 14);
    const px = Math.min(Math.max(x, layout.left + pill / 2), layout.right - pill / 2);
    add(next, "rect", { x: px - pill / 2, y: layout.axisY + 6, width: pill, height: 17, rx: 8.5, class: "dist-pill" });
    add(next, "text", { x: px, y: layout.axisY + 18, class: "dist-pill-label", "text-anchor": "middle" }, label);
    for (const { curve } of sampled) {
      const beyond = curve.law.sf(model.line.value);
      const node = add(next, "circle", { cx: x, cy: scales.chance(Math.log10(beyond)), r: 4, class: "risk-point", "data-risk": curve.id });
      add(node, "title", {}, `${curve.name}: ${formatChance(beyond, state.style)} above ${label}`);
    }
  } else if (model.line.mode === "percentile") {
    const y = scales.chance(Math.log10(1 - model.line.p));
    add(next, "line", { x1: layout.left, y1: y, x2: layout.right, y2: y, class: "risk-rule" });
    add(next, "text", { x: layout.right, y: y - 5, class: "dist-muted", "text-anchor": "end" }, model.line.label);
    for (const { curve } of sampled) {
      const amount = curve.law.quantile(model.line.p);
      const l = Math.log10(amount);
      if (!(l >= axis.min_log10 && l <= axis.max_log10)) continue;
      const x = scales.x(l);
      add(next, "line", { x1: x, y1: y, x2: x, y2: chance.bottom, class: "risk-drop", "data-risk": curve.id });
      const node = add(next, "circle", { cx: x, cy: y, r: 4, class: "risk-point", "data-risk": curve.id });
      add(node, "title", {}, `${curve.name}: ${formatAmount(amount, model.unit)} at the ${model.line.label}`);
    }
  }

  nodes.plots.setAttribute("viewBox", `0 0 ${width} ${layout.total}`);
  nodes.plots.setAttribute("width", String(width));
  nodes.plots.setAttribute("height", String(layout.total));
  nodes.plots.setAttribute(
    "aria-label",
    `The predicted distribution of ${model.curves.map((curve) => curve.name).join(", ")}: the table beside it holds the figures`
  );
  nodes.plots.classList.toggle("is-amount", state.mode === "amount");
  patchChildren(nodes.plots, next);
  return (x, y) => lineAt(axis, layout, state, x, y);
}

/**
 * Shows a message where the plots would be, keeping the risks' rows when there are any.
 * @param {RiskNodes} nodes @param {string} content @param {boolean} retry
 */
function showNote(nodes, content, retry) {
  nodes.note.hidden = false;
  nodes.noteText.textContent = content;
  nodes.retry.hidden = !retry;
  nodes.main.hidden = true;
  nodes.drag = () => null;
}

/**
 * Draws the view.
 * @param {HTMLElement} root @param {RiskView} view @param {RiskHandlers} handlers
 */
export function renderRiskPanel(root, view, handlers) {
  const nodes = mount(root);
  nodes.handlers = handlers;
  root.setAttribute("aria-busy", String(view.loading));
  const reply = view.reply;
  nodes.rowsHost.hidden = !reply;
  if (reply) syncRows(nodes, reply, view.profiles);
  if (view.message !== null || !reply) {
    // Retry belongs to a failed request alone, never to one still loading.
    showNote(nodes, view.message ?? RISK_LOADING, view.message !== null && view.retry);
    return;
  }
  const guarded = guardedRiskModel(reply, view.line, view.profiles.selected);
  if (!guarded.ok || guarded.model.empty !== null) {
    showNote(nodes, guarded.ok ? /** @type {string} */ (guarded.model.empty) : guarded.message, false);
    return;
  }
  const model = guarded.model;
  nodes.note.hidden = true;
  nodes.main.hidden = false;
  syncLine(nodes, model.line, view.line, reply, model.spreadNote);
  // A model of counts has no line to place: only how a chance is written.
  for (const node of [nodes.modeButtons[0].parentElement, nodes.box.parentElement]) {
    node?.toggleAttribute("hidden", model.counts !== null);
  }
  renderTable(nodes, model.table, model.selected);
  const width = Math.max(360, Math.round(view.width || nodes.plotsHost.clientWidth || 560));
  const height = Math.max(340, Math.round(view.height || nodes.plotsHost.clientHeight || 406));
  if (model.counts) {
    drawCounts(nodes.plots, model.counts, { style: view.line.style, width, height, floor: CHANCE_FLOOR, selected: model.selected });
    nodes.drag = () => null;
  } else {
    try {
      nodes.drag = drawPlots(nodes, model, view.line, width, height);
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      showNote(nodes, RISK_UNDRAWABLE, false);
      return;
    }
  }
  if (nodes.driversOf !== reply || nodes.driversTo !== model.selected) {
    nodes.driversOf = reply;
    nodes.driversTo = model.selected;
    renderDrivers(nodes.drivers, driversModel(reply, model.selected));
  }
}

/** The element the plots fill, whose size the caller measures. @param {HTMLElement} root */
export function riskPlotsHost(root) {
  return mount(root).plotsHost;
}

/**
 * A line to start from where the tab has none yet: the amount nearest the first risk's 95th
 * percentile, of its amounts above zero, to one significant figure.
 * @param {RiskResponse} reply @returns {LineState}
 */
export function defaultRiskLine(reply) {
  const fallback = { mode: /** @type {const} */ ("amount"), value: 1e4, style: /** @type {const} */ ("percent") };
  if (reply.kind !== "risk" || !reply.risks.length) return fallback;
  try {
    const { law } = clampedLaw(reply, lawInputs(reply), reply.risks[0].eta.edited);
    const value = Number(law.quantile(law.zero + 0.95 * (1 - law.zero)).toPrecision(1));
    return value > 0 && Number.isFinite(value) ? { ...fallback, value } : fallback;
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    return fallback;
  }
}
