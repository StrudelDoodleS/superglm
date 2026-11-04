// @ts-check
/**
 * The distribution panel: for the active feature of a SuperLSS session, each row's predicted
 * distribution as translucent ridges, a density fan across the feature, and one
 * line shared by both. The line marks an amount (each row reads the chance of a value above it)
 * or a percentile (each row reads the amount at it), and is dragged, slid or typed.
 *
 * `panelModel` is pure and holds every number the view draws. `renderDistributionPanel` draws it
 * and reports line changes through `handlers.onChange`; the caller owns the line state.
 *
 * Two inputs are clamped into the laws' domain before they are used, and the panel says so:
 *  - a gamma spread whose coefficient of variation exceeds GAMMA_CV_MAX is drawn at GAMMA_CV_MAX
 *    (under the log link its predictor is ln CV, so the predictor is capped at ln GAMMA_CV_MAX);
 *  - a percentile outside [PROBABILITY_MIN, 1 - PROBABILITY_MIN] is computed at the nearer end.
 * An amount off the axis keeps its readouts and pins only the drawn line to the axis edge. The
 * text box always shows the value as typed.
 */

import { GAMMA_CV_MAX, PROBABILITY_MIN, lawOf } from "../distribution/families.js";
import {
  formatAmount,
  formatChance,
  formatOneInN,
  ordinal,
  parseLine,
  parseOneInN,
  pinToAxis,
} from "../distribution/line.js";
import { normCdf, normQuantile } from "../distribution/math.js";
import { fmt } from "../format.js";
import { el, patchChildren, setAttributes, text } from "../chart/svg.js";
import {
  amountScale,
  decadeLabel,
  decadeTicks,
  dotPosition,
  ridgeLayout,
  ridgeShape,
} from "../chart/ridges.js";
import {
  clampToPlot,
  columnAt,
  fanBeyondRegion,
  fanPixels,
  fanPositions,
  hexToRgb,
  niceTicks,
  spreadLabels,
} from "../chart/fan.js";

/** @typedef {import('../api/contracts.js').DistributionPanelResponse} DistributionPanel */
/** @typedef {import('../api/contracts.js').DistributionLink} DistributionLink */
/** @typedef {import('../distribution/families.js').Law} Law */
/** @typedef {import('../chart/fan.js').FanModel} FanModel */
/** @typedef {import('../chart/fan.js').FanColumn} FanColumn */
/** @typedef {import('../chart/fan.js').RGB} RGB */
/** @typedef {{min_log10: number, max_log10: number, clipped_below: boolean}} PanelAxis */

/**
 * The line: an amount, or a percentile p in (0, 1); chances written as % or "1 in N".
 * @typedef {{mode: "amount" | "percentile", value: number, style: "percent" | "one_in"}} LineState
 */
/**
 * Which predictors to draw: the payload's fitted or edited set, or explicit rows and columns.
 * @typedef {"fitted" | "edited" | {rows: number[][], columns: number[][]}} EtaSet
 */
/**
 * @typedef {object} RowModel
 * @property {string} label
 * @property {Law} law
 * @property {number} median
 * @property {number} mean
 * @property {number} beyond  the chance above the line (amount mode) or the amount at it
 * @property {string} readout
 * @property {string} [ghost]  the fitted row's readout, where the drawn predictors differ from it
 * @property {number} [ghostBeyond]  the fitted row's `beyond`
 * @property {number} cut  the log10 amount where the part beyond the line starts
 * @property {boolean} spreadPinned
 * @property {string} title  the row's hover text
 */
/**
 * @typedef {object} LineModel
 * @property {"amount" | "percentile"} mode
 * @property {number} value  as typed
 * @property {number} p  the probability computed with, in percentile mode; NaN in amount mode
 * @property {number} log10  where the amount line is drawn, on the axis; NaN in percentile mode
 * @property {boolean} pinned  the drawn line or the computed probability differs from `value`
 * @property {string} text  the text box's value
 * @property {string} label  the line's name on the charts
 * @property {string | null} hint  why the line is pinned
 */
/**
 * @typedef {object} PanelModel
 * @property {string | null} empty  the reason the panel draws nothing, or null
 * @property {string} feature
 * @property {RowModel[]} rows
 * @property {FanModel} fan
 * @property {LineModel} line
 * @property {PanelAxis | null} axis
 * @property {string | null} axisNote  hover text when the axis cuts the lower tail
 * @property {string | null} spreadNote  hover text and hint when a spread is drawn capped
 * @property {string | null} unit
 * @property {string} caption
 */
/** @typedef {{onChange: (next: LineState) => void}} PanelHandlers */

export const AXIS_NOTE = "Part of the lower tail lies below the axis.";
export const DISTRIBUTION_LOADING = "Loading the distributions...";
export const DISTRIBUTION_FAILED =
  "The distribution panel could not be loaded: the editor's server did not answer. Retry to load it again.";
export const DISTRIBUTION_SUPERSEDED =
  "The distribution request was superseded by a newer change. Retry to refresh this panel.";
export const DISTRIBUTION_UNDRAWABLE =
  "The distribution panel cannot draw these curves: a predicted amount leaves the range the browser computes. Undo the last edit, or bring the curve back toward its fitted values.";
const NO_PANEL = "This feature has no distribution to draw.";

// The gamma CV is exp(eta) under its log link.
const ETA_CV_MAX = Math.log(GAMMA_CV_MAX);
const P_MAX = 1 - PROBABILITY_MIN;
// Dragging and sliding in percentile mode cover the 50th to the 99.9th percentile; typing reaches
// every computable percentile.
export const SLIDER_P = /** @type {const} */ ([0.5, 0.999]);
const SLIDER_STEPS = 1000;
// The slider moves linearly in the normal quantile z between the ends of SLIDER_P.
const SLIDER_Z = SLIDER_P.map(normQuantile);
// The dot column's fixed log10 scale of chances: 0.01% to 100%.
const CHANCE_DOMAIN = /** @type {[number, number]} */ ([-4, 0]);
const DEFAULT_PERCENTILE = 0.99;

/**
 * The positions of the mean and the spread parameter in the payload's parameter order.
 * @param {DistributionPanel} panel
 * @returns {{order: [number, number], links: DistributionLink[]}}
 */
function lawInputs(panel) {
  const average = panel.parameters.findIndex((p) => p.role === "average");
  const spread = panel.parameters.findIndex((p) => p.role === "spread");
  return {
    order: [average, spread],
    links: [panel.parameters[average].link, panel.parameters[spread].link],
  };
}

/**
 * A law of the family at one row of predictors, the gamma CV capped at GAMMA_CV_MAX.
 * @param {DistributionPanel} panel @param {{order: [number, number], links: DistributionLink[]}} inputs
 * @param {readonly number[]} eta
 * @returns {{law: Law, spreadPinned: boolean}}
 */
function clampedLaw(panel, inputs, eta) {
  const mean = eta[inputs.order[0]];
  let spread = eta[inputs.order[1]];
  const spreadPinned = panel.family === "gamma" && spread > ETA_CV_MAX;
  if (spreadPinned) spread = ETA_CV_MAX;
  const family = /** @type {"lognormal_mean" | "gamma"} */ (panel.family);
  return { law: lawOf(family, [mean, spread], inputs.links), spreadPinned };
}

/**
 * @param {DistributionPanel} panel @param {EtaSet} etaSet
 * @returns {{rows: number[][], columns: number[][]}}
 */
function pickEta(panel, etaSet) {
  if (etaSet === "fitted") return panel.eta.fitted;
  if (etaSet === "edited") return panel.eta.edited;
  return etaSet;
}

/** @type {WeakMap<object, {rows: ClampedLaw[], columns: ClampedLaw[]}>} */
const LAWS_BY_ETA = new WeakMap();

/**
 * The laws of a set of predictors, one set of laws per set of predictors: a line that moves draws
 * from the same predictors, and laws cost far more to build than to read.
 * @param {DistributionPanel} panel @param {ReturnType<typeof lawInputs>} inputs
 * @param {{rows: number[][], columns: number[][]}} eta
 */
function lawsOf(panel, inputs, eta) {
  let laws = LAWS_BY_ETA.get(eta);
  if (!laws) {
    laws = {
      rows: eta.rows.map((row) => clampedLaw(panel, inputs, row)),
      columns: eta.columns.map((column) => clampedLaw(panel, inputs, column)),
    };
    LAWS_BY_ETA.set(eta, laws);
  }
  return laws;
}

/** @typedef {{law: Law, spreadPinned: boolean}} ClampedLaw */

/** @param {readonly number[]} a @param {readonly number[]} b */
const sameEta =(a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

/**
 * The line as drawn and read.
 * @param {DistributionPanel} panel @param {LineState} state
 * @returns {LineModel}
 */
export function lineModel(panel, state) {
  const unit = panel.amount_unit;
  const axis = panel.amount_axis;
  if (state.mode === "amount") {
    const shown = formatAmount(state.value, unit);
    const pin = axis ? pinToAxis(Math.log10(state.value), axis) : { log10: Number.NaN, pinned: false };
    const edge = axis && pin.log10 === axis.max_log10 ? "upper" : "lower";
    return {
      mode: "amount",
      value: state.value,
      p: Number.NaN,
      log10: pin.log10,
      pinned: pin.pinned,
      text: shown,
      label: shown,
      hint: pin.pinned
        ? `${shown} lies off the amount axis, so the line is drawn at its ${edge} edge; the readouts use ${shown}.`
        : null,
    };
  }
  const p = Math.min(P_MAX, Math.max(PROBABILITY_MIN, state.value));
  const pinned = p !== state.value;
  const oneIn = state.style === "one_in";
  const chance = formatChance(1 - state.value, "one_in");
  return {
    mode: "percentile",
    value: state.value,
    p,
    log10: Number.NaN,
    pinned,
    text: oneIn ? formatOneInN(1 - state.value) : ordinal(state.value),
    label: oneIn ? `${chance} above` : `${ordinal(state.value)} percentile`,
    hint: pinned
      ? `The panel computes percentiles from the ${ordinal(PROBABILITY_MIN)} to the ${ordinal(P_MAX)}, so the line is drawn at the ${ordinal(p)}.`
      : null,
  };
}

/**
 * The line's reading of one law: the chance above it, or the amount at it.
 * @param {Law} law @param {LineModel} line @param {LineState} state @param {string | null} unit
 */
function readLaw(law, line, state, unit) {
  if (line.mode === "amount") {
    const beyond = law.sf(line.value);
    return { beyond, readout: formatChance(beyond, state.style), cut: Math.log10(line.value) };
  }
  const beyond = law.quantile(line.p);
  return { beyond, readout: formatAmount(beyond, unit), cut: Math.log10(beyond) };
}

/** @param {unknown} value */
const referenceValue = (value) => (typeof value === "number" ? fmt(value) : String(value));

/**
 * @param {DistributionPanel} panel
 * @returns {string}
 */
function captionOf(panel) {
  const others = panel.reference.map(({ column, value }) => `${column} ${referenceValue(value)}`);
  const sentence = others.length
    ? `Every other feature at the reference row: ${others.join(", ")}.`
    : "No other feature enters the model.";
  return panel.cap_note ? `${sentence} ${panel.cap_note}` : sentence;
}

/**
 * Every number the panel draws, for one line state and one set of predictors.
 * @param {DistributionPanel} panel @param {LineState} state @param {EtaSet} etaSet
 * @returns {PanelModel}
 */
export function panelModel(panel, state, etaSet) {
  const line = lineModel(panel, state);
  const unit = panel.amount_unit;
  const base = {
    feature: panel.feature,
    line,
    unit,
    caption: captionOf(panel),
  };
  if (panel.kind === "empty" || !panel.amount_axis) {
    return {
      ...base,
      empty: panel.empty_reason ?? NO_PANEL,
      rows: [],
      fan: { kind: "numeric", columns: [] },
      axis: null,
      axisNote: null,
      spreadNote: null,
    };
  }
  const inputs = lawInputs(panel);
  const drawn = pickEta(panel, etaSet);
  const drawnLaws = lawsOf(panel, inputs, drawn);
  const fittedLaws = lawsOf(panel, inputs, panel.eta.fitted);
  const lineName = line.mode === "amount" ? `chance above ${line.label}` : `at the ${line.label}`;
  const pinnedNote = `The spread is drawn at a CV of ${GAMMA_CV_MAX}, the largest the panel computes.`;

  const rows = panel.rows.map((row, i) => {
    const label = row.level ?? fmt(Number(row.x));
    const { law, spreadPinned } = drawnLaws.rows[i];
    const reading = readLaw(law, line, state, unit);
    /** @type {RowModel} */
    const model = {
      label,
      law,
      median: law.median,
      mean: law.mean,
      ...reading,
      spreadPinned,
      title: "",
    };
    const fittedEta = panel.eta.fitted.rows[i];
    if (!sameEta(fittedEta, drawn.rows[i])) {
      const fitted = readLaw(fittedLaws.rows[i].law, line, state, unit);
      model.ghost = fitted.readout;
      model.ghostBeyond = fitted.beyond;
    }
    model.title = [
      `${panel.feature} ${label}: median ${formatAmount(law.median, unit)}, average ${formatAmount(law.mean, unit)}; ${lineName}: ${reading.readout}`,
      model.ghost === undefined ? null : `before the edit: ${model.ghost}`,
      spreadPinned ? pinnedNote : null,
    ].filter(Boolean).join(". ");
    return model;
  });

  const columns = panel.columns.map((column, i) => {
    const { law, spreadPinned } = drawnLaws.columns[i];
    /** @type {FanColumn} */
    const model = {
      x: column.x,
      level: column.level,
      law,
      mode: law.mode,
      median: law.median,
      mean: law.mean,
      cut: readLaw(law, line, state, unit).cut,
      spreadPinned,
    };
    return model;
  });

  const pinnedLabels = rows.filter((row) => row.spreadPinned).map((row) => row.label);
  const anyPinned = pinnedLabels.length > 0 || columns.some((column) => column.spreadPinned);
  return {
    ...base,
    empty: null,
    rows,
    fan: { kind: panel.kind === "levels" ? "levels" : "numeric", columns },
    axis: panel.amount_axis,
    axisNote: panel.amount_axis.clipped_below ? AXIS_NOTE : null,
    spreadNote: anyPinned
      ? `Where the spread's CV exceeds ${GAMMA_CV_MAX}, the panel draws it at ${GAMMA_CV_MAX}, the largest it computes${pinnedLabels.length ? `: ${panel.feature} ${pinnedLabels.join(", ")}` : ""}.`
      : null,
  };
}

/**
 * The panel model, or the message to show instead when a law refuses its predictors, as one
 * whose parameters leave float64 does.
 * @param {DistributionPanel} panel @param {LineState} state @param {EtaSet} etaSet
 * @returns {{ok: true, model: PanelModel} | {ok: false, message: string}}
 */
export function guardedPanelModel(panel, state, etaSet) {
  try {
    return { ok: true, model: panelModel(panel, state, etaSet) };
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    return { ok: false, message: DISTRIBUTION_UNDRAWABLE };
  }
}

// The client errors that say nothing about the request itself: wait, then ask again.
const RETRYABLE = new Set([408, 429]);

/**
 * What the panel says when its request fails, and whether a retry can help. A refusal from the
 * server (a 4xx reply) carries its own message saying what to change, so no retry is offered; a
 * failure to answer, a timeout (408) or too many requests (429) is offered one.
 * @param {unknown} error
 * @returns {{message: string, retry: boolean}}
 */
export function distributionFailure(error) {
  const status = error && typeof error === "object" && "status" in error ? Number(error.status) : NaN;
  const message = error instanceof Error ? error.message : "";
  const refusal = status >= 400 && status < 500 && !RETRYABLE.has(status);
  if (refusal && message) return { message, retry: false };
  return { message: DISTRIBUTION_FAILED, retry: true };
}

/**
 * A line to start from: the decade nearest the middle row's 95th percentile.
 * @param {DistributionPanel} panel
 * @returns {LineState}
 */
export function defaultLineState(panel) {
  const fallback = { mode: /** @type {const} */ ("amount"), value: 1e4, style: /** @type {const} */ ("percent") };
  if (panel.kind === "empty" || !panel.amount_axis || !panel.rows.length) return fallback;
  const inputs = lawInputs(panel);
  const middle = panel.eta.edited.rows[Math.floor(panel.rows.length / 2)];
  let q = Number.NaN;
  try {
    q = clampedLaw(panel, inputs, middle).law.quantile(0.95);
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
  }
  const value = q > 0 && Number.isFinite(q) ? 10 ** Math.round(Math.log10(q)) : fallback.value;
  return { ...fallback, value };
}

// ---------------------------------------------------------------------------------------------
// Rendering

/**
 * @typedef {object} PanelNodes
 * @property {HTMLElement} bar
 * @property {HTMLButtonElement[]} modeButtons
 * @property {HTMLButtonElement[]} styleButtons
 * @property {HTMLInputElement} box
 * @property {HTMLElement} boxPrefix  the fixed "1 in" before a box that takes only N
 * @property {HTMLInputElement} slider
 * @property {HTMLElement} hint
 * @property {SVGSVGElement} ridges
 * @property {SVGSVGElement} fan
 * @property {HTMLElement} fanPlot
 * @property {HTMLCanvasElement} fanShade
 * @property {string | null} fanBox  where the canvas stands, as last placed
 * @property {FanSheets | null} fanSheets  the shading in each colour, as last made
 * @property {HTMLElement} caption
 * @property {HTMLElement} message
 * @property {HTMLElement} messageText
 * @property {HTMLButtonElement} retry
 * @property {string | null} invalid  the hint for unreadable typed text, until the next good value
 * @property {{amount: number | null, percentile: number | null}} remembered  each mode's last value
 * @property {{panel: DistributionPanel, state: LineState, handlers: PanelHandlers, model: PanelModel} | null} current
 * @property {{ridges: (x: number, y: number) => LineState | null, fan: (x: number, y: number) => LineState | null}} drag
 */

const LINE_LABEL = "Line value: an amount like 25k, or a percentile like 99.5 or 1 in 200";
const LINE_LABEL_ONE_IN_N = "Line value: N in 1 in N, such as 200";

/** @type {WeakMap<HTMLElement, PanelNodes>} */
const MOUNTED = new WeakMap();
let hintCounter = 0;

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
 * @param {string} label @param {Array<[string, string]>} options [value, text]
 * @param {string} key the data attribute naming the value
 */
function segmented(label, options, key) {
  const group = html("div", { class: "distribution-segments", role: "group", "aria-label": label });
  const buttons = options.map(([value, content]) => {
    const button = html("button", { type: "button", [`data-${key}`]: value, "aria-pressed": "false" }, content);
    group.appendChild(button);
    return button;
  });
  return { group, buttons };
}

/** @param {HTMLElement} root @returns {PanelNodes} */
function mount(root) {
  const known = MOUNTED.get(root);
  if (known) return known;
  root.textContent = "";
  const hintId = `distributionHint${++hintCounter}`;
  const bar = html("div", { class: "distribution-bar" });
  const mode = segmented("What the line marks", [["amount", "Above an amount"], ["percentile", "At a percentile"]], "mode");
  const style = segmented("Show chances as", [["percent", "%"], ["one_in", "1 in N"]], "style");
  const boxLabel = html("label", { class: "distribution-line-control" }, "Line ");
  const box = html("input", {
    type: "text",
    class: "distribution-line-input",
    inputmode: "decimal",
    spellcheck: "false",
    autocomplete: "off",
    "aria-label": LINE_LABEL,
    "aria-describedby": hintId,
  });
  const boxPrefix = html("span", { class: "distribution-line-prefix", hidden: "" }, "1 in");
  boxLabel.append(boxPrefix, box);
  const slider = html("input", {
    type: "range",
    class: "distribution-slider",
    min: "0",
    max: String(SLIDER_STEPS),
    "aria-label": "Line position",
  });
  bar.append(mode.group, boxLabel, slider, style.group);
  const hint = html("p", { id: hintId, class: "distribution-hint", "aria-live": "polite", hidden: "" });
  const ridges = /** @type {SVGSVGElement} */ (el("svg", { class: "distribution-ridges", role: "img" }));
  const fan = /** @type {SVGSVGElement} */ (el("svg", { class: "distribution-fan", role: "img" }));
  // The shading lies under the fan's lines and labels. Drawn to a canvas rather than into the SVG,
  // a frame of a drag need not encode an image for the browser to decode again.
  const fanPlot = html("div", { class: "distribution-fan-plot" });
  const fanShade = /** @type {HTMLCanvasElement} */ (/** @type {unknown} */ (html("canvas", { class: "distribution-fan-shade", "aria-hidden": "true" })));
  fanPlot.append(fanShade, fan);
  const caption = html("p", { class: "distribution-caption" });
  const message = html("div", { class: "distribution-message", hidden: "" });
  const messageText = html("p", {});
  const retry = html("button", { type: "button", class: "evidence-retry", hidden: "" }, "Retry");
  message.append(messageText, retry);
  root.append(bar, hint, ridges, fanPlot, caption, message);

  /** @type {PanelNodes} */
  const nodes = {
    bar,
    modeButtons: mode.buttons,
    styleButtons: style.buttons,
    box,
    boxPrefix,
    slider,
    hint,
    ridges,
    fan,
    fanPlot,
    fanShade,
    fanBox: null,
    fanSheets: null,
    caption,
    message,
    messageText,
    retry,
    invalid: null,
    remembered: { amount: null, percentile: null },
    current: null,
    drag: { ridges: () => null, fan: () => null },
  };
  bindControls(nodes);
  MOUNTED.set(root, nodes);
  return nodes;
}

/**
 * Reports a new line. Any change but a typed one leaves the text box to show the new line, so it
 * also ends an unreadable entry's invalid state.
 * @param {PanelNodes} nodes @param {LineState} next @param {boolean} [typed]
 */
function emit(nodes, next, typed = false) {
  if (!typed) nodes.invalid = null;
  nodes.current?.handlers.onChange(next);
}

/** @param {PanelNodes} nodes */
function bindControls(nodes) {
  for (const button of nodes.modeButtons) {
    button.addEventListener("click", () => {
      const current = nodes.current;
      const mode = /** @type {"amount" | "percentile"} */ (button.dataset.mode);
      if (!current || current.state.mode === mode) return;
      const value = nodes.remembered[mode] ??
        (mode === "amount" ? defaultLineState(current.panel).value : DEFAULT_PERCENTILE);
      emit(nodes, { ...current.state, mode, value });
    });
  }
  for (const button of nodes.styleButtons) {
    button.addEventListener("click", () => {
      const current = nodes.current;
      const style = /** @type {"percent" | "one_in"} */ (button.dataset.style);
      if (current && current.state.style !== style) emit(nodes, { ...current.state, style });
    });
  }
  nodes.slider.addEventListener("input", () => {
    const current = nodes.current;
    if (!current || !current.model.axis) return;
    const f = Number(nodes.slider.value) / SLIDER_STEPS;
    const { min_log10: lo, max_log10: hi } = current.model.axis;
    const [z0, z1] = SLIDER_Z;
    const value = current.state.mode === "amount" ? 10 ** (lo + f * (hi - lo)) : normCdf(z0 + f * (z1 - z0));
    emit(nodes, { ...current.state, value });
  });
  nodes.box.addEventListener("keydown", (event) => {
    if (event.key === "Enter") nodes.box.blur();
  });
  nodes.box.addEventListener("change", () => {
    const current = nodes.current;
    if (!current) return;
    const parsed = typesOneInN(current.state)
      ? parseOneInN(nodes.box.value)
      : parseLine(nodes.box.value, current.state.mode, current.panel.amount_unit);
    if (!parsed.ok) {
      nodes.invalid = parsed.hint;
      syncControls(nodes, current.model, current.state);
      return;
    }
    nodes.invalid = null;
    emit(nodes, { ...current.state, mode: parsed.mode, value: parsed.value }, true);
  });
  nodes.box.addEventListener("blur", () => {
    if (nodes.current && nodes.invalid === null) syncControls(nodes, nodes.current.model, nodes.current.state);
  });
  bindDrag(nodes, nodes.ridges, "ridges");
  bindDrag(nodes, nodes.fan, "fan");
}

/**
 * @param {PanelNodes} nodes @param {SVGSVGElement} svg @param {"ridges" | "fan"} which
 */
function bindDrag(nodes, svg, which) {
  let dragging = false;
  /** @param {PointerEvent} event */
  const move = (event) => {
    const box = svg.getBoundingClientRect();
    const next = nodes.drag[which](event.clientX - box.left, event.clientY - box.top);
    if (next) emit(nodes, next);
    return next !== null;
  };
  svg.addEventListener("pointerdown", (event) => {
    if (!move(event)) return;
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
 * The slider's position for the line, pinned to its ends.
 * @param {PanelModel} model @param {LineState} state
 */
function sliderPosition(model, state) {
  if (!model.axis) return 0;
  let f;
  if (state.mode === "amount") {
    const { min_log10: lo, max_log10: hi } = model.axis;
    f = (model.line.log10 - lo) / (hi - lo);
  } else {
    const [z0, z1] = SLIDER_Z;
    f = (normQuantile(model.line.p) - z0) / (z1 - z0);
  }
  return Math.round(Math.min(1, Math.max(0, f)) * SLIDER_STEPS);
}

/** A percentile shown as a chance is typed as the N of a fixed "1 in". @param {LineState} state */
function typesOneInN(state) {
  return state.mode === "percentile" && state.style === "one_in";
}

/** @param {PanelNodes} nodes @param {PanelModel} model @param {LineState} state */
function syncControls(nodes, model, state) {
  for (const button of nodes.modeButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.mode === state.mode));
  }
  for (const button of nodes.styleButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.style === state.style));
  }
  nodes.slider.value = String(sliderPosition(model, state));
  nodes.slider.setAttribute(
    "aria-valuetext",
    state.mode === "amount" ? model.line.text : `${ordinal(model.line.p)} percentile`
  );
  const box = nodes.box;
  const oneInN = typesOneInN(state);
  nodes.boxPrefix.hidden = !oneInN;
  box.classList.toggle("is-one-in-n", oneInN);
  box.setAttribute("aria-label", oneInN ? LINE_LABEL_ONE_IN_N : LINE_LABEL);
  if (nodes.invalid !== null) {
    box.setAttribute("aria-invalid", "true");
    box.title = nodes.invalid;
  } else {
    box.removeAttribute("aria-invalid");
    if (box.ownerDocument.activeElement !== box) box.value = model.line.text;
    box.title = model.line.hint ?? "";
  }
  box.toggleAttribute("data-pinned", model.line.pinned);
  const hint = [nodes.invalid ?? model.line.hint, model.spreadNote].filter(Boolean).join(" ");
  nodes.hint.textContent = hint;
  nodes.hint.hidden = hint === "";
  nodes.hint.classList.toggle("is-invalid", nodes.invalid !== null);
}

/**
 * Shows only a message: loading, a failure, or the reason a feature has no panel, with a Retry
 * button when `onRetry` is given.
 * @param {HTMLElement} root @param {string} content @param {(() => void) | null} [onRetry]
 */
export function renderDistributionMessage(root, content, onRetry = null) {
  const nodes = mount(root);
  nodes.current = null;
  for (const node of [nodes.bar, nodes.hint, nodes.ridges, nodes.fanPlot, nodes.caption]) {
    node.toggleAttribute("hidden", true);
  }
  nodes.message.hidden = false;
  nodes.messageText.textContent = content;
  nodes.retry.hidden = onRetry === null;
  nodes.retry.onclick = onRetry;
}

/**
 * Draws the panel for `state`, from `etaSet`'s predictors (the edited model by default).
 * @param {HTMLElement} root @param {DistributionPanel} panel @param {LineState} state
 * @param {PanelHandlers} handlers @param {EtaSet} [etaSet]
 * @param {number} [measuredWidth] the panel's width where the caller has it, sparing a layout
 */
export function renderDistributionPanel(root, panel, state, handlers, etaSet = "edited", measuredWidth = 0) {
  const guarded = guardedPanelModel(panel, state, etaSet);
  if (!guarded.ok) {
    renderDistributionMessage(root, guarded.message);
    return;
  }
  const model = guarded.model;
  if (model.empty !== null) {
    renderDistributionMessage(root, model.empty);
    return;
  }
  const nodes = mount(root);
  // An unreadable entry belongs to the feature it was typed for.
  if (nodes.current && nodes.current.panel.feature !== panel.feature) nodes.invalid = null;
  nodes.current = { panel, state, handlers, model };
  nodes.remembered[state.mode] = state.value;
  nodes.message.hidden = true;
  for (const node of [nodes.bar, nodes.ridges, nodes.fanPlot, nodes.caption]) {
    node.toggleAttribute("hidden", false);
  }
  const width = Math.max(MIN_WIDTH, Math.round(measuredWidth || root.clientWidth || DEFAULT_WIDTH));
  const colours = readColours(root);
  try {
    nodes.drag.ridges = drawRidges(nodes.ridges, model, state, width);
    nodes.drag.fan = drawFan(nodes, model, state, width, colours, etaSet !== "edited");
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    renderDistributionMessage(root, DISTRIBUTION_UNDRAWABLE);
    return;
  }
  nodes.caption.textContent = model.caption;
  syncControls(nodes, model, state);
}

// ---------------------------------------------------------------------------------------------
// The ridges

const DEFAULT_WIDTH = 680;
const MIN_WIDTH = 480;
const DIAMOND = "M0,-5 L5,0 L0,5 L-5,0 Z";
const LABEL_CHARS = 18;
const AXIS_END_ROOM = 24;
const CHAR_WIDTH = 6.2;

/** @param {RidgePointList} points @param {(l: number) => number} x */
const polyline = (points, x) =>
  points.map(([l, y], j) => `${j ? "L" : "M"}${x(l).toFixed(1)},${y.toFixed(1)}`).join(" ");
/** @typedef {import('../chart/ridges.js').RidgePoint[]} RidgePointList */

/** @param {RidgePointList} points @param {(l: number) => number} x @param {number} base */
const area = (points, x, base) =>
  `${polyline(points, x)} L${x(points[points.length - 1][0]).toFixed(1)},${base} L${x(points[0][0]).toFixed(1)},${base} Z`;

/**
 * @param {Element} parent @param {string} tag @param {Record<string, string | number>} attrs
 * @returns {SVGElement}
 */
function add(parent, tag, attrs) {
  const node = el(tag, attrs);
  parent.appendChild(node);
  return node;
}

/** @param {Element} parent @param {string} content */
function hover(parent, content) {
  const title = el("title", {});
  title.textContent = content;
  parent.appendChild(title);
}

/** @param {string} label */
const shortLabel = (label) => (label.length > LABEL_CHARS ? `${label.slice(0, LABEL_CHARS - 1)}…` : label);

/**
 * The dot column's scale and its ticks.
 * @param {PanelModel} model @param {LineState} state @returns {{domain: [number, number], ticks: Array<[number, string]>}}
 */
function dotScale(model, state) {
  if (state.mode === "percentile" && model.axis) {
    // Dots on the ridges' own amount scale, read from the figures beside them: labelled here, the
    // ends would repeat the ridges' axis on the same row.
    const { min_log10: lo, max_log10: hi } = model.axis;
    return { domain: [lo, hi], ticks: [] };
  }
  const oneIn = state.style === "one_in";
  return {
    domain: CHANCE_DOMAIN,
    ticks: [
      [1e-4, oneIn ? "1:10k" : "0.01%"],
      [1e-2, oneIn ? "1:100" : "1%"],
      [1, oneIn ? "1:1" : "100%"],
    ],
  };
}

/** @type {WeakMap<Law, {key: string, fill: string, line: string, yAt: (l: number) => number}>} */
const RIDGE_DRAWINGS = new WeakMap();

/**
 * A law's ridge as path strings, kept while the law lives and the plot's place holds: the line does
 * not change a ridge, and a law is rebuilt when its predictor moves.
 * @param {Law} law @param {PanelAxis} axis @param {import('../chart/ridges.js').RidgeLayout} layout @param {number} index
 * @param {{x: (l: number) => number}} scale
 */
function ridgeDrawing(law, axis, layout, index, scale) {
  const key = [axis.min_log10, axis.max_log10, layout.left, layout.right, layout.height, layout.baseOf(index)].join();
  const known = RIDGE_DRAWINGS.get(law);
  if (known?.key === key) return known;
  const { points, yAt } = ridgeShape(law, axis, layout, index);
  const drawing = { key, fill: area(points, scale.x, layout.baseOf(index)), line: polyline(points, scale.x), yAt };
  RIDGE_DRAWINGS.set(law, drawing);
  return drawing;
}

/**
 * Draws the ridges and returns the drag rule for a point on them.
 * @param {SVGSVGElement} target @param {PanelModel} model @param {LineState} state @param {number} width
 * @returns {(x: number, y: number) => LineState | null}
 */
function drawRidges(target, model, state, width) {
  const svg = el("svg", {});
  const axis = /** @type {PanelAxis} */ (model.axis);
  const rows = model.rows;
  const labelWidth = Math.min(140, Math.max(46, 14 + CHAR_WIDTH * Math.max(...rows.map((row) => shortLabel(row.label).length))));
  const layout = ridgeLayout(width, rows.length, labelWidth);
  const scale = amountScale(axis, layout);
  setAttributes(target, {
    viewBox: `0 0 ${width} ${layout.total}`,
    width,
    height: layout.total,
    "aria-label": `Distribution at each ${model.feature}, with a draggable line`,
    class: `distribution-ridges ${state.mode === "amount" ? "is-amount" : "is-percentile"}`,
  });

  text(svg, 14, 22, `Distribution at each ${model.feature}`, "dist-title", "start");
  const legendX = 14 + CHAR_WIDTH * 1.15 * (`Distribution at each ${model.feature}`.length) + 18;
  add(svg, "circle", { cx: legendX, cy: 18, r: 4, class: "ridge-median" });
  text(svg, legendX + 8, 22, "median", "dist-muted dist-small", "start");
  add(svg, "path", { d: DIAMOND, transform: `translate(${legendX + 64},18)`, class: "ridge-mean" });
  text(svg, legendX + 73, 22, "average", "dist-muted dist-small", "start");
  const header = state.mode === "amount" ? "chance above the line" : `value at the line: ${model.line.label}`;
  text(svg, layout.readoutRight, 22, header, "dist-muted dist-small", "end");

  const dots = dotScale(model, state);
  const first = layout.baseOf(0) - 22;
  for (const [value, label] of dots.ticks) {
    const x = dotPosition(value, dots.domain, layout.dotLeft, layout.dotRight);
    add(svg, "line", { x1: x, x2: x, y1: first, y2: layout.axisY - 10, class: "ridge-grid" });
    text(svg, x, layout.axisY, label, "dist-muted dist-small", "middle");
  }

  // The line is a clip: each ridge is drawn whole on both sides of it, so a line that moves changes
  // one number and not the ridge. In amount mode the line is one place for all of them.
  const clips = add(svg, "defs", {});
  /** @param {string} id @param {number} x @param {number} w */
  const clipRect = (id, x, w) => {
    const clip = add(clips, "clipPath", { id });
    add(clip, "rect", { x, y: 0, width: w, height: layout.total });
  };
  const shared = state.mode === "amount";
  rows.forEach((row, i) => {
    if (shared && i > 0) return;
    const x = scale.x(clampToPlot(row.cut, axis));
    clipRect(`ridgeBelow${i}`, 0, x);
    clipRect(`ridgeBeyond${i}`, x, width - x);
  });

  // First pass: every ridge, back to front, so the translucent fills overlap.
  const shapes = rows.map((row, i) => {
    const group = add(svg, "g", { class: "ridge-row" });
    hover(group, row.title);
    const shape = ridgeDrawing(row.law, axis, layout, i, scale);
    const base = layout.baseOf(i);
    const below = `url(#ridgeBelow${shared ? 0 : i})`;
    const beyond = `url(#ridgeBeyond${shared ? 0 : i})`;
    add(group, "path", { d: shape.fill, class: "ridge-fill", "clip-path": below });
    add(group, "path", { d: shape.fill, class: "ridge-tail-fill", "clip-path": beyond });
    add(group, "path", { d: shape.line, class: "ridge-line", "clip-path": below });
    add(group, "path", { d: shape.line, class: "ridge-tail-line", "clip-path": beyond });
    const inside = row.cut > axis.min_log10 && row.cut < axis.max_log10;
    if (state.mode === "percentile" && inside) {
      const x = scale.x(row.cut);
      add(group, "line", { x1: x, x2: x, y1: shape.yAt(row.cut) - 7, y2: base, class: "ridge-cut" });
    }
    const label = text(group, layout.left - 8, base - 3, shortLabel(row.label), "dist-muted", "end");
    if (label.textContent !== row.label) hover(label, row.label);
    return shape;
  });

  // Second pass: the marks, the dots and the readouts, above every ridge.
  rows.forEach((row, i) => {
    const group = add(svg, "g", { class: "ridge-marks" });
    hover(group, row.title);
    const base = layout.baseOf(i);
    const shape = shapes[i];
    const median = Math.log10(row.median);
    const mean = Math.log10(row.mean);
    const onAxis = (/** @type {number} */ l) => l >= axis.min_log10 && l <= axis.max_log10;
    if (onAxis(median)) add(group, "circle", { cx: scale.x(median), cy: shape.yAt(median), r: 4, class: "ridge-median" });
    if (onAxis(mean)) {
      add(group, "path", {
        d: DIAMOND,
        transform: `translate(${scale.x(mean).toFixed(1)},${shape.yAt(mean).toFixed(1)})`,
        class: "ridge-mean",
      });
    }
    const dotY = base - 8;
    const dotX = (/** @type {number} */ v) => dotPosition(v, dots.domain, layout.dotLeft, layout.dotRight);
    const cx = dotX(row.beyond);
    if (row.ghostBeyond !== undefined) {
      const gx = dotX(row.ghostBeyond);
      add(group, "line", { x1: gx, x2: cx, y1: dotY, y2: dotY, class: "ridge-dot-move" });
      add(group, "circle", { cx: gx, cy: dotY, r: 4, class: "ridge-ghost" });
    }
    add(group, "circle", { cx, cy: dotY, r: 4.5, class: "ridge-dot" });
    text(group, layout.readoutRight, base - 4, row.readout, "dist-readout", "end");
  });

  const ticks = add(svg, "g", { class: "ridge-axis" });
  for (const e of decadeTicks(axis, layout.right - layout.left, 52)) {
    // The last label ends at its tick, clear of the dot column's first.
    const x = scale.x(e);
    text(ticks, x, layout.axisY, decadeLabel(e, model.unit), "dist-muted dist-small", layout.right - x < AXIS_END_ROOM ? "end" : "middle");
  }
  if (model.axisNote) {
    hover(ticks, model.axisNote);
    const x = layout.left - 2;
    const y = layout.axisY - 4;
    const clip = add(svg, "path", { d: `M${x + 5},${y - 5} L${x},${y} L${x + 5},${y + 5}`, class: "ridge-clip" });
    hover(clip, model.axisNote);
  }

  if (state.mode === "amount") {
    const x = scale.x(model.line.log10);
    const marker = add(svg, "g", { class: model.line.pinned ? "ridge-marker is-pinned" : "ridge-marker" });
    add(marker, "line", { x1: x, x2: x, y1: 36, y2: layout.axisY + 8, class: "ridge-line-marker" });
    const w = model.line.label.length * 6.6 + 18;
    const left = Math.min(width - w - 2, Math.max(2, x - w / 2));
    add(marker, "rect", { x: left, y: layout.axisY + 8, width: w, height: 18, rx: 9, class: "dist-pill" });
    text(marker, left + w / 2, layout.axisY + 21, model.line.label, "dist-pill-label", "middle");
    if (model.line.hint) hover(marker, model.line.hint);
  }

  patchChildren(target, svg);

  return (x, y) => {
    if (x < layout.left - 10 || x > layout.right + 10 || y < 30 || y > layout.axisY + 30) return null;
    const l = clampToPlot(scale.l(x), axis);
    if (state.mode === "amount") return { ...state, value: 10 ** l };
    const i = Math.min(rows.length - 1, Math.max(0, Math.round((y - layout.top - 16) / layout.row - 1)));
    return { ...state, value: clampP(rows[i].law.cdf(10 ** l)) };
  };
}

/** @param {number} p */
const clampP = (p) => Math.min(SLIDER_P[1], Math.max(SLIDER_P[0], p));

// ---------------------------------------------------------------------------------------------
// The fan

const FAN = { left: 56, rightInset: 120, top: 34, bottom: 262, height: 300 };

/**
 * @typedef {{canvas: HTMLCanvasElement, image: ImageData}} FanSheet
 * @typedef {{below: FanSheet, beyond: FanSheet, laws: Law[], colours: FanColours, min: number, max: number}} FanSheets
 */

/**
 * @typedef {{surface: RGB, below: RGB[], beyond: RGB[]}} FanColours
 */

/** @type {{theme: string | null, colours: FanColours} | null} */
let readColoursFor = null;

/**
 * The fan's colours in the theme in force: the tokens change only with it, and asking the browser
 * for computed style on each frame of a drag makes it work out the page's styles first.
 * @param {HTMLElement} root @returns {FanColours}
 */
function readColours(root) {
  const theme = root.ownerDocument.documentElement.getAttribute("data-theme");
  if (readColoursFor?.theme === theme) return readColoursFor.colours;
  const style = getComputedStyle(root);
  /** @param {string} name */
  const token = (name) => hexToRgb(style.getPropertyValue(name));
  const surface = token("--surface");
  const colours = {
    surface,
    below: [surface, token("--fan-blue-1"), token("--fan-blue-2"), token("--fan-blue-3")],
    beyond: [surface, token("--fan-orange-1"), token("--fan-orange-2"), token("--fan-orange-3")],
  };
  readColoursFor = { theme, colours };
  return colours;
}

/**
 * A canvas holding pixels of this size, reusing `sheet`'s where it can.
 * @param {FanSheet | undefined} sheet @param {number} width @param {number} height
 * @returns {FanSheet}
 */
function sizedSheet(sheet, width, height) {
  const canvas = sheet?.canvas ?? document.createElement("canvas");
  if (sheet?.image.width === width && sheet.image.height === height) return sheet;
  canvas.width = width;
  canvas.height = height;
  return { canvas, image: new ImageData(width, height) };
}

/**
 * The fan's shading in each colour, kept while its laws, size and colours are the same.
 * @param {PanelNodes} nodes @param {FanModel} fan @param {number} width @param {number} height
 * @param {PanelAxis} axis @param {FanColours} colours
 */
function fanSheets(nodes, fan, width, height, axis, colours) {
  const known = nodes.fanSheets;
  const laws = fan.columns.map((column) => column.law);
  if (known && known.colours === colours && known.below.image.width === width && known.below.image.height === height &&
      known.min === axis.min_log10 && known.max === axis.max_log10 &&
      known.laws.length === laws.length && laws.every((law, i) => law === known.laws[i])) {
    return known;
  }
  const below = sizedSheet(known?.below, width, height);
  const beyond = sizedSheet(known?.beyond, width, height);
  for (const [sheet, ramp] of /** @type {Array<[FanSheet, RGB[]]>} */ ([[below, colours.below], [beyond, colours.beyond]])) {
    fanPixels(fan, width, height, axis, ramp, sheet.image.data);
    sheet.canvas.getContext("2d")?.putImageData(sheet.image, 0, 0);
  }
  nodes.fanSheets = { below, beyond, laws, colours, min: axis.min_log10, max: axis.max_log10 };
  return nodes.fanSheets;
}

/**
 * Draws the fan and returns the drag rule for a point on it.
 * @param {PanelNodes} nodes @param {PanelModel} model @param {LineState} state @param {number} width
 * @param {FanColours} colours @param {boolean} dragging a predictor is moving: shade at half the size
 * @returns {(x: number, y: number) => LineState | null}
 */
function drawFan(nodes, model, state, width, colours, dragging) {
  const target = nodes.fan;
  const svg = el("svg", {});
  const axis = /** @type {PanelAxis} */ (model.axis);
  const fan = model.fan;
  const left = FAN.left;
  const right = width - FAN.rightInset;
  const { top, bottom } = FAN;
  const plotW = right - left;
  const plotH = bottom - top;
  const span = axis.max_log10 - axis.min_log10;
  /** @param {number} l */
  const y = (l) => bottom - ((clampToPlot(l, axis) - axis.min_log10) / span) * plotH;
  setAttributes(target, {
    viewBox: `0 0 ${width} ${FAN.height}`,
    width,
    height: FAN.height,
    "aria-label": `Density fan of the distribution across ${model.feature}, with the same line`,
  });
  if (model.spreadNote) hover(svg, model.spreadNote);

  text(svg, 14, 22, `Distribution across ${model.feature}`, "dist-title", "start");
  text(svg, width - 14, 22, `shade: how likely, within each ${model.feature}; orange: past the line`, "dist-muted dist-small", "end");

  // The canvas is shaded at device pixels and drawn at the plot's CSS size, stopping at twice that
  // size. The shading depends on the laws and not on the line, which only clips one colour over the
  // other, so a line that moves repaints from the sheets; a moving predictor makes new ones.
  const dpr = Math.min(2, Math.max(1, globalThis.devicePixelRatio || 1));
  const deviceW = Math.round(plotW * dpr);
  const deviceH = Math.round(plotH * dpr);
  const canvas = nodes.fanShade;
  if (canvas.width !== deviceW || canvas.height !== deviceH) {
    canvas.width = deviceW;
    canvas.height = deviceH;
  }
  const box = `${left} ${top} ${plotW} ${plotH}`;
  if (nodes.fanBox !== box) {
    nodes.fanBox = box;
    Object.assign(canvas.style, { left: `${left}px`, top: `${top}px`, width: `${plotW}px`, height: `${plotH}px` });
  }
  const context = canvas.getContext("2d");
  if (context) {
    const sheets = fanSheets(nodes, fan, dragging ? Math.ceil(deviceW / 2) : deviceW, dragging ? Math.ceil(deviceH / 2) : deviceH, axis, colours);
    context.drawImage(sheets.below.canvas, 0, 0, deviceW, deviceH);
    context.save();
    context.beginPath();
    fanBeyondRegion(fan, deviceW, deviceH, axis).forEach(([x, row], i) => {
      if (i) context.lineTo(x, row);
      else context.moveTo(x, row);
    });
    context.closePath();
    context.clip();
    context.drawImage(sheets.beyond.canvas, 0, 0, deviceW, deviceH);
    context.restore();
  }

  const positions = fanPositions(fan);
  const n = fan.columns.length;
  /** @param {(column: FanColumn) => number} value */
  const pathOf = (value) => {
    if (fan.kind === "levels") {
      return fan.columns.map((column, i) => {
        const yy = y(value(column)).toFixed(1);
        return `M${(left + (i / n) * plotW).toFixed(1)},${yy} L${(left + ((i + 1) / n) * plotW).toFixed(1)},${yy}`;
      }).join(" ");
    }
    return fan.columns.map((column, i) =>
      `${i ? "L" : "M"}${(left + positions[i] * plotW).toFixed(1)},${y(value(column)).toFixed(1)}`
    ).join(" ");
  };
  const lines = [
    { value: (/** @type {FanColumn} */ c) => Math.log10(c.mean), cls: "fan-average", label: "average" },
    { value: (/** @type {FanColumn} */ c) => Math.log10(c.median), cls: "fan-median", label: "median" },
    { value: (/** @type {FanColumn} */ c) => Math.log10(c.mode), cls: "fan-mode", label: "most common" },
    { value: (/** @type {FanColumn} */ c) => c.cut, cls: "fan-cut", label: model.line.label },
  ];
  for (const line of lines) {
    const d = pathOf(line.value);
    add(svg, "path", { d, class: "fan-halo" });
    add(svg, "path", { d, class: `fan-line ${line.cls}` });
  }
  const last = fan.columns[n - 1];
  const labelYs = spreadLabels(lines.map((line) => y(line.value(last)) + 3.5), 12, top + 4, bottom + 4);
  lines.forEach((line, i) => text(svg, right + 6, labelYs[i], line.label, "dist-readout dist-small", "start"));

  for (const e of decadeTicks(axis, plotH, 16)) {
    text(svg, left - 6, y(e) + 3.5, decadeLabel(e, model.unit), "dist-muted dist-small", "end");
  }
  if (fan.kind === "levels") {
    const band = plotW / n;
    const longest = Math.max(...fan.columns.map((column) => String(column.level).length));
    const stride = Math.max(1, Math.ceil((longest * CHAR_WIDTH + 8) / band));
    fan.columns.forEach((column, i) => {
      if (i % stride) return;
      const label = text(svg, left + (i + 0.5) * band, bottom + 15, shortLabel(String(column.level)), "dist-muted dist-small", "middle");
      if (label.textContent !== column.level) hover(label, String(column.level));
    });
  } else {
    const xs = fan.columns.map((column) => Number(column.x));
    const lo = Math.min(...xs);
    const hi = Math.max(...xs);
    for (const v of niceTicks(lo, hi, Math.max(2, Math.floor(plotW / 80)))) {
      const x = hi > lo ? left + ((v - lo) / (hi - lo)) * plotW : left + plotW / 2;
      text(svg, x, bottom + 15, fmt(v), "dist-muted dist-small", "middle");
    }
  }
  text(svg, right, bottom + 29, model.feature, "dist-muted dist-small", "end");
  patchChildren(target, svg);
  return (x, yy) => {
    if (x < left || x > right || yy < top || yy > bottom) return null;
    const l = axis.min_log10 + ((bottom - yy) / plotH) * span;
    if (state.mode === "amount") return { ...state, value: 10 ** l };
    const column = fan.columns[columnAt(fan, positions, x - left, plotW)];
    return { ...state, value: clampP(column.law.cdf(10 ** l)) };
  };
}
