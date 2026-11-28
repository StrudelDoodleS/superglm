// @ts-check
/**
 * A check's face on the Assess overview and in its list: the count of what left the band, one
 * figure under it, the orange flag, and the thumbnail. Compare with the fit lives here too, as
 * the overview and the large view share it, and so does the large view with its hover readout.
 */

import { checkScene, nearestMark, patchScene, sceneMarkup } from "../chart/check_plots.js";
import { escapeHTML } from "../format.js";
import { countText } from "../reports.js";
import { assessEntry } from "../state/assess.js";
import { defineMachine } from "../state/machine.js";

/** @typedef {import('../api/contracts.js').CheckReport} CheckReport */
/** @typedef {import('../api/contracts.js').CheckPayload} CheckPayload */
/** @typedef {import('../api/contracts.js').CheckBody} CheckBody */
/** @typedef {import('../api/contracts.js').EvidenceStatus} EvidenceStatus */
/** @typedef {import('../api/contracts.js').EvidenceState} EvidenceState */
/** @typedef {import('../api/contracts.js').EditorSnapshot} EditorSnapshot */
/** @typedef {import('../chart/check_plots.js').PITStats} PITStats */
/** @typedef {import('../chart/check_plots.js').CalibrationStats} CalibrationStats */
/** @typedef {import('../chart/check_plots.js').Scene} Scene */
/** @typedef {import('../chart/check_plots.js').Mark} Mark */
/** @typedef {import('../chart/check_plots.js').SceneOptions} SceneOptions */
/**
 * The large view's nodes: its heading, the Help icon beside it, the split line, the legend
 * chips, the frame the plot is drawn in, and the polite live region that says the point the keys
 * stepped to.
 * @typedef {{title:HTMLElement, help:HTMLElement, split:HTMLElement, legend:HTMLElement,
 *   frame:HTMLElement, live?:HTMLElement|null}} CheckNodes
 */
/** @typedef {{value: string, note: string, flagged: boolean}} CheckFace */
/**
 * Compare with the fit's control: whether it can be turned, whether it is on, and its help.
 * @typedef {'hidden'|'asFitted'|'waitingForComparison'|'comparisonUnavailable'|'ready'} CompareState
 * @typedef {{state: CompareState, disabled: boolean, checked: boolean, body: string}} CompareSwitch
 */

const MISSING = "—";
/** A card's thumbnail box, which `.assess-card-plot` holds at this size. */
export const THUMB_BOX = Object.freeze({ width: 160, height: 62, thumb: true });
const NO_EDIT = "The model is as fitted; there is no edit to compare.";
const COMPARES = "Draws the model as fitted in grey beside the model as edited in blue, on every check. "
  + "It changes only what is drawn.";
/**
 * Each state of Compare with the fit, with whether it is disabled and its help, together: hidden
 * where no check is offered; as fitted where the model in force is the fit; waiting while the
 * edited model's checks are still being asked; unavailable where none of them has the fit to
 * draw beside the edit, failed included; and ready.
 * @type {Readonly<Record<CompareState, Readonly<{disabled: boolean, body: string}>>>}
 */
export const COMPARE_CONTROL = Object.freeze({
  hidden: Object.freeze({ disabled: true, body: COMPARES }),
  asFitted: Object.freeze({ disabled: true, body: NO_EDIT }),
  waitingForComparison: Object.freeze({
    disabled: true, body: "Waiting for the edited model's answer; the comparison is ready once it has loaded.",
  }),
  comparisonUnavailable: Object.freeze({
    disabled: true, body: "No check shown has the model as fitted to draw beside the edit.",
  }),
  ready: Object.freeze({ disabled: false, body: COMPARES }),
});

/** @param {unknown} value a ratio, to 2 decimals */
const ratioText = (value) => (typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : MISSING);

/** @param {unknown} value a share, as a percentage to 1 decimal */
const shareText = (value) => (typeof value === "number" && Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : MISSING);

/**
 * The model as edited's answer, when the check has one for it.
 * @param {CheckPayload|null} payload @returns {CheckBody|null}
 */
function answered(payload) {
  return payload?.available && payload.models.edited.available ? payload.models.edited : null;
}

/**
 * The number and figure of an answered check.
 * @param {CheckReport} report @param {CheckPayload} payload @param {CheckBody} body
 * @returns {{value: string, note: string}}
 */
function answeredFace(report, payload, body) {
  const outside = countText(body.outside);
  const cells = countText(body.cells);
  switch (report) {
    case "check_qq": {
      const split = payload.split;
      return {
        value: `${outside} of ${countText(body.n)} points outside the band`,
        note: split ? `${split.label}, ${countText(split.n_obs)} rows` : "",
      };
    }
    case "check_pit": {
      const { counts, band_lower, band_upper } = /** @type {CheckBody & PITStats} */ (body);
      return {
        value: `${outside} of ${cells} bins outside the band`,
        note: `Bins ${countText(Math.min(...counts))} to ${countText(Math.max(...counts))} against `
          + `${countText(band_lower)} to ${countText(band_upper)}`,
      };
    }
    case "check_ae": {
      const overall = /** @type {{ratio?: number|null}|null|undefined} */ (body.overall);
      return { value: `${outside} of ${cells} groups outside their interval`, note: `Overall ${ratioText(overall?.ratio)}` };
    }
    case "check_calibration": {
      const { p, realised_exceedance: share } = /** @type {CheckBody & CalibrationStats} */ (body).quantiles;
      const at = p.findIndex((level) => Math.abs(level - 0.99) < 1e-9);
      return {
        value: `${outside} of ${cells} quantiles outside the band`,
        note: `99th percentile exceeded by ${shareText(at < 0 ? null : share[at])}`,
      };
    }
  }
}

/**
 * A check's card face: what left its band, one figure, and whether to flag it. A check with no
 * answer for this model says why, and one not yet answered that it is loading or failed.
 * @param {CheckReport} report @param {CheckPayload|null} payload @param {EvidenceStatus} status
 * @returns {CheckFace}
 */
export function checkFace(report, payload, status) {
  if (payload === null) {
    return { value: MISSING, note: status === "error" ? "Check unavailable." : "Loading check...", flagged: false };
  }
  const body = answered(payload);
  if (body === null) {
    const reason = payload.available ? payload.models.edited.reason : payload.reason;
    return { value: MISSING, note: reason ?? "", flagged: false };
  }
  return { ...answeredFace(report, payload, body), flagged: Boolean(body.flagged) };
}

/**
 * The payload a check's evidence holds, or null.
 * @param {EvidenceState} evidence @returns {CheckPayload|null}
 */
export function checkPayload(evidence) {
  return evidence.payload === null ? null : /** @type {CheckPayload} */ (evidence.payload);
}

/**
 * A card's thumbnail: the check's scene at the card's size, an empty plot until it is answered.
 * @param {CheckReport} report @param {CheckPayload|null} payload @param {SceneOptions} options
 * @returns {string}
 */
export function checkThumbnail(report, payload, options) {
  if (payload === null) return sceneMarkup({ ...THUMB_BOX, layers: [], labels: [], marks: [] });
  return sceneMarkup(checkScene(report, payload, THUMB_BOX, options));
}

/**
 * The offered checks whose model as edited left its band, in the order given.
 * @param {Readonly<Record<CheckReport, EvidenceState>>} evidence @param {readonly CheckReport[]} offered
 * @returns {CheckReport[]}
 */
export function flaggedChecks(evidence, offered) {
  return offered.filter((name) => Boolean(answered(checkPayload(evidence[name]))?.flagged));
}

/**
 * Whether the model in force differs from the fit, as the snapshot records it: a model a step
 * replaced, or a term edited by hand (Python's `compares_edit`).
 * @param {EditorSnapshot|null} snapshot @returns {boolean}
 */
export function differsFromFit(snapshot) {
  if (!snapshot) return false;
  return !snapshot.in_force_is_original || Object.values(snapshot.terms).some((term) => term.edited === true);
}

/**
 * Whether `payload` draws the fit beside the edit: it compares an edit, and both models have an
 * answer.
 * @param {CheckPayload|null} payload
 */
const comparable = (payload) => Boolean(payload?.compares_edit && answered(payload) && payload.models.original?.available);

/**
 * Whether a check's evidence is still to be answered: never asked, being asked, or lapsed with
 * nothing failed, which the page asks again.
 * @param {EvidenceState} evidence
 */
const stillAsked = ({ status, error }) => status === "idle" || status === "updating" || (status === "stale" && error === null);

/**
 * Compare with the fit, derived from the snapshot and the offered checks' evidence
 * (`COMPARE_CONTROL`): ready while the model in force differs from the fit and an offered check
 * draws the fit beside it, and on when the analyst turned it on while ready. A payload kept from
 * an edit since undone draws no fit. The on/off preference is the store's alone.
 * @param {Readonly<Record<CheckReport, EvidenceState>>} evidence @param {readonly CheckReport[]} offered
 * @param {boolean} compareOriginal @param {EditorSnapshot|null} snapshot
 * @returns {CompareSwitch}
 */
export function compareSwitch(evidence, offered, compareOriginal, snapshot) {
  /** @type {CompareState} */
  let state = "comparisonUnavailable";
  if (offered.length === 0) state = "hidden";
  else if (!differsFromFit(snapshot)) state = "asFitted";
  else if (offered.some((name) => comparable(checkPayload(evidence[name])))) state = "ready";
  else if (offered.some((name) => stillAsked(evidence[name]))) state = "waitingForComparison";
  return { state, ...COMPARE_CONTROL[state], checked: state === "ready" && compareOriginal };
}

/**
 * Write the switch's state onto its button: hidden, disabled, on, and its help.
 * @param {HTMLElement} button @param {CompareSwitch} control
 */
export function renderCompareSwitch(button, control) {
  button.hidden = control.state === "hidden";
  button.setAttribute("aria-checked", String(control.checked));
  button.setAttribute("aria-disabled", String(control.disabled));
  button.dataset.popoverBody = control.body;
}

/**
 * Write Highlight outside the band's state onto its button; a model that offers no check hides
 * it. It is always live: it changes only how the plots colour what left a band.
 * @param {HTMLElement} button @param {boolean} checked @param {boolean} shown
 */
export function renderHighlightSwitch(button, checked, shown) {
  button.hidden = !shown;
  button.setAttribute("aria-checked", String(checked));
}

const SVG_NS = "http://www.w3.org/2000/svg";
// Each split line's markup as last written: a redraw of the same note keeps its element, and the
// keyboard focus and the popover on it.
/** @type {WeakMap<Element, string>} */
const SPLIT_WRITTEN = new WeakMap();
// What the split line adds when a held-out split's band takes the fitted parameters as exact,
// and the model it names when only one of the two drawn fell back.
const LEVEL_TEXT = "parameters taken as exact";
const LEVEL_TITLE = "Parameters taken as exact";
// The readout's text sits this far beside its point and this far above or below it, its lines
// this far apart, and this far inside the frame's visible edge. It breaks between its values.
const READOUT_GAP = 10;
const READOUT_RISE = 8;
const READOUT_LINE = 16;
const READOUT_EDGE = 4;
const READOUT_BREAK = " · ";
// Where the page cannot measure text: a character's width and the first line's ascent, at the
// readout's 12 px.
const READOUT_CHAR = 7;
const READOUT_ASCENT = 10;
/** @type {Readonly<Record<string, number>>} */
const KEY_STEPS = Object.freeze({ ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 });

/** @param {CheckBody|null|undefined} body @returns {string|null} its fallback sentence, if it has one */
const levelNote = (body) => (body?.available && typeof body.level_note === "string" && body.level_note ? body.level_note : null);

/**
 * A check's split line: the rows it reads, and on a held-out split whose band took the fitted
 * parameters as exact, the marker that says so and its sentence. When the scene draws a fitted
 * series beside the edit and only one of the two fell back, the marker names it, and when both
 * fell back for different reasons its sentence gives each. A fit with nothing to draw on the
 * whole plot does not count; one a zoom happens to leave out of view does.
 * @param {CheckPayload|null} payload @param {boolean} fitDrawn whether the scene draws the fit
 * @returns {{text: string, level: string|null, note: string|null}}
 */
export function splitLine(payload, fitDrawn) {
  const split = payload?.split;
  if (!payload || !split) return { text: "", level: null, note: null };
  const text = `${split.label}, ${countText(split.n_obs)} rows`;
  // Only drawn models count: nothing is drawn when the edit has no answer.
  const drawn = answered(payload);
  if (drawn === null) return { text, level: null, note: null };
  const edited = levelNote(drawn);
  const fitted = fitDrawn && payload.models.original?.available ? levelNote(payload.models.original) : undefined;
  // `fitted` is undefined when the fit is not drawn, and null when it is drawn and simulated.
  const note = edited && fitted && edited !== fitted ? `As edited: ${edited} As fitted: ${fitted}` : edited ?? fitted ?? null;
  if (note === null) return { text, level: null, note: null };
  // The model is named only when the fit is drawn and exactly one of the two fell back.
  const one = fitted !== undefined && (edited === null) !== (fitted === null);
  const named = !one ? "" : edited === null ? " (as fitted)" : " (as edited)";
  return { text, level: `${LEVEL_TEXT}${named}`, note };
}

/**
 * Draw a check large: its label and its Help icon, its split line, the legend when the fit is
 * drawn beside the edit, and the plot at `width` by `height` (a taller A/E grows past it, and the
 * frame scrolls), at `domain` when it is zoomed. A check with no answer for this model says why in
 * the frame. The svg is written once per call.
 * @param {CheckNodes} nodes @param {CheckReport} report @param {CheckPayload|null} payload
 * @param {SceneOptions & {width: number, height: number}} options
 * @returns {Scene|null} the scene drawn, whose marks the hover reads; null when none is
 */
export function renderCheckView(nodes, report, payload, { compare, highlight, domain = null, width, height }) {
  const label = assessEntry(report)?.label ?? "";
  nodes.title.textContent = label;
  nodes.help.dataset.helpCheck = report;
  nodes.help.setAttribute("aria-label", `About the ${label}`);
  const scene = payload === null || answered(payload) === null ? null
    : checkScene(report, payload, { width, height }, { compare, highlight, domain });
  // As the legend does, the split line counts the fit when the scene has a fitted series to draw:
  // compared, available and drawable on the whole plot, so a zoom never changes either.
  const fitDrawn = scene?.fitted === true;
  const line = splitLine(payload, fitDrawn);
  const split = escapeHTML(line.text) + (line.level === null || line.note === null ? ""
    : ` · <span class="check-level" tabindex="0" data-popover-title="${LEVEL_TITLE}"`
      + ` data-popover-body="${escapeHTML(line.note)}">${line.level}</span>`);
  if (SPLIT_WRITTEN.get(nodes.split) !== split) {
    nodes.split.innerHTML = split;
    SPLIT_WRITTEN.set(nodes.split, split);
  }
  const { frame } = nodes;
  frame.dataset.zoomed = String(scene !== null && domain !== null);
  if (payload === null || scene === null) {
    nodes.legend.hidden = true;
    for (const name of ["role", "aria-roledescription", "tabindex", "aria-label"]) frame.removeAttribute(name);
    const reason = payload === null ? null : (payload.available ? payload.models.edited.reason : payload.reason);
    frame.innerHTML = reason ? `<p class="check-reason">${escapeHTML(reason)}</p>` : "";
    return null;
  }
  nodes.legend.hidden = !fitDrawn;
  // A plot that takes its own keys, whose readout a screen reader can reach: an img's children
  // would be presentational.
  frame.setAttribute("role", "application");
  frame.setAttribute("aria-roledescription", "chart");
  frame.setAttribute("tabindex", "0");
  frame.setAttribute("aria-label", label);
  frame.innerHTML = sceneMarkup(scene);
  return scene;
}

/**
 * Draw the plot renderCheckView drew again at another domain, and nothing else: a zoom's
 * redraw, which leaves the heading, split line, legend and switches as they are. They read
 * `Scene.fitted`, which a domain does not change. The svg on screen is written in place; its
 * readout is the hover's to place on the new scene.
 * @param {HTMLElement} frame @param {CheckReport} report @param {CheckPayload} payload
 * @param {SceneOptions & {width: number, height: number}} options
 * @returns {Scene}
 */
export function redrawCheckPlot(frame, report, payload, { compare, highlight, domain = null, width, height }) {
  const scene = checkScene(report, payload, { width, height }, { compare, highlight, domain });
  frame.dataset.zoomed = String(domain !== null);
  const svg = frame.querySelector("svg");
  if (!svg) {
    frame.innerHTML = sceneMarkup(scene);
    return scene;
  }
  patchScene(svg, scene);
  return scene;
}

/**
 * The readout's ring and text, made the first time it shows.
 * @param {Element} group @returns {Element} the text
 */
function readoutText(group) {
  const found = group.querySelector("text");
  if (found) return found;
  const ring = group.ownerDocument.createElementNS(SVG_NS, "circle");
  ring.setAttribute("class", "check-readout-ring");
  ring.setAttribute("r", "6");
  const label = group.ownerDocument.createElementNS(SVG_NS, "text");
  label.setAttribute("class", "check-readout-text");
  group.append(ring, label);
  return label;
}

/**
 * `text` in lines of at most `perLine` characters, each broken after a `READOUT_BREAK`, or within
 * a value longer than a line; the lines join back to `text`.
 * @param {string} text @param {number} perLine @returns {string[]}
 */
function readoutLines(text, perLine) {
  const values = text.split(READOUT_BREAK).map((value, i, all) => (i < all.length - 1 ? value + READOUT_BREAK : value));
  const pieces = values.flatMap((value) => Array.from({ length: Math.max(1, Math.ceil(value.length / perLine)) },
    (_, j) => value.slice(j * perLine, (j + 1) * perLine)));
  /** @type {string[]} */
  const lines = [];
  for (const piece of pieces) {
    const last = lines.length - 1;
    if (last >= 0 && (lines[last] + piece).trimEnd().length <= perLine) lines[last] += piece;
    else lines.push(piece);
  }
  return lines;
}

/**
 * Write `lines` into the readout's text at its origin, and return the box they take there. One
 * line is the text's own; more are tspans.
 * @param {SVGTextElement|Element} label @param {string[]} lines @param {number} charWidth
 * @returns {{x: number, y: number, width: number, height: number}}
 */
function layoutReadout(label, lines, charWidth) {
  label.setAttribute("x", "0");
  label.setAttribute("y", "0");
  if (lines.length === 1) {
    label.textContent = lines[0];
  } else {
    label.textContent = "";
    for (const [i, line] of lines.entries()) {
      const span = label.ownerDocument.createElementNS(SVG_NS, "tspan");
      span.setAttribute("x", "0");
      span.setAttribute("dy", String(i === 0 ? 0 : READOUT_LINE));
      span.textContent = line;
      label.append(span);
    }
  }
  const box = "getBBox" in label && typeof label.getBBox === "function" ? label.getBBox() : null;
  if (box && box.width > 0) return box;
  return { x: 0, y: -READOUT_ASCENT, width: Math.max(...lines.map((line) => line.length)) * charWidth, height: lines.length * READOUT_LINE };
}

/** @param {number} value @param {number} lo @param {number} hi */
const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));

/** @typedef {{x: number, y: number}} ClientPoint the pointer, in client px */
/**
 * What the readout of a check drawn large shows: nothing; the point nearest the pointer at
 * `pointer`; mark `index` of `scene`, which the keys stepped to; or nothing while a gesture runs,
 * keeping where the pointer is over the frame, if it is, to read out there once the gesture rests.
 * @typedef {{state: 'hidden'} | {state: 'pointed', pointer: ClientPoint}
 *   | {state: 'keyboard', scene: Scene, index: number}
 *   | {state: 'suspended', pointer: ClientPoint|null}} Readout
 */
/** @typedef {'point'|'key'|'leave'|'suspend'|'rest'|'resume'|'redraw'|'scroll'|'close'} ReadoutEvent */

/**
 * The readout's lifecycle. `point` is a pointer move over the frame, `key` an arrow, Home or End
 * key, and `leave` the pointer leaving, the frame's blur or Escape. A gesture's start suspends it,
 * and its rest brings it back where the pointer is (`resume`) or hidden (`rest`). `redraw` is the
 * same check drawn again (a zoom, a resize, new evidence), `scroll` the frame's view moving, and
 * `close` the check closed or another one opened.
 */
export const CHECK_READOUT = defineMachine({
  name: "check readout",
  initial: "hidden",
  moves: {
    hidden: { point: "pointed", key: "keyboard", suspend: "suspended" },
    pointed: {
      point: "pointed", key: "keyboard", leave: "hidden", suspend: "suspended",
      redraw: "pointed", scroll: "pointed", close: "hidden",
    },
    keyboard: {
      point: "pointed", key: "keyboard", leave: "hidden", suspend: "suspended",
      redraw: "hidden", scroll: "keyboard", close: "hidden",
    },
    suspended: {
      point: "suspended", leave: "suspended", redraw: "suspended", rest: "hidden", resume: "pointed",
      close: "hidden",
    },
  },
});

/** @type {Readonly<Readout>} */
const HIDDEN = Object.freeze({ state: "hidden" });

/**
 * Read out the point nearest the pointer on the check in `frame`, and step through the points
 * with the arrow keys while the frame has focus, in the scene's reading order. A pointer move or a
 * scroll of the frame waits for the next frame, and writes only the readout group: its transform,
 * its text and whether it is hidden. A step by the keys is also said in `live`. The view tells it
 * when a gesture starts and rests (`suspend`, `rest`), when the same check is drawn again
 * (`redraw`), and when the check is closed or another opened (`close`).
 * @param {HTMLElement} frame @param {() => Scene|null} getScene the scene on screen
 * @param {{live?: HTMLElement|null}} [options]
 * @returns {{suspend(): void, rest(): void, redraw(): void, close(): void, destroy(): void}}
 */
export function bindCheckHover(frame, getScene, { live = null } = {}) {
  /** @type {Readout} */
  let readout = HIDDEN;
  let pending = 0;
  // The point drawn, the scene it belongs to, and the frame's scroll when it was placed.
  let current = -1;
  /** @type {Scene|null} */
  let shownIn = null;
  let placedAt = "";

  const group = () => frame.querySelector(".check-readout");

  /** @returns {ClientPoint|null} where the pointer is over the frame, as the readout keeps it */
  const kept = () => (readout.state === "pointed" || readout.state === "suspended" ? readout.pointer : null);

  /**
   * The one writer of the readout: `event` moves it to `value`, when the table takes `event` to
   * `value`'s state, and nothing else does. The readout group, the pending frame and the live
   * text are written here and in what this calls alone.
   * @param {ReadoutEvent} event @param {Readout} value @returns {boolean} whether it moved
   */
  function move(event, value) {
    if (CHECK_READOUT.next(readout.state, event) !== value.state) return false;
    const before = readout;
    readout = value;
    if (before.state === "keyboard" && value.state !== "keyboard") say("");
    if (value.state === "hidden" || value.state === "suspended") {
      if (pending) cancelAnimationFrame(pending);
      pending = 0;
      conceal();
    } else if (event === "redraw") {
      read();
    } else if (value.state === "keyboard" && event === "key") {
      const mark = value.scene.marks[value.index];
      // Scrolled first, so the readout is placed in the part of the frame then in view.
      reveal(value.scene, mark);
      show(value.scene, value.index);
      say(mark.aria ?? mark.text);
    } else if (!pending) {
      pending = requestAnimationFrame(onFrame);
    }
    return true;
  }

  /** @param {string} text the live region's text, written only when it changes */
  function say(text) {
    if (live && live.textContent !== text) live.textContent = text;
  }

  // The readout group hidden, and nothing drawn in it.
  function conceal() {
    current = -1;
    shownIn = null;
    const node = group();
    if (node && !node.hasAttribute("hidden")) node.setAttribute("hidden", "");
  }

  // The part of the scene the frame shows, in the scene's px, scroll included.
  /** @param {Scene} scene @returns {{left: number, top: number, right: number, bottom: number}} */
  function visible(scene) {
    const svg = frame.querySelector("svg");
    const box = svg?.getBoundingClientRect();
    if (!box || !(box.width > 0 && box.height > 0)) return { left: 0, top: 0, right: scene.width, bottom: scene.height };
    const view = frame.getBoundingClientRect();
    const kx = scene.width / box.width;
    const ky = scene.height / box.height;
    const left = (view.left - box.left) * kx;
    const top = (view.top - box.top) * ky;
    return {
      left: Math.max(0, left),
      top: Math.max(0, top),
      right: Math.min(scene.width, left + frame.clientWidth * kx),
      bottom: Math.min(scene.height, top + frame.clientHeight * ky),
    };
  }

  // The readout's text goes beside its point on the side with room, above it unless the visible
  // top is too near, and when no side has room it is centred over the point, broken into lines
  // that fit; it is clamped inside the visible part of the frame.
  /** @param {Scene} scene @param {number} index */
  function show(scene, index) {
    const node = group();
    if (!node || index < 0) return;
    const scroll = `${frame.scrollLeft} ${frame.scrollTop}`;
    if (index === current && scene === shownIn && scroll === placedAt && !node.hasAttribute("hidden")) return;
    current = index;
    shownIn = scene;
    placedAt = scroll;
    const mark = scene.marks[index];
    const label = readoutText(node);
    // Shown before it is measured: hidden text has no length.
    if (node.hasAttribute("hidden")) node.removeAttribute("hidden");
    node.setAttribute("transform", `translate(${Math.round(mark.x * 10) / 10} ${Math.round(mark.y * 10) / 10})`);
    const view = visible(scene);
    const room = view.right - view.left - 2 * READOUT_EDGE;
    label.textContent = mark.text;
    label.setAttribute("aria-label", mark.aria ?? mark.text);
    const measured = "getComputedTextLength" in label && typeof label.getComputedTextLength === "function"
      ? label.getComputedTextLength() : 0;
    const charWidth = measured > 0 ? measured / Math.max(1, mark.text.length) : READOUT_CHAR;
    let lines = [mark.text];
    let box = layoutReadout(label, lines, charWidth);
    // Two tries: the first from the mean character width, the second from the lines it drew.
    for (let tries = 0; tries < 2 && box.width > room && room > 0; tries++) {
      const longest = Math.max(...lines.map((line) => line.length));
      lines = readoutLines(mark.text, Math.max(1, Math.floor((longest * room) / box.width)));
      box = layoutReadout(label, lines, charWidth);
    }
    const lo = { x: view.left + READOUT_EDGE, y: view.top + READOUT_EDGE };
    const hi = { x: view.right - READOUT_EDGE - box.width, y: view.bottom - READOUT_EDGE - box.height };
    const right = mark.x + READOUT_GAP;
    const left = mark.x - READOUT_GAP - box.width;
    const x = right <= hi.x ? right : left >= lo.x ? left : clamp(mark.x - box.width / 2, lo.x, hi.x);
    const above = mark.y - READOUT_RISE - box.height;
    const y = clamp(above >= lo.y ? above : mark.y + READOUT_RISE, lo.y, hi.y);
    // The text's own x and y place its box's top left corner at (x, y).
    const at = { x: Math.round((x - box.x - mark.x) * 10) / 10, y: Math.round((y - box.y - mark.y) * 10) / 10 };
    label.setAttribute("x", String(at.x));
    label.setAttribute("y", String(at.y));
    for (const span of label.children) span.setAttribute("x", String(at.x));
  }

  // The point nearest the pointer read out, the pointer in the scene's px: the svg's box moves
  // with the frame's scroll. Where no point is drawn, nothing is.
  function read() {
    if (readout.state !== "pointed") return;
    const scene = getScene();
    const svg = frame.querySelector("svg");
    const box = svg?.getBoundingClientRect();
    if (!scene || !box || scene.marks.length === 0 || !(box.width > 0 && box.height > 0)) {
      conceal();
      return;
    }
    const { pointer } = readout;
    const x = ((pointer.x - box.left) * scene.width) / box.width;
    const y = ((pointer.y - box.top) * scene.height) / box.height;
    show(scene, nearestMark(scene.marks, x, y));
  }

  // A frame's work: the pointer's point read out, or the stepped point placed again in the view
  // the frame now shows, and let go once its point has left that view.
  function onFrame() {
    pending = 0;
    if (readout.state === "pointed") {
      read();
      return;
    }
    if (readout.state !== "keyboard") return;
    const { scene, index } = readout;
    const mark = scene.marks[index];
    const view = visible(scene);
    const inView = mark.x >= view.left && mark.x <= view.right && mark.y >= view.top && mark.y <= view.bottom;
    if (scene === getScene() && inView) show(scene, index);
    else move("leave", HIDDEN);
  }

  /** @param {PointerEvent} event */
  function onPointerMove(event) {
    const pointer = { x: event.clientX, y: event.clientY };
    move("point", readout.state === "suspended" ? { state: "suspended", pointer } : { state: "pointed", pointer });
  }

  const leave = () => move("leave", readout.state === "suspended" ? { state: "suspended", pointer: null } : HIDDEN);
  const onScroll = () => move("scroll", readout);

  // Scroll a point the keys stepped to into the frame's view.
  /** @param {Scene} scene @param {Mark} mark */
  function reveal(scene, mark) {
    const svg = frame.querySelector("svg");
    if (!svg) return;
    const box = svg.getBoundingClientRect();
    const y = box.top - frame.getBoundingClientRect().top + (mark.y * box.height) / scene.height;
    if (y < 0 || y > frame.clientHeight) frame.scrollTop += y - frame.clientHeight / 2;
  }

  /** @param {KeyboardEvent} event */
  function onKeyDown(event) {
    if (event.key === "Escape") {
      leave();
      return;
    }
    const scene = getScene();
    const step = KEY_STEPS[event.key];
    const jump = event.key === "Home" || event.key === "End";
    if (!scene || scene.marks.length === 0 || (step === undefined && !jump)) return;
    event.preventDefault();
    // Positions in the scene's reading order, where it has one, and else in the marks' own.
    const order = scene.steps ?? null;
    const last = scene.marks.length - 1;
    const from = scene === shownIn && current >= 0 ? (order ? order.indexOf(current) : current) : -1;
    const at = event.key === "Home" ? 0 : event.key === "End" ? last
      : from < 0 ? (step > 0 ? 0 : last) : Math.max(0, Math.min(last, from + step));
    move("key", { state: "keyboard", scene, index: order ? order[at] : at });
  }

  frame.addEventListener("pointermove", onPointerMove);
  frame.addEventListener("pointerleave", leave);
  frame.addEventListener("keydown", onKeyDown);
  frame.addEventListener("blur", leave);
  frame.addEventListener("scroll", onScroll, { passive: true });
  return {
    suspend: () => void move("suspend", { state: "suspended", pointer: kept() }),
    rest() {
      const pointer = kept();
      if (pointer) move("resume", { state: "pointed", pointer });
      else move("rest", HIDDEN);
    },
    redraw: () => void move("redraw", readout.state === "keyboard" ? HIDDEN : readout),
    close: () => void move("close", HIDDEN),
    destroy() {
      if (pending) cancelAnimationFrame(pending);
      pending = 0;
      frame.removeEventListener("pointermove", onPointerMove);
      frame.removeEventListener("pointerleave", leave);
      frame.removeEventListener("keydown", onKeyDown);
      frame.removeEventListener("blur", leave);
      frame.removeEventListener("scroll", onScroll);
    },
  };
}
