// @ts-check
/**
 * A check's face on the Assess overview and in its list: the count of what left the band, one
 * figure under it, the orange flag, and the thumbnail. Compare with the fit lives here too, as
 * the overview and the large view share it, and so does the large view with its hover readout.
 */

import { checkScene, nearestMark, sceneMarkup } from "../chart/check_plots.js";
import { escapeHTML } from "../format.js";
import { countText } from "../reports.js";
import { assessEntry } from "../state/assess.js";

/** @typedef {import('../api/contracts.js').CheckReport} CheckReport */
/** @typedef {import('../api/contracts.js').CheckPayload} CheckPayload */
/** @typedef {import('../api/contracts.js').CheckBody} CheckBody */
/** @typedef {import('../api/contracts.js').EvidenceStatus} EvidenceStatus */
/** @typedef {import('../api/contracts.js').EvidenceState} EvidenceState */
/** @typedef {import('../chart/check_plots.js').PITStats} PITStats */
/** @typedef {import('../chart/check_plots.js').CalibrationStats} CalibrationStats */
/** @typedef {import('../chart/check_plots.js').Scene} Scene */
/** @typedef {import('../chart/check_plots.js').Mark} Mark */
/**
 * The large view's nodes: its heading, the split line beside it, the legend chips, and the frame
 * the plot is drawn in.
 * @typedef {{title:HTMLElement, split:HTMLElement, legend:HTMLElement, frame:HTMLElement}} CheckNodes
 */
/** @typedef {{value: string, note: string, flagged: boolean}} CheckFace */
/** @typedef {{live: boolean, checked: boolean, body: string}} CompareSwitch */

const MISSING = "—";
/** A card's thumbnail box, which `.assess-card-plot` holds at this size. */
export const THUMB_BOX = Object.freeze({ width: 160, height: 62, thumb: true });
const NO_EDIT = "The model is as fitted; there is no edit to compare.";
const COMPARES = "Draws the model as fitted in grey beside the model as edited in blue, on every check. "
  + "It changes only what is drawn.";

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
 * @param {CheckReport} report @param {CheckPayload|null} payload @param {boolean} compare
 * @returns {string}
 */
export function checkThumbnail(report, payload, compare) {
  if (payload === null) return sceneMarkup({ ...THUMB_BOX, layers: [], labels: [], marks: [] });
  return sceneMarkup(checkScene(report, payload, THUMB_BOX, { compare }));
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
 * Compare with the fit: live once an offered check has the fitted model beside an edit, and on
 * when the analyst turned it on while it is live. Its popover says what it draws, or why it is off.
 * @param {Readonly<Record<CheckReport, EvidenceState>>} evidence @param {readonly CheckReport[]} offered
 * @param {boolean} compareOriginal
 * @returns {CompareSwitch}
 */
export function compareSwitch(evidence, offered, compareOriginal) {
  const live = offered.some((name) => Boolean(checkPayload(evidence[name])?.compares_edit));
  return { live, checked: live && compareOriginal, body: live ? COMPARES : NO_EDIT };
}

/**
 * Write the switch's state onto its button; a model that offers no check hides it.
 * @param {HTMLElement} button @param {CompareSwitch} state @param {boolean} shown
 */
export function renderCompareSwitch(button, state, shown) {
  button.hidden = !shown;
  button.setAttribute("aria-checked", String(state.checked));
  button.setAttribute("aria-disabled", String(!state.live));
  button.dataset.popoverBody = state.body;
}

const SVG_NS = "http://www.w3.org/2000/svg";
// What the split line adds when a held-out split's band takes the fitted parameters as exact,
// and the model it names when only one of the two drawn fell back.
const LEVEL_TEXT = "parameters taken as exact";
const LEVEL_TITLE = "Parameters taken as exact";
// The readout's text sits this far from its point, and below it when the point is this near the top.
const READOUT_GAP = 10;
const READOUT_TOP_ROOM = 30;
/** @type {Readonly<Record<string, number>>} */
const KEY_STEPS = Object.freeze({ ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 });

/** @param {CheckBody|null|undefined} body @returns {string|null} its fallback sentence, if it has one */
const levelNote = (body) => (body?.available && typeof body.level_note === "string" && body.level_note ? body.level_note : null);

/**
 * A check's split line: the rows it reads, and on a held-out split whose band took the fitted
 * parameters as exact, the marker that says so and its sentence. With `compare`, the fit is
 * drawn beside the edit, and when only one of the two fell back the marker names it.
 * @param {CheckPayload|null} payload @param {boolean} compare
 * @returns {{text: string, level: string|null, note: string|null}}
 */
export function splitLine(payload, compare) {
  const split = payload?.split;
  if (!payload || !split) return { text: "", level: null, note: null };
  const text = `${split.label}, ${countText(split.n_obs)} rows`;
  const edited = levelNote(answered(payload));
  const fitted = compare && payload.models.original?.available ? levelNote(payload.models.original) : undefined;
  // `fitted` is undefined when the fit is not drawn, and null when it is drawn and simulated.
  const note = edited ?? fitted ?? null;
  if (note === null) return { text, level: null, note: null };
  // The model is named only when the fit is drawn and exactly one of the two fell back.
  const one = fitted !== undefined && (edited === null) !== (fitted === null);
  const named = !one ? "" : edited === null ? " (as fitted)" : " (as edited)";
  return { text, level: `${LEVEL_TEXT}${named}`, note };
}

/**
 * Draw a check large: its label, its split line, the legend when the fit is drawn beside the
 * edit, and the plot at `width` by `height` (a taller A/E grows past it, and the frame scrolls).
 * A check with no answer for this model says why in the frame. The svg is written once per call.
 * @param {CheckNodes} nodes @param {CheckReport} report @param {CheckPayload|null} payload
 * @param {{compare: boolean, width: number, height: number}} options
 * @returns {Scene|null} the scene drawn, whose marks the hover reads; null when none is
 */
export function renderCheckView(nodes, report, payload, { compare, width, height }) {
  const label = assessEntry(report)?.label ?? "";
  nodes.title.textContent = label;
  const line = splitLine(payload, compare);
  nodes.split.innerHTML = escapeHTML(line.text) + (line.level === null || line.note === null ? ""
    : ` · <span class="check-level" tabindex="0" data-popover-title="${LEVEL_TITLE}"`
      + ` data-popover-body="${escapeHTML(line.note)}">${line.level}</span>`);
  const { frame } = nodes;
  if (payload === null || answered(payload) === null) {
    nodes.legend.hidden = true;
    for (const name of ["role", "tabindex", "aria-label"]) frame.removeAttribute(name);
    const reason = payload === null ? null : (payload.available ? payload.models.edited.reason : payload.reason);
    frame.innerHTML = reason ? `<p class="check-reason">${escapeHTML(reason)}</p>` : "";
    return null;
  }
  const scene = checkScene(report, payload, { width, height }, { compare });
  nodes.legend.hidden = !scene.layers.some((layer) => layer.role === "original");
  frame.setAttribute("role", "img");
  frame.setAttribute("tabindex", "0");
  frame.setAttribute("aria-label", label);
  frame.innerHTML = sceneMarkup(scene);
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
 * Read out the point nearest the pointer on the check in `frame`, and step through the points
 * with the arrow keys while the frame has focus. A pointer move waits for the next frame, and
 * writes only the readout group: its transform, its text and whether it is hidden.
 * @param {HTMLElement} frame @param {() => Scene|null} getScene the scene on screen
 * @returns {{destroy(): void}}
 */
export function bindCheckHover(frame, getScene) {
  /** @type {{x: number, y: number}|null} */
  let pointer = null;
  let pending = 0;
  // The point read out, and the scene it belongs to.
  let current = -1;
  /** @type {Scene|null} */
  let shownIn = null;

  const readout = () => frame.querySelector(".check-readout");

  function hide() {
    pointer = null;
    current = -1;
    const group = readout();
    if (group && !group.hasAttribute("hidden")) group.setAttribute("hidden", "");
  }

  /** @param {Scene} scene @param {number} index */
  function show(scene, index) {
    const group = readout();
    if (!group || index < 0) return;
    if (index === current && scene === shownIn && !group.hasAttribute("hidden")) return;
    current = index;
    shownIn = scene;
    const mark = scene.marks[index];
    const label = readoutText(group);
    const left = mark.x > scene.width / 2;
    const below = mark.y < READOUT_TOP_ROOM;
    group.setAttribute("transform", `translate(${Math.round(mark.x * 10) / 10} ${Math.round(mark.y * 10) / 10})`);
    label.setAttribute("x", String(left ? -READOUT_GAP : READOUT_GAP));
    label.setAttribute("y", String(below ? READOUT_GAP + 8 : -READOUT_GAP));
    label.setAttribute("text-anchor", left ? "end" : "start");
    label.textContent = mark.text;
    if (group.hasAttribute("hidden")) group.removeAttribute("hidden");
  }

  // The pointer in the scene's px: the svg's box moves with the frame's scroll.
  function onFrame() {
    pending = 0;
    const scene = getScene();
    const svg = frame.querySelector("svg");
    if (!pointer || !scene || !svg || scene.marks.length === 0) return;
    const box = svg.getBoundingClientRect();
    if (!(box.width > 0 && box.height > 0)) return;
    const x = ((pointer.x - box.left) * scene.width) / box.width;
    const y = ((pointer.y - box.top) * scene.height) / box.height;
    show(scene, nearestMark(scene.marks, x, y));
  }

  /** @param {PointerEvent} event */
  function onPointerMove(event) {
    pointer = { x: event.clientX, y: event.clientY };
    if (!pending) pending = requestAnimationFrame(onFrame);
  }

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
      hide();
      return;
    }
    const scene = getScene();
    const step = KEY_STEPS[event.key];
    const jump = event.key === "Home" || event.key === "End";
    if (!scene || scene.marks.length === 0 || (step === undefined && !jump)) return;
    event.preventDefault();
    const last = scene.marks.length - 1;
    const from = scene === shownIn ? current : -1;
    const index = event.key === "Home" ? 0 : event.key === "End" ? last
      : from < 0 ? (step > 0 ? 0 : last) : Math.max(0, Math.min(last, from + step));
    pointer = null;
    show(scene, index);
    reveal(scene, scene.marks[index]);
  }

  frame.addEventListener("pointermove", onPointerMove);
  frame.addEventListener("pointerleave", hide);
  frame.addEventListener("keydown", onKeyDown);
  frame.addEventListener("blur", hide);
  return {
    destroy() {
      if (pending) cancelAnimationFrame(pending);
      pending = 0;
      frame.removeEventListener("pointermove", onPointerMove);
      frame.removeEventListener("pointerleave", hide);
      frame.removeEventListener("keydown", onKeyDown);
      frame.removeEventListener("blur", hide);
    },
  };
}
