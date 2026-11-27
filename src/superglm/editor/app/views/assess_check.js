// @ts-check
/**
 * A check's face on the Assess overview and in its list: the count of what left the band, one
 * figure under it, the orange flag, and the thumbnail. Compare with the fit lives here too, as
 * the overview and the large view share it.
 */

import { checkScene, sceneMarkup } from "../chart/check_plots.js";
import { countText } from "../reports.js";

/** @typedef {import('../api/contracts.js').CheckReport} CheckReport */
/** @typedef {import('../api/contracts.js').CheckPayload} CheckPayload */
/** @typedef {import('../api/contracts.js').CheckBody} CheckBody */
/** @typedef {import('../api/contracts.js').EvidenceStatus} EvidenceStatus */
/** @typedef {import('../api/contracts.js').EvidenceState} EvidenceState */
/** @typedef {import('../chart/check_plots.js').PITStats} PITStats */
/** @typedef {import('../chart/check_plots.js').CalibrationStats} CalibrationStats */
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
