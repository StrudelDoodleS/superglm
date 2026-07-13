// @ts-check

const MAX_TICKS = 30;
const MAX_LABEL_WIDTH = 160;
const MIN_ANGLED_SLOT = 56;
const TICK_OFFSET = 18;
const TITLE_GAP = 18;
const OUTER_PAD = 12;
const ELLIPSIS = "…";

const GRAPHEME_SEGMENTER = typeof Intl.Segmenter === "function"
  ? new Intl.Segmenter(undefined, { granularity: "grapheme" })
  : null;

/**
 * @typedef {object} LabelMeasurement
 * @property {number} fullWidth
 * @property {readonly number[]} prefixWidths Width after each grapheme cluster.
 * @property {number} ellipsisWidth
 * @property {number} height
 */

/**
 * @template T
 * @typedef {object} CategoricalAxisOptions
 * @property {readonly T[]} values
 * @property {readonly string[]} labels
 * @property {readonly LabelMeasurement[]} measurements
 * @property {number} availableWidth
 * @property {number} svgHeight
 * @property {number} baseLeft Symmetric left/right edge inset used for label budgeting.
 * @property {number} baseBottom
 * @property {number} [titleHeight]
 */

/**
 * @template T
 * @typedef {object} CategoricalAxisTick
 * @property {number} index
 * @property {T} value
 * @property {string} fullLabel
 * @property {string} displayLabel
 * @property {0|-45} angle
 * @property {"middle"|"end"} anchor
 * @property {number} width
 * @property {number} height
 */

/**
 * @template T
 * @typedef {object} CategoricalAxisPlan
 * @property {readonly CategoricalAxisTick<T>[]} ticks
 * @property {number} bottom
 * @property {number} axisY
 * @property {number} titleY
 * @property {number} titleHeight
 * @property {number} maxLabelHeight
 * @property {number} labelBudget
 */

/**
 * Split a label into the same grapheme clusters used by measurement and truncation.
 *
 * @param {string} label
 * @returns {string[]}
 */
export function splitLabelGraphemes(label) {
  if (typeof label !== "string") throw new TypeError("label must be a string");
  if (!GRAPHEME_SEGMENTER) return Array.from(label);
  return Array.from(GRAPHEME_SEGMENTER.segment(label), (part) => part.segment);
}

/**
 * Return at most `maximum` stable indices, retaining both edges whenever possible.
 *
 * @param {number} count
 * @param {number} maximum
 * @returns {number[]}
 */
export function evenlySpacedIndices(count, maximum) {
  assertNonnegativeInteger("count", count);
  assertNonnegativeInteger("maximum", maximum);
  if (count === 0 || maximum === 0) return [];
  if (count <= maximum) return Array.from({ length: count }, (_, index) => index);
  if (maximum === 1) return [0];
  const indices = [];
  for (let position = 0; position < maximum; position += 1) {
    indices.push(Math.round(position * (count - 1) / (maximum - 1)));
  }
  return Array.from(new Set(indices)).sort((left, right) => left - right);
}

/**
 * Fit measured text to a pixel budget without changing the source label.
 *
 * @param {string} label
 * @param {LabelMeasurement} measurement
 * @param {number} budget
 * @returns {string}
 */
export function fitMeasuredLabel(label, measurement, budget) {
  const graphemes = validateMeasurement(label, measurement);
  assertNonnegativeFinite("budget", budget);
  if (measurement.fullWidth <= budget) return label;
  if (measurement.ellipsisWidth > budget) return "";

  // Search from the longest prefix because shaped-font widths need not be monotonic.
  for (let length = graphemes.length - 1; length >= 0; length -= 1) {
    const prefixWidth = length === 0 ? 0 : measurement.prefixWidths[length - 1];
    if (prefixWidth + measurement.ellipsisWidth <= budget) {
      return `${graphemes.slice(0, length).join("")}${ELLIPSIS}`;
    }
  }
  return "";
}

/**
 * Project an axis-aligned box after rotation.
 *
 * @param {number} width
 * @param {number} height
 * @param {number} degrees
 * @returns {{width:number, height:number}}
 */
export function rotatedExtent(width, height, degrees) {
  assertNonnegativeFinite("width", width);
  assertNonnegativeFinite("height", height);
  assertFinite("degrees", degrees);
  const radians = Math.abs(degrees) * Math.PI / 180;
  return {
    width: Math.abs(width * Math.cos(radians)) + Math.abs(height * Math.sin(radians)),
    height: Math.abs(width * Math.sin(radians)) + Math.abs(height * Math.cos(radians)),
  };
}

/**
 * Plan categorical ticks and the bottom gutter from measured, immutable labels.
 *
 * @template T
 * @param {CategoricalAxisOptions<T>} options
 * @returns {CategoricalAxisPlan<T>}
 */
export function planCategoricalAxis({
  values,
  labels,
  measurements,
  availableWidth,
  svgHeight,
  baseLeft,
  baseBottom,
  titleHeight = 14,
}) {
  if (values.length !== labels.length || labels.length !== measurements.length) {
    throw new RangeError("values, labels, and measurements must have the same length");
  }
  assertPositiveFinite("availableWidth", availableWidth);
  assertPositiveFinite("svgHeight", svgHeight);
  assertNonnegativeFinite("baseLeft", baseLeft);
  assertNonnegativeFinite("baseBottom", baseBottom);
  assertNonnegativeFinite("titleHeight", titleHeight);
  if (baseBottom >= svgHeight) {
    throw new RangeError("baseBottom must be smaller than svgHeight");
  }
  measurements.forEach((measurement, index) => {
    validateMeasurement(labels[index], measurement);
  });

  const densityLimit = Math.max(2, Math.floor(availableWidth / MIN_ANGLED_SLOT) + 1);
  const indices = evenlySpacedIndices(labels.length, Math.min(MAX_TICKS, densityLimit));
  const slot = availableWidth / Math.max(indices.length - 1, 1);
  const horizontalBudget = Math.max(0, slot - 10);
  const maxMeasuredHeight = Math.max(0, ...indices.map((index) => measurements[index].height));
  const rotate = indices.some((index) => measurements[index].fullWidth > horizontalBudget);
  /** @type {0|-45} */
  const angle = rotate ? -45 : 0;
  const radians = Math.abs(angle) * Math.PI / 180;
  const projectedHeight = maxMeasuredHeight * Math.sin(radians);
  const cosine = Math.cos(radians);
  const angledEdgeBudget = rotate
    ? Math.max(0, (baseLeft - OUTER_PAD - projectedHeight) / cosine)
    : Number.POSITIVE_INFINITY;
  const angledSlotBudget = rotate
    ? Math.max(0, (slot - projectedHeight) / cosine)
    : Number.POSITIVE_INFINITY;
  const centeredEdgeBudget = Math.max(0, 2 * (baseLeft - OUTER_PAD));
  const labelBudget = Math.max(0, Math.min(
    MAX_LABEL_WIDTH,
    rotate ? angledEdgeBudget : centeredEdgeBudget,
    rotate ? angledSlotBudget : horizontalBudget,
  ));

  /** @type {CategoricalAxisTick<T>[]} */
  const ticks = indices.map((index) => {
    const fullLabel = labels[index];
    const measurement = measurements[index];
    const displayLabel = fitMeasuredLabel(fullLabel, measurement, labelBudget);
    const displayWidth = measuredDisplayWidth(fullLabel, displayLabel, measurement);
    const extent = rotatedExtent(displayWidth, measurement.height, angle);
    /** @type {"middle"|"end"} */
    const anchor = rotate ? "end" : "middle";
    return {
      index,
      value: values[index],
      fullLabel,
      displayLabel,
      angle,
      anchor,
      width: extent.width,
      height: extent.height,
    };
  });

  const maxLabelHeight = Math.max(0, ...ticks.map((tick) => tick.height));
  const requiredBottom = Math.ceil(
    TICK_OFFSET + maxLabelHeight + TITLE_GAP + titleHeight + OUTER_PAD,
  );
  const bottom = Math.max(baseBottom, requiredBottom);
  if (bottom >= svgHeight) {
    throw new RangeError("categorical axis content does not fit within svgHeight");
  }
  const axisY = svgHeight - bottom;
  const titleY = axisY + TICK_OFFSET + maxLabelHeight + TITLE_GAP;
  return {
    ticks,
    bottom,
    axisY,
    titleY,
    titleHeight,
    maxLabelHeight,
    labelBudget,
  };
}

/**
 * @param {string} fullLabel
 * @param {string} displayLabel
 * @param {LabelMeasurement} measurement
 * @returns {number}
 */
function measuredDisplayWidth(fullLabel, displayLabel, measurement) {
  if (displayLabel === fullLabel) return measurement.fullWidth;
  if (displayLabel === "") return 0;
  const hasEllipsis = displayLabel.endsWith(ELLIPSIS);
  const prefix = hasEllipsis ? displayLabel.slice(0, -ELLIPSIS.length) : displayLabel;
  const graphemeCount = splitLabelGraphemes(prefix).length;
  const prefixWidth = graphemeCount === 0 ? 0 : measurement.prefixWidths[graphemeCount - 1];
  return prefixWidth + (hasEllipsis ? measurement.ellipsisWidth : 0);
}

/**
 * @param {string} label
 * @param {LabelMeasurement} measurement
 * @returns {string[]}
 */
function validateMeasurement(label, measurement) {
  const graphemes = splitLabelGraphemes(label);
  assertNonnegativeFinite("measurement.fullWidth", measurement.fullWidth);
  assertNonnegativeFinite("measurement.ellipsisWidth", measurement.ellipsisWidth);
  assertNonnegativeFinite("measurement.height", measurement.height);
  if (!Array.isArray(measurement.prefixWidths)) {
    throw new TypeError("measurement.prefixWidths must be an array");
  }
  if (measurement.prefixWidths.length !== graphemes.length) {
    throw new RangeError("measurement.prefixWidths must contain one width per grapheme");
  }
  measurement.prefixWidths.forEach((width, index) => {
    assertNonnegativeFinite(`measurement.prefixWidths[${index}]`, width);
  });
  return graphemes;
}

/** @param {string} name @param {number} value */
function assertFinite(name, value) {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
}

/** @param {string} name @param {number} value */
function assertNonnegativeFinite(name, value) {
  assertFinite(name, value);
  if (value < 0) throw new RangeError(`${name} must be nonnegative`);
}

/** @param {string} name @param {number} value */
function assertPositiveFinite(name, value) {
  assertFinite(name, value);
  if (value <= 0) throw new RangeError(`${name} must be positive`);
}

/** @param {string} name @param {number} value */
function assertNonnegativeInteger(name, value) {
  assertNonnegativeFinite(name, value);
  if (!Number.isInteger(value)) throw new RangeError(`${name} must be an integer`);
}
