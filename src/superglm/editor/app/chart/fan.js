// @ts-check
/**
 * The distribution panel's density fan: the feature across, the log10 response amount up, each
 * column shaded by the density of log10 Y relative to its own peak, blue below the line and orange
 * beyond it. The pixels are computed here as a pure function, so the canvas only receives them.
 */

/** @typedef {import('../distribution/families.js').Law} Law */
/** @typedef {{min_log10: number, max_log10: number}} AmountAxis */
/** @typedef {[number, number, number]} RGB */

/**
 * @typedef {object} FanColumn
 * @property {number | null} x  the feature's value, for a numeric fan
 * @property {string | null} level  the level, for a fan of levels
 * @property {Law} law
 * @property {number} median  the median of the amounts above zero, which the shading draws
 * @property {number} mean  the mean of the amounts above zero
 * @property {number} cut  the log10 amount where the part beyond the line starts
 * @property {boolean} spreadPinned  the CV was clamped to the largest the panel computes
 */

/** @typedef {{kind: "numeric" | "levels", columns: FanColumn[]}} FanModel */

/**
 * A log10 amount held inside the plot; an amount of zero or none sits on the bottom edge.
 * @param {number} l @param {AmountAxis} axis
 */
export function clampToPlot(l, axis) {
  if (Number.isNaN(l)) return axis.min_log10;
  return Math.min(axis.max_log10, Math.max(axis.min_log10, l));
}

/**
 * The colour at `t` in [0, 1] on evenly spaced stops, linear in each channel.
 * @param {number} t @param {RGB[]} stops
 * @returns {RGB}
 */
export function rampColour(t, stops) {
  const p = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(p));
  const f = p - i;
  const [a, b] = [stops[i], stops[i + 1]];
  return /** @type {RGB} */ ([0, 1, 2].map((j) => a[j] * (1 - f) + b[j] * f));
}

/**
 * Each column's place across the fan in [0, 1]: by value for a numeric fan, and the middle of
 * its band for levels.
 * @param {FanModel} fan
 * @returns {number[]}
 */
export function fanPositions(fan) {
  const n = fan.columns.length;
  if (fan.kind === "levels") return fan.columns.map((_, i) => (i + 0.5) / n);
  const xs = fan.columns.map((column) => Number(column.x));
  const lo = Math.min(...xs);
  const span = Math.max(...xs) - lo;
  return xs.map((x) => (span > 0 ? (x - lo) / span : 0.5));
}

/**
 * The column a pixel column shows: the nearest by value, or the level band it falls in.
 * @param {FanModel} fan @param {number[]} positions from `fanPositions`
 * @param {number} px @param {number} width
 */
export function columnAt(fan, positions, px, width) {
  const n = fan.columns.length;
  if (fan.kind === "levels") return Math.min(n - 1, Math.floor(((px + 0.5) / width) * n));
  const f = width > 1 ? px / (width - 1) : 0.5;
  let best = 0;
  for (let i = 1; i < n; i += 1) {
    if (Math.abs(positions[i] - f) < Math.abs(positions[best] - f)) best = i;
  }
  return best;
}

/**
 * The fan's RGBA pixels in one ramp, row 0 at the top of the axis.
 * @param {FanModel} fan @param {number} width @param {number} height
 * @param {AmountAxis} axis @param {RGB[]} ramp
 * @param {Uint8ClampedArray<ArrayBuffer>} [into] pixels of this size to write over
 * @returns {Uint8ClampedArray<ArrayBuffer>}
 */
export function fanPixels(fan, width, height, axis, ramp, into) {
  const pixels = into ?? new Uint8ClampedArray(width * height * 4);
  if (!fan.columns.length) return pixels;
  const positions = fanPositions(fan);
  const words = new Uint32Array(pixels.buffer, pixels.byteOffset, width * height);
  const colours = rampWords(ramp);
  const ascending = positions.every((p, i) => i === 0 || positions[i - 1] <= p);
  let at = 0;
  for (let x = 0; x < width; x += 1) {
    let i;
    if (fan.kind === "numeric" && ascending) {
      const f = width > 1 ? x / (width - 1) : 0.5;
      while (at + 1 < positions.length && Math.abs(positions[at + 1] - f) < Math.abs(positions[at] - f)) at += 1;
      i = at;
    } else {
      i = columnAt(fan, positions, x, width);
    }
    const column = fan.columns[i];
    const shades = lawShades(column.law, height, axis);
    for (let y = 0; y < height; y += 1) {
      words[y * width + x] = colours[Math.round(shades[y] * RAMP_TOP)];
    }
  }
  return pixels;
}

/**
 * The part of the plot past the line, as a polygon of plot pixels with row 0 at the top of the
 * axis: the plot's top edge, then back along the line, which runs as the fan's cut line does.
 * @param {FanModel} fan @param {number} width @param {number} height @param {AmountAxis} axis
 * @returns {Array<[number, number]>}
 */
export function fanBeyondRegion(fan, width, height, axis) {
  const span = axis.max_log10 - axis.min_log10;
  const n = fan.columns.length;
  /** @param {number} cut */
  const rowOf = (cut) => {
    const row = ((axis.max_log10 - cut) / span) * height;
    return Number.isNaN(row) ? height : Math.max(0, Math.min(height, row));
  };
  /** @type {Array<[number, number]>} */
  const along = [];
  if (fan.kind === "levels") {
    fan.columns.forEach((column, i) => {
      const row = rowOf(column.cut);
      along.push([(i / n) * width, row], [((i + 1) / n) * width, row]);
    });
  } else {
    const positions = fanPositions(fan);
    const order = positions.map((_, i) => i).sort((a, b) => positions[a] - positions[b]);
    for (const i of order) along.push([positions[i] * width, rowOf(fan.columns[i].cut)]);
    if (along.length) {
      along.unshift([0, along[0][1]]);
      along.push([width, along[along.length - 1][1]]);
    }
  }
  return [[0, 0], [width, 0], ...along.reverse()];
}

// A column's shade changes smoothly down the axis, so past this many samples more add nothing to
// the eye and only slow a drag; the rows between two samples take their straight line.
const MAX_SHADE_SAMPLES = 150;
const RAMP_TOP = 255;

/** @type {WeakMap<Law, {height: number, min: number, max: number, shades: Float32Array}>} */
const LAW_SHADES = new WeakMap();

/**
 * A law's density down the axis as a fraction of its own peak, row 0 at the top. It does not
 * depend on the line, so a line that moves redraws from these without asking the law again.
 * @param {Law} law @param {number} height @param {AmountAxis} axis
 * @returns {Float32Array}
 */
function lawShades(law, height, axis) {
  const known = LAW_SHADES.get(law);
  if (known && known.height === height && known.min === axis.min_log10 && known.max === axis.max_log10) {
    return known.shades;
  }
  const last = height - 1;
  const stride = Math.max(1, Math.round(height / MAX_SHADE_SAMPLES));
  /** @param {number} y */
  const levelAt = (y) => axis.max_log10 - (last > 0 ? y / last : 0) * (axis.max_log10 - axis.min_log10);
  const rows = [];
  for (let y = 0; y < last; y += stride) rows.push(y);
  rows.push(Math.max(0, last));
  const densities = rows.map((y) => law.log10Density(levelAt(y)));
  const peak = Math.max(...densities);
  const shades = new Float32Array(height);
  for (let k = 0; k < rows.length; k += 1) {
    const value = peak > -Infinity ? 10 ** (densities[k] - peak) : 0;
    shades[rows[k]] = value;
    if (k === 0) continue;
    const from = rows[k - 1];
    const gap = rows[k] - from;
    for (let y = from + 1; y < rows[k]; y += 1) {
      shades[y] = shades[from] + ((y - from) / gap) * (value - shades[from]);
    }
  }
  LAW_SHADES.set(law, { height, min: axis.min_log10, max: axis.max_log10, shades });
  return shades;
}

/** @type {WeakMap<RGB[], Uint32Array>} */
const RAMP_WORDS = new WeakMap();

/**
 * A ramp's opaque colours as pixel words, one per level of shade.
 * @param {RGB[]} stops
 * @returns {Uint32Array}
 */
function rampWords(stops) {
  const known = RAMP_WORDS.get(stops);
  if (known) return known;
  const words = new Uint32Array(RAMP_TOP + 1);
  const bytes = new Uint8ClampedArray(words.buffer);
  for (let level = 0; level <= RAMP_TOP; level += 1) {
    const [r, g, b] = rampColour(level / RAMP_TOP, stops);
    bytes.set([r, g, b, 255], 4 * level);
  }
  RAMP_WORDS.set(stops, words);
  return words;
}

/**
 * Label positions pushed apart to at least `gap`, in their input order, inside [top, bottom]
 * when there is room for them all.
 * @param {number[]} ys @param {number} gap @param {number} top @param {number} bottom
 * @returns {number[]}
 */
export function spreadLabels(ys, gap, top, bottom) {
  const order = ys.map((y, i) => [y, i]).sort((a, b) => a[0] - b[0]);
  const placed = order.map(([y]) => Math.min(bottom, Math.max(top, y)));
  for (let k = 1; k < placed.length; k += 1) placed[k] = Math.max(placed[k], placed[k - 1] + gap);
  if (placed.length) placed[placed.length - 1] = Math.min(placed[placed.length - 1], bottom);
  for (let k = placed.length - 2; k >= 0; k -= 1) placed[k] = Math.min(placed[k], placed[k + 1] - gap);
  const out = new Array(ys.length);
  order.forEach(([, i], k) => {
    out[i] = Math.max(top, placed[k]);
  });
  return out;
}

/**
 * Round tick values for the fan's feature axis: steps of 1, 2 or 5 times a power of ten.
 * @param {number} lo @param {number} hi @param {number} count  about how many ticks
 * @returns {number[]}
 */
export function niceTicks(lo, hi, count) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(1, count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? 10 * power;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
    ticks.push(Number(v.toPrecision(12)));
  }
  return ticks;
}

/**
 * A `#rrggbb` colour as channels.
 * @param {string} hex
 * @returns {RGB}
 */
export function hexToRgb(hex) {
  const h = hex.trim().replace(/^#/, "");
  return /** @type {RGB} */ ([0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)));
}
