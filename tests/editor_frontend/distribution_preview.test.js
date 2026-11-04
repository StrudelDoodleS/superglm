// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { toDisplay } from "../../src/superglm/editor/app/chart/display.js";
import { previewEta } from "../../src/superglm/editor/app/distribution/preview.js";
import { panelModel } from "../../src/superglm/editor/app/views/distribution_panel.js";

const EPS = Number.EPSILON;
const LOG = { kind: "log" };
const FLOORED = { kind: "floored_log", floor: 0.05 };
const P99 = { mode: "percentile", value: 0.99, style: "percent" };
const SPREAD = { kind: "lognormal_cv", reference_eta: -0.5, floor: 0.05 };
const GRID = [18, 30, 45, 60, 85];
const BASE_EFFECT = [-0.1, 0, 0.05, 0.1, 0.2];
// The drag's change on the link scale at each grid point.
const DRAG = [0.15, 0.3, 0.5, 0.2, 0.25];

/** A log-normal panel over age; row 0 lies below the term's grid and row 4 above it. */
function agePanel() {
  const rows = [[7.0, -0.5], [7.1, -0.4], [7.2, -0.3], [7.4, -0.45], [7.5, -0.2]];
  const columns = [[7.0, -0.5], [7.2, -0.3], [7.5, -0.2]];
  return {
    feature: "age",
    family: "lognormal_mean",
    kind: "numeric",
    empty_reason: null,
    rows: [10, 24, 37.5, 52.5, 95].map((x) => ({ label: String(x), x, level: null })),
    columns: [18, 45, 85].map((x) => ({ x, level: null })),
    parameters: [
      { name: "mean", label: "Average", role: "average", link: LOG },
      { name: "scale", label: "Spread", role: "spread", link: FLOORED },
    ],
    term_keys: { mean: "mean:age", scale: "scale:age" },
    eta: { fitted: { rows, columns }, edited: { rows, columns } },
    amount_axis: { min_log10: 1, max_log10: 6, clipped_below: false },
    amount_unit: "€",
    reference: [],
    cap_note: null,
  };
}

/** A numeric term on GRID whose effect is BASE_EFFECT plus `shift`. */
function curve(display, shift = [0, 0, 0, 0, 0]) {
  return {
    x: GRID,
    levels: null,
    display,
    y: BASE_EFFECT.map((e, j) => toDisplay(display, e + shift[j])),
  };
}

/** np.interp: linear between grid points, the end value beyond them. */
function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  const j = xs.findIndex((v) => v > x);
  const t = (x - xs[j - 1]) / (xs[j] - xs[j - 1]);
  return ys[j - 1] + t * (ys[j] - ys[j - 1]);
}

const close = (actual, expected, scale = 1) =>
  Math.abs(actual - expected) <= 64 * EPS * Math.max(scale, Math.abs(expected));

test("a spread drag leaves every row's mean unchanged while the 99th percentile moves", () => {
  const panel = agePanel();
  const eta = previewEta(panel, "scale:age", curve(SPREAD), curve(SPREAD, DRAG));
  const before = panelModel(panel, P99, "edited");
  const during = panelModel(panel, P99, eta);
  during.rows.forEach((row, i) => {
    const mean = before.rows[i].mean;
    assert.ok(Math.abs(row.mean - mean) <= 64 * EPS * mean, `row ${i} mean`);
  });
  during.rows.forEach((row, i) => {
    assert.ok(row.beyond > before.rows[i].beyond, `row ${i} reads a higher 99th percentile`);
  });
  panel.rows.forEach((row, i) => {
    const expected = panel.eta.edited.rows[i][1] + interp(row.x, GRID, DRAG);
    assert.ok(close(eta.rows[i][1], expected), `row ${i} spread eta`);
  });
  // Beyond the grid the change is the end point's.
  assert.ok(close(eta.rows[0][1], panel.eta.edited.rows[0][1] + DRAG[0]), "below the grid");
  assert.ok(close(eta.rows[4][1], panel.eta.edited.rows[4][1] + DRAG[4]), "above the grid");
});

test("a term with one grid point shifts every row and column by its change", () => {
  const panel = agePanel();
  const point = (shift) => ({ x: [40], levels: null, display: SPREAD, y: [toDisplay(SPREAD, 0.1 + shift)] });
  const eta = previewEta(panel, "scale:age", point(0), point(0.4));
  for (const [set, edited] of [[eta.rows, panel.eta.edited.rows], [eta.columns, panel.eta.edited.columns]]) {
    set.forEach((row, i) => assert.ok(close(row[1], edited[i][1] + 0.4), `point ${i}`));
  }
});

test("a mean drag shifts each row's mean by exp of the interpolated change", () => {
  const panel = agePanel();
  const eta = previewEta(panel, "mean:age", curve(undefined), curve(undefined, DRAG));
  const during = panelModel(panel, P99, eta);
  panel.rows.forEach((row, i) => {
    const [mean, spread] = panel.eta.edited.rows[i];
    const expected = Math.exp(mean + interp(row.x, GRID, DRAG));
    assert.ok(Math.abs(during.rows[i].mean - expected) <= 64 * EPS * expected, `row ${i} mean`);
    assert.equal(eta.rows[i][1], spread, "the spread is untouched");
  });
  panel.columns.forEach((column, i) => {
    const expected = panel.eta.edited.columns[i][0] + interp(column.x, GRID, DRAG);
    assert.ok(close(eta.columns[i][0], expected), `column ${i} mean eta`);
  });
});

test("a drag on another feature's term leaves the edited predictors as they are", () => {
  const panel = agePanel();
  const other = { x: [0, 1], levels: ["North", "South"], display: undefined, y: [1, 1] };
  const moved = { ...other, y: [1.5, 0.5] };
  assert.deepEqual(previewEta(panel, "mean:region", other, moved), panel.eta.edited);
  assert.deepEqual(previewEta(panel, "scale:region", other, moved), panel.eta.edited);
});

test("a level term shifts the rows and columns of its own level only", () => {
  const rows = [[7.0, -0.5], [7.2, -0.3], [7.4, -0.1]];
  const panel = {
    ...agePanel(),
    feature: "region",
    kind: "levels",
    rows: ["North", "South", "Central"].map((level) => ({ label: level, x: null, level })),
    columns: ["North", "South", "Central"].map((level) => ({ x: null, level })),
    term_keys: { mean: "mean:region", scale: "scale:region" },
    eta: { fitted: { rows, columns: rows }, edited: { rows, columns: rows } },
  };
  const levels = ["East", "North", "South", "West"];
  const effects = [-0.2, 0, 0.1, 0.3];
  const level = (shift) => ({
    x: [0, 1, 2, 3],
    levels,
    display: SPREAD,
    y: effects.map((e, j) => toDisplay(SPREAD, e + shift[j])),
  });
  const effect = 0.25;
  // Every listed level moves except North; Central, which the term does not list, stays.
  const eta = previewEta(
    panel, "scale:region", level([0, 0, 0, 0]), level([0.3, 0, effect, 0.3])
  );
  for (const set of [eta.rows, eta.columns]) {
    assert.deepEqual(set[0], rows[0]);
    assert.deepEqual(set[2], rows[2], "an unlisted level does not move");
    assert.equal(set[1][0], rows[1][0]);
    assert.ok(close(set[1][1], rows[1][1] + effect), "South's spread moves by the drag");
  }
});
