// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { toDisplay } from "../../src/superglm/editor/app/chart/display.js";
import { fanBeyondRegion, fanPixels } from "../../src/superglm/editor/app/chart/fan.js";
import { ridgeLayout, ridgeShape } from "../../src/superglm/editor/app/chart/ridges.js";
import { lawOf } from "../../src/superglm/editor/app/distribution/families.js";
import { amountInsideAxis, formatAmount, formatChance } from "../../src/superglm/editor/app/distribution/line.js";
import { previewEta } from "../../src/superglm/editor/app/distribution/preview.js";
import {
  DISTRIBUTION_UNDRAWABLE,
  ZERO_NOTE,
  clampedLaw,
  defaultLineState,
  guardedPanelModel,
  lawInputs,
  lineModel,
  panelModel,
} from "../../src/superglm/editor/app/views/distribution_panel.js";

const LOG = { kind: "log" };
const WALLS = { kind: "bounded_logit", lower: 1.05, upper: 1.95 };
const AMOUNT = { mode: "amount", value: 2000, style: "percent" };
const percentile = (value) => ({ mode: "percentile", value, style: "percent" });
const AXIS = { min_log10: -1, max_log10: 5, clipped_below: false };
const PARAMETERS = [
  { name: "mean", label: "Mean", role: "average", link: LOG },
  { name: "dispersion", label: "Dispersion", role: "spread", link: LOG },
  { name: "power", label: "Power", role: "shape", link: WALLS },
];

/** The predictors of a row with this mean, chance of exactly zero and power, from the model's formulas. */
function etaOf(mean, zero, power = 1.5) {
  const rate = -Math.log(zero);
  const dispersion = mean ** (2 - power) / (rate * (2 - power));
  return [Math.log(mean), Math.log(dispersion), Math.log((power - 1.05) / (1.95 - power))];
}
/** The model's chance of zero, exp(-mean^(2-p) / (dispersion (2-p))), at predictors [mean, dispersion, power 1.5]. */
const zeroOf = (eta) => Math.exp(-(Math.exp(eta[0]) ** 0.5) / (Math.exp(eta[1]) * 0.5));
const close = (actual, expected, tolerance = 1e-12) =>
  assert.ok(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} vs ${expected}`);

/**
 * A Tweedie panel over age. The rows' means and chances of zero are 50 and 95%, 200 and 50%, 800 and
 * 2%; the power has no term, so `term_keys` has no entry for it and its predictor is the same on
 * every row. Row 1 is edited.
 */
function tweediePanel(rowSpecs = [[50, 0.95], [200, 0.5], [800, 0.02]], overrides = {}) {
  const fitted = rowSpecs.map(([mean, zero]) => etaOf(mean, zero));
  const edited = fitted.map((eta, i) => (i === 1 ? [eta[0] + 0.3, eta[1], eta[2]] : eta));
  const columns = [20, 30, 40, 50, 60].map((_, i) => etaOf(30 * 3 ** i, 0.97 * 0.35 ** i + 0.01));
  return {
    feature: "age",
    family: "tweedie",
    kind: "numeric",
    empty_reason: null,
    rows: [20, 40, 60].map((x) => ({ label: String(x), x, level: null })),
    columns: [20, 30, 40, 50, 60].map((x) => ({ x, level: null })),
    parameters: PARAMETERS,
    term_keys: { mean: "mean:age", dispersion: "dispersion:age" },
    eta: { fitted: { rows: fitted, columns }, edited: { rows: edited, columns } },
    amount_axis: AXIS,
    amount_unit: "€",
    reference: [],
    cap_note: null,
    ...overrides,
  };
}

test("three predictors are read by role, in whatever order the payload lists them", () => {
  const panel = tweediePanel();
  const model = panelModel(panel, AMOUNT, "edited");
  const expectedZero = [0.95, 0.5, 0.02];
  model.rows.forEach((row, i) => {
    close(row.zero, i === 1 ? zeroOf(panel.eta.edited.rows[1]) : expectedZero[i], 1e-9);
    close(row.law.mean, Math.exp(panel.eta.edited.rows[i][0]), 1e-14);
    assert.equal(row.beyond, lawOf("tweedie", panel.eta.edited.rows[i], [LOG, LOG, WALLS]).sf(AMOUNT.value));
  });
  // The power first, then the mean, then the dispersion: the same rows.
  const order = [2, 0, 1];
  const shuffle = (rows) => rows.map((eta) => order.map((at) => eta[at]));
  const shuffled = tweediePanel(undefined, {
    parameters: order.map((at) => PARAMETERS[at]),
    eta: {
      fitted: { rows: shuffle(panel.eta.fitted.rows), columns: shuffle(panel.eta.fitted.columns) },
      edited: { rows: shuffle(panel.eta.edited.rows), columns: shuffle(panel.eta.edited.columns) },
    },
  });
  const again = panelModel(shuffled, AMOUNT, "edited");
  assert.deepEqual(again.rows.map((row) => [row.zero, row.beyond, row.mean]), model.rows.map((row) => [row.zero, row.beyond, row.mean]));
});

test("the law helpers take a payload-like object for every continuous family", () => {
  const axis = AXIS;
  const tweedie = { family: "tweedie", parameters: PARAMETERS, amount_unit: "€", amount_axis: axis };
  const inputs = lawInputs(tweedie);
  assert.deepEqual(inputs.order, [0, 1, 2]);
  assert.deepEqual(inputs.links, [LOG, LOG, WALLS]);
  const { law, spreadPinned } = clampedLaw(tweedie, inputs, etaOf(50, 0.95));
  assert.equal(spreadPinned, false);
  close(law.zero, 0.95, 1e-9);
  assert.equal(lineModel(tweedie, AMOUNT).log10, Math.log10(AMOUNT.value));

  const pair = (family, spread) => ({
    family,
    parameters: [PARAMETERS[0], { ...PARAMETERS[1], link: spread }],
    amount_unit: "€",
    amount_axis: axis,
  });
  const gamma = pair("gamma", LOG);
  const gammaInputs = lawInputs(gamma);
  assert.deepEqual(gammaInputs.order, [0, 1]);
  assert.equal(clampedLaw(gamma, gammaInputs, [Math.log(100), Math.log(5)]).spreadPinned, false);
  const capped = clampedLaw(gamma, gammaInputs, [Math.log(100), Math.log(30)]);
  assert.equal(capped.spreadPinned, true);
  assert.equal(capped.law.zero, 0);
  const lognormal = pair("lognormal_mean", { kind: "floored_log", floor: 0.05 });
  const lognormalInputs = lawInputs(lognormal);
  assert.equal(clampedLaw(lognormal, lognormalInputs, [7, -0.5]).spreadPinned, false);
});

test("a percentile inside the atom reads 0 with one note; the others read an amount and no note", () => {
  const panel = tweediePanel();
  const inside = panelModel(panel, percentile(0.9), "edited");
  // Row 0 is 95% zero, so its 90th percentile is zero; rows 1 and 2 have 50% (a bit less, edited) and 2%.
  assert.deepEqual(inside.rows.map((row) => row.readsZero), [true, false, false]);
  const [first, second] = inside.rows;
  assert.equal(first.beyond, 0);
  assert.equal(first.readout, formatAmount(0, "€"));
  assert.equal(first.cut, -Infinity);
  assert.ok(second.beyond > 0 && Number.isFinite(second.cut));
  assert.equal(inside.zeroNote, ZERO_NOTE);
  assert.match(first.title, new RegExp(`chance of exactly zero ${formatChance(first.zero, "percent")}`));
  // The fan's columns carry the same rule: the first is nearly all zeros, the last has almost none.
  assert.equal(inside.fan.columns[0].cut, -Infinity);
  assert.ok(Number.isFinite(inside.fan.columns[4].cut));
  // Above every chance of zero, and in amount mode, nothing reads zero.
  for (const state of [percentile(0.99), AMOUNT]) {
    const outside = panelModel(panel, state, "edited");
    assert.ok(outside.rows.every((row) => !row.readsZero), JSON.stringify(state));
    assert.equal(outside.zeroNote, null);
  }
  // The ridge and the fan draw the amounts above zero, so their median and mean marks are those of
  // the amounts above zero: a law 95% zero has a median of 0 and a mean far below its ridge.
  assert.equal(inside.aboveZero, true);
  assert.equal(inside.rows[0].law.median, 0);
  for (const marked of [...inside.rows, ...inside.fan.columns]) {
    const { law } = marked;
    close(law.cdf(marked.median), law.zero + 0.5 * (1 - law.zero), 1e-9);
    close(marked.mean, law.mean / (1 - law.zero), 1e-14);
  }
  assert.ok(inside.rows[0].mean > 10 * inside.rows[0].law.mean);
});

test("every row of a Tweedie panel reports its chance of zero, in amount mode too", () => {
  const model = panelModel(tweediePanel(), AMOUNT, "edited");
  const chances = model.rows.map((row) => row.zero);
  assert.ok(chances[0] > chances[1] && chances[1] > chances[2] && chances[2] > 0.01);
  for (const row of model.rows) {
    assert.match(row.title, /chance of exactly zero \d/);
    assert.equal(row.readsZero, false);
  }
  // The continuous families have no atom, so their rows carry none.
  const gamma = tweediePanel(undefined, {
    family: "gamma",
    parameters: PARAMETERS.slice(0, 2),
    eta: { fitted: { rows: [[6, -0.5]], columns: [[6, -0.5]] }, edited: { rows: [[6, -0.5]], columns: [[6, -0.5]] } },
    rows: [{ label: "20", x: 20, level: null }],
    columns: [{ x: 20, level: null }],
  });
  const row = panelModel(gamma, AMOUNT, "edited").rows[0];
  assert.equal(row.zero, 0);
  assert.doesNotMatch(row.title, /exactly zero/);
});

test("the ridges and the fan draw a Tweedie panel with no value outside their frames", () => {
  const panel = tweediePanel();
  const state = percentile(0.9);
  const model = panelModel(panel, state, "edited");
  const layout = ridgeLayout(600, model.rows.length);
  model.rows.forEach((row, i) => {
    const { points } = ridgeShape(row.law, model.axis, layout, i);
    const base = layout.baseOf(i);
    for (const [l, y] of points) assert.ok(Number.isFinite(l) && y <= base && y >= base - layout.height - 1e-9, `row ${i}`);
  });
  const ramp = [[255, 255, 255], [180, 200, 255], [90, 130, 230], [20, 60, 200]];
  const pixels = fanPixels(model.fan, 40, 60, model.axis, ramp);
  assert.equal(pixels.length, 40 * 60 * 4);
  assert.ok(pixels.every((byte, k) => (k % 4 === 3 ? byte === 255 : byte >= 0)));
  const region = fanBeyondRegion(model.fan, 40, 60, model.axis);
  for (const [x, y] of region) assert.ok(x >= 0 && x <= 40 && y >= 0 && y <= 60, `${x}, ${y}`);
  // A column inside the atom is orange down to the foot of the axis.
  assert.ok(region.some(([, y]) => y === 60));
});

test("the default line is a decade of the amounts when the middle row is mostly zero", () => {
  // The middle row is 99% zero, so its 95th percentile is 0; the default reads its amounts instead.
  const panel = tweediePanel([[50, 0.95], [100, 0.99], [800, 0.02]]);
  const middle = lawOf("tweedie", panel.eta.edited.rows[1], [LOG, LOG, WALLS]);
  assert.equal(middle.quantile(0.95), 0);
  const { value } = defaultLineState(panel);
  const expected = middle.quantile(middle.zero + 0.95 * (1 - middle.zero));
  const decade = 10 ** Math.round(Math.log10(expected));
  // The decade is the axis's top edge, where no tail past the line would be drawn and the slider
  // could not move right: the line starts at the round amount just inside it.
  assert.equal(decade, 10 ** AXIS.max_log10);
  assert.equal(value, 5 * 10 ** (AXIS.max_log10 - 1));
  // With the middle row's 95th percentile above zero it is that decade, as for the other families.
  const usual = tweediePanel();
  const row = lawOf("tweedie", usual.eta.edited.rows[1], [LOG, LOG, WALLS]);
  const usualDecade = 10 ** Math.round(Math.log10(row.quantile(0.95)));
  assert.equal(defaultLineState(usual).value, amountInsideAxis(usualDecade, AXIS));
  assert.ok(Math.log10(defaultLineState(usual).value) < AXIS.max_log10);
});

test("a row whose predictors leave the law's range makes the guarded model a message", () => {
  const wild = [[Math.log(100), Math.log(1e-6), 0], ...tweediePanel().eta.edited.rows.slice(1)];
  const panel = tweediePanel(undefined, {
    eta: { fitted: tweediePanel().eta.fitted, edited: { ...tweediePanel().eta.edited, rows: wild } },
  });
  const result = guardedPanelModel(panel, AMOUNT, "edited");
  assert.equal(result.ok, false);
  assert.equal(result.message, DISTRIBUTION_UNDRAWABLE);
  assert.equal(guardedPanelModel(tweediePanel(), AMOUNT, "edited").ok, true);
});

const GRID = [20, 40, 60];
const BASE_EFFECT = [-0.1, 0, 0.1];
/** A numeric term on GRID whose effect is BASE_EFFECT plus `shift`, shown as a relativity. */
const curve = (shift = [0, 0, 0]) => ({
  x: GRID,
  levels: null,
  display: undefined,
  y: BASE_EFFECT.map((e, j) => toDisplay(undefined, e + shift[j])),
});

test("a drag preview moves the mean or the dispersion predictor and never the power", () => {
  const panel = tweediePanel();
  const edited = panel.eta.edited;
  const drag = [0.2, 0.4, 0.1];
  const mean = previewEta(panel, "mean:age", curve(), curve(drag));
  const dispersion = previewEta(panel, "dispersion:age", curve(), curve(drag));
  // The rows sit on the grid, so each moves by its own grid point's change.
  panel.rows.forEach((_, i) => {
    close(mean.rows[i][0], edited.rows[i][0] + drag[i], 1e-14);
    assert.equal(mean.rows[i][1], edited.rows[i][1]);
    assert.equal(mean.rows[i][2], edited.rows[i][2]);
    assert.equal(dispersion.rows[i][0], edited.rows[i][0]);
    close(dispersion.rows[i][1], edited.rows[i][1] + drag[i], 1e-14);
    assert.equal(dispersion.rows[i][2], edited.rows[i][2], "the power has no term");
  });
  // The power has no term to drag.
  assert.deepEqual(previewEta(panel, "power:age", curve(), curve(drag)), edited);
  // The preview draws: a higher mean raises the row's mean and lowers its chance of zero.
  const during = panelModel(panel, AMOUNT, mean);
  const before = panelModel(panel, AMOUNT, "edited");
  during.rows.forEach((row, i) => {
    close(row.law.mean, before.rows[i].law.mean * Math.exp(drag[i]), 1e-13);
    close(row.zero, zeroOf(mean.rows[i]), 1e-9);
    assert.ok(row.zero < before.rows[i].zero);
    assert.notEqual(row.ghost, undefined);
  });
  // A larger dispersion lowers the rate, so the chance of zero rises while the mean stays put.
  const spread = panelModel(panel, AMOUNT, dispersion);
  spread.rows.forEach((row, i) => {
    close(row.law.mean, before.rows[i].law.mean, 1e-13);
    close(row.zero, zeroOf(dispersion.rows[i]), 1e-9);
    assert.ok(row.zero > before.rows[i].zero);
  });
});
