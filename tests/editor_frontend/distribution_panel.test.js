// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  GAMMA_CV_MAX,
  PROBABILITY_MIN,
  lawOf,
} from "../../src/superglm/editor/app/distribution/families.js";
import { formatAmount, formatChance } from "../../src/superglm/editor/app/distribution/line.js";
import { createEditorClient } from "../../src/superglm/editor/app/api/client.js";
import {
  DISTRIBUTION_FAILED,
  DISTRIBUTION_UNDRAWABLE,
  defaultLineState,
  distributionFailure,
  guardedPanelModel,
  panelModel,
} from "../../src/superglm/editor/app/views/distribution_panel.js";
import {
  clampToPlot,
  fanBeyondRegion,
  fanPixels,
  rampColour,
  spreadLabels,
} from "../../src/superglm/editor/app/chart/fan.js";
import {
  decadeTicks,
  dotPosition,
  ridgeLayout,
  ridgeShape,
} from "../../src/superglm/editor/app/chart/ridges.js";

const LOG = { kind: "log" };
const FLOORED = { kind: "floored_log", floor: 0.05 };

/** A log-normal panel: row 1 is edited, rows 0 and 2 are as fitted. */
function lognormalPanel(overrides = {}) {
  const fitted = [[7.0, -0.2], [7.2, 0.0], [7.5, 0.1]];
  const edited = [[7.0, -0.2], [7.2, 0.3], [7.5, 0.1]];
  return {
    feature: "age",
    family: "lognormal_mean",
    kind: "numeric",
    empty_reason: null,
    rows: [
      { label: "18", x: 18, level: null },
      { label: "1.25e+04", x: 12500, level: null },
      { label: "60", x: 60, level: null },
    ],
    columns: [{ x: 18, level: null }, { x: 39, level: null }, { x: 60, level: null }],
    parameters: [
      { name: "mean", label: "Mean", role: "average", link: LOG },
      { name: "scale", label: "CV", role: "spread", link: FLOORED },
    ],
    term_keys: { mean: "mean:age", scale: "scale:age" },
    eta: { fitted: { rows: fitted, columns: fitted }, edited: { rows: edited, columns: edited } },
    amount_axis: { min_log10: 1, max_log10: 6, clipped_below: false },
    amount_unit: "€",
    reference: [{ column: "region", value: "North" }],
    cap_note: null,
    ...overrides,
  };
}

const LINKS = [LOG, FLOORED];
const lawAt = (eta) => lawOf("lognormal_mean", eta, LINKS);
const AMOUNT = { mode: "amount", value: 25000, style: "percent" };

test("in amount mode beyond is each row's chance above the line", () => {
  const panel = lognormalPanel();
  const model = panelModel(panel, AMOUNT, "edited");
  assert.equal(model.rows.length, 3);
  model.rows.forEach((row, i) => {
    const expected = lawAt(panel.eta.edited.rows[i]).sf(AMOUNT.value);
    assert.equal(row.beyond, expected);
    assert.equal(row.readout, formatChance(expected, "percent"));
  });
  assert.equal(model.line.pinned, false);
});

test("in percentile mode the readout is the amount at the percentile", () => {
  const panel = lognormalPanel();
  const state = { mode: "percentile", value: 0.995, style: "percent" };
  const model = panelModel(panel, state, "edited");
  model.rows.forEach((row, i) => {
    const amount = lawAt(panel.eta.edited.rows[i]).quantile(0.995);
    assert.equal(row.beyond, amount);
    assert.equal(row.readout, formatAmount(amount, "€"));
  });
});

test("the parameters are read by role, whatever their payload order", () => {
  const panel = lognormalPanel();
  const swap = (rows) => rows.map(([mean, spread]) => [spread, mean]);
  const swapped = lognormalPanel({
    parameters: [...panel.parameters].reverse(),
    eta: {
      fitted: { rows: swap(panel.eta.fitted.rows), columns: swap(panel.eta.fitted.columns) },
      edited: { rows: swap(panel.eta.edited.rows), columns: swap(panel.eta.edited.columns) },
    },
  });
  const a = panelModel(panel, AMOUNT, "edited").rows.map((row) => row.beyond);
  const b = panelModel(swapped, AMOUNT, "edited").rows.map((row) => row.beyond);
  assert.deepEqual(b, a);
});

test("ghost is the fitted readout, present only where the fitted eta differs", () => {
  const panel = lognormalPanel();
  const edited = panelModel(panel, AMOUNT, "edited");
  assert.equal(edited.rows[0].ghost, undefined);
  assert.equal(edited.rows[2].ghost, undefined);
  const fittedChance = lawAt(panel.eta.fitted.rows[1]).sf(AMOUNT.value);
  assert.equal(edited.rows[1].ghost, formatChance(fittedChance, "percent"));
  assert.equal(edited.rows[1].ghostBeyond, fittedChance);

  const fitted = panelModel(panel, AMOUNT, "fitted");
  assert.ok(fitted.rows.every((row) => row.ghost === undefined));

  // Task 17's preview: explicit rows and columns.
  const preview = {
    rows: [[7.0, -0.2], [7.2, 0.0], [7.6, 0.1]],
    columns: panel.eta.edited.columns,
  };
  const previewed = panelModel(panel, AMOUNT, preview);
  assert.deepEqual(previewed.rows.map((row) => row.ghost !== undefined), [false, false, true]);
  assert.equal(previewed.rows[2].beyond, lawAt(preview.rows[2]).sf(AMOUNT.value));
});

test("a ghost appears for any eta change, even one that prints the same readout", () => {
  const panel = lognormalPanel();
  // A change far below the readout's printed digits is still a change.
  const edited = panel.eta.fitted.rows.map(([mean, spread], i) => [i === 0 ? mean + 1e-12 : mean, spread]);
  const nudged = lognormalPanel({ eta: { ...panel.eta, edited: { rows: edited, columns: edited } } });
  const model = panelModel(nudged, AMOUNT, "edited");
  assert.equal(model.rows[0].readout, model.rows[0].ghost);
  assert.equal(model.rows[1].ghost, undefined);
});

test("an amount outside the axis keeps its readouts and pins the line", () => {
  const panel = lognormalPanel();
  for (const value of [5e7, 0.5]) {
    const model = panelModel(panel, { mode: "amount", value, style: "one_in" }, "edited");
    assert.equal(model.line.pinned, true);
    assert.equal(model.line.value, value);
    assert.equal(model.line.log10, value > 1 ? 6 : 1);
    model.rows.forEach((row, i) => {
      const expected = lawAt(panel.eta.edited.rows[i]).sf(value);
      assert.equal(row.beyond, expected);
      assert.equal(row.readout, formatChance(expected, "one_in"));
    });
    assert.match(model.line.hint, /axis/);
    // The text box keeps the typed amount, not the axis edge.
    // The box holds the number alone; the unit is in the line's label.
    assert.equal(model.line.text, formatAmount(value, null));
    assert.equal(model.line.label, formatAmount(value, "€"));
  }
});

test("a probability outside the computable range is clamped and shown pinned", () => {
  const panel = lognormalPanel();
  for (const [value, clamped] of [[1 - 1e-15, 1 - PROBABILITY_MIN], [1e-15, PROBABILITY_MIN]]) {
    // Unclamped, the law refuses it.
    assert.throws(() => lawAt(panel.eta.edited.rows[0]).quantile(value), RangeError);
    const model = panelModel(panel, { mode: "percentile", value, style: "percent" }, "edited");
    assert.equal(model.line.pinned, true);
    assert.equal(model.line.p, clamped);
    assert.equal(model.line.value, value);
    assert.ok(model.line.hint);
    model.rows.forEach((row, i) => {
      assert.equal(row.beyond, lawAt(panel.eta.edited.rows[i]).quantile(clamped));
    });
  }
  const inside = panelModel(panel, { mode: "percentile", value: 0.99, style: "percent" }, "edited");
  assert.equal(inside.line.pinned, false);
  assert.equal(inside.line.hint, null);
});

test("a gamma CV above GAMMA_CV_MAX is drawn at GAMMA_CV_MAX and shown pinned", () => {
  const tooWide = Math.log(GAMMA_CV_MAX) + 1;
  const rows = [[6.0, Math.log(0.8)], [6.0, tooWide]];
  const panel = lognormalPanel({
    family: "gamma",
    rows: [{ label: "a", x: null, level: "a" }, { label: "b", x: null, level: "b" }],
    columns: [{ x: null, level: "a" }, { x: null, level: "b" }],
    parameters: [
      { name: "mean", label: "Mean", role: "average", link: LOG },
      { name: "scale", label: "CV", role: "spread", link: LOG },
    ],
    kind: "levels",
    eta: { fitted: { rows, columns: rows }, edited: { rows, columns: rows } },
  });
  // Unclamped, the law refuses it.
  assert.throws(() => lawOf("gamma", rows[1], [LOG, LOG]), RangeError);
  const model = panelModel(panel, AMOUNT, "edited");
  assert.equal(model.rows[0].spreadPinned, false);
  assert.equal(model.rows[1].spreadPinned, true);
  const capped = lawOf("gamma", [6.0, Math.log(GAMMA_CV_MAX)], [LOG, LOG]);
  assert.equal(model.rows[1].beyond, capped.sf(AMOUNT.value));
  assert.ok(
    model.rows[1].title.endsWith(
      `The CV is drawn at ${GAMMA_CV_MAX}, the largest the panel computes.`
    )
  );
  assert.ok(!model.rows[0].title.includes("The CV is drawn"));
  assert.equal(
    model.spreadNote,
    `Where the CV exceeds ${GAMMA_CV_MAX}, the panel draws it at ${GAMMA_CV_MAX}, the largest it computes: age b.`
  );
  assert.ok(model.fan.columns[1].spreadPinned);
  // Level rows are labelled by their level.
  assert.deepEqual(model.rows.map((row) => row.label), ["a", "b"]);
});

test("numeric rows are labelled from x with the editor's number format", () => {
  const model = panelModel(lognormalPanel(), AMOUNT, "edited");
  assert.deepEqual(model.rows.map((row) => row.label), ["18", "12500", "60"]);
});

test("a count payload renders only its empty reason", () => {
  const panel = lognormalPanel({
    family: "negative_binomial",
    kind: "empty",
    empty_reason: "Count distributions are not drawn in this panel.",
    amount_axis: null,
  });
  const model = panelModel(panel, AMOUNT, "edited");
  assert.equal(model.empty, panel.empty_reason);
  assert.deepEqual(model.rows, []);
  assert.deepEqual(model.fan.columns, []);

  // The kind alone decides: an empty payload that still carries an axis draws nothing either.
  const withAxis = panelModel({ ...panel, amount_axis: lognormalPanel().amount_axis }, AMOUNT, "edited");
  assert.equal(withAxis.empty, panel.empty_reason);
  assert.deepEqual(withAxis.rows, []);
  assert.deepEqual(withAxis.fan.columns, []);
});

test("predictors a law refuses give the panel's message instead of a throw", () => {
  const panel = lognormalPanel();
  // exp(1000) overflows float64, so lawOf refuses the mean.
  const huge = panel.eta.edited.rows.map(([, spread]) => [1000, spread]);
  assert.throws(() => panelModel(panel, AMOUNT, { rows: huge, columns: huge }), RangeError);
  assert.deepEqual(guardedPanelModel(panel, AMOUNT, { rows: huge, columns: huge }), {
    ok: false,
    message: DISTRIBUTION_UNDRAWABLE,
  });
  const fine = guardedPanelModel(panel, AMOUNT, "edited");
  assert.equal(fine.ok, true);
  assert.equal(fine.model.rows.length, 3);
});

test("a server refusal keeps its own message; a failure to answer offers a retry", () => {
  const refusal = Object.assign(new Error("Refit it with a log link to see its distribution here."), {
    status: 400,
  });
  assert.deepEqual(distributionFailure(refusal), {
    message: "Refit it with a log link to see its distribution here.",
    retry: false,
  });
  assert.deepEqual(distributionFailure(Object.assign(new Error("Internal"), { status: 500 })), {
    message: DISTRIBUTION_FAILED,
    retry: true,
  });
  assert.deepEqual(distributionFailure(new TypeError("Failed to fetch")), {
    message: DISTRIBUTION_FAILED,
    retry: true,
  });
});

test("a timeout or too many requests is offered a retry, whatever the reply's text", () => {
  for (const status of [408, 429]) {
    assert.deepEqual(distributionFailure(Object.assign(new Error(`HTTP ${status}`), { status })), {
      message: DISTRIBUTION_FAILED,
      retry: true,
    });
  }
});

test("a clipped axis carries the lower-tail hover note", () => {
  const clipped = lognormalPanel({ amount_axis: { min_log10: 1, max_log10: 6, clipped_below: true } });
  assert.equal(panelModel(clipped, AMOUNT, "edited").axisNote, "Part of the lower tail lies below the axis.");
  assert.equal(panelModel(lognormalPanel(), AMOUNT, "edited").axisNote, null);
});

test("the fan carries one column per payload column, each cut at the line", () => {
  const panel = lognormalPanel();
  const model = panelModel(panel, AMOUNT, "edited");
  assert.equal(model.fan.columns.length, 3);
  // A law with no chance of zero is drawn whole, and its marks are its own median and mean.
  assert.equal(model.aboveZero, false);
  model.fan.columns.forEach((column, i) => {
    const law = lawAt(panel.eta.edited.columns[i]);
    assert.equal(column.median, law.median);
    assert.equal(column.mean, law.mean);
    assert.equal(column.cut, Math.log10(AMOUNT.value));
  });
  const pct = panelModel(panel, { mode: "percentile", value: 0.9, style: "percent" }, "edited");
  assert.equal(pct.fan.columns[2].cut, Math.log10(lawAt(panel.eta.edited.columns[2]).quantile(0.9)));
});

test("fan lines are clamped to the plot", () => {
  const axis = { min_log10: 1, max_log10: 6 };
  assert.equal(clampToPlot(-Infinity, axis), 1);
  assert.equal(clampToPlot(9, axis), 6);
  assert.equal(clampToPlot(3.5, axis), 3.5);
  assert.equal(clampToPlot(Number.NaN, axis), 1);
});

test("fan pixels shade by density in the ramp they are given", () => {
  const panel = lognormalPanel();
  const model = panelModel(panel, AMOUNT, "edited");
  const width = 30;
  const height = 50;
  const pixels = fanPixels(model.fan, width, height, panel.amount_axis, [[255, 255, 255], [0, 0, 255]]);
  assert.equal(pixels.length, width * height * 4);
  const column = [];
  for (let y = 0; y < height; y += 1) column.push(Array.from(pixels.slice(4 * (y * width + 15), 4 * (y * width + 15) + 4)));
  // Along the white-to-blue ramp, reaching its dark end at the law's peak.
  assert.ok(column.every(([r, g, b, a]) => r === g && b === 255 && a === 255));
  assert.ok(Math.min(...column.map(([r]) => r)) <= 5);
});

test("the region past the line is the plot above the cut, whichever way the fan runs", () => {
  const axis = { min_log10: 1, max_log10: 6 };
  const numeric = { kind: "numeric", columns: [{ x: 0, cut: 4 }, { x: 1, cut: 4 }, { x: 2, cut: 2 }] };
  assert.deepEqual(fanBeyondRegion(numeric, 100, 50, axis), [
    [0, 0], [100, 0], [100, 40], [100, 40], [50, 20], [0, 20], [0, 20],
  ]);
  const levels = { kind: "levels", columns: [{ level: "a", cut: 6 }, { level: "b", cut: 1 }] };
  assert.deepEqual(fanBeyondRegion(levels, 100, 50, axis), [
    [0, 0], [100, 0], [100, 50], [50, 50], [50, 0], [0, 0],
  ]);
  // A cut off the axis stays on the plot's edge, and one that is not a number passes nothing.
  const off = { kind: "levels", columns: [{ level: "a", cut: 9 }, { level: "b", cut: Number.NaN }] };
  assert.deepEqual(fanBeyondRegion(off, 100, 50, axis), [
    [0, 0], [100, 0], [100, 50], [50, 50], [50, 0], [0, 0],
  ]);
});

test("ramp colours interpolate between their stops", () => {
  const stops = [[0, 0, 0], [100, 200, 50]];
  assert.deepEqual(rampColour(0, stops), [0, 0, 0]);
  assert.deepEqual(rampColour(1, stops), [100, 200, 50]);
  assert.deepEqual(rampColour(0.5, stops), [50, 100, 25]);
  assert.deepEqual(rampColour(7, stops), [100, 200, 50]);
});

test("fan labels are pushed apart and kept inside the plot", () => {
  const ys = spreadLabels([100, 101, 103, 300], 12, 20, 260);
  const sorted = [...ys].sort((a, b) => a - b);
  for (let i = 1; i < sorted.length; i += 1) assert.ok(sorted[i] - sorted[i - 1] >= 12 - 1e-9);
  assert.ok(ys.every((y) => y >= 20 && y <= 260));
});

test("a ridge stays inside its row, peaking at its height", () => {
  const layout = ridgeLayout(680, 3);
  const law = lawAt([7.2, 0.0]);
  const axis = { min_log10: 1, max_log10: 6 };
  const shape = ridgeShape(law, axis, layout, 1);
  const base = layout.baseOf(1);
  assert.equal(shape.points[0][0], 1);
  assert.equal(shape.points.at(-1)[0], 6);
  const ys = shape.points.map(([, y]) => y);
  assert.ok(ys.every((y) => y <= base && y >= base - layout.height - 1e-9));
  assert.ok(Math.min(...ys) <= base - layout.height * 0.99);
});

test("the dot column clamps to its fixed log scale", () => {
  assert.equal(dotPosition(1e-9, [-4, 0], 548, 612), 548);
  assert.equal(dotPosition(1, [-4, 0], 548, 612), 612);
  assert.equal(dotPosition(1e-2, [-4, 0], 548, 612), 580);
  assert.equal(dotPosition(0, [-4, 0], 548, 612), 548);
});

test("decade ticks thin out so their labels do not collide", () => {
  const all = decadeTicks({ min_log10: 1, max_log10: 6 }, 500, 40);
  assert.deepEqual(all, [1, 2, 3, 4, 5, 6]);
  const thin = decadeTicks({ min_log10: -4, max_log10: 6 }, 200, 40);
  for (let i = 1; i < thin.length; i += 1) assert.ok((thin[i] - thin[i - 1]) * 20 >= 40);
  assert.ok(thin.includes(6));
});

test("the client posts the feature to /distribution", async () => {
  const requests = [];
  const client = createEditorClient({
    fetchImpl: async (url, options) => {
      requests.push({ url: String(url), body: JSON.parse(String(options?.body)) });
      return new Response(JSON.stringify({ status: "superseded", model_revision: 2 }), { status: 200 });
    },
  });
  assert.deepEqual(await client.distribution("scale:age"), { status: "superseded", model_revision: 2 });
  assert.deepEqual(requests, [{ url: "/distribution", body: { feature: "scale:age" } }]);
});

test("the line starts at the decade nearest the middle row's 95th percentile", () => {
  const panel = lognormalPanel();
  const q = lawAt(panel.eta.edited.rows[1]).quantile(0.95);
  assert.deepEqual(defaultLineState(panel), {
    mode: "amount",
    value: 10 ** Math.round(Math.log10(q)),
    style: "percent",
  });
});
