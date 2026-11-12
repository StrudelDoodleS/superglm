import assert from "node:assert/strict";
import test from "node:test";

import {
  RISK_REQUEST,
  createRiskProfiles,
  createRiskRequest,
  moveRiskRequest,
  riskRequestKey,
  withRiskAdded,
  withRiskRemoved,
  withRiskSelected,
  withRiskValue,
  withRiskValues
} from "../../src/superglm/editor/app/state/risk.js";
import { createRiskController, riskFailure } from "../../src/superglm/editor/app/views/risk_controller.js";
import {
  RISK_FAILED,
  RISK_SUPERSEDED,
  amountTicks,
  defaultRiskLine,
  densityTicks,
  guardedRiskModel,
  lineAt,
  outsideRange,
  plotAxis,
  riskLayout,
  riskModel,
  riskScales,
  sampleLaw
} from "../../src/superglm/editor/app/views/risk_panel.js";
import { driversModel, parameterValue } from "../../src/superglm/editor/app/views/risk_drivers.js";
import { countReach } from "../../src/superglm/editor/app/views/risk_counts.js";
/** @typedef {import('../../src/superglm/editor/app/api/contracts.js').RiskProfiles} RiskProfiles */
import { lawOf } from "../../src/superglm/editor/app/distribution/families.js";
import { nbPmf, nbSf } from "../../src/superglm/editor/app/distribution/math.js";

/** @typedef {import('../../src/superglm/editor/app/api/contracts.js').RiskResponse} RiskResponse */
/** @typedef {import('../../src/superglm/editor/app/state/risk.js').RiskEvent} RiskEvent */
/** @typedef {import('../../src/superglm/editor/app/views/distribution_panel.js').LineState} LineState */

const LOG = /** @type {const} */ ({ kind: "log" });
/** @type {LineState} */
const LINE = { mode: "amount", value: 1000, style: "percent" };

/**
 * A gamma reply: risk A at the reference row, and risk B when `second` gives its predictors.
 * @param {{fitted?: number[], edited?: number[], second?: number[]}} [eta]
 * @returns {RiskResponse}
 */
function reply({ fitted = [7, -0.5], edited = fitted, second } = {}) {
  const still = [[0, 0], [0, 0]];
  const risks = [{ values: { age: 40, region: "R0" }, eta: { fitted, edited }, effects: { fitted: still, edited: still } }];
  if (second) {
    const moved = [second.map((v, p) => v - fitted[p]), [0, 0]];
    risks.push({ values: { age: 22, region: "R4" }, eta: { fitted: second, edited: second }, effects: { fitted: moved, edited: moved } });
  }
  return {
    family: "gamma",
    parameters: [
      { name: "mean", label: "Mean", role: "average", link: LOG },
      { name: "scale", label: "CV", role: "spread", link: LOG }
    ],
    amount_unit: "€",
    kind: "risk",
    features: [
      { column: "age", kind: "numeric", min: 18, max: 90, reference: 40, levels: null, terms: { mean: "mean:age", scale: "scale:age" } },
      { column: "region", kind: "levels", min: null, max: null, reference: "R0", levels: ["R0", "R4"], terms: { mean: "mean:region" } }
    ],
    reference: { fitted, edited },
    risks,
    amount_axis: { min_log10: 0, max_log10: 5, clipped_below: false }
  };
}

/** @param {number[]} eta the law the browser builds for these gamma predictors */
const gamma = (eta) => lawOf("gamma", eta, [LOG, LOG]);

// ---------------------------------------------------------------------------------------------
// The request's lifecycle

test("the request's table names every state and event, and every move lands where it says", () => {
  /** @type {Record<string, RiskEvent>} */
  const events = {
    ask: { type: "ask", key: "k2" },
    answer: { type: "answer", key: "k1", reply: reply() },
    fail: { type: "fail", key: "k1", message: "no", retry: true },
    close: { type: "close" }
  };
  assert.deepEqual([...RISK_REQUEST.events].sort(), Object.keys(events).sort());
  for (const status of RISK_REQUEST.states) {
    for (const type of RISK_REQUEST.events) {
      const before = { ...createRiskRequest(), status, key: "k1" };
      const after = moveRiskRequest(before, events[type]);
      const target = RISK_REQUEST.next(status, type);
      if (target === null) assert.strictEqual(after, before, `${status} takes no ${type}`);
      else assert.equal(after.status, target, `${status} --${type}--> ${target}`);
    }
  }
});

test("an answer for a request the view has moved on from changes nothing", () => {
  const first = moveRiskRequest(createRiskRequest(), { type: "ask", key: "k1" });
  const second = moveRiskRequest(first, { type: "ask", key: "k2" });

  assert.strictEqual(moveRiskRequest(second, { type: "answer", key: "k1", reply: reply() }), second);
  assert.strictEqual(moveRiskRequest(second, { type: "fail", key: "k1", message: "no", retry: true }), second);
  assert.strictEqual(moveRiskRequest(second, { type: "ask", key: "k2" }), second);
  assert.equal(moveRiskRequest(second, { type: "answer", key: "k2", reply: reply() }).status, "shown");
});

test("the last answer stays through a newer request and a failure, and goes when the view closes", () => {
  const drawn = reply();
  let request = moveRiskRequest(createRiskRequest(), { type: "ask", key: "k1" });
  request = moveRiskRequest(request, { type: "answer", key: "k1", reply: drawn });
  request = moveRiskRequest(request, { type: "ask", key: "k2" });
  assert.equal(request.status, "loading");
  assert.strictEqual(request.reply, drawn);

  request = moveRiskRequest(request, { type: "fail", key: "k2", message: "No level R9.", retry: false });
  assert.deepEqual(
    { status: request.status, message: request.message, retry: request.retry },
    { status: "failed", message: "No level R9.", retry: false }
  );
  assert.strictEqual(request.reply, drawn);
  assert.deepEqual(moveRiskRequest(request, { type: "close" }), createRiskRequest());
});

test("risks change one value at a time, and a new risk copies the selected one under the next letter", () => {
  const start = createRiskProfiles();
  const aged = withRiskValue(start, "a", "age", 30);

  assert.deepEqual(aged, { risks: [{ id: "a", values: { age: 30 } }], selected: "a" });
  assert.strictEqual(withRiskValue(aged, "a", "age", 30), aged);
  assert.strictEqual(withRiskValue(aged, "b", "age", 50), aged, "there is no risk B to change");
  const two = withRiskAdded(aged);
  assert.deepEqual(two, { risks: [{ id: "a", values: { age: 30 } }, { id: "b", values: { age: 30 } }], selected: "b" });
  assert.notStrictEqual(two.risks[1].values, two.risks[0].values);
  // Given values, the new risk takes them and not the selected risk's.
  assert.deepEqual(withRiskAdded(aged, { age: 61, region: "R4" }).risks[1], { id: "b", values: { age: 61, region: "R4" } });
  assert.deepEqual(withRiskValues(two, "b", {}).risks[1], { id: "b", values: {} });
  assert.strictEqual(withRiskSelected(two, "b"), two);
  assert.equal(withRiskSelected(two, "a").selected, "a");
  assert.strictEqual(withRiskSelected(two, "c"), two, "there is no risk C to select");
});

test("a removed risk frees its letter, the others keep theirs, and the last one stays", () => {
  let profiles = createRiskProfiles();
  for (let i = 0; i < 3; i += 1) profiles = withRiskAdded(profiles);
  assert.deepEqual(profiles.risks.map((risk) => risk.id), ["a", "b", "c", "d"]);

  const without = withRiskRemoved(withRiskSelected(profiles, "b"), "b");
  assert.deepEqual(without.risks.map((risk) => risk.id), ["a", "c", "d"]);
  assert.equal(without.selected, "a", "the risk before the removed one is selected");
  assert.equal(withRiskRemoved(profiles, "b").selected, "d", "another risk's removal leaves the selection");
  // The next risk takes the freed letter, so C and D keep their colours.
  assert.deepEqual(withRiskAdded(without).risks.map((risk) => risk.id), ["a", "c", "d", "b"]);

  const one = createRiskProfiles();
  assert.strictEqual(withRiskRemoved(one, "a"), one);
  let full = one;
  for (let i = 0; i < 10; i += 1) full = withRiskAdded(full);
  assert.deepEqual(full.risks.map((risk) => risk.id), ["a", "b", "c", "d", "e", "f", "g", "h"]);
  assert.strictEqual(withRiskAdded(full), full, "the tab holds eight");
});

test("a new model, a redrawn chart or a changed risk asks again, and a selection does not", () => {
  const profiles = withRiskAdded(createRiskProfiles());
  const key = riskRequestKey(3, 7, profiles);

  assert.equal(riskRequestKey(3, 7, withRiskSelected(profiles, "a")), key);
  assert.notEqual(riskRequestKey(4, 7, profiles), key);
  assert.notEqual(riskRequestKey(3, 8, profiles), key);
  assert.notEqual(riskRequestKey(3, 7, withRiskValue(profiles, "a", "age", 30)), key);
  assert.notEqual(riskRequestKey(3, 7, withRiskAdded(profiles)), key);
});

// ---------------------------------------------------------------------------------------------
// The controller

/** @param {(risks: unknown[]) => Promise<unknown>} risk */
function controllerWith(risk) {
  let draws = 0;
  const controller = createRiskController({
    client: { risk, riskRow: async () => ({ values: { age: 61, region: "R4" }, index: 5 }) },
    draw: () => { draws += 1; }
  });
  return { controller, draws: () => draws };
}

const WANT = { active: true, generation: 1, revision: 1, profiles: createRiskProfiles() };

test("the controller asks once for what the view wants and draws the answer", async () => {
  /** @type {unknown[][]} */
  const asked = [];
  const answer = reply();
  const { controller, draws } = controllerWith(async (risks) => {
    asked.push(risks);
    return answer;
  });

  controller.sync(WANT);
  controller.sync(WANT);
  assert.equal(controller.view().loading, true);
  await new Promise((resolve) => setImmediate(resolve));

  assert.deepEqual(asked, [[{}]]);
  // The answer takes the letters it was asked under.
  assert.deepEqual(controller.view().reply, { ...answer, risks: [{ ...answer.risks[0], id: "a" }] });
  assert.equal(controller.view().loading, false);
  assert.equal(controller.view().message, null);
  assert.equal(draws(), 2);

  const three = withRiskAdded(withRiskRemoved(withRiskAdded(withRiskAdded(WANT.profiles)), "b"));
  controller.sync({ ...WANT, profiles: three });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(asked[1], [{}, {}, {}]);
  controller.sync({ ...WANT, profiles: withRiskSelected(three, "a") });
  assert.equal(asked.length, 2, "selecting a risk asks nothing");
  controller.sync({ ...WANT, active: false });
  assert.equal(controller.view().reply, null);
  assert.equal(controller.request().status, "idle");
});

test("a refusal shows the server's sentence with no Retry; no answer offers Retry, which asks again", async () => {
  let calls = 0;
  const refusal = Object.assign(new Error("age must be a finite number."), { status: 400 });
  const { controller } = controllerWith(async () => {
    calls += 1;
    if (calls === 1) throw refusal;
    if (calls === 2) throw new TypeError("Failed to fetch");
    if (calls === 3) return { status: "superseded", model_revision: 2 };
    return reply();
  });
  const settle = () => new Promise((resolve) => setImmediate(resolve));

  controller.sync(WANT);
  await settle();
  assert.deepEqual(
    { message: controller.view().message, retry: controller.view().retry },
    { message: "age must be a finite number.", retry: false }
  );
  controller.retry();
  await settle();
  assert.equal(calls, 2, "Retry asks again for the request that failed");
  assert.deepEqual(
    { message: controller.view().message, retry: controller.view().retry },
    { message: RISK_FAILED, retry: true }
  );
  controller.retry();
  await settle();
  assert.equal(controller.view().message, RISK_SUPERSEDED);
  controller.retry();
  await settle();
  assert.equal(controller.view().message, null);
  assert.equal(riskFailure(Object.assign(new Error("slow"), { status: 429 })).retry, true);
});

test("a late answer to an older request is dropped", async () => {
  /** @type {Array<(value: unknown) => void>} */
  const resolvers = [];
  const { controller } = controllerWith(() => new Promise((resolve) => { resolvers.push(resolve); }));
  const older = reply({ fitted: [6, -0.5] });
  const newer = reply({ fitted: [8, -0.5] });

  controller.sync(WANT);
  controller.sync({ ...WANT, revision: 2 });
  resolvers[1](newer);
  resolvers[0](older);
  await new Promise((resolve) => setImmediate(resolve));

  assert.deepEqual(controller.view().reply?.risks[0].eta.fitted, newer.risks[0].eta.fitted);
});

test("a row from the book answers its values, or nothing where the server does not", async () => {
  const { controller } = controllerWith(async () => reply());
  assert.deepEqual(await controller.bookRow(), { age: 61, region: "R4" });
  const silent = createRiskController({
    client: { risk: async () => reply(), riskRow: async () => { throw new Error("down"); } },
    draw: () => {}
  });
  assert.equal(await silent.bookRow(), null);
});

// ---------------------------------------------------------------------------------------------
// What the view shows

/**
 * The table's cells by row label and column label.
 * @param {import('../../src/superglm/editor/app/views/risk_panel.js').RiskModel} model
 * @returns {Record<string, Record<string, string>>}
 */
const tableCells = (model) => Object.fromEntries(model.table.rows.map((row) => [
  row.label,
  Object.fromEntries(model.table.columns.map((column, c) => [column.label, row.cells[c]]))
]));
const NUMBER = new Intl.NumberFormat("en-GB", { maximumSignificantDigits: 3 });

test("one unedited risk is a row of the table, read from its own law, with the unit shown once", () => {
  const model = riskModel(reply(), LINE);
  const law = gamma([7, -0.5]);

  assert.equal(model.table.unit, "€");
  assert.deepEqual(model.table.columns.map((column) => column.label), ["Mean", "Median", "90th", "99th", "99.5th", "Above 1,000", "CV"]);
  assert.deepEqual(model.table.rows.map((row) => [row.swatch, row.label]), [["a", "A"]]);
  assert.equal(model.curves[0].original, null);
  assert.equal(model.selected, "a");
  const cells = tableCells(model).A;
  assert.equal(cells.Mean, NUMBER.format(law.mean));
  assert.equal(cells["99.5th"], NUMBER.format(law.quantile(0.995)));
  assert.equal(cells["Above 1,000"], `${(100 * law.sf(1000)).toFixed(1)}%`);
  assert.equal(cells.CV, Math.exp(-0.5).toFixed(3));
});

test("two risks are followed by the ratio of the second to the first", () => {
  const model = riskModel(reply({ second: [6.4, 0.1] }), LINE);
  const a = gamma([7, -0.5]);
  const b = gamma([6.4, 0.1]);

  assert.deepEqual(model.table.rows.map((row) => [row.swatch, row.label]), [["a", "A"], ["b", "B"], [null, "B ÷ A"]]);
  const ratio = tableCells(model)["B ÷ A"];
  assert.equal(ratio.Mean, `×${(b.mean / a.mean).toPrecision(3)}`);
  assert.equal(ratio["99.5th"], `×${(b.quantile(0.995) / a.quantile(0.995)).toPrecision(3)}`);
  assert.equal(ratio.CV, `×${Math.exp(0.6).toPrecision(3)}`);
  // The lower mean with the higher CV overtakes in the tail: the point of the view.
  assert.ok(b.mean < a.mean && b.quantile(0.9999) > a.quantile(0.9999));
});

test("three risks keep their letters and have no ratio, and the selection falls back to the first", () => {
  const two = reply({ second: [6.4, 0.1] });
  const three = {
    ...two,
    risks: [{ ...two.risks[0], id: /** @type {const} */ ("a") }, { ...two.risks[1], id: /** @type {const} */ ("d") }, { ...two.risks[1], id: /** @type {const} */ ("b") }]
  };
  const model = riskModel(three, LINE, "d");

  assert.deepEqual(model.curves.map((curve) => [curve.id, curve.name]), [["a", "Risk A"], ["d", "Risk D"], ["b", "Risk B"]]);
  assert.deepEqual(model.table.rows.map((row) => row.label), ["A", "D", "B"]);
  assert.equal(model.selected, "d");
  assert.equal(riskModel(three, LINE, "c").selected, "a", "risk C is not in the reply");
});

test("one edited risk is read beside the original model's, and several are not", () => {
  const model = riskModel(reply({ fitted: [7, -0.5], edited: [7, -0.1] }), LINE);

  assert.deepEqual(
    model.table.rows.map((row) => [row.swatch, row.label]),
    [["original", "A, original"], ["a", "A"], [null, "Edited ÷ original"]]
  );
  assert.ok(model.curves[0].original);
  const ratio = tableCells(model)["Edited ÷ original"];
  assert.equal(ratio.CV, `×${Math.exp(0.4).toPrecision(3)}`);
  assert.equal(ratio.Mean, "×1.00");
});

test("a percentile line reads each risk's amount at it, headed 1 in N when written so", () => {
  /** @type {LineState} */
  const line = { mode: "percentile", value: 0.999, style: "one_in" };
  const model = riskModel(reply({ second: [6.4, 0.1] }), line);
  const cells = tableCells(model).A;

  assert.ok(model.table.columns.some((column) => column.label === "1 in 1,000"));
  assert.ok(!model.table.columns.some((column) => column.label.startsWith("At ")));
  assert.notEqual(cells["1 in 1,000"], cells["99th"]);
  const percent = riskModel(reply({ second: [6.4, 0.1] }), { ...line, style: "percent" });
  assert.ok(percent.table.columns.some((column) => column.label === "At 99.9th percentile"));
});

test("a percentile line at a column the table always has adds no second column", () => {
  const fixed = ["Mean", "Median", "90th", "99th", "99.5th", "CV"];
  /** @type {Array<[number, "percent" | "one_in"]>} */
  const lines = [
    [0.5, "percent"], [0.9, "percent"], [0.99, "percent"], [0.995, "percent"], [0.99, "one_in"],
    [199 / 200, "one_in"],
  ];
  for (const [value, style] of lines) {
    const model = riskModel(reply({ second: [6.4, 0.1] }), { mode: "percentile", value, style });
    assert.deepEqual(model.table.columns.map((column) => column.label), fixed, `${value} ${style}`);
  }
  const other = riskModel(reply({ second: [6.4, 0.1] }), { mode: "percentile", value: 0.97, style: "percent" });
  assert.deepEqual(other.table.columns.map((column) => column.label), [
    "Mean", "Median", "90th", "99th", "99.5th", "At 97th percentile", "CV",
  ]);
});

/**
 * A negative binomial reply: risk A with mean exp(-3) and theta 2, and risk B where `second` gives
 * its predictors.
 * @param {{edited?: number[], second?: number[]}} [eta]
 * @returns {RiskResponse}
 */
function countsReply({ edited, second } = {}) {
  const base = reply({ fitted: [-3, LN2], edited: edited ?? [-3, LN2], second });
  return {
    ...base,
    family: "negative_binomial",
    kind: "counts",
    amount_unit: null,
    amount_axis: null,
    parameters: [
      { name: "mean", label: "Mean", role: "average", link: LOG },
      { name: "theta", label: "Overdispersion", role: "spread", link: LOG }
    ]
  };
}
const LN2 = Math.log(2);

test("a model of counts reads each risk's chance of a number of claims or more", () => {
  const model = riskModel(countsReply({ second: [-2, 0] }), { ...LINE, style: "percent" });
  const cells = tableCells(model);
  const [a, b] = [{ mean: Math.exp(-3), theta: 2 }, { mean: Math.exp(-2), theta: 1 }];

  assert.equal(model.empty, null);
  assert.deepEqual(model.curves, []);
  assert.deepEqual(model.counts?.map((curve) => [curve.id, curve.inForce, curve.original]), [["a", a, null], ["b", b, null]]);
  assert.deepEqual(model.table.columns.map((column) => column.label), ["Mean", "No claim", "1 or more", "2 or more", "3 or more", "Overdispersion"]);
  assert.equal(model.table.unit, null);
  assert.equal(cells.A.Mean, Math.exp(-3).toPrecision(3));
  assert.equal(cells.B["No claim"], `${(100 * nbPmf(0, b.mean, b.theta)).toFixed(1)}%`);
  assert.equal(cells["B ÷ A"]["2 or more"], `×${(nbSf(2, b.mean, b.theta) / nbSf(2, a.mean, a.theta)).toPrecision(3)}`);
  // theta 2 against theta 1: the overdispersion 1/theta doubles.
  assert.deepEqual([cells.A.Overdispersion, cells.B.Overdispersion, cells["B ÷ A"].Overdispersion], ["0.500", "1.00", "×2.00"]);
  // The chances of none and of one or more are the whole of it.
  assert.ok(Math.abs(nbPmf(0, a.mean, a.theta) + nbSf(1, a.mean, a.theta) - 1) < 1e-12);
});

test("an edited model of counts is read beside the original, and in 1 in N where asked", () => {
  const model = riskModel(countsReply({ edited: [-3, 0] }), { ...LINE, style: "one_in" });
  assert.deepEqual(model.table.rows.map((row) => row.label), ["A, original", "A", "Edited ÷ original"]);
  assert.deepEqual(model.counts?.[0].original, { mean: Math.exp(-3), theta: 2 });
  assert.match(tableCells(model).A["1 or more"], /^1 in /);
});

test("the count plot reaches the last count some risk gives a chance above its floor, within its limits", () => {
  const rare = { mean: 0.05, theta: 1 };
  assert.equal(countReach([rare], -4), 3);
  const common = { mean: 1.5, theta: 1 };
  const reach = countReach([rare, common], -4);
  assert.ok(reach > 3 && nbPmf(reach, common.mean, common.theta) >= 1e-4);
  assert.ok(reach === 12 || nbPmf(reach + 1, common.mean, common.theta) < 1e-4);
});

test("predictors the browser cannot compute say why instead of drawing", () => {
  const guarded = guardedRiskModel(reply({ fitted: [800, -0.5] }), LINE);
  assert.equal(guarded.ok, false);
});

test("the plots span the risks' first percentile to their 1 in 10,000 amount, on quarter decades", () => {
  const model = riskModel(reply({ second: [6.4, 0.1] }), LINE);
  const axis = /** @type {NonNullable<typeof model.axis>} */ (model.axis);
  const laws = [gamma([7, -0.5]), gamma([6.4, 0.1])];
  const low = Math.min(...laws.map((law) => Math.log10(law.quantile(0.01))));
  const high = Math.max(...laws.map((law) => Math.log10(law.quantile(0.9999))));

  assert.ok(axis.min_log10 <= Math.max(low, 0) && Math.max(low, 0) - axis.min_log10 < 0.25);
  assert.ok(axis.max_log10 >= high && axis.max_log10 - high < 0.25);
  assert.equal((axis.min_log10 * 4) % 1, 0);
  assert.deepEqual(plotAxis(model.curves, { min_log10: 2.5, max_log10: 5, clipped_below: true }).min_log10, 2.5);
});

test("a wide decade is labelled at 1, 2 and 5, and a narrow one at the decades alone", () => {
  const axis = { min_log10: 2, max_log10: 4, clipped_below: false };

  assert.deepEqual(amountTicks(axis, 600, "€").map((tick) => tick.label), ["€100", "€200", "€500", "€1k", "€2k", "€5k", "€10k"]);
  assert.deepEqual(amountTicks(axis, 200, "€").map((tick) => tick.label), ["€100", "€1k", "€10k"]);
});

test("a sampled law's chance falls as the amount rises and its density integrates to one", () => {
  const axis = { min_log10: -2, max_log10: 6, clipped_below: false };
  const { l, density, chance } = sampleLaw(gamma([7, -0.5]), axis, 800);
  const step = l[1] - l[0];

  assert.ok(chance.every((value, i) => i === 0 || value <= chance[i - 1]));
  assert.ok(Math.abs(density.reduce((sum, value) => sum + value * step, 0) - 1) < 1e-3);
});

test("a press on the plots sets the line to the amount, or to the chance on the lower plot", () => {
  const axis = { min_log10: 2, max_log10: 4, clipped_below: false };
  const layout = riskLayout(600);
  const scales = riskScales(axis, layout, 1);

  assert.deepEqual(lineAt(axis, layout, LINE, scales.x(3), layout.density.top + 10), { ...LINE, value: 1000 });
  assert.equal(lineAt(axis, layout, LINE, layout.left - 5, layout.density.top + 10), null);
  /** @type {LineState} */
  const percentile = { mode: "percentile", value: 0.5, style: "percent" };
  assert.deepEqual(lineAt(axis, layout, percentile, scales.x(3), scales.chance(-2)), { ...percentile, value: 0.99 });
  assert.equal(lineAt(axis, layout, percentile, scales.x(3), layout.density.top + 10), null);
});

test("the two plots share the height the titles and the axis leave, and never fall below a floor", () => {
  const tall = riskLayout(600, 706);
  assert.equal(tall.total, 706);
  assert.equal(tall.density.bottom - tall.density.top, tall.chance.bottom - tall.chance.top);
  assert.equal(tall.density.bottom - tall.density.top, 293);
  assert.equal(riskLayout(600, 100).density.bottom - riskLayout(600, 100).density.top, 110);
});

test("the line starts near risk A's 95th percentile where the panel has none", () => {
  const line = defaultRiskLine(reply());

  assert.equal(line.mode, "amount");
  assert.equal(line.value, Number(gamma([7, -0.5]).quantile(0.95).toPrecision(1)));
});

// ---------------------------------------------------------------------------------------------
// The walk from the starting risk

const BASE = [7, -0.5];
const LN = Math.log;

/**
 * A gamma reply whose reference risk (age 40, region R0) has the predictors `BASE`. A risk's
 * predictors are the reference's plus each feature's own move, plus `joint` where the features
 * add more together than apart.
 * @param {Array<{values: {age: number, region: string}, fitted: number[][], edited?: number[][], joint?: number[]}>} risks
 * @returns {RiskResponse}
 */
function walked(risks) {
  /** @param {number[][]} moves @param {number[]} joint */
  const sum = (moves, joint) => BASE.map((v, p) => v + moves[0][p] + moves[1][p] + joint[p]);
  return {
    ...reply(),
    reference: { fitted: BASE, edited: BASE },
    risks: risks.map(({ values, fitted, edited = fitted, joint = [0, 0] }) => ({
      values,
      eta: { fitted: sum(fitted, joint), edited: sum(edited, joint) },
      effects: { fitted, edited }
    }))
  };
}

test("the first risk is walked from the reference risk, a feature at a time", () => {
  const model = driversModel(walked([
    { values: { age: 22, region: "R0" }, fitted: [[LN(0.8), LN(1.25)], [0, 0]] }
  ]));

  assert.equal(model.title, "Reference → A");
  assert.equal(model.id, "a");
  assert.deepEqual(model.columns, ["Mean", "CV"]);
  assert.deepEqual(model.start, {
    label: "Reference", swatch: null,
    values: [{ text: "€1,100", was: null }, { text: "0.607", was: null }]
  });
  // region keeps the reference's value, so it moves nothing and has no row.
  assert.deepEqual(model.rows, [{
    label: "age", detail: "40 → 22",
    steps: [{ move: LN(0.8), text: "×0.800", was: null }, { move: LN(1.25), text: "×1.25", was: null }]
  }]);
  assert.equal(model.together, null);
  assert.deepEqual(model.end, {
    label: "A", swatch: "a",
    values: [{ text: "€877", was: null }, { text: "0.758", was: null }]
  });
});

test("a walk where nothing differs from its start is one line, not two identical rows", () => {
  const REFERENCE = { age: 40, region: "R0" };
  const atReference = driversModel(walked([{ values: REFERENCE, fitted: [[0, 0], [0, 0]] }]));
  assert.equal(atReference.title, "Reference → A");
  assert.equal(
    atReference.empty,
    "Risk A is the reference risk. Change a value or add a risk to see what moves it."
  );
  // Two risks that agree have no steps between them either.
  const twins = walked([
    { values: { age: 30, region: "R4" }, fitted: [[LN(1.1), 0], [LN(0.5), 0]] },
    { values: { age: 30, region: "R4" }, fitted: [[LN(1.1), 0], [LN(0.5), 0]] }
  ]);
  twins.risks.forEach((risk, i) => { risk.id = /** @type {const} */ (["a", "b"])[i]; });
  assert.equal(
    driversModel(twins, "b").empty,
    "Risk B has the same values as risk A. Change a value to see what moves it."
  );
  // A walk with a step, a joint step, or an edit that moved an end keeps its table.
  const moved = driversModel(walked([{ values: { age: 22, region: "R0" }, fitted: [[LN(0.8), 0], [0, 0]] }]));
  assert.equal(moved.empty, null);
  assert.equal(driversModel(walked([{ values: REFERENCE, fitted: [[0, 0], [0, 0]], joint: [LN(1.2), 0] }])).empty, null);
  const edited = driversModel(walked([{ values: REFERENCE, fitted: [[0, 0], [0, 0]], edited: [[0, LN(1.5)], [0, 0]] }]));
  assert.equal(edited.empty, null);
});

test("a numeric value outside the training rows' range is flagged, one inside is not", () => {
  /** @type {import('../../src/superglm/editor/app/api/contracts.js').RiskFeature} */
  const age = { column: "age", kind: "numeric", min: 18, max: 90, reference: 40, levels: null, terms: {} };
  assert.equal(outsideRange(age, 40), null);
  assert.equal(outsideRange(age, 18), null);
  assert.equal(outsideRange(age, 90), null);
  const below = "Outside the training rows (18 to 90): the model extrapolates here";
  assert.equal(outsideRange(age, 5), below);
  assert.equal(outsideRange(age, 91), below);
  assert.equal(outsideRange(age, "120"), below);
  // No range known, a blank box and a column of levels flag nothing.
  assert.equal(outsideRange({ ...age, min: null, max: null }, 5), null);
  assert.equal(outsideRange(age, ""), null);
  assert.equal(outsideRange({ ...age, kind: "levels", min: null, max: null }, "R4"), null);
});

test("a full bar is the largest step drawn, and never less than a quarter", () => {
  const model = driversModel(walked([
    { values: { age: 22, region: "R4" }, fitted: [[LN(1.02), LN(1.01)], [LN(0.5), 0]] }
  ]));
  assert.deepEqual(model.reach, [LN(2), LN(1.25)]);
  // region has no term in the CV, so that cell stays empty.
  assert.deepEqual(model.rows[1].steps.map((step) => step.text), ["×0.500", ""]);
});

test("another risk is walked from risk A by what each feature puts on it over A", () => {
  const reply3 = walked([
    { values: { age: 30, region: "R4" }, fitted: [[LN(1.1), 0], [LN(0.5), 0]] },
    { values: { age: 22, region: "R4" }, fitted: [[LN(0.8), LN(1.25)], [LN(0.5), 0]] },
    { values: { age: 30, region: "R0" }, fitted: [[LN(1.1), 0], [0, 0]] }
  ]);
  reply3.risks.forEach((risk, i) => { risk.id = /** @type {const} */ (["a", "d", "b"])[i]; });
  const model = driversModel(reply3, "d");

  assert.equal(model.title, "A → D");
  assert.equal(model.id, "d");
  assert.deepEqual([model.start.label, model.start.swatch, model.end.label, model.end.swatch], ["A", "a", "D", "d"]);
  // Only age differs between A and D.
  assert.deepEqual(model.rows.map((row) => [row.label, row.detail]), [["age", "30 → 22"]]);
  assert.deepEqual(model.rows[0].steps.map((step) => step.text), [`×${(0.8 / 1.1).toFixed(3)}`, "×1.25"]);
  assert.equal(model.together, null);
  // The start times every step is the end.
  assert.equal(model.end.values[0].text, `€${Math.round(Math.exp(BASE[0]) * 1.1 * 0.5 * (0.8 / 1.1))}`);

  const third = driversModel(reply3, "b");
  assert.equal(third.title, "A → B");
  assert.deepEqual(third.rows.map((row) => [row.label, row.detail, row.steps[0].text]), [["region", "R4 → R0", "×2.00"]]);
  // Risk A itself, and a risk the reply does not hold, are walked from the reference risk.
  assert.equal(driversModel(reply3, "a").title, "Reference → A");
  assert.equal(driversModel(reply3, "h").title, "Reference → A");
});

test("what the features add only together is a row of its own", () => {
  const model = driversModel(walked([
    { values: { age: 22, region: "R4" }, fitted: [[LN(0.8), 0], [LN(0.5), 0]], joint: [LN(1.2), 0] }
  ]));

  assert.ok(model.together);
  assert.equal(model.together.label, "Together");
  assert.deepEqual(model.together.steps.map((step) => step.text), ["×1.20", "×1.00"]);
  const walkedTo = [model.rows[0], model.rows[1], model.together].reduce((sum, row) => sum + row.steps[0].move, BASE[0]);
  assert.ok(Math.abs(Math.exp(walkedTo) - Math.exp(BASE[0]) * 0.8 * 0.5 * 1.2) < 1e-9);
});

test("a step the session's edits moved says what it was under the original model", () => {
  const model = driversModel(walked([{
    values: { age: 22, region: "R0" },
    fitted: [[LN(0.8), LN(1.25)], [0, 0]],
    edited: [[LN(0.8), LN(1.5)], [0, 0]]
  }]));

  assert.deepEqual(model.rows[0].steps.map((step) => [step.text, step.was]), [["×0.800", null], ["×1.50", "×1.25"]]);
  assert.deepEqual(model.end.values[1], { text: "0.910", was: "0.758" });
  assert.equal(model.start.values[1].was, null);
});

test("theta is read as the overdispersion 1/theta, and a log-normal's scale as its sigma", () => {
  const counts = /** @type {RiskResponse} */ ({
    ...walked([{ values: { age: 22, region: "R0" }, fitted: [[0, LN(2)], [0, 0]] }]),
    family: "negative_binomial",
    parameters: [
      { name: "mean", label: "Mean", role: "average", link: LOG },
      { name: "theta", label: "Overdispersion", role: "spread", link: LOG, display: { kind: "inverse_theta", link: "log" } }
    ],
    features: [{ ...reply().features[0], terms: { mean: "mean:age", theta: "theta:age" } }, reply().features[1]]
  });
  assert.equal(parameterValue(counts, 1, LN(4)), 0.25);
  // Doubling theta halves the overdispersion.
  const step = driversModel(counts).rows[0].steps[1];
  assert.deepEqual([step.text, step.move], ["×0.500", -LN(2)]);

  const floored = /** @type {const} */ ({ kind: "floored_log", floor: 0.01 });
  const lognormal = /** @type {RiskResponse} */ ({
    ...walked([{ values: { age: 22, region: "R0" }, fitted: [[0, 0], [0, 0]] }]),
    family: "lognormal_mean",
    parameters: [
      { name: "mean", label: "Mean", role: "average", link: LOG },
      { name: "scale", label: "CV", role: "spread", link: floored, value_label: "σ" }
    ]
  });
  // A log-normal's walk is in sigma, the quantity its steps multiply above the floor, not in the
  // CV the table shows: a x2 step must read as x2 on the column it sits in.
  assert.equal(parameterValue(lognormal, 1, -0.5), 0.01 + Math.exp(-0.5));
  const moved = driversModel({
    ...lognormal,
    features: [{ ...reply().features[0], terms: { mean: "mean:age", scale: "scale:age" } }, reply().features[1]],
    risks: [{ ...lognormal.risks[0], eta: { fitted: [0, LN(2)], edited: [0, LN(2)] },
      effects: { fitted: [[0, LN(2)], [0, 0]], edited: [[0, LN(2)], [0, 0]] } }],
    reference: { fitted: [0, 0], edited: [0, 0] },
  });
  assert.deepEqual(moved.columns, ["Mean", "σ"]);
  const [start, end] = [moved.start.values[1].text, moved.end.values[1].text];
  assert.deepEqual([moved.rows[0].steps[1].text, start, end], ["×2.00", "1.01", "2.01"]);
});

test("an additive parameter's step reads as the amount added, and an uninverted link leaves its value blank", () => {
  const base = walked([{ values: { age: 22, region: "R0" }, fitted: [[150, 0.2], [0, 0]] }]);
  const gaussian = /** @type {RiskResponse} */ ({
    ...base,
    family: null,
    kind: "unavailable",
    reason: "Not drawn yet.",
    amount_axis: null,
    parameters: [
      { name: "location", label: "Location", role: "average", link: { kind: "identity" }, display: { kind: "additive", link: "identity" } },
      { name: "scale", label: "Scale", role: "spread", link: { kind: "other" }, display: { kind: "additive", link: "inverse" } }
    ],
    features: [{ ...reply().features[0], terms: { location: "location:age", scale: "scale:age" } }, reply().features[1]],
    reference: { fitted: [1000, 1], edited: [1000, 1] },
    risks: [{ ...base.risks[0], eta: { fitted: [1150, 1.2], edited: [1150, 1.2] },
      effects: { fitted: [[150, 0.2], [0, 0]], edited: [[150, 0.2], [0, 0]] } }]
  });
  assert.equal(parameterValue(gaussian, 0, 1150), 1150);
  assert.ok(Number.isNaN(parameterValue(gaussian, 1, 1.2)));
  const model = driversModel(gaussian);
  assert.deepEqual(model.rows[0].steps.map((step) => [step.text, step.move]), [["+150", 150], ["+0.200", 0.2]]);
  assert.deepEqual(model.end.values.map((value) => value.text), ["€1,150", ""]);
  // A full bar is a quarter of the parameter's own value, or its largest step where that is
  // larger: an amount added has no fixed unit, as a factor of x1.25 is.
  assert.deepEqual(model.reach, [0.25 * 1150, 0.25 * 1.2]);
  // A step of 3e-4 on an inverse link whose value is 1e-3 is most of a full bar, not none of it.
  const tiny = driversModel({
    ...gaussian,
    reference: { fitted: [1000, 1e-3], edited: [1000, 1e-3] },
    risks: [{ ...gaussian.risks[0], eta: { fitted: [1150, 1.3e-3], edited: [1150, 1.3e-3] },
      effects: { fitted: [[150, 3e-4], [0, 0]], edited: [[150, 3e-4], [0, 0]] } }]
  });
  assert.ok(tiny.rows[0].steps[1].move / tiny.reach[1] > 0.9, String(tiny.reach[1]));
  // An unavailable reply says why the view draws no distribution.
  assert.equal(riskModel(gaussian, { mode: "amount", value: 100, style: "percent" }).empty, "Not drawn yet.");
});

// ---------------------------------------------------------------------------------------------
// A Tweedie: a chance of no claim, and a power no feature moves

const WALLS = /** @type {const} */ ({ kind: "bounded_logit", lower: 1.05, upper: 1.95 });

/**
 * A Tweedie reply: risk A at the reference risk, mean 100, dispersion 20 and the power midway
 * between its walls (1.5); risk B where `second` gives its predictors.
 * @param {number[]} [second]
 * @returns {RiskResponse}
 */
function tweedieReply(second) {
  const eta = [LN(100), LN(20), 0];
  const base = reply({ fitted: eta, second });
  return {
    ...base,
    family: "tweedie",
    parameters: [
      { name: "mean", label: "Mean", role: "average", link: LOG },
      { name: "dispersion", label: "Dispersion", role: "spread", link: LOG },
      { name: "power", label: "Power", role: "shape", link: WALLS }
    ],
    features: [
      { ...base.features[0], terms: { mean: "mean:age", dispersion: "dispersion:age" } },
      base.features[1]
    ],
    risks: base.risks.map((risk, i) => ({
      ...risk,
      id: /** @type {const} */ (["a", "b"])[i],
      effects: { fitted: risk.effects.fitted.map((row) => [...row, 0].slice(0, 3)), edited: risk.effects.edited.map((row) => [...row, 0].slice(0, 3)) }
    })),
    amount_axis: { min_log10: 0, max_log10: 4, clipped_below: false }
  };
}

test("a Tweedie risk reads its chance of no claim, and its CV from mean, dispersion and power", () => {
  const model = riskModel(tweedieReply(), LINE);
  const law = lawOf("tweedie", [LN(100), LN(20), 0], [LOG, LOG, WALLS]);
  const cells = tableCells(model).A;

  // Compound Poisson: the chance of no claim is exp(-mean^(2-p) / (dispersion (2-p))).
  assert.ok(Math.abs(law.zero - Math.exp(-(100 ** 0.5) / (20 * 0.5))) < 1e-12);
  assert.deepEqual(model.table.columns.map((column) => column.label).slice(0, 3), ["Mean", "No claim", "Median"]);
  assert.equal(cells.Mean, "100");
  assert.equal(cells["No claim"], `${(100 * law.zero).toFixed(1)}%`);
  // Variance = dispersion x mean^power, so CV = sqrt(20 x 100^-0.5).
  assert.equal(cells.CV, Math.sqrt(20 / 10).toFixed(2));
  assert.equal(parameterValue(tweedieReply(), 2, 0), 1.5);
  // The plots start from the amounts above zero, and so does the line the tab opens with.
  assert.ok(model.axis && model.axis.min_log10 >= 0);
  const line = defaultRiskLine(tweedieReply());
  assert.equal(line.value, Number(law.quantile(law.zero + 0.95 * (1 - law.zero)).toPrecision(1)));
});

test("a Tweedie's walk has no column for its constant power", () => {
  const model = driversModel(tweedieReply([LN(150), LN(30), 0]), "b");
  assert.deepEqual(model.columns, ["Mean", "Dispersion"]);
  // age acts on both; region has a term in the mean alone, and moves nothing here.
  assert.deepEqual(model.rows.map((row) => [row.label, ...row.steps.map((step) => step.text)]), [
    ["age", "×1.50", "×1.50"],
    ["region", "×1.00", ""]
  ]);
  assert.deepEqual(model.end.values.map((value) => value.text), ["€150", "30.0"]);
});

test("a Tweedie's curve is its density given a claim, so its area is one whatever its chance of a claim", () => {
  const axis = { min_log10: -3, max_log10: 5, clipped_below: false };
  for (const eta of [[LN(100), LN(20), 0], [LN(400), LN(60), 0]]) {
    const law = lawOf("tweedie", eta, [LOG, LOG, WALLS]);
    const sampled = sampleLaw(law, axis, 4000);
    const step = (axis.max_log10 - axis.min_log10) / 4000;
    const area = sampled.density.reduce((sum, d) => sum + d * step, 0);
    assert.ok(law.zero > 0.05 && law.zero < 0.95);
    assert.ok(Math.abs(area - 1) < 2e-3, `area ${area} at chance of zero ${law.zero}`);
    // The chance plot is not conditional: it starts from the chance of any claim.
    assert.ok(Math.abs(10 ** sampled.chance[0] - (1 - law.zero)) < 1e-3);
  }
});

test("the density axis is labelled at round steps below its peak", () => {
  assert.deepEqual(densityTicks(0.87), [0.2, 0.4, 0.6, 0.8]);
  assert.deepEqual(densityTicks(1.3), [0.5, 1]);
  assert.deepEqual(densityTicks(0.031), [0.01, 0.02, 0.03]);
  assert.deepEqual(densityTicks(6.4), [2, 4, 6]);
  assert.deepEqual(densityTicks(0), []);
});
