// @ts-nocheck

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { CHECK_REPORTS } from "../../src/superglm/editor/app/api/contracts.js";
import { checkScene, nearestMark, sceneMarkup } from "../../src/superglm/editor/app/chart/check_plots.js";
import { neutralCopy } from "./neutral_copy.js";

const FIXTURE = JSON.parse(readFileSync(new URL("./fixtures/assess_checks.json", import.meta.url)));
const BOX = { width: 820, height: 520 };
const THUMB = { width: 160, height: 62, thumb: true };

/** The layer drawn in `role`, or undefined. */
const layer = (scene, role) => scene.layers.find((l) => l.role === role);
/** Each `M` subpath of a path as its numbers, in order. */
const subpaths = (d) => d.split("M").filter((s) => s.trim()).map((s) => [...s.matchAll(/-?\d+(?:\.\d+)?/g)].map(Number));
/**
 * The absolute points a path visits, as [x, y] with null for the axis a command leaves alone; a
 * point's arcs are relative to its centre, which is listed.
 */
const absolutePoints = (d) => [...d.matchAll(/([MLHVZmaz])([^MLHVZmaz]*)/g)].flatMap(([, command, args]) => {
  const v = [...args.matchAll(/-?\d+(?:\.\d+)?/g)].map(Number);
  if (command === "M" || command === "L") return [[v[0], v[1]]];
  if (command === "H") return [[v[0], null]];
  if (command === "V") return [[null, v[0]]];
  return [];
});
/** The pixel map through two known (value, px) pairs. */
const affine = (v1, p1, v2, p2) => (v) => p1 + ((v - v1) * (p2 - p1)) / (v2 - v1);
/** Path coordinates are written to 0.01 px. */
const near = (actual, expected, tol = 0.011) => assert.ok(Math.abs(actual - expected) < tol, `${actual} vs ${expected}`);

test("the Q-Q scene puts each drawn point at its residual and keeps y = x on the diagonal", () => {
  const payload = FIXTURE.calm.check_qq, body = payload.models.edited;
  const scene = checkScene("check_qq", payload, BOX, { compare: false });
  assert.equal(scene.marks.length, body.index.length);
  const [a, b] = [scene.marks[0], scene.marks.at(-1)];
  const sx = (b.x - a.x) / (body.theoretical.at(-1) - body.theoretical[0]);
  for (const [i, mark] of scene.marks.entries()) {
    assert.ok(Math.abs(mark.x - (a.x + sx * (body.theoretical[i] - body.theoretical[0]))) < 1e-6);
  }
  // The vertical map, read from the points, sends the diagonal's ends to the values the
  // horizontal map gives them: both axes share one scale.
  const y = affine(body.observed[0], a.y, body.observed.at(-1), b.y);
  for (const [i, mark] of scene.marks.entries()) assert.ok(Math.abs(mark.y - y(body.observed[i])) < 1e-6);
  const xValue = (px) => body.theoretical[0] + (px - a.x) / sx;
  const [x0, y0, x1, y1] = subpaths(layer(scene, "reference").d)[0];
  near(y0, y(xValue(x0)), 0.03);
  near(y1, y(xValue(x1)), 0.03);
  assert.ok(!scene.layers.some((l) => l.role === "original"));
});

test("a flagged Q-Q draws its outside points in their own role", () => {
  const scene = checkScene("check_qq", FIXTURE.wrong.check_qq, BOX, { compare: false });
  const outside = layer(scene, "outside");
  assert.equal((outside.d.match(/M/g) ?? []).length, FIXTURE.wrong.check_qq.models.edited.outside);
  // 912 points drawn, 556 of them outside the band.
  assert.equal((layer(scene, "edited").d.match(/M/g) ?? []).length, 912 - 556);
});

test("compare draws the model as fitted only when the payload has it", () => {
  const edited = FIXTURE.edited.check_ae;
  assert.ok(checkScene("check_ae", edited, BOX, { compare: true }).layers.some((l) => l.role === "original"));
  assert.ok(!checkScene("check_ae", edited, BOX, { compare: false }).layers.some((l) => l.role === "original"));
  assert.ok(!checkScene("check_ae", FIXTURE.calm.check_ae, BOX, { compare: true }).layers.some((l) => l.role === "original"));
  for (const report of CHECK_REPORTS) {
    assert.ok(layer(checkScene(report, FIXTURE.edited[report], BOX, { compare: true }), "original"), report);
  }
});

test("PIT bars read count over expected against the band", () => {
  // The wrong case: 1000 rows in 20 bins, 50 expected in each, and a band of 28 to 76 values.
  const payload = FIXTURE.wrong.check_pit, body = payload.models.edited;
  assert.equal(body.expected, 1000 / 20);
  const scene = checkScene("check_pit", payload, BOX, { compare: false });
  // Bin 1 holds 9 values, 9 / 50 = 0.18 of expected; bin 5 holds 85, 85 / 50 = 1.7.
  const y = affine(9 / 50, scene.marks[0].y, 85 / 50, scene.marks[4].y);
  const bars = ["edited", "outside"].flatMap((role) => subpaths(layer(scene, role).d)).sort((p, q) => p[0] - q[0]);
  assert.equal(bars.length, 20);
  for (const [i, [, base, top]] of bars.entries()) {
    near(top, y(body.counts[i] / 50));
    near(base, y(0));
  }
  const [, upper, , lower] = subpaths(layer(scene, "band").d)[0];
  near(upper, y(76 / 50)); // 1.52
  near(lower, y(28 / 50)); // 0.56
  near(subpaths(layer(scene, "reference").d)[0][1], y(1));
  // 9 and 23 lie below 28, 84 and 85 above 76: four bars outside.
  assert.equal(subpaths(layer(scene, "outside").d).length, 4);
});

test("A/E draws a panel per table with its interval around each ratio", () => {
  const payload = FIXTURE.edited.check_ae, tables = payload.models.edited.tables;
  // Band 1 of predicted Mean: 36.36 actual over 34.49 expected is 1.054, interval 0.714 to 1.394.
  assert.ok(Math.abs(tables[0].actual[0] / tables[0].expected[0] - tables[0].ratio[0]) < 1e-12);
  const scene = checkScene("check_ae", payload, BOX, { compare: false });
  const references = subpaths(layer(scene, "reference").d);
  const intervals = subpaths(layer(scene, "interval").d);
  assert.equal(references.length, tables.length);
  for (const table of tables) {
    const marks = scene.marks.filter((m) => m.text.startsWith(`${table.covariate} `));
    assert.equal(marks.length, table.ratio.length, table.covariate);
    const j = table.ratio.reduce((best, r, k) => (Math.abs(r - table.ratio[0]) > Math.abs(table.ratio[best] - table.ratio[0]) ? k : best), 0);
    const y = affine(table.ratio[0], marks[0].y, table.ratio[j], marks[j].y);
    // Panels in one column share their left and right: the reference is the one at y(1).
    const spanning = references.filter(([left, , right]) => left <= marks[0].x && marks.at(-1).x <= right);
    assert.ok(spanning.some(([, at]) => Math.abs(at - y(1)) < 0.011), table.covariate);
    for (const [i, mark] of marks.entries()) {
      assert.ok(Math.abs(mark.y - y(table.ratio[i])) < 1e-6);
      const tick = intervals.find(([x, a, b]) => Math.abs(x - mark.x) < 0.011 && Math.min(a, b) <= mark.y && mark.y <= Math.max(a, b));
      near(tick[1], y(table.lower[i]));
      near(tick[2], y(table.upper[i]));
    }
  }
  const texts = scene.labels.map((l) => l.text);
  for (const name of ["1", "10", "a", "b", "c"]) assert.ok(texts.includes(name), name);
});

test("quantile calibration reads shares on a log axis against 1 - p", () => {
  const payload = FIXTURE.wrong.check_calibration, body = payload.models.edited, q = body.quantiles;
  const scene = checkScene("check_calibration", payload, BOX, { compare: false });
  assert.equal(scene.marks.length, 11);
  const step = scene.marks[1].x - scene.marks[0].x;
  for (const [i, mark] of scene.marks.entries()) near(mark.x, scene.marks[0].x + i * step, 1e-6);
  // 99.1% of rows exceed the 5% quantile and 1.5% exceed the 99.5% quantile: equal ratios of
  // share are equal steps up the axis.
  const log = affine(Math.log10(0.991), scene.marks[1].y, Math.log10(0.015), scene.marks[10].y);
  for (const [i, mark] of scene.marks.entries()) near(mark.y, log(Math.log10(q.realised_exceedance[i])), 1e-6);
  const reference = subpaths(layer(scene, "reference").d)[0];
  for (const [i, expected] of q.expected.entries()) near(reference[2 * i + 1], log(Math.log10(expected)));
  assert.equal(subpaths(layer(scene, "outside").d).length, body.outside);
  const texts = scene.labels.map((l) => l.text);
  for (const name of ["1%", "50%", "97.5%", "99.5%"]) assert.ok(texts.includes(name), name);
});

test("each readout says what its point is, in the stated form", () => {
  const calm = FIXTURE.calm;
  const first = (report) => checkScene(report, calm[report], BOX, { compare: false }).marks;
  assert.equal(first("check_qq")[0].text, "Rank 1 of 200 · expected -2.7 · observed -2.91 · band -4.84 to -1.71");
  assert.equal(first("check_pit")[0].text, "0 to 0.05 · 10 values · band 2 to 21");
  const ae = first("check_ae").map((m) => m.text);
  assert.ok(ae.includes("predicted Mean band 1 · 1.05 · interval 0.714 to 1.39 · 20 rows"), ae.join("\n"));
  assert.ok(ae.includes("g a · 1.15 · interval 0.94 to 1.36 · 53 rows"), ae.join("\n"));
  assert.equal(first("check_calibration")[0].text, "Quantile 1% · exceeded by 99% · band 96% to 100%");
});

test("a thumbnail has no text, at most 60 Q-Q points, and stays in its box", () => {
  for (const report of CHECK_REPORTS) {
    const scene = checkScene(report, FIXTURE.wrong[report], THUMB, { compare: false });
    const markup = sceneMarkup(scene);
    assert.ok(!/<text/.test(markup), report);
    assert.ok(markup.startsWith('<svg class="check-plot" viewBox="0 0 160 62" width="160" height="62">'), report);
    for (const { d } of scene.layers) {
      for (const [x, y] of absolutePoints(d)) {
        if (x !== null) assert.ok(x >= -1 && x <= 161, `${report}: x ${x}`);
        if (y !== null) assert.ok(y >= -1 && y <= 63, `${report}: y ${y}`);
      }
    }
  }
  const qq = checkScene("check_qq", FIXTURE.wrong.check_qq, THUMB, { compare: false });
  const count = (role) => (layer(qq, role)?.d.match(/M/g) ?? []).length;
  assert.ok(count("edited") + count("outside") <= 60);
  assert.equal(count("outside"), 20);
  // The thumbnail of A/E is its predicted-Mean bands alone.
  const ae = checkScene("check_ae", FIXTURE.wrong.check_ae, THUMB, { compare: false });
  assert.equal(subpaths(layer(ae, "reference").d).length, 1);
  assert.equal(ae.marks.length, 10);
});

test("the markup draws one path per layer, escaped labels and an empty hidden readout", () => {
  const payload = structuredClone(FIXTURE.calm.check_ae);
  payload.models.edited.tables[2].levels = ["a<b", "b", "c"];
  const scene = checkScene("check_ae", payload, BOX, { compare: false });
  const markup = sceneMarkup(scene);
  assert.ok(markup.startsWith('<svg class="check-plot" viewBox="0 0 820 520"'));
  assert.equal((markup.match(/<path /g) ?? []).length, scene.layers.length);
  for (const { role } of scene.layers) assert.ok(markup.includes(`<path class="check-${role}" d="`), role);
  assert.ok(markup.includes(">a&lt;b</text>"));
  assert.ok(markup.endsWith('<g class="check-readout" hidden></g></svg>'));
});

test("an unavailable check draws nothing", () => {
  const body = { available: false, reason: "No rows.", flagged: false, outside: 0, cells: 0 };
  const payload = { ...FIXTURE.calm.check_qq, available: false, models: { edited: body, original: null } };
  const scene = checkScene("check_qq", payload, BOX, { compare: true });
  assert.deepEqual([scene.layers, scene.labels, scene.marks], [[], [], []]);
});

test("nearestMark is the closest mark by x, or in the plane when given y", () => {
  const marks = checkScene("check_qq", FIXTURE.calm.check_qq, BOX, { compare: false }).marks;
  let seed = 7; const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < 200; i++) {
    const x = random() * BOX.width;
    const brute = marks.reduce((best, m, k) => (Math.abs(m.x - x) < Math.abs(marks[best].x - x) ? k : best), 0);
    assert.equal(Math.abs(marks[nearestMark(marks, x)].x - x), Math.abs(marks[brute].x - x));
  }
  // A/E's panels stack two rows: in the plane, the nearest is the mark under the pointer.
  const grid = checkScene("check_ae", FIXTURE.wrong.check_ae, BOX, { compare: false }).marks;
  const distance = (m, x, y) => Math.hypot(m.x - x, m.y - y);
  for (let i = 0; i < 200; i++) {
    const [x, y] = [random() * BOX.width, random() * BOX.height];
    const brute = grid.reduce((best, m, k) => (distance(m, x, y) < distance(grid[best], x, y) ? k : best), 0);
    assert.equal(distance(grid[nearestMark(grid, x, y)], x, y), distance(grid[brute], x, y));
  }
  assert.equal(nearestMark([], 10), -1);
});

test("axis titles and readouts use neutral words", () => {
  for (const report of CHECK_REPORTS) {
    const scene = checkScene(report, FIXTURE.edited[report], BOX, { compare: true });
    assert.deepEqual(neutralCopy([...scene.labels.map((l) => l.text), ...scene.marks.map((m) => m.text)].join(" ")), []);
  }
});
