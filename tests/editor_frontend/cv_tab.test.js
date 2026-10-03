// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  createCVTab,
  curveChartMarkup,
  cvSourceLine,
  cvTabMarkup,
  exposureProfile,
  filterTerms,
  foldEnvelope,
  highlightFold,
  jobLine,
  levelChartMarkup,
  newerJob,
  niceTicks,
  termListMarkup
} from "../../src/superglm/editor/app/views/cv_tab.js";

function foldRow(index, deviance, gini) {
  return {
    fold: index,
    n_train: 36000,
    n_test: 9000,
    fit_time_s: 1.25,
    converged: true,
    n_iter: 7,
    effective_df: 31.4,
    scores: { deviance, gini }
  };
}

const brand = Object.freeze({
  name: "VehBrand",
  kind: "levels",
  x: null,
  levels: ["B1", "B2", "B10"],
  weights: [3, 2, 1],
  folds: [
    { label: "Fold 1", values: [1.1, 0.9, 1.0] },
    { label: "Fold 2", values: [1.2, 0.8, 1.05] }
  ],
  fit: [1.15, 0.85, 1.02],
  edited: null,
  spread: 0.05,
  min_correlation: 0.9
});

const age = Object.freeze({
  name: "DrivAge",
  kind: "continuous",
  x: [18, 30, 50, 85],
  levels: null,
  weights: [1, 4, 3, 1],
  folds: [
    { label: "Fold 1", values: [1.4, 1.0, 0.9, 1.1] },
    { label: "Fold 2", values: [1.5, 1.05, 0.88, 1.0] }
  ],
  fit: [1.45, 1.02, 0.89, 1.05],
  edited: [1.3, 1.02, 0.89, 1.05],
  spread: 0.01,
  min_correlation: 0.99
});

function cvPayload(overrides = {}) {
  return {
    report: "cv",
    title: "Cross-validation",
    note: "",
    model_revision: 3,
    header: { supplied: true, n_folds: 2, splitter: "KFold", n_rows: 45000 },
    pending: 0,
    run_cv: { available: true, reason: null, note: null },
    final_fit: { available: true, reason: null, note: null, done: false, stale: false, n_rows: null },
    metrics: [
      { name: "deviance", label: "Mean deviance", lower_is_better: true },
      { name: "gini", label: "Gini", lower_is_better: false }
    ],
    results: [{
      label: "As supplied",
      origin: "supplied",
      model_revision: null,
      stale: false,
      folds: [foldRow(0, 0.31, 0.2), foldRow(1, 0.3, 0.22)],
      mean: { deviance: 0.305, gini: 0.21 },
      std: { deviance: 0.005, gini: 0.01 },
      pooled: { deviance: 0.3049 }
    }],
    relativities: { available: true, origin: "supplied", stale: false, note: null, terms: [brand, age] },
    jobs: { cv: null, final_fit: null },
    ...overrides
  };
}

const idle = () => ({
  term: "",
  query: "",
  jobs: { cv: null, final_fit: null },
  errors: { cv: "", final_fit: "" }
});

function count(markup, pattern) {
  return (markup.match(pattern) ?? []).length;
}

test("the source line names the folds, splitter, rows and edit() call", () => {
  assert.equal(
    cvSourceLine(cvPayload()),
    "2 folds · KFold · 45,000 rows · supplied with edit(model, cv=result)"
  );
  assert.equal(
    cvSourceLine(cvPayload({ header: { supplied: false, n_folds: 0, splitter: null, n_rows: null } })),
    "No cross-validation result supplied"
  );
});

test("Run CV is disabled with its reason while changes wait", () => {
  const markup = cvTabMarkup(cvPayload({
    pending: 2,
    run_cv: { available: false, reason: "Refit first: 2 changes are waiting.", note: null }
  }), idle());

  assert.match(markup, /data-cv-start="cv"\s+disabled/);
  assert.doesNotMatch(markup, /data-cv-start="final_fit"\s+disabled/);
  assert.match(markup, /<p class="cv-reason">Refit first: 2 changes are waiting\.<\/p>/);
  assert.match(markup, /2 changes waiting for refit/);
});

test("performance cards show mean ± sd, pooled, and one dot per fold for each run", () => {
  const current = { ...cvPayload().results[0], label: "Current model", origin: "run", pooled: {} };
  const markup = cvTabMarkup(cvPayload({ results: [cvPayload().results[0], current] }), idle());
  const deviance = markup.slice(markup.indexOf("Mean deviance"), markup.indexOf(">Gini"));

  assert.match(deviance, /<strong>0\.3050<\/strong>\s*<span class="cv-card-spread">± 0\.0050 · pooled 0\.3049<\/span>/);
  assert.equal(count(deviance, /<circle /g), 4);
  assert.equal(count(deviance, /class="cv-card-row"/g), 2);
});

test("the fold table lists the latest run's folds and a mean row", () => {
  const markup = cvTabMarkup(cvPayload(), idle());
  const table = markup.slice(markup.indexOf('aria-label="Fold scores"'), markup.indexOf("</table>"));

  assert.equal(count(table, /<tr>/g), 3);
  assert.match(table, /Fold 2<\/td>\s*<td>36,000<\/td><td>9,000<\/td>/);
  assert.match(table, /<th>Deviance<\/th><th>Gini<\/th>/);
  assert.match(table, /Mean ± sd/);
});

test("the term list filters by name, ignoring case, in the server's order", () => {
  assert.deepEqual(filterTerms([brand, age], "").map((term) => term.name), ["VehBrand", "DrivAge"]);
  assert.deepEqual(filterTerms([brand, age], "  drIV ").map((term) => term.name), ["DrivAge"]);
  const markup = cvTabMarkup(cvPayload(), { ...idle(), query: "brand" });
  assert.match(markup, /Veh<mark>Brand<\/mark>/);
  assert.equal(count(markup, /class="cv-term"/g), 1);
  assert.match(cvTabMarkup(cvPayload(), { ...idle(), query: "zzz" }), /No terms match\./);
});

test("a level chart keeps the model's level order with a whisker and dots per fold", () => {
  const markup = levelChartMarkup(brand);

  assert.ok(markup.indexOf(">B1<") < markup.indexOf(">B2<"));
  assert.ok(markup.indexOf(">B2<") < markup.indexOf(">B10<"));
  assert.equal(count(markup, /class="cv-range"/g), 3);
  assert.equal(count(markup, /class="cv-fold-dot"/g), 6);
  assert.equal(count(markup, /class="cv-fit"/g), 3);
  assert.equal(count(markup, /class="exposure"/g), 3);
  assert.equal(count(markup, /class="cv-edited"/g), 0);
});

test("a curve chart draws each fold, the envelope, the fit and the edited curve", () => {
  const markup = curveChartMarkup(age);

  assert.equal(count(markup, /class="cv-fold-line"/g), 2);
  assert.equal(count(markup, /class="cv-envelope"/g), 1);
  assert.equal(count(markup, /class="cv-fit-line"/g), 1);
  assert.equal(count(markup, /class="cv-edited-line"/g), 1);
  assert.doesNotMatch(markup, /NaN/);
});

test("the fold envelope and the axis ticks", () => {
  assert.deepEqual(foldEnvelope(brand), { lo: [1.1, 0.8, 1.0], hi: [1.2, 0.9, 1.05] });
  assert.deepEqual(niceTicks(0.82, 1.21, 5), [0.9, 1, 1.1, 1.2]);
  assert.deepEqual(niceTicks(18, 85, 6), [20, 40, 60, 80]);
});

test("a job line says what the job is doing and how it ended", () => {
  const running = { job_id: "cv-1", kind: "cv", status: "running", progress: [], result: null };
  assert.equal(jobLine("cv", running), "Run CV: starting…");
  assert.equal(
    jobLine("cv", { ...running, progress: [{ phase: "fold", fold: 2, n_folds: 5 }] }),
    "Run CV: fold 2 of 5…"
  );
  assert.equal(jobLine("cv", { ...running, cancel_requested: true }), "Run CV: cancelling after this step…");
  assert.equal(jobLine("cv", { ...running, status: "cancelled" }), "Run CV was cancelled. Nothing was kept.");
  assert.equal(
    jobLine("final_fit", { ...running, kind: "final_fit", status: "done", result: { n_rows: 50000 } }),
    "Final fit finished on 50,000 rows. Export offers it as Final fit model."
  );
  assert.equal(
    jobLine("cv", { ...running, status: "failed", error: "Refit first: 1 change is waiting." }),
    "Run CV failed: Refit first: 1 change is waiting."
  );
});

test("the newer status wins: a later job, then a finished one, then more progress", () => {
  const first = { job_id: "cv-1", status: "running", progress: [{}] };
  const later = { job_id: "cv-3", status: "running", progress: [] };
  assert.equal(newerJob(first, later), later);
  assert.equal(newerJob(later, first), later);
  assert.equal(newerJob(first, { ...first, status: "done" }).status, "done");
  assert.equal(newerJob({ ...first, status: "done" }, first).status, "done");
  assert.equal(newerJob(first, { ...first, progress: [] }), first);
  assert.equal(newerJob(null, first), first);
});

function fakeFrame() {
  return {
    innerHTML: "",
    querySelector: () => null,
    addEventListener() {},
    removeEventListener() {}
  };
}

test("a started job is polled until it settles, then reported once", async () => {
  const statuses = [
    { job_id: "cv-1", kind: "cv", status: "running", progress: [{ phase: "fold", fold: 1, n_folds: 2 }], result: null },
    { job_id: "cv-1", kind: "cv", status: "done", progress: [], result: { n_folds: 2 } }
  ];
  const calls = [];
  const settled = [];
  const frame = fakeFrame();
  const tab = createCVTab({
    frame,
    client: {
      async jobStart(kind) {
        calls.push(["start", kind]);
        return { job_id: "cv-1", kind, status: "running", progress: [], result: null };
      },
      async jobStatus(jobId) {
        calls.push(["status", jobId]);
        return statuses.shift();
      },
      async jobCancel() {
        throw new Error("not called");
      }
    },
    onJobSettled: (kind, job) => settled.push([kind, job.status]),
    pause: async () => {}
  });
  tab.render(cvPayload());

  await tab.start("cv");

  assert.deepEqual(calls, [["start", "cv"], ["status", "cv-1"], ["status", "cv-1"]]);
  assert.deepEqual(settled, [["cv", "done"]]);
  assert.match(frame.innerHTML, /Run CV finished\./);
});

test("cancel posts the running job's id and shows it is stopping", async () => {
  const cancelled = [];
  let release;
  const frame = fakeFrame();
  const running = { job_id: "cv-2", kind: "cv", status: "running", progress: [], result: null };
  const tab = createCVTab({
    frame,
    client: {
      async jobStart() {
        return running;
      },
      async jobStatus() {
        await new Promise((resolve) => { release = resolve; });
        return { ...running, status: "cancelled" };
      },
      async jobCancel(jobId) {
        cancelled.push(jobId);
        return { job_id: jobId, status: "running", cancel_requested: true };
      }
    },
    onJobSettled: () => {},
    pause: async () => {}
  });
  tab.render(cvPayload());
  const started = tab.start("cv");
  await new Promise((resolve) => setImmediate(resolve));

  await tab.cancel("cv");
  const stopping = frame.innerHTML;
  release();
  await started;

  assert.deepEqual(cancelled, ["cv-2"]);
  assert.match(stopping, /Run CV: cancelling after this step…/);
  assert.doesNotMatch(stopping, /data-cv-cancel="cv"/);
  assert.match(frame.innerHTML, /Run CV was cancelled\. Nothing was kept\./);
});

test("a refused start shows the server's sentence", async () => {
  const frame = fakeFrame();
  const tab = createCVTab({
    frame,
    client: {
      async jobStart() {
        throw new Error("Refit first: 1 change is waiting.");
      },
      async jobStatus() {},
      async jobCancel() {}
    },
    onJobSettled: () => {},
    pause: async () => {}
  });
  tab.render(cvPayload());

  await tab.start("cv");

  assert.match(frame.innerHTML, /data-status="failed">Refit first: 1 change is waiting\.<\/p>/);
});

test("a job that settles while another report is shown leaves that report alone", async () => {
  const frame = fakeFrame();
  let shown = true;
  const settled = [];
  const tab = createCVTab({
    frame,
    client: {
      async jobStart(kind) {
        return { job_id: "cv-1", kind, status: "running", progress: [], result: null };
      },
      async jobStatus(jobId) {
        // The user opens the Validation tab while the job runs.
        shown = false;
        frame.innerHTML = "<section>Validation report</section>";
        return { job_id: jobId, kind: "cv", status: "done", progress: [], result: { n_folds: 2 } };
      },
      async jobCancel() {}
    },
    onJobSettled: (kind, job) => settled.push([kind, job.status]),
    pause: async () => {},
    isShown: () => shown
  });
  tab.render(cvPayload());

  await tab.start("cv");

  assert.equal(frame.innerHTML, "<section>Validation report</section>");
  assert.deepEqual(settled, [["cv", "done"]]);
});

// ── A fold with no value at a level, and telling the folds apart ──

/** Fold dots as [fold, level, cx], and each level label's x, the level's centre. */
function dotPlaces(markup) {
  const dots = [...markup.matchAll(/<circle class="cv-fold-dot" data-fold="(\d+)" data-level="(\d+)"\s+cx="([-\d.]+)"/g)]
    .map((match) => [Number(match[1]), Number(match[2]), Number(match[3])]);
  const centres = [...markup.matchAll(/<text class="cv-level" x="([-\d.]+)"/g)].map((match) => Number(match[1]));
  return { dots, centres };
}

const threeFolds = Object.freeze({
  ...brand,
  folds: [
    { fold: 0, label: "Fold 1", values: [1.1, 0.9, 1.0] },
    { fold: 1, label: "Fold 2", values: [null, 0.8, 1.05] },
    { fold: 2, label: "Fold 3", values: [1.0, 0.95, 0.98] }
  ]
});

test("a fold with no value at a level leaves a gap there, not a mark", () => {
  const markup = levelChartMarkup(threeFolds);
  const { dots } = dotPlaces(markup);

  assert.equal(dots.length, 8);
  assert.deepEqual(dots.filter(([, level]) => level === 0).map(([fold]) => fold), [0, 2]);
  // The whisker at that level spans the folds that have a value there.
  assert.equal(count(markup, /class="cv-range"/g), 3);
  assert.doesNotMatch(markup, /NaN|null|undefined|Infinity/);
  assert.deepEqual(
    foldEnvelope({ ...brand, folds: [{ values: [1, null, 2] }, { values: [1.5, null, 2.5] }] }),
    { lo: [1, null, 2], hi: [1.5, null, 2.5] }
  );
  const noLevel = levelChartMarkup({ ...brand, folds: [{ fold: 0, label: "Fold 1", values: [1.1, null, 1.0] }] });
  assert.equal(count(noLevel, /class="cv-range"/g), 2);
  assert.doesNotMatch(noLevel, /NaN|null|undefined|Infinity/);

  const curve = curveChartMarkup({
    ...age,
    folds: [{ fold: 0, label: "Fold 1", values: [1.4, null, 0.9, 1.1] }, { ...age.folds[1], fold: 1 }]
  });
  const line = /<path class="cv-fold-line" data-fold="0"[^>]*\sd="([^"]+)"/.exec(curve)[1];
  assert.equal(count(line, /M/g), 2);
  assert.doesNotMatch(curve, /NaN|null|undefined|Infinity/);
});

test("each fold keeps its own place within every level, fold 1 leftmost", () => {
  const { dots, centres } = dotPlaces(levelChartMarkup(threeFolds));
  const offsets = new Map();
  for (const [fold, level, cx] of dots) {
    offsets.set(fold, [...(offsets.get(fold) ?? []), cx - centres[level]]);
  }
  const place = [0, 1, 2].map((fold) => offsets.get(fold)[0]);
  // The same offset at every level, rounding to 0.1 px aside, even beside
  // fold 2's gap at the first level.
  for (const [fold, values] of offsets) {
    for (const value of values) assert.ok(Math.abs(value - place[fold]) <= 0.15, `fold ${fold}: ${values}`);
  }
  assert.ok(place[0] < place[1] && place[1] < place[2], `offsets ${place}`);

  // A fold missing from the term keeps its own place and colour.
  const missing = levelChartMarkup({ ...threeFolds, folds: [threeFolds.folds[0], threeFolds.folds[2]] });
  const third = dotPlaces(missing).dots.filter(([fold]) => fold === 2);
  assert.ok(third.every(([, level, cx]) => Math.abs(cx - centres[level] - place[2]) <= 0.15));
  assert.match(missing, /data-fold="2" data-level="0"[^>]*style="fill: var\(--trace-2\)"/);
  assert.doesNotMatch(missing, /var\(--trace-1\)/);
});

test("the legend names each fold, and its marks carry the fold they belong to", () => {
  for (const markup of [levelChartMarkup(threeFolds), curveChartMarkup({
    ...age,
    folds: age.folds.map((fold, index) => ({ ...fold, fold: index }))
  })]) {
    const keys = [...markup.matchAll(/<button type="button" class="cv-fold-key" data-fold="(\d+)"[^>]*>[\s\S]*?<\/span>([^<]+)<\/button>/g)]
      .map((match) => [Number(match[1]), match[2].trim()]);
    const marks = [...markup.matchAll(/class="cv-fold-(?:dot|line)" data-fold="(\d+)"/g)].map((match) => Number(match[1]));
    const folds = keys.map(([fold]) => fold);
    assert.deepEqual(keys.map(([, label]) => label), folds.map((fold) => `Fold ${fold + 1}`));
    assert.ok(marks.length > 0 && marks.every((fold) => folds.includes(fold)));
    assert.match(markup, /class="cv-legend"[^>]*aria-label="[^"]*fold[^"]*"/i);
  }
});

test("picking out a fold lights its marks and dims the other folds'", () => {
  function mark(fold) {
    const classes = new Set();
    return {
      getAttribute: (name) => (name === "data-fold" ? fold : null),
      classList: { toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)) },
      classes
    };
  }
  const marks = ["0", "1", "2", "1"].map(mark);
  const root = {
    querySelectorAll(selector) {
      assert.equal(selector, "[data-fold]");
      return marks;
    }
  };

  highlightFold(root, "1");
  assert.deepEqual(marks.map((node) => [...node.classes].sort()), [["is-dimmed"], ["is-lit"], ["is-dimmed"], ["is-lit"]]);
  highlightFold(root, null);
  assert.deepEqual(marks.map((node) => [...node.classes]), [[], [], [], []]);
});

test("a fold's colour follows its number in the cards and the fold table", () => {
  const markup = cvTabMarkup(cvPayload(), idle());
  const table = markup.slice(markup.indexOf('aria-label="Fold scores"'), markup.indexOf("</table>"));
  assert.match(table, /background: var\(--trace-1\)"><\/span>Fold 2/);
  assert.match(markup, /fill: var\(--trace-1\)"><title>Fold 2: 0\.3000<\/title>/);
});

test("a term whose spread could not be measured reads as --", () => {
  const markup = termListMarkup([{ ...brand, spread: null, min_correlation: null }], "VehBrand", "");
  assert.equal(count(markup, /<span>--<\/span>/g), 2);
});

test("a hand-edited term held on every fold reads as held, and says why on hover", () => {
  const held = { ...age, held: true, spread: null, min_correlation: null };
  const [measured, edited] = termListMarkup([brand, held], "VehBrand", "").split("<button").slice(1);
  assert.match(measured, /<span>0\.050<\/span><span>0\.90<\/span>/);
  assert.match(edited, /<span class="cv-term-held" data-popover-title="Hand-edited"\s+data-popover-body="The same curve on every fold\.">held<\/span>/);
  assert.doesNotMatch(edited, /--|\d\.\d/);
});

test("the exposure behind a curve is smooth where the rows sit on whole units", () => {
  // A grid twice as fine as the data: every other point holds no rows.
  const xs = Array.from({ length: 121 }, (_unused, index) => 18 + index / 2);
  const weights = xs.map((_x, index) => (index % 2 ? 0 : 1));
  const inner = exposureProfile(xs, weights).slice(20, -20);
  assert.ok(Math.max(...inner) / Math.min(...inner) < 1.01, `${Math.min(...inner)} to ${Math.max(...inner)}`);
  assert.deepEqual(exposureProfile([5], [2]), [2]);
});

test("short level labels stay level, long ones turn", () => {
  /** @param {string[]} levels */
  const term = (levels) => ({
    ...brand,
    levels,
    weights: levels.map(() => 1),
    fit: levels.map(() => 1),
    folds: [{ fold: 0, label: "Fold 1", values: levels.map(() => 1) }]
  });
  // Eleven brands, B1 to B14, as on the board: each label fits its level.
  const brands = ["B1", "B10", "B11", "B12", "B13", "B14", "B2", "B3", "B4", "B5", "B6"];
  assert.doesNotMatch(levelChartMarkup(term(brands)), /rotate\(/);
  const regions = Array.from({ length: 40 }, (_unused, index) => `Region ${index}`);
  assert.equal(count(levelChartMarkup(term(regions)), /rotate\(/g), 40);
});
