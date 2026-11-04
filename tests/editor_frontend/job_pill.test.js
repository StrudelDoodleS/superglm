// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { jobPillState, pillJob } from "../../src/superglm/editor/app/views/job_pill.js";

function job(overrides = {}) {
  return {
    job_id: "cv-1",
    kind: "cv",
    status: "running",
    elapsed_s: 62.7,
    started_at: 10,
    result: null,
    progress: [
      { phase: "fold", fold: 1, n_folds: 5 },
      { phase: "fold_done", fold: 1, n_folds: 5 },
      { phase: "fold", fold: 2, n_folds: 5 },
    ],
    ...overrides,
  };
}

test("a running Run CV names its fold and time, with a tick for each fold", () => {
  const state = jobPillState(job(), false);
  assert.equal(state.text, "Run CV: fold 2 of 5");
  assert.equal(state.time, "1:02");
  assert.equal(state.tone, "running");
  assert.equal(state.view, "cv");
  assert.deepEqual(state.ticks.map((tick) => tick.fill), ["done", "now", "waiting", "waiting", "waiting"]);
});

test("each stage of a run reads as what it is doing", () => {
  assert.equal(jobPillState(job({ progress: [] }), false).text, "Run CV: starting");
  const fiveDone = Array.from({ length: 5 }, (_, i) => ({ phase: "fold_done", fold: i + 1, n_folds: 5 }));
  assert.equal(jobPillState(job({ progress: fiveDone }), false).text, "Run CV: reading the fold curves");
  assert.equal(jobPillState(job({ cancel_requested: true }), false).text, "Run CV: cancelling");
  const final = jobPillState(job({ kind: "final_fit", job_id: "final_fit-1", progress: [{ phase: "fitting", n_rows: 9 }] }), false);
  assert.equal(final.text, "Final fit: fitting all rows");
  assert.deepEqual(final.ticks, []);
  assert.equal(final.view, "final");
});

test("a finished run says how it ended until its tab has been opened", () => {
  const done = job({ status: "done", result: { headline: { label: "Log score", value: 0.208751 } } });
  assert.equal(jobPillState(done, false).text, "Run CV finished: Log score 0.2088");
  assert.equal(jobPillState(done, false).tone, "finished");
  assert.equal(jobPillState(done, true), null);
  const large = job({ status: "done", result: { headline: { label: "CRPS", value: 1007.691 } } });
  assert.equal(jobPillState(large, false).text, "Run CV finished: CRPS 1007.69");
  assert.equal(jobPillState(job({ status: "done" }), false).text, "Run CV finished");
});

test("a failed job sends you to the tab that says why, and a cancelled one leaves no pill", () => {
  const failed = jobPillState(job({ kind: "final_fit", status: "failed" }), false);
  assert.equal(failed.text, "Final fit failed");
  assert.equal(failed.tone, "failed");
  assert.equal(failed.view, "cv");
  assert.equal(jobPillState(job({ status: "cancelled" }), false), null);
  assert.equal(jobPillState(null, false), null);
});

test("many folds drop the ticks, which could no longer be counted", () => {
  const progress = [{ phase: "fold", fold: 1, n_folds: 25 }];
  const state = jobPillState(job({ progress }), false);
  assert.deepEqual(state.ticks, []);
  assert.equal(state.text, "Run CV: fold 1 of 25");
});

test("the pill follows the running job, else the one that started last", () => {
  const cv = job({ status: "done", started_at: 20 });
  const final = job({ kind: "final_fit", job_id: "final_fit-1", status: "running", started_at: 10 });
  assert.equal(pillJob({ cv, final_fit: final }), final);
  assert.equal(pillJob({ cv, final_fit: { ...final, status: "done" } }), cv);
  assert.equal(pillJob({ cv: null, final_fit: null }), null);
});
