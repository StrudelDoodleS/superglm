// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  FINAL_FIT_RUN_HINT,
  finalFitFailure,
  finalFitRunState,
} from "../../src/superglm/editor/app/views/final_fit_run.js";

const ready = { available: true, reason: null, note: null, done: false, stale: false, n_rows: null };

test("a final fit that can run enables the button with the standard hover text", () => {
  assert.deepEqual(finalFitRunState({ final: ready }), {
    disabled: false, hint: FINAL_FIT_RUN_HINT, note: "",
  });
});

test("a final fit that cannot run disables the button and says why on hover", () => {
  const state = finalFitRunState({ final: { ...ready, available: false, reason: "No train or validation rows are kept." } });
  assert.equal(state.disabled, true);
  assert.equal(state.hint, "No train or validation rows are kept.");
});

test("a running job disables the button; an unknown plan does not", () => {
  assert.equal(finalFitRunState({ final: ready, running: true }).disabled, true);
  assert.equal(finalFitRunState({ final: null }).disabled, false);
});

test("a refused start is told as what failed and what to do, and keeps the button usable", () => {
  const state = finalFitRunState({ final: ready, error: "The fit stopped at fold 2" });
  assert.equal(state.disabled, false);
  assert.equal(state.note, "Final fit could not run. The fit stopped at fold 2. Change the model if needed, then run it again.");
  assert.equal(
    finalFitRunState({ final: ready, error: "Not enough rows." }).note,
    "Final fit could not run. Not enough rows. Change the model if needed, then run it again.",
  );
});

test("a failed or cancelled job is told; a finished or running one says nothing", () => {
  assert.equal(
    finalFitFailure({ status: "failed", error: "Solver did not converge." }),
    "Solver did not converge.",
  );
  assert.equal(finalFitFailure({ status: "failed", error: null }), "The editor hit an internal error.");
  assert.equal(finalFitFailure({ status: "cancelled" }), "The job was cancelled and nothing was kept.");
  assert.equal(finalFitFailure({ status: "done" }), "");
  assert.equal(finalFitFailure({ status: "running" }), "");
});
