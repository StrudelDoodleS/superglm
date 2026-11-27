// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { createInitialEditorState } from "../../src/superglm/editor/app/state/store.js";
import {
  cvCard,
  finalCard,
  overviewCards,
  overviewGroups,
  overviewMarkup,
  smoothingCard,
  validationCard
} from "../../src/superglm/editor/app/views/assess_overview.js";

// Trimmed from a SuperLSS session's /report (demo 8767) and a bare Poisson GLM's.
const LSS_LABELS = { log_score: "Log score", crps: "CRPS", actual_expected: "Actual / expected" };
const LSS_TRAIN = {
  name: "train", label: "Train", n_obs: 1342,
  metrics: { edited: { log_score: 10.412507243415122, crps: 8304.057816712328, actual_expected: 0.9884475876742346 } }
};
const LSS_VALIDATION_SPLIT = {
  name: "validation", label: "Validation", n_obs: 331,
  metrics: { edited: { log_score: 10.388618691811986, crps: 7790.3547609357865, actual_expected: 0.915438108068378 } }
};
const LSS_VALIDATION = {
  available: true, report: "validation",
  note: "Each split scored by the model as edited. Δ is the change from the last fit.",
  metric_labels: LSS_LABELS, compares_edit: false, splits: [LSS_TRAIN, LSS_VALIDATION_SPLIT]
};
const TRAIN_ONLY_NOTE = "No explicit split data supplied; metrics are in-sample on retained fit data.";
const GLM_TRAIN_ONLY = {
  available: true, report: "validation", note: TRAIN_ONLY_NOTE,
  metric_labels: {
    deviance: "Deviance", aic: "AIC", bic: "BIC", log_likelihood: "Log Likelihood",
    explained_deviance: "Explained Deviance", pearson_chi2: "Pearson Chi2", effective_df: "Total EDF"
  },
  compares_edit: false,
  splits: [{ name: "train", label: "Train", n_obs: 800, metrics: { edited: { deviance: 512.3, aic: 1604.5 } } }]
};
const NO_DATA = "No retained fit data is available for metrics.";
// A split's refusal, as lss_split_reason words it and split_metrics_payload sends it in "error".
const NO_QUOTE = "The Validation split cannot be scored with its weights: the model's family has no likelihood for weights like these. Pass the split without weights to score it.";
const LSS_CV = {
  available: true, report: "cv", note: "",
  header: { supplied: true, n_folds: 5, splitter: "KFold", n_rows: 1342 },
  metrics: [
    { name: "nll", label: "Log score", short: "Log score", lower_is_better: true },
    { name: "crps", label: "CRPS", short: "CRPS", lower_is_better: true }
  ],
  final_fit: { available: true, reason: null, note: null, done: false, stale: false, n_rows: null, has_test_split: false },
  results: [{
    label: "As supplied", origin: "supplied", model_revision: null, stale: false, held_out: true,
    folds: [0, 1, 2, 3, 4].map((fold) => ({ fold })),
    mean: { nll: 10.418178314830914, crps: 8348.629629222844 },
    std: { nll: 0.04863675282064414, crps: 296.53517660807745 }
  }]
};
const NO_CV = {
  ...LSS_CV,
  note: "No cross-validation result was supplied. Pass cv=cross_validate(..., return_estimators=True) to edit().",
  header: { supplied: false, n_folds: 0, splitter: null, n_rows: null },
  results: []
};
const NO_FINAL_ROWS = "Final fit needs the rows the model was fitted on.";
// final_fit.note on a book with no validation rows, before and after a final fit (bare GLM).
const TRAIN_ONLY_FINAL = "No validation data was supplied, so Final fit uses the train rows only.";
function trainOnly(payload) {
  return { ...payload, final_fit: { ...payload.final_fit, note: TRAIN_ONLY_FINAL } };
}

function withDelta(delta) {
  return { ...LSS_VALIDATION_SPLIT, metrics: { ...LSS_VALIDATION_SPLIT.metrics, delta: { log_score: delta } } };
}
function staleRun(payload) {
  const run = { ...payload.results[0], label: "Current model", origin: "run", stale: true };
  return { ...payload, results: [...payload.results, run] };
}
function finalDone(payload, rows, stale) {
  return { ...payload, final_fit: { ...payload.final_fit, done: true, stale, n_rows: rows } };
}
function refused(payload, reason) {
  return { ...payload, final_fit: { ...payload.final_fit, available: false, reason } };
}
const VERDICTS = {
  converged: "Converged. The smoothing search passed its own check.",
  loose: "Converged loosely. The search stopped near its tolerance.",
  not_converged: "Did not converge. The smoothing search stopped before its check passed.",
  fixed: "No smoothing search. The smoothing is held where REML chose it."
};
function fits(statuses) {
  return {
    report: "diagnose", available: true, note: "Folds from the cross-validation result you supplied.",
    fits: statuses.map((status, i) => ({ id: `fit-${i}`, label: `Fit ${i}`, status, verdict: VERDICTS[status] }))
  };
}
function snapshot(revision = 0) {
  return {
    model_revision: revision, selected_term: "age",
    terms: {
      age: {
        kind: "spline", term_type: "spline", x: [1], y: [1], original_y: [1], previous_y: null,
        levels: null, n_points: 1, controls: null, group_display: null, impact: {},
        shape: { available: true, reason: null, ranges: [], support: null, specials: [] }
      }
    },
    selection: { age: [0] }, undo_redo: { undo: null, redo: null }, timeline: [{ kind: "marker" }],
    in_force_is_original: true
  };
}

test("Validation reads the report's first score on the validation split", () => {
  assert.deepEqual(validationCard(LSS_VALIDATION),
    { value: "Log score 10.4", note: "Validation, 331 rows" });
  assert.deepEqual(validationCard({ ...LSS_VALIDATION, compares_edit: true, splits: [withDelta(-0.25)] }).note,
    "Validation, 331 rows · Δ -0.25");
  assert.deepEqual(validationCard(GLM_TRAIN_ONLY),
    { value: "Deviance 512.3", note: TRAIN_ONLY_NOTE });
  assert.deepEqual(validationCard({ available: false, splits: [], metric_labels: {}, note: NO_DATA }),
    { value: "—", note: NO_DATA });
});

test("a Validation split the model could not score shows the report's reason, with no change", () => {
  const refusedSplit = { name: "validation", label: "Validation", n_obs: 331, source: "validation", available: false, error: NO_QUOTE };
  assert.deepEqual(validationCard({ ...LSS_VALIDATION, splits: [LSS_TRAIN, refusedSplit] }),
    { value: "—", note: NO_QUOTE });
  assert.deepEqual(validationCard({ ...LSS_VALIDATION, compares_edit: true, splits: [LSS_TRAIN, refusedSplit] }),
    { value: "—", note: NO_QUOTE });
  // No split scored at all: the report names no score column, and the refusal still shows.
  const unscoredTrain = {
    ...LSS_TRAIN, metrics: { edited: { log_score: null, crps: null, actual_expected: null } }
  };
  assert.deepEqual(validationCard({ ...LSS_VALIDATION, splits: [unscoredTrain, refusedSplit] }),
    { value: "—", note: NO_QUOTE });
});

test("Cross-validation reads the newest result's first score", () => {
  assert.deepEqual(cvCard(LSS_CV), { value: "Log score 10.4182 ± 0.0486", note: "As supplied · 5 folds" });
  assert.deepEqual(cvCard(staleRun(LSS_CV)).note, "Current model, earlier model · 5 folds");
  assert.deepEqual(cvCard(NO_CV), { value: "—", note: "No cross-validation result supplied" });
});

test("Final fit reads the cv report's final_fit", () => {
  assert.deepEqual(finalCard(LSS_CV), { value: "Not run", note: "Run Final fit on all rows from Cross-validation." });
  assert.deepEqual(finalCard(finalDone(LSS_CV, 400, false)), { value: "400 rows", note: "Refitted on train and validation rows." });
  assert.equal(finalCard(finalDone(LSS_CV, 400, true)).note, "The model changed after the final fit.");
  assert.equal(finalCard(refused(LSS_CV, NO_FINAL_ROWS)).note, NO_FINAL_ROWS);
});

test("Final fit on a book with no validation rows says it refits on the train rows", () => {
  assert.deepEqual(finalCard(trainOnly(LSS_CV)), { value: "Not run", note: TRAIN_ONLY_FINAL });
  const done = finalCard(trainOnly(finalDone(LSS_CV, 800, false)));
  assert.deepEqual(done, { value: "800 rows", note: TRAIN_ONLY_FINAL });
  assert.doesNotMatch(done.note, /train and validation/);
  // The model moving past the final fit, and a refusal, say so over the note.
  assert.equal(finalCard(trainOnly(finalDone(LSS_CV, 800, true))).note, "The model changed after the final fit.");
  assert.equal(finalCard(refused(trainOnly(LSS_CV), NO_FINAL_ROWS)).note, NO_FINAL_ROWS);
});

test("Smoothing search counts the fits", () => {
  assert.equal(smoothingCard(fits(["converged", "converged", "loose"])).value, "3 of 3 fits converged");
  assert.equal(smoothingCard(fits(["converged", "not_converged", "converged"])).value, "1 of 3 fits did not converge");
  assert.equal(smoothingCard(fits(["fixed"])).value, "No smoothing search.");
  assert.equal(smoothingCard(fits(["converged"])).value, "Converged.");
  assert.equal(smoothingCard(fits(["converged"])).note, "Folds from the cross-validation result you supplied.");
});

test("a card with nothing yet says it is loading, or unavailable", () => {
  const evidence = createInitialEditorState(snapshot(0)).request.evidence;
  const cards = overviewCards(evidence);
  assert.deepEqual(cards.map((c) => [c.id, c.value, c.note]), [
    ["validation", "—", "Loading report..."], ["cv", "—", "Loading report..."],
    ["final", "—", "Loading report..."], ["diagnose", "—", "Loading report..."]]);
  const failed = overviewCards({ ...evidence, cv: { ...evidence.cv, status: "error", error: "boom" } });
  assert.deepEqual(failed.filter((c) => c.id === "cv" || c.id === "final").map((c) => [c.value, c.note, c.freshness]), [
    ["—", "Report unavailable.", "error"], ["—", "Report unavailable.", "error"]]);
});

test("a card shows its report's payload, and Final fit takes the freshness of the cv report", () => {
  const evidence = createInitialEditorState(snapshot(0)).request.evidence;
  const cards = overviewCards({
    ...evidence,
    validation: { ...evidence.validation, status: "current", payload: LSS_VALIDATION },
    cv: { ...evidence.cv, status: "stale", payload: finalDone(LSS_CV, 400, false) },
    diagnose: { ...evidence.diagnose, status: "updating", payload: fits(["converged", "loose"]) }
  });
  assert.deepEqual(cards.map((c) => [c.id, c.value, c.freshness]), [
    ["validation", "Log score 10.4", "current"],
    ["cv", "Log score 10.4182 ± 0.0486", "stale"],
    ["final", "400 rows", "stale"],
    ["diagnose", "2 of 2 fits converged", "updating"]]);
});

test("a number from an earlier report says so in words, in sight where its update failed", () => {
  const evidence = createInitialEditorState(snapshot(0)).request.evidence;
  const held = (status, error = null) => ({ ...evidence.validation, status, payload: LSS_VALIDATION, error });
  const card = (validation) => overviewCards({ ...evidence, validation })[0];
  const face = (c) => [c.value, c.note, c.marker, c.markerShown];

  assert.deepEqual(face(card(held("current"))), ["Log score 10.4", "Validation, 331 rows", "", false]);
  // Asked again: the earlier number stays, muted, and a screen reader hears it is earlier.
  assert.deepEqual(face(card(held("updating"))),
    ["Log score 10.4", "Validation, 331 rows", "Earlier result, updating", false]);
  assert.deepEqual(face(card(held("stale"))),
    ["Log score 10.4", "Validation, 331 rows", "Earlier result, may be out of date", false]);
  // The update failed over a payload kept: the card says the number is earlier, and why.
  assert.deepEqual(face(card(held("stale", "Python stopped answering."))),
    ["Log score 10.4", "Update failed. Python stopped answering.", "Earlier result", true]);
  // A failure with nothing to keep, and a first request, show no number.
  assert.deepEqual(face(card({ ...evidence.validation, status: "error", error: "boom" })),
    ["—", "Report unavailable.", "", false]);
  assert.deepEqual(face(card(evidence.validation)), ["—", "Loading report...", "", false]);
});

test("the card's markup carries its marker, clipped unless it is shown", () => {
  const evidence = createInitialEditorState(snapshot(0)).request.evidence;
  const failed = { ...evidence.validation, status: "stale", payload: LSS_VALIDATION, error: "offline" };
  const updating = { ...evidence.cv, status: "updating", payload: LSS_CV };
  const markup = overviewMarkup(overviewGroups({ ...evidence, validation: failed, cv: updating }));
  assert.match(markup, /data-card="validation"[^]*?<span class="assess-card-marker" data-shown="true">Earlier result<\/span><span class="assess-card-note">Update failed\. offline<\/span>/);
  assert.match(markup, /data-card="cv"[^]*?<span class="assess-card-marker" data-shown="false">Earlier result, updating<\/span>/);
});
