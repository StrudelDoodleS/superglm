// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { CHECK_REPORTS } from "../../src/superglm/editor/app/api/contracts.js";
import { countText } from "../../src/superglm/editor/app/reports.js";
import { createInitialEditorState } from "../../src/superglm/editor/app/state/store.js";
import {
  checkFace,
  compareSwitch,
  flaggedChecks
} from "../../src/superglm/editor/app/views/assess_check.js";
import { CHECK_FIXTURE as FIXTURE, withCurrentChecks } from "./check_evidence.js";
import { neutralCopy } from "./neutral_copy.js";

function snapshot() {
  return {
    model_revision: 0, selected_term: "x",
    terms: { x: { kind: "spline", term_type: "spline", x: [1], y: [1], original_y: [1], previous_y: null,
      levels: null, n_points: 1, controls: null, group_display: null, impact: {},
      shape: { available: true, reason: null, ranges: [], support: null, specials: [] } } },
    selection: { x: [0] }, undo_redo: { undo: null, redo: null }, timeline: [{ kind: "marker" }],
    in_force_is_original: true, checks: [...CHECK_REPORTS]
  };
}

/** The page's evidence with every check of `reports` current. */
function evidenceFrom(reports) {
  return withCurrentChecks(createInitialEditorState(snapshot()).request.evidence, reports);
}

test("each card face counts what left its band and names one figure", () => {
  const calm = FIXTURE.calm;
  const qq = calm.check_qq.models.edited;
  assert.deepEqual(checkFace("check_qq", calm.check_qq, "current"),
    { value: `${countText(qq.outside)} of ${countText(qq.n)} points outside the band`, note: "Validation, 200 rows", flagged: qq.outside > 0 });
  const pit = calm.check_pit.models.edited;
  assert.equal(checkFace("check_pit", calm.check_pit, "current").note,
    `Bins ${Math.min(...pit.counts)} to ${Math.max(...pit.counts)} against ${pit.band_lower} to ${pit.band_upper}`);
  assert.equal(checkFace("check_qq", FIXTURE.wrong.check_qq, "current").flagged, true);
});

test("the four faces read their counts, the overall ratio and the 99th percentile's share", () => {
  const wrong = FIXTURE.wrong;
  const qq = wrong.check_qq.models.edited;
  // The flag counts all n, not the points sent to draw.
  assert.ok(qq.index.length < qq.n);
  assert.deepEqual(checkFace("check_qq", wrong.check_qq, "current"),
    { value: `${countText(qq.outside)} of 1,000 points outside the band`, note: "Validation, 1,000 rows", flagged: true });
  const pit = wrong.check_pit.models.edited;
  assert.deepEqual(checkFace("check_pit", wrong.check_pit, "current").value, `${pit.outside} of 20 bins outside the band`);
  const ae = FIXTURE.calm.check_ae.models.edited;
  assert.deepEqual(checkFace("check_ae", FIXTURE.calm.check_ae, "current"),
    { value: `0 of ${ae.cells} groups outside their interval`, note: `Overall ${ae.overall.ratio.toFixed(2)}`, flagged: false });
  const q = wrong.check_calibration.models.edited.quantiles;
  const share = q.realised_exceedance[q.p.indexOf(0.99)];
  assert.deepEqual(checkFace("check_calibration", wrong.check_calibration, "current"), {
    value: `${wrong.check_calibration.models.edited.outside} of 11 quantiles outside the band`,
    note: `99th percentile exceeded by ${(100 * share).toFixed(1)}%`,
    flagged: true
  });
  // Edited by hand: the face reads the model as edited, not the one fitted beside it.
  const edited = FIXTURE.edited.check_ae.models.edited;
  assert.equal(checkFace("check_ae", FIXTURE.edited.check_ae, "current").note, `Overall ${edited.overall.ratio.toFixed(2)}`);
  const off = structuredClone(FIXTURE.calm.check_ae);
  off.models.edited.overall.ratio = null;
  assert.equal(checkFace("check_ae", off, "current").note, "Overall —");
});

test("a check with no answer for this model says why, with no flag", () => {
  const off = { ...FIXTURE.calm.check_ae, available: false, reason: "R", models: { edited: { available: false, reason: "R", flagged: false, outside: 0, cells: 0 }, original: null } };
  assert.deepEqual(checkFace("check_ae", off, "current"), { value: "—", note: "R", flagged: false });
  assert.deepEqual(checkFace("check_ae", null, "updating"), { value: "—", note: "Loading check...", flagged: false });
  assert.deepEqual(checkFace("check_ae", null, "error"), { value: "—", note: "Check unavailable.", flagged: false });
  // Unavailable for this model alone: the model's own reason, and a flag it carries does not show.
  const body = { ...FIXTURE.wrong.check_qq, models: { edited: { available: false, reason: "Only one row.", flagged: true, outside: 0, cells: 0 }, original: null } };
  assert.deepEqual(checkFace("check_qq", body, "current"), { value: "—", note: "Only one row.", flagged: false });
});

test("every face is in neutral words", () => {
  for (const [name, reports] of Object.entries(FIXTURE)) {
    for (const report of CHECK_REPORTS) {
      const face = checkFace(report, reports[report], "current");
      assert.deepEqual(neutralCopy(`${face.value} ${face.note}`), [], `${name} ${report}`);
    }
  }
});

test("the flagged checks are the offered ones whose model as edited left its band", () => {
  const wrong = evidenceFrom(FIXTURE.wrong);
  assert.deepEqual(flaggedChecks(wrong, CHECK_REPORTS), CHECK_REPORTS.filter((name) => FIXTURE.wrong[name].models.edited.flagged));
  assert.deepEqual(flaggedChecks(wrong, ["check_ae", "check_qq"]), ["check_qq"]);
  assert.deepEqual(flaggedChecks(evidenceFrom(FIXTURE.calm), CHECK_REPORTS), []);
  assert.deepEqual(flaggedChecks({ ...wrong, check_qq: { ...wrong.check_qq, payload: null } }, ["check_qq"]), []);
});

test("Compare with the fit is live only when an offered check compares an edit", () => {
  const calm = compareSwitch(evidenceFrom(FIXTURE.calm), CHECK_REPORTS, true);
  assert.deepEqual(calm, { live: false, checked: false, body: "The model is as fitted; there is no edit to compare." });
  const edited = evidenceFrom(FIXTURE.edited);
  assert.deepEqual(compareSwitch(edited, CHECK_REPORTS, false), {
    live: true, checked: false,
    body: "Draws the model as fitted in grey beside the model as edited in blue, on every check. It changes only what is drawn."
  });
  assert.equal(compareSwitch(edited, CHECK_REPORTS, true).checked, true);
  // A model offering no checks has nothing to compare, whatever the evidence holds.
  assert.equal(compareSwitch(edited, [], true).live, false);
  assert.deepEqual(neutralCopy(compareSwitch(edited, CHECK_REPORTS, true).body), []);
});
