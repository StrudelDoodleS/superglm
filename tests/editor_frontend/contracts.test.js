import assert from "node:assert/strict";
import test from "node:test";

import {
  CHECK_REPORTS,
  EVIDENCE_PANELS,
  REPORT_NAMES,
  REPORT_VIEWS,
  createEmptyEvidenceState,
  isCheckReport,
  isReportName,
  isReportView
} from "../../src/superglm/editor/app/api/contracts.js";

test("frontend tests execute as native ES modules", () => {
  assert.deepEqual({ runner: "node", modules: "esm" }, {
    runner: "node",
    modules: "esm"
  });
});

test("foundation contracts define all evidence panels", () => {
  assert.deepEqual(EVIDENCE_PANELS, [
    "metrics", "summary", "validation", "cv", "final", "diagnose",
    "check_pit", "check_ae", "check_calibration", "check_qq"
  ]);
  assert.equal(Object.isFrozen(EVIDENCE_PANELS), true);
  assert.deepEqual(createEmptyEvidenceState(), {
    status: "idle",
    revision: null,
    inForce: null,
    sequence: 0,
    payload: null,
    error: null,
    retry: null
  });

  const first = createEmptyEvidenceState();
  const second = createEmptyEvidenceState();
  assert.notStrictEqual(first, second);
  first.status = "error";
  assert.equal(second.status, "idle");
});

test("every panel asked through /report is a report view or a check, never both", () => {
  assert.deepEqual([...REPORT_NAMES], [...REPORT_VIEWS, ...CHECK_REPORTS]);
  for (const names of [REPORT_VIEWS, CHECK_REPORTS, REPORT_NAMES]) assert.equal(Object.isFrozen(names), true);
  for (const name of REPORT_VIEWS) {
    assert.ok(isReportName(name)); assert.ok(isReportView(name)); assert.ok(!isCheckReport(name));
  }
  for (const value of ["metrics", "assess", "check", "", null, undefined, 3]) {
    assert.ok(!isReportName(value)); assert.ok(!isReportView(value)); assert.ok(!isCheckReport(value));
  }
});
