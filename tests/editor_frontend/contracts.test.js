import assert from "node:assert/strict";
import test from "node:test";

import {
  EVIDENCE_PANELS,
  createEmptyEvidenceState
} from "../../src/superglm/editor/app/api/contracts.js";

test("frontend tests execute as native ES modules", () => {
  assert.deepEqual({ runner: "node", modules: "esm" }, {
    runner: "node",
    modules: "esm"
  });
});

test("foundation contracts define all evidence panels", () => {
  assert.deepEqual(EVIDENCE_PANELS, ["metrics", "summary", "report"]);
  assert.equal(Object.isFrozen(EVIDENCE_PANELS), true);
  assert.deepEqual(createEmptyEvidenceState(), {
    status: "idle",
    revision: null,
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
