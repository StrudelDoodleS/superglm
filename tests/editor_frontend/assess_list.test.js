// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { REPORT_VIEWS } from "../../src/superglm/editor/app/api/contracts.js";
import { assessListMarkup, stepEntry } from "../../src/superglm/editor/app/views/assess_list.js";

test("the list marks the open entry and keeps one tab stop", () => {
  const html = assessListMarkup("diagnose");
  assert.equal((html.match(/aria-current="true"/g) ?? []).length, 1);
  assert.match(html, /data-assess-entry="diagnose"[^>]*aria-current="true"[^>]*tabindex="0"/);
  assert.equal((html.match(/tabindex="0"/g) ?? []).length, 1);
  assert.match(html, /data-popover-title="Smoothing search"/);
});

test("arrow keys, Home and End step through the entries", () => {
  const ids = ["assess", ...REPORT_VIEWS];
  assert.equal(stepEntry(ids, "assess", "ArrowDown"), "validation");
  assert.equal(stepEntry(ids, "assess", "ArrowUp"), "assess");
  assert.equal(stepEntry(ids, "cv", "End"), "diagnose");
  assert.equal(stepEntry(ids, "cv", "Home"), "assess");
  assert.equal(stepEntry(ids, "cv", "a"), null);
});

test("the list hides checks the model does not offer and marks flagged ones", () => {
  const markup = assessListMarkup("assess", { offered: ["check_qq"], flagged: ["check_qq"] });
  assert.match(markup, /data-assess-entry="check_qq"[^>]*data-flagged="true"/);
  assert.match(markup, /data-assess-entry="check_pit"[^>]*hidden/);
  assert.match(markup, /data-assess-group="calibration"[^>]*hidden/);
  assert.match(markup, /data-assess-entry="check_qq"[^>]*aria-label="Q-Q plot, flagged"/);
  assert.doesNotMatch(markup, /data-assess-group="distribution"[^>]*hidden/);
  assert.doesNotMatch(markup, /data-assess-entry="check_qq"[^>]*hidden/);
  // A row neither hidden nor flagged keeps its own name.
  assert.match(markup, /data-assess-entry="validation"[^>]*data-flagged="false"[^>]*aria-label="Validation"/);
  const none = assessListMarkup("assess");
  assert.equal((none.match(/ hidden/g) ?? []).length, 6);
});
