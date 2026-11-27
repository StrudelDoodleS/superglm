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
