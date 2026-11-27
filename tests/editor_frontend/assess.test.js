// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import { REPORT_NAMES } from "../../src/superglm/editor/app/api/contracts.js";
import {
  ASSESS_GROUPS,
  OVERVIEW_ENTRY,
  assessEntry,
  openableView,
  tabOf,
} from "../../src/superglm/editor/app/state/assess.js";
import { HELP_SECTIONS } from "../../src/superglm/editor/app/views/help_content.js";

test("every entry is a report the page can ask for, once, in list order", () => {
  const ids = ASSESS_GROUPS.flatMap((group) => group.entries.map((entry) => entry.id));
  assert.deepEqual(ids, [...REPORT_NAMES]);
  assert.deepEqual(ASSESS_GROUPS.flatMap((g) => g.entries.map((e) => e.label)),
    ["Validation", "Cross-validation", "Final fit", "Smoothing search"]);
  for (const entry of [OVERVIEW_ENTRY, ...ASSESS_GROUPS[0].entries]) {
    assert.match(entry.icon, /^M/); assert.ok(entry.hint.endsWith("."));
  }
});

test("each view has its tab, and an unknown or Editor view in an Assess window is mapped", () => {
  assert.deepEqual(["editor", "assess", "validation", "cv", "final", "diagnose", "risk"].map(tabOf),
    ["editor", "assess", "assess", "assess", "assess", "assess", "risk"]);
  assert.equal(openableView("bogus", { assessWindow: false }), "editor");
  assert.equal(openableView("cv", { assessWindow: false }), "cv");
  assert.equal(openableView("editor", { assessWindow: true }), "assess");
  assert.equal(openableView("risk", { assessWindow: true }), "assess");
});

test("an entry is found by its view, and a view outside Assess has none", () => {
  assert.equal(assessEntry("assess"), OVERVIEW_ENTRY);
  assert.equal(assessEntry("diagnose")?.label, "Smoothing search");
  assert.equal(assessEntry("editor"), null);
  assert.equal(assessEntry("risk"), null);
});

test("Help names Assess and no removed tab", () => {
  const text = JSON.stringify(HELP_SECTIONS);
  assert.doesNotMatch(text, /(Validation|Cross-validation|Final Fit|Final fit|Diagnose) tab/);
  assert.ok(HELP_SECTIONS.some((s) => s.title === "Assess"));
  assert.ok(HELP_SECTIONS.some((s) => s.title === "Smoothing search"));
});
