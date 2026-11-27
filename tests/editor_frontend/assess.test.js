// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  CHECK_REPORTS,
  REPORT_NAMES,
  REPORT_VIEWS,
  isCheckReport,
  isReportName,
  isReportView,
} from "../../src/superglm/editor/app/api/contracts.js";
import {
  ASSESS_GROUPS,
  OVERVIEW_ENTRY,
  OVERVIEW_REPORTS,
  assessEntry,
  openableView,
  overviewReports,
  tabOf,
} from "../../src/superglm/editor/app/state/assess.js";
import {
  CHECK_HELP,
  CONTROL_HELP,
  HELP_SECTIONS,
  helpForElement,
} from "../../src/superglm/editor/app/views/help_content.js";
import { BANNED_WORDS, neutralCopy } from "./neutral_copy.js";

test("every entry is a report the page can ask for, once, in list order", () => {
  const ids = ASSESS_GROUPS.flatMap((group) => group.entries.map((entry) => entry.id));
  assert.deepEqual(ids, [...REPORT_VIEWS, "check_qq", "check_pit", "check_ae", "check_calibration"]);
  assert.deepEqual([...ids].sort(), [...REPORT_NAMES].sort());
  assert.deepEqual(ASSESS_GROUPS.flatMap((g) => g.entries.map((e) => e.label)),
    ["Validation", "Cross-validation", "Final fit", "Smoothing search",
      "Q-Q plot", "PIT histogram", "Actual over expected", "Quantile calibration"]);
  for (const entry of [OVERVIEW_ENTRY, ...ASSESS_GROUPS.flatMap((g) => g.entries)]) {
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

test("check reports are asked like reports and drawn by their own view", () => {
  assert.deepEqual([...CHECK_REPORTS], ["check_pit", "check_ae", "check_calibration", "check_qq"]);
  assert.deepEqual([...REPORT_VIEWS], ["validation", "cv", "final", "diagnose"]);
  for (const name of CHECK_REPORTS) {
    assert.ok(isReportName(name)); assert.ok(isCheckReport(name)); assert.ok(!isReportView(name));
    assert.equal(tabOf(name), "assess");
    assert.equal(openableView(name, { assessWindow: true }), name);
  }
});

test("the overview asks the reports, then each offered check, cheapest first", () => {
  assert.deepEqual([...OVERVIEW_REPORTS], ["validation", "cv", "diagnose"]);
  assert.deepEqual(overviewReports(["check_qq", "check_pit"]), ["validation", "cv", "diagnose", "check_pit", "check_qq"]);
  assert.deepEqual(overviewReports([...CHECK_REPORTS].reverse()),
    ["validation", "cv", "diagnose", "check_pit", "check_ae", "check_calibration", "check_qq"]);
  assert.deepEqual(overviewReports([]), ["validation", "cv", "diagnose"]);
});

test("Distribution and Calibration follow Reports, in neutral words", () => {
  assert.deepEqual(ASSESS_GROUPS.map((group) => group.name), ["Reports", "Distribution", "Calibration"]);
  assert.deepEqual(ASSESS_GROUPS.map((group) => group.key), ["reports", "distribution", "calibration"]);
  assert.deepEqual(ASSESS_GROUPS[1].entries.map((e) => [e.id, e.label]), [["check_qq", "Q-Q plot"], ["check_pit", "PIT histogram"]]);
  assert.deepEqual(ASSESS_GROUPS[2].entries.map((e) => [e.id, e.label]), [["check_ae", "Actual over expected"], ["check_calibration", "Quantile calibration"]]);
  assert.deepEqual(neutralCopy(JSON.stringify(ASSESS_GROUPS)), []);
});

test("each check's popover says what it shows, and its large view is headed by its label", () => {
  const checks = ASSESS_GROUPS.slice(1).flatMap((group) => group.entries);
  assert.deepEqual(Object.fromEntries(checks.map((e) => [e.id, e.hint])), {
    check_qq: "Sorted quantile residuals against their expected values, with a 95% band that holds for every point at once.",
    check_pit: "Where each observed value falls in its own predicted distribution, in 20 bins, with a 95% band that holds for every bin at once.",
    check_ae: "Observed total over predicted total, by band of predicted Mean and by each input, with 95% intervals that hold for every group at once.",
    check_calibration: "How often each predicted quantile is exceeded against how often it should be, with a 95% band that holds for every quantile at once.",
  });
  assert.deepEqual(Object.fromEntries(checks.map((e) => [e.id, e.icon])), {
    check_qq: "M4 20 20 4M6 17l2-1M10 14l2-1M14 10l2-2",
    check_pit: "M4 20V9h3v11M10.5 20V7h3v13M17 20V10h3v10M3 20h18",
    check_ae: "M3 12h18M6 8v8M12 5v8M18 10v6",
    check_calibration: "M4 20h16M4 20V4M7 17l3-4 3 1 3-5 2-3",
  });
  // A heading is a plain noun label: no entry carries a second, longer title.
  for (const entry of checks) assert.deepEqual(Object.keys(entry).sort(), ["hint", "icon", "id", "label"]);
  assert.equal(assessEntry("check_ae")?.label, "Actual over expected");
});

test("the banned-word check finds each word whole, in any case or plural, and passes neutral text", () => {
  assert.deepEqual(neutralCopy("Claims per POLICIES; burning  cost"), ["claim", "policy", "burning cost"]);
  assert.deepEqual(neutralCopy("Losses"), ["loss", "losses"]);
  assert.deepEqual(neutralCopy("A bookmark, a closed claimant, unpriced"), []);
  for (const word of BANNED_WORDS) assert.ok(neutralCopy(`the ${word.toUpperCase()} here`).includes(word), word);
});

const SHARED_ENDING = "Orange marks a check with something outside its band. Checks read the validation rows, else the training rows. "
  + "On validation rows the band allows for the uncertainty in the fitted parameters; on the training rows it takes them as exact, "
  + "so a validation split is the sharper test.";

/** A check's or a control's Help entry, as the Help pane shows it. */
const helpEntry = (id) => CHECK_HELP[id] ?? CONTROL_HELP[id];
const helpText = (id) => `${helpEntry(id).title} ${helpEntry(id).body}`;

test("Help for every check is in neutral words", () => {
  for (const id of [...CHECK_REPORTS, "compare_original"]) assert.deepEqual(neutralCopy(helpText(id)), [], id);
});

test("each check and Compare with the fit has its Help entry, each ending the same way", () => {
  const bodies = {
    check_qq: "Each point is a sorted quantile residual against the value it would have if each row's predicted distribution were right. "
      + "The band holds for every point at once: a model that fits leaves it only 5% of the time, so one point outside is evidence. "
      + "A curve above the band at the top end means larger values come more often than the model allows. "
      + "Band: equal local levels (Aldor-Noiman et al. 2013; Weine, McPeek and Abney 2023).",
    check_pit: "Each observed value is placed at its own predicted distribution's cumulative probability, and the 20 bars count them. "
      + "A model that fits gives flat bars. A hump means its predicted distributions are too wide, a U too narrow, a slope a shifted Mean, "
      + "and one end alone a tail. The band holds for all 20 bars at once.",
    check_ae: "Observed total over predicted total, for each band of predicted Mean and each group of every input. 1 is on target. "
      + "Each interval comes from the predicted distribution's own variance, and all of them hold at once, so a group outside its "
      + "interval is evidence.",
    check_calibration: "For each predicted quantile, the share of rows above it against the share that should be. "
      + "The 99% quantile should be exceeded by 1% of rows. The band is exact binomial and holds for all 11 quantiles at once.",
    compare_original: "Draws the model as fitted in grey beside the model as edited in blue, on every check. It changes only what is drawn.",
  };
  const titles = { check_qq: "Q-Q plot", check_pit: "PIT histogram", check_ae: "Actual over expected",
    check_calibration: "Quantile calibration", compare_original: "Compare with the fit" };
  for (const [id, body] of Object.entries(bodies)) {
    assert.deepEqual({ ...helpEntry(id) }, { title: titles[id], body: `${body} ${SHARED_ENDING}` }, id);
  }
  // Each Help title is the entry's list label.
  for (const entry of ASSESS_GROUPS.slice(1).flatMap((group) => group.entries)) assert.equal(CHECK_HELP[entry.id].title, entry.label);
  const section = HELP_SECTIONS.find((s) => s.title === "Checks");
  assert.deepEqual(section?.keys, ["check_qq", "check_pit", "check_ae", "check_calibration", "compare_original"]);
  assert.deepEqual(neutralCopy(JSON.stringify(section)), []);
});

class FakeHelpElement {
  constructor(dataset, attributes = {}) { this.dataset = dataset; this.attributes = attributes; }
  getAttribute(name) { return this.attributes[name] ?? null; }
  hasAttribute(name) { return name in this.attributes; }
}

test("a disabled control's own reason outranks its Help, as a tool's does", () => {
  const previous = globalThis.HTMLElement;
  globalThis.HTMLElement = FakeHelpElement;
  try {
    const reason = "The model is as fitted; there is no edit to compare.";
    const dataset = { helpControl: "compare_original", popoverTitle: "Compare with the fit", popoverBody: reason };
    assert.deepEqual({ ...helpForElement(new FakeHelpElement(dataset, { "aria-disabled": "true" })) },
      { title: "Compare with the fit", body: reason });
    assert.deepEqual({ ...helpForElement(new FakeHelpElement(dataset, { disabled: "" })) },
      { title: "Compare with the fit", body: reason });
    assert.equal(helpForElement(new FakeHelpElement(dataset, { "aria-disabled": "false" })), CONTROL_HELP.compare_original);
    // A disabled control with no reason of its own keeps its Help.
    assert.equal(helpForElement(new FakeHelpElement({ helpControl: "new_window" }, { disabled: "" })), CONTROL_HELP.new_window);
  } finally {
    globalThis.HTMLElement = previous;
  }
});
