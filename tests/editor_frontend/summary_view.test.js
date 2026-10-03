// @ts-nocheck
import assert from "node:assert/strict";
import test from "node:test";

const summaryViewModulePath = "../../src/superglm/editor/app/views/summary_view.js";
const {
  bindSummarySearch,
  highlightMatches,
  highlightRowName,
  summaryCountText,
  summaryRowTerm,
  summaryViewModel,
  waitingCounts
} = await import(summaryViewModulePath);

// Compact rows as Python sends them, in model order.
const ROWS = [
  { name: "Intercept", group: "" },
  { name: "VehAge", group: "VehAge", kind: "spline" },
  { name: "VehBrand[B1]", group: "VehBrand" },
  { name: "VehBrand[B10]", group: "VehBrand" },
  { name: "VehBrand[B2]", group: "VehBrand" },
  // Integer-typed levels arrive as their text, in the model's order 1, 2, 10.
  { name: "bonus[1]", group: "bonus" },
  { name: "bonus[2]", group: "bonus" },
  { name: "bonus[10]", group: "bonus" },
  // A polynomial prints under a decorated group.
  { name: "x_poly[P1]", group: "x_poly P(2)" },
  { name: "x_poly[P2]", group: "x_poly P(2)" }
];
const TERMS = ["VehAge", "VehBrand", "bonus", "x_poly"];

function search(query) {
  return summaryViewModel(ROWS, { query, termNames: TERMS });
}

/** The row names a search shows, in the order shown. */
function shown(query) {
  return search(query).sections.flatMap((section) => section.rows
    .filter((entry) => !entry.hidden)
    .map((entry) => ROWS[entry.index].name));
}

test("each row belongs to its editor term, a decorated polynomial group included", () => {
  assert.equal(summaryRowTerm(ROWS[8], TERMS), "x_poly");
  assert.equal(summaryRowTerm(ROWS[3], TERMS), "VehBrand");
  assert.equal(summaryRowTerm(ROWS[0], TERMS), "Intercept");
  assert.equal(summaryRowTerm({ name: "x:z", group: "x:z" }, TERMS), "x:z");
  const sections = search("").sections;
  assert.deepEqual(sections.map((section) => [section.term, section.label]), [
    ["Intercept", "Intercept"],
    ["VehAge", "VehAge"],
    ["VehBrand", "VehBrand"],
    ["bonus", "bonus"],
    ["x_poly", "x_poly P(2)"]
  ]);
});

test("an empty search shows every row", () => {
  assert.deepEqual(shown(""), ROWS.map((row) => row.name));
  assert.equal(summaryCountText(search(""), ""), "");
});

test("a term-name match keeps all its rows and the intercept leaves while searching", () => {
  assert.deepEqual(shown("veh"), ["VehAge", "VehBrand[B1]", "VehBrand[B10]", "VehBrand[B2]"]);
  assert.equal(summaryCountText(search("veh"), "veh"), "2 terms · 4 rows");
  // Model order is kept, not string order (1, 10, 2).
  assert.deepEqual(shown("BONUS"), ["bonus[1]", "bonus[2]", "bonus[10]"]);
});

test("numeric-looking levels match as text: 1 finds 1 and 10, 10 finds only 10", () => {
  assert.deepEqual(shown("1"), ["VehBrand[B1]", "VehBrand[B10]", "bonus[1]", "bonus[10]", "x_poly[P1]"]);
  assert.equal(summaryCountText(search("1"), "1"), "3 terms · 5 rows");
  assert.deepEqual(shown("10"), ["VehBrand[B10]", "bonus[10]"]);
  // The decorated group's "(2)" is not a level: only the level P2 matches 2.
  assert.deepEqual(shown("2"), ["VehBrand[B2]", "bonus[2]", "x_poly[P2]"]);
  assert.equal(summaryCountText(search(" 10 "), " 10 "), "2 terms · 2 rows");
});

test("a search with no match says so, and its text is taken literally", () => {
  assert.equal(summaryCountText(search("zzz"), "zzz"), "No terms match.");
  assert.deepEqual(shown("("), []);
  assert.deepEqual(shown("."), []);
  assert.equal(search("zzz").sections.every((section) => section.hidden), true);
});

test("matches are marked in the term and the level, and the rest is escaped", () => {
  assert.equal(highlightMatches("<B1>b1", "b1"), "&lt;<mark>B1</mark>&gt;<mark>b1</mark>");
  assert.equal(highlightRowName("bonus[10]", "bonus", "1"), "bonus[<mark>1</mark>0]");
  assert.equal(highlightRowName("VehBrand[B1]", "VehBrand", "veh"), "<mark>Veh</mark>Brand[B1]");
  assert.equal(highlightRowName("VehBrand[B1]", "VehBrand", ""), "VehBrand[B1]");
});

test("the search box reports each input and Escape clears it without closing the inspector", () => {
  const listeners = new Map();
  const search = {
    value: "",
    addEventListener(name, listener) { listeners.set(name, listener); },
    removeEventListener(name) { listeners.delete(name); }
  };
  const queries = [];
  const binding = bindSummarySearch(search, (query) => queries.push(query));
  const key = (name) => {
    const event = {
      key: name,
      prevented: false,
      stopped: false,
      preventDefault() { this.prevented = true; },
      stopPropagation() { this.stopped = true; }
    };
    listeners.get("keydown")(event);
    return event;
  };

  search.value = "veh";
  listeners.get("input")();
  const escape = key("Escape");
  assert.deepEqual(queries, ["veh", ""]);
  assert.equal(search.value, "");
  assert.equal(escape.prevented && escape.stopped, true);

  // An empty box lets Escape through, so a narrow inspector still closes.
  const passed = key("Escape");
  assert.equal(passed.prevented || passed.stopped, false);
  assert.deepEqual(queries, ["veh", ""]);

  binding.destroy();
  assert.equal(listeners.size, 0);
});

// A small book: a spline with its whole-term test, two categoricals.
const BOOK_ROWS = [
  { name: "Intercept", group: "", kind: "coef", edf: 1, p_value: 1e-6, sig_class: "sig-strong", sig_code: "***" },
  { name: "DrivAge", group: "DrivAge", kind: "spline", edf: 11.2, p_value: 2e-5, sig_class: "sig-strong", sig_code: "***" },
  { name: "VehBrand[B1]", group: "VehBrand", kind: "reference", edf: null, p_value: null, sig_class: "sig-reference" },
  { name: "VehBrand[B2]", group: "VehBrand", kind: "coef", edf: 10, p_value: 0.03, sig_class: "sig-standard", sig_code: "*" },
  { name: "VehBrand[B10]", group: "VehBrand", kind: "coef", edf: null, p_value: 0.004, sig_class: "sig-medium", sig_code: "**" },
  { name: "Area[A]", group: "Area", kind: "reference", edf: null, p_value: null, sig_class: "sig-reference" },
  { name: "Area[B]", group: "Area", kind: "coef", edf: 1, p_value: 0.5, sig_class: "sig-none", sig_code: "" }
];
const BOOK = {
  query: "",
  termNames: ["DrivAge", "VehBrand", "Area"],
  currentTerm: "VehBrand",
  kinds: { DrivAge: "spline", VehBrand: "categorical" }
};

function book(patch = {}) {
  return summaryViewModel(BOOK_ROWS, { ...BOOK, ...patch });
}

function lines(model) {
  return model.sections.filter((section) => section.header && !section.hidden).map((section) => section.term);
}

function openLines(model) {
  return model.sections
    .filter((section) => section.header && !section.hidden && section.open)
    .map((section) => section.term);
}

function bookRows(model) {
  return model.sections.flatMap((section) => section.rows
    .filter((entry) => !entry.hidden)
    .map((entry) => BOOK_ROWS[entry.index].name));
}

test("the chart's term is open, the others fold to a line that still says how they fit", () => {
  const model = book();

  assert.deepEqual(lines(model), ["DrivAge", "VehBrand", "Area"]);
  assert.deepEqual(openLines(model), ["VehBrand"]);
  assert.deepEqual(bookRows(model), ["Intercept", "VehBrand[B1]", "VehBrand[B2]", "VehBrand[B10]"]);
  const [, drivAge, vehBrand, area] = model.sections;
  assert.deepEqual(
    [drivAge.kind, drivAge.edf, drivAge.chip, drivAge.current],
    ["spline", 11.2, { p: 2e-5, sigClass: "sig-strong", sigCode: "***" }, false]
  );
  // A categorical has no whole-term test, so its line has no p chip: the
  // smallest of its level p-values would overstate the term's significance.
  assert.deepEqual([vehBrand.edf, vehBrand.chip, vehBrand.current], [10, null, true]);
  // A term the editor does not show still gets its line.
  assert.deepEqual([area.kind, area.edf, area.chip], ["", 1, null]);
});

test("a section opened or closed by hand stays so; without a chart term every section is open", () => {
  const toggled = new Map([["Area", true], ["VehBrand", false]]);
  assert.deepEqual(openLines(book({ toggled })), ["Area"]);
  assert.deepEqual(openLines(book({ currentTerm: "DrivAge" })), ["DrivAge"]);
  assert.deepEqual(openLines(book({ currentTerm: "" })), ["DrivAge", "VehBrand", "Area"]);
});

test("a search opens every section it finds", () => {
  const model = book({ query: "b1" });
  assert.deepEqual(lines(model), ["VehBrand"]);
  assert.deepEqual(bookRows(model), ["VehBrand[B1]", "VehBrand[B10]"]);

  const every = book({ query: "a" });
  assert.deepEqual(openLines(every), ["DrivAge", "VehBrand", "Area"]);
  assert.equal(summaryCountText(every, "a"), "3 terms · 6 rows");
});

test("Edited and Waiting keep the terms with hand edits or changes waiting for refit", () => {
  const edited = book({ filter: "edited", edited: ["DrivAge"] });
  assert.deepEqual(lines(edited), ["DrivAge"]);
  assert.deepEqual(bookRows(edited), []);

  const waiting = book({ filter: "waiting", waiting: { Area: 2 } });
  assert.deepEqual(lines(waiting), ["Area"]);
  assert.equal(waiting.sections[3].waiting, 2);

  // The filter and the search combine.
  const neither = book({ filter: "waiting", waiting: { Area: 2 }, query: "veh" });
  assert.deepEqual(lines(neither), []);
  assert.equal(summaryCountText(neither, "veh"), "No terms match.");
  // An empty result says so in the table too; the plain view never does.
  assert.equal(neither.empty, true);
  assert.equal(book({ filter: "edited" }).empty, true);
  assert.equal(book().empty, false);
});

test("waiting changes are counted per term", () => {
  assert.deepEqual(
    waitingCounts([{ term: "Area" }, { term: "VehBrand" }, { term: "Area" }]),
    { Area: 2, VehBrand: 1 }
  );
  assert.deepEqual(waitingCounts([]), {});
});
