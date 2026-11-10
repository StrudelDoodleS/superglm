// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  amountInsideAxis,
  formatAmount,
  formatChance,
  ordinal,
  formatOneInN,
  parseLine,
  parseOneInN,
  pinToAxis,
} from "../../src/superglm/editor/app/distribution/line.js";

const HINT = "Type an amount such as 25k or a percentile such as 99.5th or 1 in 200.";
const amount = (value) => ({ ok: true, mode: "amount", value });
const percentile = (value) => ({ ok: true, mode: "percentile", value });
const refused = { ok: false, hint: HINT };

const EPS = Number.EPSILON;

/**
 * Asserts that each `[text, mode, unit, want]` row parses to `want`. A value is the parsed number
 * text divided by 100 or scaled by 1e3 or 1e6: at most three roundings, so within 2 eps relative.
 */
function expectParses(rows) {
  for (const [text, mode, unit, want] of rows) {
    const got = parseLine(text, mode, unit);
    const label = `${JSON.stringify(text)} in ${mode} mode`;
    assert.equal(got.ok, want.ok, label);
    if (!want.ok) {
      assert.deepEqual(got, want, label);
      continue;
    }
    assert.equal(got.mode, want.mode, label);
    const bound = 2 * EPS * Math.abs(want.value);
    assert.ok(Math.abs(got.value - want.value) <= bound, `${label}: ${got.value}`);
  }
}

function expectRefused(texts, mode, unit = null) {
  for (const text of texts) {
    assert.deepEqual(parseLine(text, mode, unit), refused, `${JSON.stringify(text)} in ${mode}`);
  }
}

test("the spec's amount examples, typed in amount mode, are exact", () => {
  assert.deepEqual(parseLine("25k", "amount", null), amount(25000));
  assert.deepEqual(parseLine("€25,000", "amount", null), amount(25000));
  assert.deepEqual(parseLine("1.5m", "amount", null), amount(1500000));
});

test("the spec's percentile examples, typed in amount mode", () => {
  expectParses([
    ["99.5th", "amount", null, percentile(0.995)],
    ["1 in 200", "amount", null, percentile(0.995)],
  ]);
});

test("a bare number follows the mode, so 99.5 is a percentile in percentile mode", () => {
  expectParses([
    ["99.5", "percentile", null, percentile(0.995)],
    ["99.5", "amount", null, amount(99.5)],
    ["25000", "amount", null, amount(25000)],
    ["25000", "percentile", null, refused],
  ]);
});

test("unreadable input is refused with the one hint, in either mode", () => {
  for (const mode of ["amount", "percentile"]) {
    expectRefused(
      ["abc", "", "   ", "€", ",", "25kk", "k", "25 x", "1e6", "1..5", "1 in", "1 in x", "th"],
      mode
    );
  }
});

test("amounts and percentiles are recognised in either mode", () => {
  expectParses([
    ["25k", "percentile", null, amount(25000)],
    ["1.5m", "percentile", null, amount(1500000)],
    ["99.5th", "percentile", null, percentile(0.995)],
    ["1 in 200", "percentile", null, percentile(0.995)],
    [".5k", "amount", null, amount(500)],
    ["0.25m", "amount", null, amount(250000)],
  ]);
});

test("a currency symbol or unit in the text forces amount mode; a unit only set does not", () => {
  expectParses([
    ["€25,000", "percentile", null, amount(25000)],
    ["£40", "percentile", null, amount(40)],
    ["$1.5m", "percentile", null, amount(1500000)],
    ["25", "percentile", "€", percentile(0.25)],
    ["€25", "percentile", "€", amount(25)],
    ["25kr", "percentile", "kr", amount(25)],
    ["25k kr", "percentile", "kr", amount(25000)],
    ["kr 25", "percentile", "kr", amount(25)],
  ]);
});

test("a unit made of letters and a symbol is read, and formatAmount's output reads back", () => {
  expectParses([
    ["US$25k", "amount", "US$", amount(25000)],
    ["us$ 25,000", "percentile", "US$", amount(25000)],
    ["R$25", "percentile", "R$", amount(25)],
    ["A$1.5m", "amount", "A$", amount(1500000)],
    ["CA$ 40", "percentile", "CA$", amount(40)],
    ["25k US$", "percentile", "US$", amount(25000)],
  ]);
  for (const unit of ["US$", "R$", "A$", "CA$", "€", "£", "$", "EUR", "kr "]) {
    for (const mode of ["amount", "percentile"]) {
      // formatAmount keeps three significant figures, so 1234567 reads back as 1230000.
      for (const [v, read] of [[25000, 25000], [1234567, 1230000], [0.0123, 0.0123], [7, 7]]) {
        const text = formatAmount(v, unit);
        assert.deepEqual(parseLine(text, mode, unit), amount(read), `${text} in ${mode} mode`);
      }
    }
  }
});

test("the unit and the symbols are read at the ends of the text, never from its middle", () => {
  // The same unit mid-string is left in place, so what is typed is not rearranged into a number.
  expectRefused(["2kr5", "25kr25", "kr25kr25"], "amount", "kr");
  expectRefused(["2€5", "2£5", "25$25"], "amount");
  expectRefused(["2€5"], "percentile", "€");
  expectParses([
    ["1 in 200", "amount", "in", percentile(0.995)],
    ["25€", "percentile", null, amount(25)],
    ["€25€", "percentile", null, amount(25)],
    ["25 kr", "amount", "kr", amount(25)],
  ]);
});

test("the unit is stripped without regard to case", () => {
  expectParses([
    ["EUR 25k", "amount", "eur", amount(25000)],
    ["eur25,000", "percentile", "EUR", amount(25000)],
    ["25 Eur", "percentile", "EUR", amount(25)],
  ]);
});

test("a unit that is only whitespace or empty strips nothing", () => {
  expectParses([
    ["99.5", "percentile", "", percentile(0.995)],
    ["99.5", "percentile", "  ", percentile(0.995)],
    ["99.5", "percentile", null, percentile(0.995)],
  ]);
});

test("every percentile suffix is read, and means p = v / 100", () => {
  expectParses([
    ["1st", "amount", null, percentile(0.01)],
    ["2nd", "amount", null, percentile(0.02)],
    ["3rd", "amount", null, percentile(0.03)],
    ["99th", "amount", null, percentile(0.99)],
    ["90%", "amount", null, percentile(0.9)],
    ["50pct", "amount", null, percentile(0.5)],
    ["0.5%", "amount", null, percentile(0.005)],
    ["99.9%", "percentile", null, percentile(0.999)],
  ]);
});

test("1 in N is p = 1 - 1/N", () => {
  expectParses([
    ["1 in 2", "amount", null, percentile(0.5)],
    ["1 in 4", "amount", null, percentile(0.75)],
    ["1 in 1000", "percentile", null, percentile(0.999)],
    ["1 in 1.25", "percentile", null, percentile(0.2)],
    ["1 in 2,000", "percentile", null, percentile(0.9995)],
  ]);
  // p is (N - 1) / N, one rounding: 1 - 1/3 and 1 - 1/7 are not the doubles nearest 2/3 and 6/7.
  assert.deepEqual(parseLine("1 in 3", "amount", null), percentile(2 / 3));
  assert.deepEqual(parseLine("1 in 7", "amount", null), percentile(6 / 7));
  // The spec's examples are exact.
  assert.deepEqual(parseLine("1 in 200", "amount", null), percentile(0.995));
  assert.deepEqual(parseLine("99.5th", "amount", null), percentile(0.995));
  assert.deepEqual(parseLine("99.5", "percentile", null), percentile(0.995));
});

test("suffixes are case-insensitive, and spaces and commas inside are ignored", () => {
  expectParses([
    ["25K", "amount", null, amount(25000)],
    ["1.5M", "amount", null, amount(1500000)],
    ["99.5PCT", "amount", null, percentile(0.995)],
    ["99.5TH", "amount", null, percentile(0.995)],
    ["1 IN 200", "amount", null, percentile(0.995)],
    ["25 k", "amount", null, amount(25000)],
    ["  25k  ", "amount", null, amount(25000)],
    ["99.5 th", "amount", null, percentile(0.995)],
    ["1in200", "amount", null, percentile(0.995)],
    ["1 , 500 k", "amount", null, amount(1500000)],
    ["€ 25 , 000", "amount", null, amount(25000)],
  ]);
});

test("a percentile must lie strictly between 0 and 1", () => {
  expectRefused(
    ["0th", "100th", "0%", "100%", "0pct", "150%", "0", "100", "101", "0.0"],
    "percentile"
  );
  expectRefused(["0th", "100th", "100%", "0pct"], "amount");
  expectParses([
    ["0.001%", "percentile", null, percentile(0.00001)],
    ["99.999th", "amount", null, percentile(0.99999)],
  ]);
});

test("1 in N needs N > 1: N = 1 (p = 0), N = 0, N < 1, negative and non-numeric N refuse", () => {
  expectRefused(
    ["1 in 1", "1 in 1.0", "1 in 0", "1 in 0.5", "1 in -5", "1 in -200", "1 in abc", "1 in 20x"],
    "amount"
  );
  expectRefused(["1 in 1", "1 in 0", "1 in -5", "1 in abc"], "percentile");
  // N beyond the double range reads as infinity, not as a certain percentile.
  expectRefused([`1 in ${"9".repeat(400)}`], "amount");
});

test("an amount must be positive and finite", () => {
  expectRefused(["0", "0k", "0m", "€0", "-25k", "-5", "€-5", "0.0", "0.000k"], "amount");
  expectRefused(["0k", "0m", "€0", "-25k"], "percentile");
  expectRefused([`1${"0".repeat(400)}k`, `1${"0".repeat(400)}`], "amount");
  expectParses([["0.001", "amount", null, amount(0.001)]]);
});

test("a refusal carries a fresh result each time", () => {
  const first = parseLine("abc", "amount", null);
  first.hint = "changed";
  assert.deepEqual(parseLine("abc", "amount", null), refused);
});

test("formatChance, percent style: one decimal from 1%, two significant figures from 1e-6", () => {
  const rows = [
    [1, "100.0%"],
    [0.5, "50.0%"],
    [0.123, "12.3%"],
    [0.05, "5.0%"],
    [0.01, "1.0%"],
    [0.0099999, "1.0%"],
    [0.005, "0.50%"],
    [0.0012345, "0.12%"],
    [0.00033, "0.033%"],
    [0.0000012, "0.00012%"],
    [1e-6, "0.00010%"],
  ];
  for (const [p, want] of rows) assert.equal(formatChance(p, "percent"), want, `p = ${p}`);
});

test("formatChance, percent style: below 1e-6 it says so rather than printing zero", () => {
  for (const p of [9.9e-7, 1e-9, 1e-12, 5e-324, 0]) {
    assert.equal(formatChance(p, "percent"), "<0.0001%", `p = ${p}`);
  }
});

test("formatChance, one_in style: 1 in N with two significant figures and en-GB grouping", () => {
  const rows = [
    [0.49, "1 in 2.0"],
    [0.11, "1 in 9.1"],
    [0.1, "1 in 10"],
    [0.01, "1 in 100"],
    [0.005, "1 in 200"],
    [0.00123, "1 in 810"],
    [0.00033, "1 in 3,000"],
    [0.0001, "1 in 10,000"],
    [0.000037, "1 in 27,000"],
    [1e-6, "1 in 1,000,000"],
  ];
  for (const [p, want] of rows) assert.equal(formatChance(p, "one_in"), want, `p = ${p}`);
});

test("formatChance, one_in style: a chance of one half or more is a percentage, not 1 in 1.0", () => {
  const rows = [[1, "100.0%"], [0.954, "95.4%"], [0.8, "80.0%"], [0.5, "50.0%"]];
  for (const [p, want] of rows) assert.equal(formatChance(p, "one_in"), want, `p = ${p}`);
  // The percent style is unchanged, and so is the first chance below a half.
  assert.equal(formatChance(0.954, "percent"), "95.4%");
  assert.equal(formatChance(0.4999, "one_in"), "1 in 2.0");
});

test("formatChance, one_in style: below 1e-6 it says 1 in >1,000,000", () => {
  for (const p of [9.9e-7, 1e-9, 5e-324, 0]) {
    assert.equal(formatChance(p, "one_in"), "1 in >1,000,000", `p = ${p}`);
  }
});

test("formatAmount: three significant figures, en-GB grouping, unit as a plain prefix", () => {
  const rows = [
    [25000, null, "25,000"],
    [25000, "€", "€25,000"],
    [1234567, null, "1,230,000"],
    [1234567, "€", "€1,230,000"],
    [0.0123, null, "0.0123"],
    [0.000123456, null, "0.000123"],
    [12.345, null, "12.3"],
    [999.6, null, "1,000"],
    [123456789, null, "123,000,000"],
    [7, null, "7"],
    [0.5, "£", "£0.5"],
    [25000, "EUR", "EUR25,000"],
    [25000, "kr ", "kr 25,000"],
    [25000, "", "25,000"],
  ];
  for (const [v, unit, want] of rows) {
    assert.equal(formatAmount(v, unit), want, `${v} with unit ${JSON.stringify(unit)}`);
  }
});

test("ordinal is the prototype's from the 0.5th to the 99.95th: whole, then one decimal", () => {
  const rows = [
    [0.5, "50th"],
    [0.01, "1st"],
    [0.02, "2nd"],
    [0.03, "3rd"],
    [0.04, "4th"],
    [0.11, "11th"],
    [0.12, "12th"],
    [0.13, "13th"],
    [0.21, "21st"],
    [0.22, "22nd"],
    [0.23, "23rd"],
    [0.9, "90th"],
    [0.99, "99th"],
    [0.9904, "99th"],
    [0.991, "99.1th"],
    [0.995, "99.5th"],
    [0.999, "99.9th"],
    [0.99949, "99.9th"],
    [0.005, "1st"],
  ];
  for (const [p, want] of rows) assert.equal(ordinal(p), want, `p = ${p}`);
});

test("ordinal widens its decimals at the ends and never prints 0th or 100th", () => {
  const rows = [
    [1e-6, "0.0001th"],
    [1e-4, "0.01th"],
    [0.001, "0.1th"],
    [0.004, "0.4th"],
    [0.0049, "0.5th"],
    [0.9995, "99.95th"],
    [0.99951, "99.95th"],
    [0.9999, "99.99th"],
    [0.999999, "99.9999th"],
    [1 - 1e-9, "99.9999999th"],
  ];
  for (const [p, want] of rows) {
    const text = ordinal(p);
    assert.equal(text, want, `p = ${p}`);
    // The reader takes the printed digits as they stand, so the text commits again unchanged.
    const digits = text.slice(0, -2);
    assert.deepEqual(parseLine(text, "percentile", null), percentile(Number(digits) / 100), text);
    assert.deepEqual(parseLine(text, "amount", null), percentile(Number(digits) / 100), text);
  }
});

test("ordinal reads back for every power of ten from both ends, and never prints an end", () => {
  const ps = [];
  for (let e = 2; e <= 100; e += 1) ps.push(10 ** -e);
  for (let e = 2; e <= 15; e += 1) ps.push(1 - 10 ** -e);
  ps.push(1 - 2 ** -53);
  for (const p of ps) {
    const text = ordinal(p);
    const digits = text.slice(0, -2);
    assert.ok(Number(digits) > 0 && Number(digits) < 100, `p = ${p}: ${text}`);
    assert.match(text, /^\d+(\.\d+)?(st|nd|rd|th)$/, `p = ${p}: ${text}`);
    assert.deepEqual(parseLine(text, "percentile", null), percentile(Number(digits) / 100), text);
  }
});

test("ordinal uses the fewest decimals that keep the percentile off 0 and 100", () => {
  // 0.4 needs one decimal; 0.04, two; and so on. 99.96 needs two; 99.9996, four.
  assert.equal(ordinal(0.004), "0.4th");
  assert.equal(ordinal(0.0004), "0.04th");
  assert.equal(ordinal(0.00004), "0.004th");
  assert.equal(ordinal(0.9996), "99.96th");
  assert.equal(ordinal(0.999996), "99.9996th");
});

test("ordinal text parses back to the percentile it names", () => {
  for (let k = 1; k <= 99; k += 1) {
    expectParses([[ordinal(k / 100), "amount", null, percentile(k / 100)]]);
  }
  for (let j = 1; j <= 9; j += 1) {
    const text = ordinal(0.99 + j / 1000);
    expectParses([[text, "percentile", null, percentile((99 + j / 10) / 100)]]);
  }
  // Past the 99.95th and below the 0.5th the digits widen, and still read back.
  for (const p of [0.9995, 0.9999, 0.999999, 0.004, 0.001, 1e-4, 1e-5, 1e-6]) {
    const digits = ordinal(p).slice(0, -2);
    expectParses([[ordinal(p), "percentile", null, percentile(Number(digits) / 100)]]);
  }
});

test("a starting amount on the axis's top edge moves to the round amount just inside it", () => {
  const axis = { min_log10: -1, max_log10: 4 };
  assert.equal(amountInsideAxis(1e4, axis), 5e3);
  assert.equal(amountInsideAxis(1e5, axis), 5e3);
  assert.equal(amountInsideAxis(1e3, axis), 1e3);
  assert.equal(amountInsideAxis(5e3, axis), 5e3);
  // The result is a 1, 2 or 5 times a power of ten, whatever the top is.
  assert.equal(amountInsideAxis(1e5, { min_log10: 0, max_log10: 5 }), 5e4);
  assert.equal(amountInsideAxis(1e3, { min_log10: -1, max_log10: 3.3 }), 1e3);
  assert.equal(amountInsideAxis(2e3, { min_log10: -1, max_log10: 3.3 }), 1e3);
  for (const top of [1, 2, 2.3, 3.7, 4, 5.25]) {
    const value = amountInsideAxis(10 ** Math.ceil(top), { min_log10: -1, max_log10: top });
    assert.ok(Math.log10(value) < top, `top ${top}`);
    assert.match(String(value), /^[125]0*$|^0\.[125]$|^[125]e/, `top ${top}: ${value}`);
  }
});

test("pinToAxis leaves a value on the axis alone and reports pinned: false", () => {
  const axis = { min_log10: 2, max_log10: 5, clipped_below: false };
  assert.deepEqual(pinToAxis(3.5, axis), { log10: 3.5, pinned: false });
  assert.deepEqual(pinToAxis(2, axis), { log10: 2, pinned: false });
  assert.deepEqual(pinToAxis(5, axis), { log10: 5, pinned: false });
});

test("pinToAxis clamps a value off either end to the edge and reports pinned: true", () => {
  const axis = { min_log10: 2, max_log10: 5, clipped_below: false };
  assert.deepEqual(pinToAxis(1.2, axis), { log10: 2, pinned: true });
  assert.deepEqual(pinToAxis(6, axis), { log10: 5, pinned: true });
  assert.deepEqual(pinToAxis(-Infinity, axis), { log10: 2, pinned: true });
  assert.deepEqual(pinToAxis(Infinity, axis), { log10: 5, pinned: true });
});

test("pinToAxis ignores clipped_below", () => {
  const clipped = { min_log10: 2, max_log10: 5, clipped_below: true };
  assert.deepEqual(pinToAxis(3, clipped), { log10: 3, pinned: false });
  assert.deepEqual(pinToAxis(1, clipped), { log10: 2, pinned: true });
});

test("a typed amount outside the axis keeps its value for the readouts and pins the line", () => {
  const axis = { min_log10: 2, max_log10: 5, clipped_below: false };
  const typed = parseLine("2m", "amount", null);
  assert.equal(typed.value, 2000000);
  assert.deepEqual(pinToAxis(Math.log10(typed.value), axis), { log10: 5, pinned: true });
  assert.equal(formatAmount(typed.value, "€"), "€2,000,000");
});

test("Help states the panel's typing rules and limits as the panel applies them", async () => {
  const { HELP_SECTIONS } = await import("../../src/superglm/editor/app/views/help_content.js");
  const { GAMMA_CV_MAX } = await import("../../src/superglm/editor/app/distribution/families.js");
  const section = HELP_SECTIONS.find((candidate) => candidate.title === "Distribution panel");
  assert.ok(section, "Help has a Distribution panel section");
  const text = section.items.join(" ");
  for (const typed of ["25k", "€25,000", "1.5m", "99.5th", "1 in 200"]) {
    assert.ok(text.includes(typed), typed);
    assert.equal(parseLine(typed, "amount", null).ok, true, typed);
  }
  assert.ok(text.includes("1,5m reads as 15,000,000"));
  assert.deepEqual(parseLine("1,5m", "amount", null), amount(15000000));
  assert.ok(text.includes(`gamma CV above ${GAMMA_CV_MAX} is drawn at ${GAMMA_CV_MAX}`));
});

test("the box beside a fixed 1 in reads only N", () => {
  const value = (n) => ({ ok: true, mode: "percentile", value: (n - 1) / n });
  assert.deepEqual(parseOneInN("200"), value(200));
  assert.deepEqual(parseOneInN(" 2,000 "), value(2000));
  assert.deepEqual(parseOneInN("10k"), value(10000));
  assert.deepEqual(parseOneInN("1.5m"), value(1500000));
  assert.deepEqual(parseOneInN("1.25"), value(1.25));
  assert.deepEqual(parseOneInN("1 in 200"), value(200));
});

test("the box beside a fixed 1 in refuses what is not an N above 1", () => {
  const refused = { ok: false, hint: "Type N as a number above 1, such as 200 or 10k." };
  for (const text of ["", "abc", "1", "1.0", "0", "0.5", "-5", "99.5th", "50%", "€200", ">1,000,000"]) {
    assert.deepEqual(parseOneInN(text), refused, JSON.stringify(text));
  }
});

test("the N of a chance is what formatChance writes after 1 in", () => {
  for (const p of [0.4, 0.05, 0.005, 1e-5, 1e-7]) {
    assert.equal(`1 in ${formatOneInN(p)}`, formatChance(p, "one_in"));
  }
  assert.equal(formatOneInN(0.005), "200");
  assert.equal(formatOneInN(1e-7), ">1,000,000");
});
