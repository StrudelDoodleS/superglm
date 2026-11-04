// @ts-check
// Sizes in the editor's stylesheets come from the scales in tokens.css.
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";

const APP = join(import.meta.dirname, "../../src/superglm/editor/app");
const SHEETS = [
  join(APP, "styles.css"),
  ...readdirSync(join(APP, "styles")).map((name) => join(APP, "styles", name)),
];

/** @param {RegExp} pattern @returns {string[]} */
function matches(pattern) {
  return SHEETS.flatMap((sheet) =>
    [...readFileSync(sheet, "utf8").matchAll(pattern)].map((match) => `${sheet}: ${match[0]}`)
  );
}

test("every font size comes from the type scale", () => {
  assert.deepEqual(matches(/font-size:(?!\s*var\(--font-size-|\s*inherit)[^;}]+/g), []);
  assert.deepEqual(matches(/(?<![-\w])font:\s*\d[^;}]*/g), []);
});

test("every corner comes from the radius scale, a circle or none", () => {
  assert.deepEqual(matches(/border(?:-[a-z]+)*-radius:(?!\s*var\(--radius-|\s*50%|\s*0\s*[;}])[^;}]+/g), []);
});

test("type comes in two weights", () => {
  assert.deepEqual(matches(/font-weight:(?!\s*400|\s*600|\s*inherit)[^;}]+/g), []);
});
