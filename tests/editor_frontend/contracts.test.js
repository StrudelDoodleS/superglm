import assert from "node:assert/strict";
import test from "node:test";

test("frontend tests execute as native ES modules", () => {
  assert.deepEqual({ runner: "node", modules: "esm" }, {
    runner: "node",
    modules: "esm"
  });
});
