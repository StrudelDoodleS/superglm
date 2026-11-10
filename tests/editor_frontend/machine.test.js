import assert from "node:assert/strict";
import test from "node:test";

import { defineMachine } from "../../src/superglm/editor/app/state/machine.js";

const door = () => defineMachine({
  name: "door",
  initial: /** @type {"open"|"closed"|"locked"} */ ("closed"),
  moves: {
    open: { push: "closed" },
    closed: { pull: "open", lock: "locked", push: "closed" },
    locked: { unlock: "closed" }
  }
});

test("a machine answers each move from its table and refuses the rest", () => {
  const machine = door();

  assert.equal(machine.next("closed", "pull"), "open");
  assert.equal(machine.next("closed", "push"), "closed");
  assert.equal(machine.next("locked", "pull"), null);
  assert.equal(machine.next("open", "lock"), null);
  assert.deepEqual([...machine.states], ["open", "closed", "locked"]);
  assert.deepEqual([...machine.events].sort(), ["lock", "pull", "push", "unlock"]);
  assert.equal(machine.initial, "closed");
});

test("a declared table cannot be changed afterwards", () => {
  const machine = door();

  assert.throws(() => {
    /** @type {Record<string, unknown>} */ (machine.moves.locked).pull = "open";
  }, TypeError);
  assert.throws(() => {
    /** @type {Record<string, unknown>} */ (machine.moves).ajar = {};
  }, TypeError);
  assert.equal(machine.next("locked", "pull"), null);
});

test("a table that moves to a state it does not define is refused", () => {
  assert.throws(
    () => defineMachine({
      name: "door",
      initial: "closed",
      moves: /** @type {Record<string, Record<string, string>>} */ ({
        closed: { pull: "ajar" }
      })
    }),
    /closed --pull--> ajar, which is not a state/
  );
});

test("a table with a state nothing leaves is refused", () => {
  assert.throws(
    () => defineMachine({
      name: "door",
      initial: "closed",
      moves: { closed: { lock: "locked" }, locked: { knock: "locked" } }
    }),
    /no move leaves locked/
  );
});

test("a table with a state nothing reaches is refused", () => {
  assert.throws(
    () => defineMachine({
      name: "door",
      initial: "closed",
      moves: /** @type {Record<string, Record<string, string>>} */ ({
        closed: { pull: "open" }, open: { push: "closed" }, locked: { unlock: "closed" }
      })
    }),
    /no move from closed reaches locked/
  );
});
