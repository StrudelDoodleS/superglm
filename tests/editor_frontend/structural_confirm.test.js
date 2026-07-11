// @ts-nocheck

import assert from "node:assert/strict";
import test from "node:test";

import {
  bindStructuralConfirm,
  structuralImpact,
} from "../../src/superglm/editor/app/views/structural_confirm.js";

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

function snapshot(historyCount = 2) {
  return {
    model_revision: 7,
    selected_term: "region",
    terms: {
      region: {
        kind: "categorical",
        term_type: "categorical",
        x: [0, 1, 2],
        y: [1, 1.1, 1.2],
        original_y: [1, 1.1, 1.2],
        previous_y: null,
        levels: ["A", "B", "C"],
        n_points: 3,
        controls: null,
        group_display: null,
        impact: {},
      },
    },
    selection: { region: [1, 2] },
    can_uncollapse_levels: false,
    last_collapse: null,
    history: {
      active: Array.from({ length: historyCount }, (_, index) => ({ index })),
      redo: [],
    },
  };
}

test("empty active history needs no structural confirmation", () => {
  const current = deepFreeze(snapshot(0));
  const operation = deepFreeze({
    name: "collapse levels",
    path: "/collapse_levels",
    payload: { term: "region", method: "auto" },
  });

  assert.deepEqual(structuralImpact(current, operation), {
    requiresConfirmation: false,
  });
});

test("collapse impact copies exact selected category labels and history count", () => {
  const current = deepFreeze(snapshot());
  const operation = deepFreeze({
    name: "collapse levels",
    path: "/collapse_levels",
    payload: { term: "region", method: "auto" },
  });

  const impact = structuralImpact(current, operation);

  assert.deepEqual(impact, {
    requiresConfirmation: true,
    historyCount: 2,
    selectedTerm: "region",
    selectedLabels: ["B", "C"],
    operationTitle: "Collapse levels",
    message: "Collapse levels B, C in region? This refit clears 2 manual edit history entries.",
  });
  assert.notStrictEqual(impact.selectedLabels, current.terms.region.levels);
});

test("structural impact uses exact operation copy for ungroup and restore", () => {
  const current = deepFreeze(snapshot());
  const cases = [
    {
      operation: {
        name: "ungroup levels",
        path: "/ungroup_levels",
        payload: { term: "region", method: "auto" },
      },
      title: "Ungroup levels",
      message: "Ungroup levels B, C in region? This refit clears 2 manual edit history entries.",
    },
    {
      operation: {
        name: "restore collapsed levels",
        path: "/uncollapse_levels",
        payload: {},
      },
      title: "Restore previous collapse",
      message:
        "Restore the previous collapse in region? This refit clears 2 manual edit history entries.",
    },
  ];

  for (const { operation, title, message } of cases) {
    const frozenOperation = deepFreeze(operation);
    const impact = structuralImpact(current, frozenOperation);
    assert.equal(impact.operationTitle, title);
    assert.equal(impact.message, message);
    assert.equal(impact.selectedTerm, "region");
    assert.deepEqual(impact.selectedLabels, ["B", "C"]);
    assert.equal(impact.historyCount, 2);
  }
});

test("structural impact uses singular history grammar without mutating inputs", () => {
  const current = deepFreeze(snapshot(1));
  const operation = deepFreeze({
    name: "ungroup levels",
    path: "/ungroup_levels",
    payload: { term: "region", method: "auto" },
  });
  const before = JSON.stringify(current);

  const impact = structuralImpact(current, operation);

  assert.equal(
    impact.message,
    "Ungroup levels B, C in region? This refit clears 1 manual edit history entry.",
  );
  assert.equal(JSON.stringify(current), before);
  assert.deepEqual(operation, {
    name: "ungroup levels",
    path: "/ungroup_levels",
    payload: { term: "region", method: "auto" },
  });
});

class FakeLauncher {
  constructor() {
    this.focusCalls = [];
    this.isConnected = true;
  }

  focus(options) {
    this.focusCalls.push(options);
    globalThis.document.activeElement = this;
  }
}

class FakeDialog {
  constructor() {
    this.open = false;
    this.returnValue = "stale";
    this.showCalls = 0;
    this.title = { textContent: "Confirm structural refit" };
    this.message = { textContent: "" };
    this.listeners = new Map();
  }

  querySelector(selector) {
    if (selector === "#structuralConfirmTitle") return this.title;
    if (selector === "#structuralConfirmMessage") return this.message;
    return null;
  }

  addEventListener(name, listener) {
    const listeners = this.listeners.get(name) ?? new Set();
    listeners.add(listener);
    this.listeners.set(name, listeners);
  }

  listenerCount(name) {
    return this.listeners.get(name)?.size ?? 0;
  }

  showModal() {
    assert.equal(this.open, false);
    this.open = true;
    this.showCalls += 1;
  }

  dispatch(name) {
    for (const listener of this.listeners.get(name) ?? []) {
      listener({ type: name, target: this });
    }
  }

  close(value) {
    if (value !== undefined) this.returnValue = value;
    this.open = false;
    this.dispatch("close");
  }

  escape() {
    this.dispatch("cancel");
    this.open = false;
    this.dispatch("close");
  }
}

function installDocument(t, launcher) {
  const originalDocument = globalThis.document;
  const originalHTMLElement = globalThis.HTMLElement;
  globalThis.document = { activeElement: launcher };
  globalThis.HTMLElement = FakeLauncher;
  t.after(() => {
    if (originalDocument === undefined) delete globalThis.document;
    else globalThis.document = originalDocument;
    if (originalHTMLElement === undefined) delete globalThis.HTMLElement;
    else globalThis.HTMLElement = originalHTMLElement;
  });
}

function impact() {
  return {
    requiresConfirmation: true,
    historyCount: 2,
    selectedTerm: "region",
    selectedLabels: ["B", "C"],
    operationTitle: "Collapse levels",
    message: "Collapse levels B, C in region? This refit clears 2 manual edit history entries.",
  };
}

test("confirmation bypass resolves true without opening the dialog", async () => {
  const dialog = new FakeDialog();
  const controller = bindStructuralConfirm(dialog);

  assert.equal(await controller.confirm({ requiresConfirmation: false }), true);
  assert.equal(dialog.showCalls, 0);
});

test("cancel populates text safely, resolves false, and restores launcher focus", async (t) => {
  const launcher = new FakeLauncher();
  installDocument(t, launcher);
  const dialog = new FakeDialog();
  const controller = bindStructuralConfirm(dialog);

  const pending = controller.confirm(impact());
  assert.equal(dialog.showCalls, 1);
  assert.equal(dialog.title.textContent, "Collapse levels");
  assert.equal(
    dialog.message.textContent,
    "Collapse levels B, C in region? This refit clears 2 manual edit history entries.",
  );
  dialog.close("cancel");

  assert.equal(await pending, false);
  assert.deepEqual(launcher.focusCalls, [{ preventScroll: true }]);
});

test("Escape cannot reuse a prior return value and resolves false", async (t) => {
  const launcher = new FakeLauncher();
  installDocument(t, launcher);
  const dialog = new FakeDialog();
  dialog.returnValue = "confirm";
  const controller = bindStructuralConfirm(dialog);

  const pending = controller.confirm(impact());
  assert.equal(dialog.returnValue, "");
  dialog.escape();

  assert.equal(await pending, false);
  assert.deepEqual(launcher.focusCalls, [{ preventScroll: true }]);
});

test("Continue alone resolves true and repeated calls do not duplicate listeners", async (t) => {
  const firstLauncher = new FakeLauncher();
  installDocument(t, firstLauncher);
  const dialog = new FakeDialog();
  const controller = bindStructuralConfirm(dialog);

  const first = controller.confirm(impact());
  const duplicate = controller.confirm(impact());
  assert.equal(await duplicate, false);
  dialog.close("confirm");
  assert.equal(await first, true);

  const secondLauncher = new FakeLauncher();
  globalThis.document.activeElement = secondLauncher;
  const second = controller.confirm({
    ...impact(),
    operationTitle: "Ungroup levels",
    message: "Ungroup levels B, C in region? This refit clears 2 manual edit history entries.",
  });
  dialog.close("cancel");

  assert.equal(await second, false);
  assert.equal(dialog.showCalls, 2);
  assert.equal(dialog.listenerCount("close"), 1);
  assert.equal(dialog.listenerCount("cancel"), 1);
  assert.deepEqual(firstLauncher.focusCalls, [{ preventScroll: true }]);
  assert.deepEqual(secondLauncher.focusCalls, [{ preventScroll: true }]);
});
