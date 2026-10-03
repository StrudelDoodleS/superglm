// @ts-nocheck

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  BUILD_DURATION_RANGE,
  DEFAULT_SETTINGS,
  SETTINGS_STORAGE_KEY,
  bindSettingsPane,
  createSettingsStore,
  formatBuildDuration,
  normaliseSettings,
  renderSettingsPane,
} from "../../src/superglm/editor/app/views/settings.js";

function memoryStorage(entries = []) {
  const store = new Map(entries);
  return {
    store,
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => store.set(key, String(value)),
  };
}

const BLOCKED = Object.freeze({
  getItem() { throw new Error("storage disabled"); },
  setItem() { throw new Error("storage disabled"); },
});

test("the defaults are the contract's, frozen, under one key", () => {
  assert.equal(SETTINGS_STORAGE_KEY, "superglm.editor.settings");
  assert.deepEqual(DEFAULT_SETTINGS, {
    refitEveryChange: false,
    keepReference: true,
    followBrowserTheme: true,
    groupsDefault: "expanded",
    buildDurationMs: 10000,
    showTimings: false,
  });
  assert.equal(Object.isFrozen(DEFAULT_SETTINGS), true);
  assert.deepEqual(createSettingsStore({ storage: memoryStorage() }).load(), DEFAULT_SETTINGS);
});

test("a change is saved as one JSON object, reported once, and the other keys are left alone", () => {
  const storage = memoryStorage([
    ["superglm.editor.theme", "dark"],
    ["superglm.editor.shapeJoin", "kink"],
  ]);
  const settings = createSettingsStore({ storage });
  const heard = [];
  const unsubscribe = settings.subscribe((value) => heard.push(value));

  const saved = settings.save({ refitEveryChange: true, buildDurationMs: 6000 });
  assert.deepEqual(saved, { ...DEFAULT_SETTINGS, refitEveryChange: true, buildDurationMs: 6000 });
  assert.deepEqual(JSON.parse(storage.store.get(SETTINGS_STORAGE_KEY)), saved);
  assert.equal(storage.store.get("superglm.editor.theme"), "dark");
  assert.equal(storage.store.get("superglm.editor.shapeJoin"), "kink");
  assert.deepEqual(heard, [saved]);

  // Saving what is already there changes nothing and reports nothing.
  assert.strictEqual(settings.save({ refitEveryChange: true }), saved);
  assert.equal(heard.length, 1);

  unsubscribe();
  settings.save({ showTimings: true });
  assert.equal(heard.length, 1);

  // A later page reads back what this one saved.
  assert.deepEqual(createSettingsStore({ storage }).load(), { ...saved, showTimings: true });
});

test("an old, partial or hand-edited entry keeps its valid fields and defaults the rest", () => {
  const storage = memoryStorage([[SETTINGS_STORAGE_KEY, JSON.stringify({
    refitEveryChange: "yes",
    keepReference: false,
    groupsDefault: "sideways",
    buildDurationMs: 99999,
    retired: true,
  })]]);
  assert.deepEqual(createSettingsStore({ storage }).load(), {
    ...DEFAULT_SETTINGS,
    keepReference: false,
    buildDurationMs: BUILD_DURATION_RANGE.max,
  });
  assert.equal(normaliseSettings({ buildDurationMs: 6234 }).buildDurationMs, 6000);
  assert.equal(normaliseSettings({ buildDurationMs: 100 }).buildDurationMs, BUILD_DURATION_RANGE.min);
  assert.deepEqual(normaliseSettings([true]), DEFAULT_SETTINGS);
  assert.deepEqual(normaliseSettings(null), DEFAULT_SETTINGS);
  const corrupt = memoryStorage([[SETTINGS_STORAGE_KEY, "{not json"]]);
  assert.deepEqual(createSettingsStore({ storage: corrupt }).load(), DEFAULT_SETTINGS);
});

test("blocked storage gives the defaults, and a change lasts for the page without throwing", () => {
  const settings = createSettingsStore({ storage: BLOCKED });
  assert.deepEqual(settings.load(), DEFAULT_SETTINGS);
  const heard = [];
  settings.subscribe((value) => heard.push(value.refitEveryChange));
  assert.doesNotThrow(() => settings.save({ refitEveryChange: true }));
  assert.equal(settings.load().refitEveryChange, true);
  assert.deepEqual(heard, [true]);

  // Storage whose very access throws, as a blocked origin's does.
  const hostile = {
    get getItem() { throw new Error("SecurityError"); },
    get setItem() { throw new Error("SecurityError"); },
  };
  const page = createSettingsStore({ storage: hostile });
  assert.deepEqual(page.load(), DEFAULT_SETTINGS);
  assert.doesNotThrow(() => page.save({ showTimings: true }));
  assert.equal(page.load().showTimings, true);
});

class FakeElement {
  constructor({ tag = "div", dataset = {}, attributes = {}, name = "", value = "", parent = null } = {}) {
    this.tagName = tag.toUpperCase();
    this.dataset = dataset;
    this.attributes = new Map(Object.entries(attributes));
    this.name = name;
    this.value = value;
    this.checked = false;
    this.hidden = false;
    this.textContent = "";
    this.parentNode = parent;
    this.listeners = new Map();
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  closest(selector) {
    if (selector !== '[role="switch"][data-setting]') throw new Error(`fake DOM cannot match ${selector}`);
    for (let node = this; node; node = node.parentNode) {
      if (node.getAttribute("role") === "switch" && node.dataset.setting) return node;
    }
    return null;
  }

  contains(node) {
    for (let current = node; current; current = current.parentNode) {
      if (current === this) return true;
    }
    return false;
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type, listener) {
    this.listeners.get(type)?.delete(listener);
  }

  // Bubbles from `target` to the root, like the real event path.
  emit(type, target = this) {
    const event = { type, target };
    for (let node = target; node; node = node.parentNode) {
      for (const listener of node.listeners.get(type) ?? []) listener(event);
    }
  }
}

class FakeInput extends FakeElement {}

globalThis.Element = FakeElement;
globalThis.HTMLElement = FakeElement;
globalThis.HTMLInputElement = FakeInput;

const SWITCH_KEYS = ["refitEveryChange", "keepReference", "followBrowserTheme", "showTimings"];

function pane() {
  const root = new FakeElement();
  const switches = SWITCH_KEYS.map((setting) => new FakeElement({
    tag: "button",
    dataset: { setting },
    attributes: { role: "switch", "aria-checked": "false" },
    parent: root,
  }));
  const radios = ["expanded", "collapsed"].map((value) => new FakeInput({
    tag: "input",
    name: "settingGroupsDefault",
    value,
    parent: root,
  }));
  root.querySelectorAll = (selector) => {
    if (selector === '[role="switch"][data-setting]') return switches;
    if (selector === 'input[name="settingGroupsDefault"]') return radios;
    throw new Error(`fake DOM cannot match ${selector}`);
  };
  const nodes = {
    root,
    buildDuration: new FakeInput({ tag: "input", value: "10000", parent: root }),
    buildDurationValue: new FakeElement({ parent: root }),
    timing: new FakeElement({ parent: root }),
  };
  const checked = () => Object.fromEntries(
    switches.map((node) => [node.dataset.setting, node.getAttribute("aria-checked")]),
  );
  return { nodes, switches, radios, checked };
}

test("the pane shows each setting", () => {
  const { nodes, radios, checked } = pane();
  renderSettingsPane(nodes, {
    settings: {
      ...DEFAULT_SETTINGS,
      refitEveryChange: true,
      followBrowserTheme: false,
      groupsDefault: "collapsed",
      buildDurationMs: 6500,
    },
  });
  assert.deepEqual(checked(), {
    refitEveryChange: "true",
    keepReference: "true",
    followBrowserTheme: "false",
    showTimings: "false",
  });
  assert.deepEqual(radios.map((radio) => radio.checked), [false, true]);
  assert.equal(nodes.buildDuration.value, "6500");
  assert.equal(nodes.buildDurationValue.textContent, "6.5 s");
  assert.equal(nodes.timing.hidden, true);

  renderSettingsPane(nodes, { settings: { ...DEFAULT_SETTINGS, showTimings: true } });
  assert.equal(nodes.timing.hidden, false);
  assert.equal(checked().followBrowserTheme, "true");
  assert.equal(formatBuildDuration(10000), "10 s");
});

test("with storage blocked the pane renders the defaults", () => {
  const { nodes, radios, checked } = pane();
  renderSettingsPane(nodes, { settings: createSettingsStore({ storage: BLOCKED }).load() });
  assert.deepEqual(checked(), {
    refitEveryChange: "false",
    keepReference: "true",
    followBrowserTheme: "true",
    showTimings: "false",
  });
  assert.deepEqual(radios.map((radio) => radio.checked), [true, false]);
  assert.equal(nodes.buildDurationValue.textContent, "10 s");
});

test("a switch reports its key, a groups choice its value, and the slider saves on release", () => {
  const { nodes, switches, radios } = pane();
  const calls = [];
  const binding = bindSettingsPane(nodes, {
    onToggle: (key) => calls.push(["toggle", key]),
    onGroupsDefault: (value) => calls.push(["groups", value]),
    onBuildDuration: (ms) => calls.push(["build", ms]),
  });

  nodes.root.emit("click", switches[0]);
  nodes.root.emit("click", switches[2]);
  radios[1].checked = true;
  nodes.root.emit("change", radios[1]);
  nodes.buildDuration.value = "7500";
  nodes.buildDuration.emit("input");
  // While the slider moves only its label follows; nothing is saved yet.
  assert.equal(nodes.buildDurationValue.textContent, "7.5 s");
  assert.equal(calls.length, 3);
  nodes.root.emit("change", nodes.buildDuration);

  assert.deepEqual(calls, [
    ["toggle", "refitEveryChange"],
    ["toggle", "followBrowserTheme"],
    ["groups", "collapsed"],
    ["build", 7500],
  ]);
  binding.destroy();
  nodes.root.emit("click", switches[0]);
  assert.equal(calls.length, 4);
});

test("index.html carries one control per setting, and the slider's range is the module's", () => {
  const html = readFileSync(
    new URL("../../src/superglm/editor/app/index.html", import.meta.url),
    "utf8",
  );
  for (const key of SWITCH_KEYS) {
    assert.equal(html.split(`data-setting="${key}"`).length - 1, 1, key);
  }
  assert.equal(html.split('name="settingGroupsDefault"').length - 1, 2);
  const { min, max, step } = BUILD_DURATION_RANGE;
  assert.ok(html.includes(`id="buildDuration" type="range" min="${min}" max="${max}" step="${step}"`));
  assert.ok(html.includes('id="settingsPane"'));
  assert.ok(!html.includes('id="advancedPane"'));
});
