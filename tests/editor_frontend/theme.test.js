// @ts-nocheck

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  THEME_STORAGE_KEY,
  describeThemeControl,
  mountThemeControl,
  nextThemeChoice,
  readThemeChoice,
  resolveTheme,
  storeThemeChoice,
} from "../../src/superglm/editor/app/views/theme.js";

class FakeIcon {
  constructor(choice) {
    this.choice = choice;
    this.hidden = false;
  }

  getAttribute(name) {
    return name === "data-theme-icon" ? this.choice : null;
  }

  toggleAttribute(name, force) {
    if (name === "hidden") this.hidden = force;
  }
}

class FakeButton {
  constructor() {
    this.dataset = {};
    this.attributes = new Map();
    this.icons = [new FakeIcon("auto"), new FakeIcon("light"), new FakeIcon("dark")];
    this.listeners = new Map();
  }

  setAttribute(name, value) {
    this.attributes.set(name, value);
  }

  querySelectorAll(selector) {
    return selector === "[data-theme-icon]" ? this.icons : [];
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  removeEventListener(type) {
    this.listeners.delete(type);
  }

  click() {
    this.listeners.get("click")?.();
  }

  shownIcons() {
    return this.icons.filter((icon) => !icon.hidden).map((icon) => icon.choice);
  }
}

class FakeMedia {
  constructor(matches) {
    this.matches = matches;
    this.listeners = new Map();
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  removeEventListener(type) {
    this.listeners.delete(type);
  }

  set(matches) {
    this.matches = matches;
    this.listeners.get("change")?.();
  }
}

function memoryStorage() {
  const store = new Map();
  return {
    store,
    getItem: (key) => (store.has(key) ? store.get(key) : null),
    setItem: (key, value) => store.set(key, value),
    removeItem: (key) => store.delete(key),
  };
}

function appFile(path) {
  return readFileSync(new URL(`../../src/superglm/editor/app/${path}`, import.meta.url), "utf8");
}

test("Auto resolves to the browser's setting and a click first shows the other theme", () => {
  assert.deepEqual(
    [resolveTheme("auto", false), resolveTheme("auto", true), resolveTheme("light", true), resolveTheme("dark", false)],
    ["light", "dark", "light", "dark"],
  );
  // A light browser: Auto shows light, so the clicks go Dark, Light, Auto.
  assert.deepEqual(["auto", "dark", "light"].map((choice) => nextThemeChoice(choice, false)), ["dark", "light", "auto"]);
  assert.deepEqual(["auto", "light", "dark"].map((choice) => nextThemeChoice(choice, true)), ["light", "dark", "auto"]);
});

test("the control names its state and what a click does", () => {
  assert.deepEqual(describeThemeControl("auto", false), {
    label: "Theme: Auto",
    body: "Follows the browser's setting, light now. Click for Dark.",
  });
  assert.deepEqual(describeThemeControl("dark", false), { label: "Theme: Dark", body: "Click for Light." });
  assert.deepEqual(describeThemeControl("light", false), {
    label: "Theme: Light",
    body: "Click for Auto, which follows the browser's setting.",
  });
});

test("the choice is remembered in storage and is Auto without one or with storage blocked", () => {
  const storage = memoryStorage();
  assert.equal(readThemeChoice(storage), "auto");
  storeThemeChoice("dark", storage);
  assert.deepEqual([...storage.store.entries()], [["superglm.editor.theme", "dark"]]);
  assert.equal(readThemeChoice(storage), "dark");
  storage.store.set("superglm.editor.theme", "sepia");
  assert.equal(readThemeChoice(storage), "auto");
  storeThemeChoice("auto", storage);
  assert.equal(storage.store.size, 0);

  const blocked = {
    getItem() { throw new Error("storage disabled"); },
    setItem() { throw new Error("storage disabled"); },
    removeItem() { throw new Error("storage disabled"); },
  };
  assert.equal(readThemeChoice(blocked), "auto");
  assert.doesNotThrow(() => storeThemeChoice("dark", blocked));
  assert.doesNotThrow(() => storeThemeChoice("auto", blocked));
});

test("mounting applies the remembered choice, a click cycles it, and Auto follows the browser", () => {
  const storage = memoryStorage();
  storage.setItem(THEME_STORAGE_KEY, "dark");
  const button = new FakeButton();
  const root = { dataset: {} };
  const media = new FakeMedia(false);
  const control = mountThemeControl({ button, root, media, storage });
  assert.deepEqual([root.dataset.theme, button.dataset.choice, button.shownIcons()], ["dark", "dark", ["dark"]]);
  assert.equal(button.attributes.get("aria-label"), "Theme: Dark");
  assert.deepEqual([button.dataset.popoverTitle, button.dataset.popoverBody], ["Theme: Dark", "Click for Light."]);

  button.click();
  assert.deepEqual([root.dataset.theme, storage.getItem(THEME_STORAGE_KEY)], ["light", "light"]);
  button.click();
  assert.deepEqual([root.dataset.theme, button.dataset.choice, storage.getItem(THEME_STORAGE_KEY)], ["light", "auto", null]);
  media.set(true);
  assert.equal(root.dataset.theme, "dark");
  assert.equal(button.dataset.popoverBody, "Follows the browser's setting, dark now. Click for Light.");
  assert.deepEqual(button.shownIcons(), ["auto"]);

  control.destroy();
  assert.equal(button.listeners.size + media.listeners.size, 0);
});

test("the first-paint script in index.html reads the key the control writes", () => {
  const html = appFile("index.html");
  assert.ok(html.includes(`localStorage.getItem("${THEME_STORAGE_KEY}")`));
  assert.ok(html.includes('id="themeAction"'));
});

test("the dark palette restates every colour token of the light one and no other", () => {
  const names = (css) => new Set([...css.matchAll(/^\s+(--[\w-]+):/gm)].map((match) => match[1]));
  const light = names(appFile("styles/tokens.css"));
  const dark = names(appFile("styles/dark.css"));
  const layout = /^--(font|radius|space|control|feature|inspector|tooltip)/;
  // An alias of another token follows it in both themes.
  const aliases = new Set(["--green", "--trace-best"]);
  const missing = [...light].filter((name) => !dark.has(name) && !layout.test(name) && !aliases.has(name));
  assert.deepEqual(missing, []);
  assert.deepEqual([...dark].filter((name) => !light.has(name)), []);
});
