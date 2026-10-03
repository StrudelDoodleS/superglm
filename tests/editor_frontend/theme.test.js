// @ts-nocheck

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  THEME_STORAGE_KEY,
  describeThemeSwitch,
  mountThemeSwitch,
  readThemeChoice,
  resolveTheme,
  storeThemeChoice,
} from "../../src/superglm/editor/app/views/theme.js";

class FakeSwitch {
  constructor() {
    this.dataset = {};
    this.attributes = new Map();
    this.listeners = new Map();
  }

  setAttribute(name, value) {
    this.attributes.set(name, value);
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

  get checked() {
    return this.attributes.get("aria-checked") === "true";
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

const BLOCKED = {
  getItem() { throw new Error("storage disabled"); },
  setItem() { throw new Error("storage disabled"); },
  removeItem() { throw new Error("storage disabled"); },
};

// The settings store as views/settings.js keeps it: the current settings in
// memory, every listener told synchronously after a save.
function memorySettings(followBrowserTheme) {
  let current = { followBrowserTheme };
  const listeners = new Set();
  return {
    saves: [],
    load: () => ({ ...current }),
    save(patch) {
      current = { ...current, ...patch };
      this.saves.push(patch);
      for (const listener of [...listeners]) listener({ ...current });
      return { ...current };
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    listenerCount: () => listeners.size,
  };
}

function mount({ stored = null, follow = true, prefersDark = false, storage = memoryStorage(), settings } = {}) {
  if (stored !== null) storage.setItem(THEME_STORAGE_KEY, stored);
  const button = new FakeSwitch();
  const root = { dataset: {} };
  const media = new FakeMedia(prefersDark);
  const store = settings ?? memorySettings(follow);
  const control = mountThemeSwitch({ button, root, media, settings: store, storage });
  return { button, root, media, settings: store, storage, control };
}

function appFile(path) {
  return readFileSync(new URL(`../../src/superglm/editor/app/${path}`, import.meta.url), "utf8");
}

test("a stored choice is explicit, and no stored choice or blocked storage follows the browser", () => {
  assert.deepEqual(
    [resolveTheme("auto", false), resolveTheme("auto", true), resolveTheme("light", true), resolveTheme("dark", false)],
    ["light", "dark", "light", "dark"],
  );
  const storage = memoryStorage();
  assert.equal(readThemeChoice(storage), "auto");
  storeThemeChoice("dark", storage);
  assert.deepEqual([...storage.store.entries()], [["superglm.editor.theme", "dark"]]);
  assert.equal(readThemeChoice(storage), "dark");
  storage.store.set("superglm.editor.theme", "sepia");
  assert.equal(readThemeChoice(storage), "auto");
  storeThemeChoice("auto", storage);
  assert.equal(storage.store.size, 0);
  assert.equal(readThemeChoice(BLOCKED), "auto");
  assert.doesNotThrow(() => storeThemeChoice("dark", BLOCKED));
  assert.doesNotThrow(() => storeThemeChoice("auto", BLOCKED));
});

test("the switch is on for Night and says what a click does", () => {
  assert.deepEqual(describeThemeSwitch("auto", false), {
    checked: false,
    title: "Theme: Day",
    body: "Follows the browser's setting. Click for Night; the theme then stays as you set it.",
  });
  assert.deepEqual(describeThemeSwitch("auto", true), {
    checked: true,
    title: "Theme: Night",
    body: "Follows the browser's setting. Click for Day; the theme then stays as you set it.",
  });
  assert.deepEqual(describeThemeSwitch("dark", false), {
    checked: true,
    title: "Theme: Night",
    body: "Click for Day. Settings can follow the browser's setting again.",
  });
  assert.deepEqual(describeThemeSwitch("light", true), {
    checked: false,
    title: "Theme: Day",
    body: "Click for Night. Settings can follow the browser's setting again.",
  });
});

test("a flip stores the theme and stops following the browser", () => {
  const { button, root, media, settings, storage } = mount();
  assert.deepEqual([root.dataset.theme, button.checked, button.dataset.popoverTitle], ["light", false, "Theme: Day"]);
  assert.deepEqual(settings.saves, []);

  button.click();
  assert.deepEqual([root.dataset.theme, button.checked, storage.getItem(THEME_STORAGE_KEY)], ["dark", true, "dark"]);
  assert.deepEqual(settings.saves, [{ followBrowserTheme: false }]);
  // A flipped switch stays put when the browser changes.
  media.set(true);
  media.set(false);
  assert.equal(root.dataset.theme, "dark");

  button.click();
  assert.deepEqual([root.dataset.theme, storage.getItem(THEME_STORAGE_KEY)], ["light", "light"]);
  assert.deepEqual(settings.saves, [{ followBrowserTheme: false }, { followBrowserTheme: false }]);
});

test("while following the browser the switch moves with it", () => {
  const { button, root, media } = mount({ prefersDark: false });
  media.set(true);
  assert.deepEqual([root.dataset.theme, button.checked], ["dark", true]);
  assert.equal(button.dataset.popoverBody, "Follows the browser's setting. Click for Day; the theme then stays as you set it.");
});

test("Settings hands the theme back to the browser, and turning that off keeps the theme on screen", () => {
  const { button, root, media, settings, storage } = mount({ prefersDark: false });
  button.click();
  settings.save({ followBrowserTheme: true });
  assert.deepEqual([root.dataset.theme, storage.getItem(THEME_STORAGE_KEY)], ["light", null]);
  media.set(true);
  assert.equal(root.dataset.theme, "dark");

  settings.save({ followBrowserTheme: false });
  assert.equal(storage.getItem(THEME_STORAGE_KEY), "dark");
  media.set(false);
  assert.equal(root.dataset.theme, "dark");
});

test("the theme key decides, and the setting is brought into line with it on mount", () => {
  // A choice made before the setting existed: still explicit.
  const legacy = mount({ stored: "dark", follow: true, prefersDark: false });
  assert.equal(legacy.root.dataset.theme, "dark");
  assert.deepEqual(legacy.settings.saves, [{ followBrowserTheme: false }]);
  // No stored choice: the browser's theme, whatever the setting said.
  const unset = mount({ stored: null, follow: false, prefersDark: true });
  assert.equal(unset.root.dataset.theme, "dark");
  assert.deepEqual(unset.settings.saves, [{ followBrowserTheme: true }]);
  // In line already: nothing is saved.
  assert.deepEqual(mount({ stored: "light", follow: false }).settings.saves, []);
});

test("with storage blocked the switch follows the browser, flips for the page, and nothing throws", () => {
  const { button, root, media } = mount({ storage: BLOCKED, prefersDark: true });
  assert.equal(root.dataset.theme, "dark");
  assert.doesNotThrow(() => button.click());
  assert.equal(root.dataset.theme, "light");
  media.set(false);
  media.set(true);
  assert.equal(root.dataset.theme, "light");

  // A settings store that keeps nothing echoes its defaults back; the flip holds.
  const listeners = new Set();
  const forgetful = {
    load: () => ({ followBrowserTheme: true }),
    save() {
      for (const listener of listeners) listener({ followBrowserTheme: true });
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  const page = mount({ storage: BLOCKED, prefersDark: false, settings: forgetful });
  page.button.click();
  assert.equal(page.root.dataset.theme, "dark");
});

test("destroy removes every listener", () => {
  const { button, media, settings, control } = mount();
  control.destroy();
  assert.equal(button.listeners.size + media.listeners.size + settings.listenerCount(), 0);
});

test("index.html paints the stored theme first and carries the switch", () => {
  const html = appFile("index.html");
  assert.ok(html.includes(`localStorage.getItem("${THEME_STORAGE_KEY}")`));
  assert.ok(!html.includes('id="themeAction"'));
  const start = html.indexOf('<button id="themeSwitch"');
  assert.notEqual(start, -1);
  const markup = html.slice(start, html.indexOf("</button>", start));
  for (const part of ['role="switch"', 'aria-checked="false"', 'aria-label="Dark theme"', ">DAY<", ">NIGHT<", 'data-icon="sun"', 'data-icon="moon"']) {
    assert.ok(markup.includes(part), part);
  }
  assert.match(html, /fonts\.googleapis\.com\/css2\?family=Bangers&family=Source\+Sans\+3/);
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
