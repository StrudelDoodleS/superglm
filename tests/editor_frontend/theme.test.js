// @ts-nocheck

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

import {
  FADE_CLASS,
  FLIP_CLASS,
  THEME_STORAGE_KEY,
  describeThemeSwitch,
  mountThemeSwitch,
  readThemeChoice,
  resolveTheme,
  storeThemeChoice,
} from "../../src/superglm/editor/app/views/theme.js";

class FakeClassList {
  constructor() {
    this.names = new Set();
  }

  add(name) {
    this.names.add(name);
  }

  remove(name) {
    this.names.delete(name);
  }

  contains(name) {
    return this.names.has(name);
  }
}

class FakeSwitch {
  constructor() {
    this.dataset = {};
    this.attributes = new Map();
    this.classList = new FakeClassList();
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

  animationEnd(animationName) {
    this.listeners.get("animationend")?.({ animationName });
  }

  get checked() {
    return this.attributes.get("aria-checked") === "true";
  }

  get flipping() {
    return this.classList.contains(FLIP_CLASS);
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
  const root = { dataset: {}, classList: new FakeClassList(), style: { colorScheme: "" } };
  const media = new FakeMedia(prefersDark);
  const store = settings ?? memorySettings(follow);
  const control = mountThemeSwitch({ button, root, media, settings: store, storage });
  return { button, root, media, settings: store, storage, control };
}

const fading = (root) => root.classList.contains(FADE_CLASS);

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

test("a flip plays the switch and fades the page until the knob lands, and nothing else plays", () => {
  const { button, root, media, settings } = mount();
  assert.deepEqual([button.flipping, fading(root)], [false, false]);

  button.click();
  assert.deepEqual([root.dataset.theme, button.flipping, fading(root)], ["dark", true, true]);
  // The fade ends when the knob lands, not when another keyframe does.
  button.animationEnd("theme-icon-in");
  assert.equal(fading(root), true);
  button.animationEnd("theme-knob-to-night");
  assert.equal(fading(root), false);
  // The next flip keeps the class; its keyframes take the other theme's names.
  button.click();
  assert.deepEqual([root.dataset.theme, button.flipping, fading(root)], ["light", true, true]);

  // Settings hands the theme back without motion, and the browser leads without it.
  settings.save({ followBrowserTheme: true });
  assert.deepEqual([button.flipping, fading(root)], [false, false]);
  media.set(true);
  assert.deepEqual([root.dataset.theme, button.flipping, fading(root)], ["dark", false, false]);
});

test("while the page fades it keeps the colour scheme it had, and takes the new one when the knob lands", () => {
  const { button, root, settings } = mount();
  button.click();
  assert.deepEqual([root.dataset.theme, root.style.colorScheme], ["dark", "light"]);
  // A second click mid-flip keeps the scheme the page still has.
  button.click();
  assert.deepEqual([root.dataset.theme, root.style.colorScheme], ["light", "light"]);
  button.animationEnd("theme-knob-to-day");
  assert.deepEqual([fading(root), root.style.colorScheme], [false, ""]);

  // Settings handing the theme back mid-flip ends the fade and the pin with it.
  button.click();
  assert.equal(root.style.colorScheme, "light");
  settings.save({ followBrowserTheme: true });
  assert.deepEqual([fading(root), root.style.colorScheme], [false, ""]);
});

test("destroy removes every listener", () => {
  const { button, media, settings, control } = mount();
  control.destroy();
  assert.equal(button.listeners.size + media.listeners.size + settings.listenerCount(), 0);
});

/**
 * Run index.html's first-paint script alone, as the browser does before any
 * module loads, and return the theme it writes.
 * @param {{stored?: string|null, prefersDark: boolean, blocked?: boolean}} page
 */
function firstPaintTheme({ stored = null, prefersDark, blocked = false }) {
  const source = /<script>([\s\S]*?)<\/script>/.exec(appFile("index.html"))?.[1];
  assert.ok(source, "index.html has an inline first-paint script");
  const root = { dataset: {} };
  const storage = { getItem: (key) => (key === THEME_STORAGE_KEY ? stored : null) };
  const context = {
    document: { documentElement: root },
    matchMedia: (query) => ({ matches: query === "(prefers-color-scheme: dark)" && prefersDark }),
  };
  Object.defineProperty(context, "localStorage", {
    get() {
      if (blocked) throw new Error("The operation is insecure.");
      return storage;
    },
  });
  vm.runInNewContext(source, context);
  return root.dataset.theme;
}

test("the first-paint script paints the theme theme.js will keep", () => {
  const pages = [
    { stored: "dark", prefersDark: false },
    { stored: "light", prefersDark: true },
    { stored: null, prefersDark: true },
    { stored: null, prefersDark: false },
    { stored: "sepia", prefersDark: true },
    { blocked: true, prefersDark: true },
    { blocked: true, prefersDark: false },
  ];
  const painted = pages.map(firstPaintTheme);
  assert.deepEqual(painted, ["dark", "light", "dark", "light", "dark", "dark", "light"]);
  // theme.js takes over from the same key, so nothing changes when the app loads.
  const kept = pages.map(({ stored = null, prefersDark, blocked = false }) =>
    resolveTheme(readThemeChoice(blocked ? BLOCKED : { getItem: () => stored }), prefersDark));
  assert.deepEqual(painted, kept);
});

test("index.html carries the switch", () => {
  const html = appFile("index.html");
  assert.ok(!html.includes('id="themeAction"'));
  const start = html.indexOf('<button id="themeSwitch"');
  assert.notEqual(start, -1);
  const markup = html.slice(start, html.indexOf("</button>", start));
  for (const part of ['role="switch"', 'aria-checked="false"', 'aria-label="Dark theme"', ">DAY<", ">NIGHT<", 'data-icon="sun"', 'data-icon="moon"']) {
    assert.ok(markup.includes(part), part);
  }
  assert.match(html, /fonts\.googleapis\.com\/css2\?family=Bangers&family=Source\+Sans\+3/);
});

test("the flip's keyframes are named for the theme reached, and reduced motion lands them at once", () => {
  const shell = appFile("styles/shell.css");
  for (const name of [
    "theme-knob-to-night", "theme-knob-to-day", "theme-track-to-night", "theme-track-to-day",
    "theme-icon-in", "theme-icon-out", "theme-label-in", "theme-label-out",
  ]) {
    assert.ok(shell.includes(`@keyframes ${name} {`), name);
  }
  // tokens.css zeroes every animation and transition; shell.css drops the switch's delays.
  assert.match(
    appFile("styles/tokens.css"),
    /@media \(prefers-reduced-motion: reduce\) \{\s*\*, \*::before, \*::after \{[^}]*animation-duration: 0\.001ms !important;[^}]*transition-duration: 0\.001ms !important;/,
  );
  assert.match(shell, /@media \(prefers-reduced-motion: reduce\) \{\s*#themeSwitch,\s*#themeSwitch \* \{\s*animation-delay: 0s !important;/);
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
