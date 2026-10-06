// @ts-check
// Editor preferences and the Settings pane. The preferences live in this
// browser under one localStorage key, as one JSON object. Storage that is
// blocked or broken never stops the page: a read falls back to the defaults
// and a change lasts for the page. The theme keeps its own key, which the
// first-paint script in index.html reads; "Follow the browser" mirrors it, and
// views/theme.js keeps the two equal.

/** @typedef {"expanded"|"collapsed"} GroupsDefault */
/**
 * @typedef {object} EditorSettings
 * @property {boolean} refitEveryChange refit after every structural change instead of waiting for Refit
 * @property {boolean} keepReference collapsing and ungrouping keep the reference level
 * @property {boolean} followBrowserTheme the theme follows the browser's light or dark setting
 * @property {GroupsDefault} groupsDefault how a grouped term's levels are drawn when it opens
 * @property {number} buildDurationMs how long a Build animation runs
 * @property {boolean} showTimings show the request-timing readout
 */
/** @typedef {"refitEveryChange"|"keepReference"|"followBrowserTheme"|"showTimings"} SettingsFlag */
/** @typedef {Pick<Storage, 'getItem'|'setItem'>} SettingsStorage */
/**
 * @typedef {object} SettingsStore
 * @property {()=>Readonly<EditorSettings>} load
 * @property {(patch:Partial<EditorSettings>)=>Readonly<EditorSettings>} save
 * @property {(listener:(settings:Readonly<EditorSettings>)=>void)=>()=>void} subscribe
 */
/**
 * @typedef {object} SettingsPaneNodes
 * @property {HTMLElement} root the Settings pane
 * @property {HTMLInputElement} buildDuration
 * @property {HTMLElement} buildDurationValue
 * @property {HTMLElement} timing the request-timing readout
 */

export const SETTINGS_STORAGE_KEY = "superglm.editor.settings";

/** The Build animation's range in ms; the slider in index.html states the same. */
export const BUILD_DURATION_RANGE = Object.freeze({ min: 4000, max: 30000, step: 500 });

/** @type {Readonly<EditorSettings>} */
export const DEFAULT_SETTINGS = Object.freeze({
  refitEveryChange: false,
  keepReference: true,
  followBrowserTheme: true,
  groupsDefault: "expanded",
  buildDurationMs: 10000,
  showTimings: false,
});

/** @type {ReadonlySet<string>} */
const SWITCHES = new Set(["refitEveryChange", "keepReference", "followBrowserTheme", "showTimings"]);
const GROUPS_INPUT = "settingGroupsDefault";

/**
 * Settings from a stored value: each field present with a valid value, the
 * default for every other, so an old, partial or hand-edited entry still
 * loads. The Build animation's length is held to its slider's range and step.
 * @param {unknown} value
 * @returns {Readonly<EditorSettings>}
 */
export function normaliseSettings(value) {
  /** @type {Record<string, unknown>} */
  const source = value !== null && typeof value === "object" && !Array.isArray(value)
    ? /** @type {Record<string, unknown>} */ (value)
    : {};
  /** @param {SettingsFlag} key */
  const flag = (key) =>
    typeof source[key] === "boolean" ? Boolean(source[key]) : DEFAULT_SETTINGS[key];
  const groups = source.groupsDefault;
  return Object.freeze({
    refitEveryChange: flag("refitEveryChange"),
    keepReference: flag("keepReference"),
    followBrowserTheme: flag("followBrowserTheme"),
    groupsDefault: groups === "expanded" || groups === "collapsed"
      ? groups
      : DEFAULT_SETTINGS.groupsDefault,
    buildDurationMs: buildDuration(source.buildDurationMs),
    showTimings: flag("showTimings"),
  });
}

/** @param {unknown} value */
function buildDuration(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return DEFAULT_SETTINGS.buildDurationMs;
  const { min, max, step } = BUILD_DURATION_RANGE;
  return Math.min(max, Math.max(min, Math.round(value / step) * step));
}

/** @param {Readonly<EditorSettings>} left @param {Readonly<EditorSettings>} right */
function sameSettings(left, right) {
  return left.refitEveryChange === right.refitEveryChange &&
    left.keepReference === right.keepReference &&
    left.followBrowserTheme === right.followBrowserTheme &&
    left.groupsDefault === right.groupsDefault &&
    left.buildDurationMs === right.buildDurationMs &&
    left.showTimings === right.showTimings;
}

/** @param {SettingsStorage|undefined} storage @returns {Readonly<EditorSettings>} */
function readStored(storage) {
  try {
    const stored = (storage ?? localStorage).getItem(SETTINGS_STORAGE_KEY);
    return stored === null ? DEFAULT_SETTINGS : normaliseSettings(JSON.parse(stored));
  } catch {
    // Blocked, unusable or corrupt storage: the defaults.
    return DEFAULT_SETTINGS;
  }
}

/** @param {Readonly<EditorSettings>} settings @param {SettingsStorage|undefined} storage */
function writeStored(settings, storage) {
  try {
    (storage ?? localStorage).setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Unusable storage: the settings last this page only.
  }
}

/**
 * A settings store over `storage`, which is localStorage when omitted and is
 * read on first use. It keeps the page's settings itself, so a change holds
 * for the page even where storage refuses it, and it tells its listeners once
 * per change.
 * @param {{storage?:SettingsStorage}} [options]
 * @returns {SettingsStore}
 */
export function createSettingsStore({ storage } = {}) {
  /** @type {Readonly<EditorSettings>|null} */
  let current = null;
  /** @type {Set<(settings:Readonly<EditorSettings>)=>void>} */
  const listeners = new Set();

  function load() {
    if (current === null) current = readStored(storage);
    return current;
  }

  /** @param {Partial<EditorSettings>} patch */
  function save(patch) {
    const previous = load();
    const next = normaliseSettings({ ...previous, ...patch });
    if (sameSettings(next, previous)) return previous;
    current = next;
    writeStored(next, storage);
    for (const listener of [...listeners]) listener(next);
    return next;
  }

  /** @param {(settings:Readonly<EditorSettings>)=>void} listener */
  function subscribe(listener) {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }

  return Object.freeze({ load, save, subscribe });
}

const browserSettings = createSettingsStore();

/** The page's settings, read from this browser's storage on first use. */
export function loadSettings() {
  return browserSettings.load();
}

/** @param {Partial<EditorSettings>} patch */
export function saveSettings(patch) {
  return browserSettings.save(patch);
}

/** @param {(settings:Readonly<EditorSettings>)=>void} listener @returns {()=>void} */
export function onSettingsChange(listener) {
  return browserSettings.subscribe(listener);
}

/** @param {number} ms @returns {string} */
export function formatBuildDuration(ms) {
  const seconds = ms / 1000;
  return `${Number.isInteger(seconds) ? seconds : seconds.toFixed(1)} s`;
}

/** @param {string|undefined} value @returns {value is SettingsFlag} */
function isSwitch(value) {
  return value !== undefined && SWITCHES.has(value);
}

/**
 * Show the settings in the pane: each switch, the groups choice, the Build
 * animation's length, and whether the timing readout shows.
 * @param {SettingsPaneNodes} nodes
 * @param {{settings:Readonly<EditorSettings>}} state
 */
export function renderSettingsPane(nodes, { settings }) {
  for (const element of nodes.root.querySelectorAll('[role="switch"][data-setting]')) {
    if (!(element instanceof HTMLElement)) continue;
    const key = element.dataset.setting;
    if (isSwitch(key)) element.setAttribute("aria-checked", String(settings[key]));
  }
  for (const element of nodes.root.querySelectorAll(`input[name="${GROUPS_INPUT}"]`)) {
    if (element instanceof HTMLInputElement) element.checked = element.value === settings.groupsDefault;
  }
  nodes.buildDuration.value = String(settings.buildDurationMs);
  nodes.buildDurationValue.textContent = formatBuildDuration(settings.buildDurationMs);
  nodes.timing.hidden = !settings.showTimings;
}

/**
 * Bind the pane. A switch reports its setting, the groups choice its value,
 * and the slider its length when released. While it moves, only its label
 * follows. The settings themselves live with the caller, which renders them
 * back.
 * @param {SettingsPaneNodes} nodes
 * @param {{
 *   onToggle:(key:SettingsFlag)=>unknown,
 *   onGroupsDefault:(value:GroupsDefault)=>unknown,
 *   onBuildDuration:(ms:number)=>unknown
 * }} handlers
 * @returns {{destroy:()=>void}}
 */
export function bindSettingsPane(nodes, { onToggle, onGroupsDefault, onBuildDuration }) {
  /** @param {Event} event */
  function onClick(event) {
    const target = event.target instanceof Element
      ? event.target.closest('[role="switch"][data-setting]')
      : null;
    if (!(target instanceof HTMLElement) || !nodes.root.contains(target)) return;
    const key = target.dataset.setting;
    if (isSwitch(key)) onToggle(key);
  }

  /** @param {Event} event */
  function onChange(event) {
    const target = event.target;
    if (target === nodes.buildDuration) {
      onBuildDuration(Number(nodes.buildDuration.value));
      return;
    }
    if (
      target instanceof HTMLInputElement &&
      target.name === GROUPS_INPUT &&
      target.checked &&
      (target.value === "expanded" || target.value === "collapsed")
    ) {
      onGroupsDefault(target.value);
    }
  }

  function onInput() {
    nodes.buildDurationValue.textContent = formatBuildDuration(Number(nodes.buildDuration.value));
  }

  nodes.root.addEventListener("click", onClick);
  nodes.root.addEventListener("change", onChange);
  nodes.buildDuration.addEventListener("input", onInput);
  return Object.freeze({
    destroy() {
      nodes.root.removeEventListener("click", onClick);
      nodes.root.removeEventListener("change", onChange);
      nodes.buildDuration.removeEventListener("input", onInput);
    },
  });
}
