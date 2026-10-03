// @ts-check
// The theme switch in the app bar: a comic pill reading DAY or NIGHT. Until
// it is flipped the editor follows the browser's colour-scheme setting, which
// inside a notebook is not always the notebook's own theme; a flip is an
// explicit choice that outlives the page through localStorage when it can,
// and Settings can hand the theme back to the browser. The resolved theme is
// written to <html data-theme>, which styles/dark.css keys the dark palette on
// and the switch's rest position follows; index.html writes the same attribute
// before first paint.
//
// Which store decides: the theme key. A stored "light" or "dark" is an
// explicit choice and no stored value means follow the browser, so the
// first-paint script reads that key alone. The followBrowserTheme setting
// mirrors it, and this module keeps the two equal: it reconciles them on
// mount, a flip clears the setting, and turning the setting on or off in
// Settings removes the key or stores the theme on screen.
//
// A flip plays the switch's keyframes and cross-fades the page's grounds
// (styles/shell.css). A change from the browser or from Settings lands
// without motion.

/** @typedef {"auto"|"light"|"dark"} ThemeChoice */
/** @typedef {"light"|"dark"} Theme */
/** @typedef {Pick<Storage, 'getItem'|'setItem'|'removeItem'>} ThemeStorage */
/** @typedef {Pick<MediaQueryList, 'matches'|'addEventListener'|'removeEventListener'>} DarkMedia */
/** @typedef {{followBrowserTheme: boolean}} FollowSetting */
/**
 * The part of views/settings.js the switch uses, passed in by main.js.
 * @typedef {{
 *   load: () => FollowSetting,
 *   save: (patch: FollowSetting) => unknown,
 *   subscribe: (listener: (settings: FollowSetting) => void) => () => void,
 * }} ThemeSettings
 */

export const THEME_STORAGE_KEY = "superglm.editor.theme";
/** On the switch while a flip's keyframes may play. */
export const FLIP_CLASS = "is-flipping";
/** On <html> while the page's grounds cross-fade after a flip. */
export const FADE_CLASS = "theme-fading";

/** @param {unknown} value @returns {value is ThemeChoice} */
export function isThemeChoice(value) {
  return value === "auto" || value === "light" || value === "dark";
}

/**
 * The remembered choice, or Auto with no stored choice or unusable storage
 * (private mode, a blocked origin).
 * @param {Pick<ThemeStorage, 'getItem'>} [storage] defaults to localStorage, whose access may itself throw
 * @returns {ThemeChoice}
 */
export function readThemeChoice(storage) {
  try {
    const stored = (storage ?? localStorage).getItem(THEME_STORAGE_KEY);
    return isThemeChoice(stored) ? stored : "auto";
  } catch {
    return "auto";
  }
}

/**
 * Remember the choice; Auto is the absence of one.
 * @param {ThemeChoice} choice @param {ThemeStorage} [storage]
 */
export function storeThemeChoice(choice, storage) {
  try {
    const store = storage ?? localStorage;
    if (choice === "auto") store.removeItem(THEME_STORAGE_KEY);
    else store.setItem(THEME_STORAGE_KEY, choice);
  } catch {
    // Unusable storage: the choice lasts this page only.
  }
}

/** @param {ThemeChoice} choice @param {boolean} prefersDark @returns {Theme} */
export function resolveTheme(choice, prefersDark) {
  if (choice === "auto") return prefersDark ? "dark" : "light";
  return choice;
}

/**
 * What the switch says of itself: on for Night, its popover, what a click does.
 * @param {ThemeChoice} choice @param {boolean} prefersDark
 * @returns {{checked: boolean, title: string, body: string}}
 */
export function describeThemeSwitch(choice, prefersDark) {
  const dark = resolveTheme(choice, prefersDark) === "dark";
  const next = dark ? "Day" : "Night";
  return {
    checked: dark,
    title: `Theme: ${dark ? "Night" : "Day"}`,
    body: choice === "auto"
      ? `Follows the browser's setting. Click for ${next}; the theme then stays as you set it.`
      : `Click for ${next}. Settings can follow the browser's setting again.`,
  };
}

/**
 * Show the choice on the switch: its state and its popover.
 * @param {HTMLElement} button @param {ThemeChoice} choice @param {boolean} prefersDark
 */
export function renderThemeSwitch(button, choice, prefersDark) {
  const { checked, title, body } = describeThemeSwitch(choice, prefersDark);
  button.dataset.choice = choice;
  button.setAttribute("aria-checked", String(checked));
  button.dataset.popoverTitle = title;
  button.dataset.popoverBody = body;
}

/**
 * Mount the switch: apply the remembered choice, flip it on a click, follow
 * the browser while no choice is stored, and keep the follow setting equal to
 * that.
 * @param {{button:HTMLElement, root:HTMLElement, media:DarkMedia, settings:ThemeSettings, storage?:ThemeStorage}} options
 * @returns {{destroy:()=>void}}
 */
export function mountThemeSwitch({ button, root, media, settings, storage }) {
  let choice = readThemeChoice(storage);
  // Set while this module saves the setting, so its own echo is not taken
  // for a change made in Settings.
  let saving = false;

  function mirror() {
    saving = true;
    try {
      settings.save({ followBrowserTheme: choice === "auto" });
    } finally {
      saving = false;
    }
  }

  function render() {
    root.dataset.theme = resolveTheme(choice, media.matches);
    renderThemeSwitch(button, choice, media.matches);
  }

  function onClick() {
    choice = resolveTheme(choice, media.matches) === "dark" ? "light" : "dark";
    storeThemeChoice(choice, storage);
    mirror();
    button.classList.add(FLIP_CLASS);
    root.classList.add(FADE_CLASS);
    render();
  }

  // While the browser leads the switch carries no flip: a flip ends the
  // following, and Settings clears the flip when it hands the theme back.
  function onBrowserChange() {
    if (choice === "auto") render();
  }

  /** @param {FollowSetting} next */
  function onSettingsChange(next) {
    if (saving || next.followBrowserTheme === (choice === "auto")) return;
    choice = next.followBrowserTheme ? "auto" : resolveTheme(choice, media.matches);
    storeThemeChoice(choice, storage);
    // Kept, the last flip's keyframes would replay under the new theme's names.
    button.classList.remove(FLIP_CLASS);
    root.classList.remove(FADE_CLASS);
    render();
  }

  /** The knob has landed, and the fade with it. @param {Event} event */
  function onAnimationEnd(event) {
    if (/** @type {AnimationEvent} */ (event).animationName.startsWith("theme-knob-")) {
      root.classList.remove(FADE_CLASS);
    }
  }

  if (settings.load().followBrowserTheme !== (choice === "auto")) mirror();
  button.addEventListener("click", onClick);
  button.addEventListener("animationend", onAnimationEnd);
  media.addEventListener("change", onBrowserChange);
  const unsubscribe = settings.subscribe(onSettingsChange);
  render();
  return Object.freeze({
    destroy() {
      button.removeEventListener("click", onClick);
      button.removeEventListener("animationend", onAnimationEnd);
      media.removeEventListener("change", onBrowserChange);
      unsubscribe();
    },
  });
}
