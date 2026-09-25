// @ts-check
// The theme control in the app bar: one icon that cycles Auto, Light and Dark.
// Auto follows the browser's colour-scheme setting, which inside a notebook is
// not always the notebook's own theme, so an explicit choice wins over it and
// outlives the page through localStorage when it can. The resolved theme is
// written to <html data-theme>, which styles/dark.css keys the dark palette
// on; index.html writes the same attribute before first paint.

/** @typedef {"auto"|"light"|"dark"} ThemeChoice */
/** @typedef {"light"|"dark"} Theme */
/** @typedef {Pick<Storage, 'getItem'|'setItem'|'removeItem'>} ThemeStorage */
/** @typedef {Pick<MediaQueryList, 'matches'|'addEventListener'|'removeEventListener'>} DarkMedia */

export const THEME_STORAGE_KEY = "superglm.editor.theme";

const NAMES = Object.freeze({ auto: "Auto", light: "Light", dark: "Dark" });

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
 * The choice after a click: first the theme Auto is not showing, then the one
 * it is, then Auto again, so the first click always changes what is on screen.
 * @param {ThemeChoice} choice @param {boolean} prefersDark @returns {ThemeChoice}
 */
export function nextThemeChoice(choice, prefersDark) {
  const browser = prefersDark ? "dark" : "light";
  const other = prefersDark ? "light" : "dark";
  if (choice === "auto") return other;
  return choice === other ? browser : "auto";
}

/**
 * What the control says of itself: its state, and what a click does.
 * @param {ThemeChoice} choice @param {boolean} prefersDark
 * @returns {{label:string, body:string}}
 */
export function describeThemeControl(choice, prefersDark) {
  const next = nextThemeChoice(choice, prefersDark);
  const now = choice === "auto"
    ? `Follows the browser's setting, ${resolveTheme(choice, prefersDark)} now. `
    : "";
  const target = next === "auto" ? "Auto, which follows the browser's setting" : NAMES[next];
  return { label: `Theme: ${NAMES[choice]}`, body: `${now}Click for ${target}.` };
}

/**
 * Show the choice on the control: its icon, its name, and its popover.
 * @param {HTMLElement} button @param {ThemeChoice} choice @param {boolean} prefersDark
 */
export function renderThemeControl(button, choice, prefersDark) {
  const { label, body } = describeThemeControl(choice, prefersDark);
  button.dataset.choice = choice;
  button.setAttribute("aria-label", label);
  button.dataset.popoverTitle = label;
  button.dataset.popoverBody = body;
  for (const icon of button.querySelectorAll("[data-theme-icon]")) {
    icon.toggleAttribute("hidden", icon.getAttribute("data-theme-icon") !== choice);
  }
}

/**
 * Mount the control: apply the remembered choice, cycle it on a click, and
 * follow the browser's setting while the choice is Auto.
 * @param {{button:HTMLElement, root:HTMLElement, media:DarkMedia, storage?:ThemeStorage}} options
 * @returns {{destroy:()=>void}}
 */
export function mountThemeControl({ button, root, media, storage }) {
  let choice = readThemeChoice(storage);

  function render() {
    root.dataset.theme = resolveTheme(choice, media.matches);
    renderThemeControl(button, choice, media.matches);
  }

  function onClick() {
    choice = nextThemeChoice(choice, media.matches);
    storeThemeChoice(choice, storage);
    render();
  }

  button.addEventListener("click", onClick);
  media.addEventListener("change", render);
  render();
  return Object.freeze({
    destroy() {
      button.removeEventListener("click", onClick);
      media.removeEventListener("change", render);
    },
  });
}
