// @ts-check

/** @typedef {'select'|'move'|'zoom'|'handles'} ToolMode */

/** @type {Readonly<Record<string, ToolMode>>} */
const SHORTCUT_MODES = Object.freeze({
  v: "select",
  m: "move",
  z: "zoom",
  h: "handles",
});

const ROVING_KEYS = new Set([
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "Home",
  "End",
]);

/**
 * Bind the chart tool rail without making the DOM a second source of mode state.
 *
 * @param {object} options
 * @param {HTMLElement} options.root
 * @param {(mode:ToolMode)=>unknown} options.onMode
 * @param {()=>unknown} options.onHelp
 * @param {Document} [options.shortcutRoot]
 * @returns {{destroy:()=>void}}
 */
export function bindToolRail({ root, onMode, onHelp, shortcutRoot = document }) {
  /** @returns {HTMLButtonElement[]} */
  function enabledRadios() {
    /** @type {HTMLButtonElement[]} */
    const radios = [];
    for (const element of root.querySelectorAll('[role="radio"]')) {
      if (element instanceof HTMLButtonElement && !element.disabled) radios.push(element);
    }
    return radios;
  }

  /** @param {MouseEvent} event */
  function onClick(event) {
    const element = event.target instanceof Element ? event.target.closest("[data-tool]") : null;
    if (!(element instanceof HTMLButtonElement) || !root.contains(element) || element.disabled) {
      return;
    }
    const tool = element.dataset.tool;
    if (tool === "help") {
      onHelp();
    } else if (isToolMode(tool)) {
      onMode(tool);
    }
  }

  /** @param {KeyboardEvent} event */
  function onRailKeyDown(event) {
    if (!(event.target instanceof HTMLButtonElement) || !ROVING_KEYS.has(event.key)) return;
    const radios = enabledRadios();
    const index = radios.indexOf(event.target);
    if (index < 0 || radios.length === 0) return;
    event.preventDefault();
    const forward = event.key === "ArrowDown" || event.key === "ArrowRight";
    const next = event.key === "Home"
      ? 0
      : event.key === "End"
        ? radios.length - 1
        : (index + (forward ? 1 : -1) + radios.length) % radios.length;
    const button = radios[next];
    button.focus();
    const tool = button.dataset.tool;
    if (isToolMode(tool)) onMode(tool);
  }

  /** @param {KeyboardEvent} event */
  function onShortcutKeyDown(event) {
    if (
      event.defaultPrevented ||
      event.ctrlKey ||
      event.metaKey ||
      event.altKey ||
      isEditableTarget(event.target)
    ) {
      return;
    }
    if (event.key === "?") {
      event.preventDefault();
      onHelp();
      return;
    }
    const mode = SHORTCUT_MODES[event.key.toLowerCase()];
    if (!mode) return;
    const button = root.querySelector(`[data-tool="${mode}"]`);
    if (!(button instanceof HTMLButtonElement) || button.disabled) return;
    event.preventDefault();
    onMode(mode);
  }

  root.addEventListener("click", onClick);
  root.addEventListener("keydown", onRailKeyDown);
  shortcutRoot.addEventListener("keydown", onShortcutKeyDown);

  return Object.freeze({
    destroy() {
      root.removeEventListener("click", onClick);
      root.removeEventListener("keydown", onRailKeyDown);
      shortcutRoot.removeEventListener("keydown", onShortcutKeyDown);
    },
  });
}

/**
 * @param {HTMLElement} root
 * @param {{mode:ToolMode, handlesAvailable:boolean}} state
 */
export function renderToolRail(root, { mode, handlesAvailable }) {
  const effectiveMode = mode === "handles" && !handlesAvailable ? "select" : mode;
  for (const element of root.querySelectorAll('[role="radio"]')) {
    if (!(element instanceof HTMLButtonElement)) continue;
    if (element.dataset.tool === "handles") element.disabled = !handlesAvailable;
    const active = element.dataset.tool === effectiveMode;
    element.setAttribute("aria-checked", String(active));
    element.tabIndex = active ? 0 : -1;
    element.classList.toggle("active", active);
  }
}

/** @param {string|undefined} value @returns {value is ToolMode} */
function isToolMode(value) {
  return value === "select" || value === "move" || value === "zoom" || value === "handles";
}

/** @param {EventTarget|null} target */
function isEditableTarget(target) {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return target.isContentEditable || tag === "input" || tag === "select" || tag === "textarea";
}
