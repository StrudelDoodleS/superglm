// @ts-check
// The join toggle beside the shape icons: how a shaped range meets the free
// curve at its edges. Tangent (the curve leaves the shape along its slope) is
// the default; Corner lets the slope change there. Python knows Corner as
// "kink". The choice outlives the page through localStorage when it can.

/** @typedef {"tangent"|"kink"} ShapeJoin */

const STORAGE_KEY = "superglm.editor.shapeJoin";
const ROVING_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"]);

/** @param {unknown} value @returns {value is ShapeJoin} */
export function isShapeJoin(value) {
  return value === "tangent" || value === "kink";
}

/**
 * The remembered join, or Tangent with no stored choice or unusable storage
 * (private mode, a blocked origin).
 * @param {Pick<Storage, 'getItem'>} [storage] defaults to localStorage, whose access may itself throw
 * @returns {ShapeJoin}
 */
export function readShapeJoin(storage) {
  try {
    const stored = (storage ?? localStorage).getItem(STORAGE_KEY);
    return isShapeJoin(stored) ? stored : "tangent";
  } catch {
    return "tangent";
  }
}

/** @param {ShapeJoin} join @param {Pick<Storage, 'setItem'>} [storage] */
export function storeShapeJoin(join, storage) {
  try {
    (storage ?? localStorage).setItem(STORAGE_KEY, join);
  } catch {
    // Unusable storage: the choice lasts this page only.
  }
}

/**
 * Bind the toggle's two radios: a click chooses, the arrow keys step between
 * them. The choice itself lives with the caller, which renders it back.
 * @param {HTMLElement} root
 * @param {{onChange:(join:ShapeJoin)=>unknown}} handlers
 * @returns {{destroy:()=>void}}
 */
export function bindJoinToggle(root, { onChange }) {
  /** @returns {HTMLButtonElement[]} */
  function radios() {
    /** @type {HTMLButtonElement[]} */
    const found = [];
    for (const element of root.querySelectorAll("[data-join]")) {
      if (element instanceof HTMLButtonElement) found.push(element);
    }
    return found;
  }

  /** @param {MouseEvent} event */
  function onClick(event) {
    const element = event.target instanceof Element ? event.target.closest("[data-join]") : null;
    if (!(element instanceof HTMLButtonElement) || !root.contains(element)) return;
    const join = element.dataset.join;
    if (isShapeJoin(join)) onChange(join);
  }

  /** @param {KeyboardEvent} event */
  function onKeyDown(event) {
    if (!(event.target instanceof HTMLButtonElement) || !ROVING_KEYS.has(event.key)) return;
    const options = radios();
    const index = options.indexOf(event.target);
    if (index < 0) return;
    event.preventDefault();
    const forward = event.key === "ArrowRight" || event.key === "ArrowDown";
    const next = event.key === "Home"
      ? 0
      : event.key === "End"
        ? options.length - 1
        : (index + (forward ? 1 : -1) + options.length) % options.length;
    options[next].focus();
    const join = options[next].dataset.join;
    if (isShapeJoin(join)) onChange(join);
  }

  root.addEventListener("click", onClick);
  root.addEventListener("keydown", onKeyDown);
  return Object.freeze({
    destroy() {
      root.removeEventListener("click", onClick);
      root.removeEventListener("keydown", onKeyDown);
    },
  });
}

/**
 * Mark the chosen join: it is the checked radio and the group's one tab stop.
 * @param {HTMLElement} root @param {ShapeJoin} join
 */
export function renderJoinToggle(root, join) {
  for (const element of root.querySelectorAll("[data-join]")) {
    if (!(element instanceof HTMLButtonElement)) continue;
    const active = element.dataset.join === join;
    element.setAttribute("aria-checked", String(active));
    element.tabIndex = active ? 0 : -1;
  }
}
