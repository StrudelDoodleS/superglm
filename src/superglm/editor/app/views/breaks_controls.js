// @ts-check

import { MAX_POLYNOMIAL_DEGREE, formHint } from "../breaks.js";

/** @typedef {import('../api/contracts.js').BreakDraft} BreakDraft */
/** @typedef {import('../api/contracts.js').BreakForm} BreakForm */

/** @param {HTMLElement} root @param {string} selector @returns {HTMLElement} */
function required(root, selector) {
  const node = root.querySelector(selector);
  if (!(node instanceof HTMLElement)) throw new Error(`Breaks controls are missing ${selector}`);
  return node;
}

/** @param {string|undefined} value @returns {value is BreakForm} */
function isBreakForm(value) {
  return value === "piecewise" || value === "spline" || value === "polynomial";
}

/**
 * Bind the Breaks controls in the chart action bar; the store stays the only draft state.
 *
 * @param {object} options
 * @param {HTMLElement} options.root
 * @param {(form:BreakForm)=>unknown} options.onForm
 * @param {(step:number)=>unknown} options.onDegree
 * @param {()=>unknown} options.onClear
 * @param {()=>unknown} options.onTransform
 * @returns {{destroy:()=>void}}
 */
export function bindBreaksControls({ root, onForm, onDegree, onClear, onTransform }) {
  const clear = required(root, "#clearBreaks");
  const transform = required(root, "#transformTerm");

  /** @param {MouseEvent} event */
  function onClick(event) {
    const element = event.target instanceof Element ? event.target : null;
    const form = element?.closest("[data-form]");
    if (form instanceof HTMLButtonElement && isBreakForm(form.dataset.form)) {
      onForm(form.dataset.form);
      return;
    }
    const step = element?.closest("[data-degree-step]");
    if (step instanceof HTMLButtonElement) onDegree(Number(step.dataset.degreeStep));
  }

  root.addEventListener("click", onClick);
  clear.addEventListener("click", onClear);
  transform.addEventListener("click", onTransform);

  return Object.freeze({
    destroy() {
      root.removeEventListener("click", onClick);
      clear.removeEventListener("click", onClear);
      transform.removeEventListener("click", onTransform);
    },
  });
}

/**
 * @param {object} options
 * @param {HTMLElement} options.root
 * @param {BreakDraft} options.draft
 * @param {string|null} options.problem
 * @param {boolean} options.busy
 */
export function renderBreaksControls({ root, draft, problem, busy }) {
  for (const button of root.querySelectorAll("[data-form]")) {
    if (!(button instanceof HTMLButtonElement)) continue;
    button.setAttribute("aria-checked", String(button.dataset.form === draft.form));
  }
  const polynomial = draft.form === "polynomial";
  required(root, "#polynomialDegreeWrap").hidden = !polynomial;
  required(root, "#polynomialDegree").textContent = String(draft.degree);
  for (const button of root.querySelectorAll("[data-degree-step]")) {
    if (!(button instanceof HTMLButtonElement)) continue;
    const next = draft.degree + Number(button.dataset.degreeStep);
    button.disabled = next < 1 || next > MAX_POLYNOMIAL_DEGREE;
  }
  const hint = required(root, "#breaksHint");
  hint.textContent = problem ?? formHint(draft);
  hint.classList.toggle("warn", problem !== null);
  const clear = required(root, "#clearBreaks");
  clear.hidden = polynomial;
  if (clear instanceof HTMLButtonElement) clear.disabled = draft.breaks.length === 0;
  const transform = required(root, "#transformTerm");
  if (transform instanceof HTMLButtonElement) transform.disabled = problem !== null || busy;
}
