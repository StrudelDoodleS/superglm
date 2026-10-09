// @ts-check
// Knots mode's controls in the toolbar: the count stepper, the rule that
// places the knots, the tempered quantiles' alpha and Reset knots. Also the
// context chip that names a term's knots in every mode, and the status line's
// sentence while Knots mode is on.

import {
  DEFAULT_ALPHA,
  knotAxis,
  knotChip,
  ruleParams,
  shownKnots,
  stepRule,
  stepperState
} from "../knots.js";
import { waitingLabel } from "./context_bar.js";

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('../api/contracts.js').KnotParams} KnotParams */
/** @typedef {import('../api/contracts.js').KnotRule} KnotRule */
/**
 * @typedef {object} KnotBarNodes
 * @property {HTMLElement} root
 * @property {HTMLButtonElement} fewer
 * @property {HTMLElement} count
 * @property {HTMLButtonElement} more
 * @property {HTMLSelectElement} rule
 * @property {HTMLOptionElement} hand the disabled "Hand" choice, shown for positions set by hand or in code
 * @property {HTMLElement} alphaWrap
 * @property {HTMLInputElement} alpha
 * @property {HTMLButtonElement} reset
 */

const RULES = new Set(["uniform", "quantile", "quantile_rows", "quantile_tempered"]);
const RESET_BODY = "Back to the knots declared in code. It waits for Refit.";
const RESET_NOTHING = "The knots are the ones declared in code.";

/**
 * Show the controls for ``term`` while Knots mode is on for it, else hide them.
 * @param {KnotBarNodes} nodes @param {TermPayload|null} term @param {boolean} visible
 */
export function renderKnotBar(nodes, term, visible) {
  const shown = visible && term ? shownKnots(term) : null;
  const axis = shown && term ? knotAxis(term) : null;
  nodes.root.hidden = !shown || !axis || !term;
  if (!shown || !axis || !term) return;
  nodes.count.textContent = String(shown.count);
  const stepper = stepperState(shown, axis, stepRule(term).strategy);
  renderAction(nodes.fewer, "One knot fewer", stepper.fewer.enabled, stepper.fewer.body);
  renderAction(nodes.more, "One knot more", stepper.more.enabled, stepper.more.body);
  const explicit = shown.strategy === "explicit";
  nodes.hand.hidden = !explicit;
  nodes.rule.value = shown.strategy;
  nodes.alphaWrap.hidden = shown.strategy !== "quantile_tempered";
  // An alpha being typed is left alone until it is sent.
  if (nodes.alpha.ownerDocument.activeElement !== nodes.alpha) {
    nodes.alpha.value = String(shown.alpha ?? DEFAULT_ALPHA);
  }
  const resettable = Boolean(term.knots?.resettable);
  renderAction(nodes.reset, "Reset knots", resettable, resettable ? RESET_BODY : RESET_NOTHING);
}

/**
 * A control that stays focusable and hoverable while it cannot act, so its
 * popover can say why.
 * @param {HTMLButtonElement} button @param {string} title @param {boolean} enabled @param {string} body
 */
function renderAction(button, title, enabled, body) {
  button.setAttribute("aria-disabled", String(!enabled));
  button.dataset.popoverTitle = title;
  button.dataset.popoverBody = body;
}

/**
 * Bind the controls: each sends one knot change. A control that cannot act
 * says why on the status line instead.
 * @param {KnotBarNodes} nodes
 * @param {object} options
 * @param {()=>TermPayload|null} options.term the term the controls show
 * @param {(params:KnotParams)=>unknown} options.onChange stages the change
 * @param {(message:string)=>void} options.onRefuse
 * @param {()=>void} options.onSettled redraws the controls once a change is
 *   staged or refused, so they show what is in force
 */
export function bindKnotBar(nodes, { term, onChange, onRefuse, onSettled }) {
  /** @param {KnotParams} params */
  async function send(params) {
    try {
      await onChange(params);
    } finally {
      onSettled();
    }
  }

  /** @param {HTMLButtonElement} button @param {number} delta */
  function step(button, delta) {
    const current = term();
    const shown = current ? shownKnots(current) : null;
    if (!current || !shown) return;
    if (button.getAttribute("aria-disabled") === "true") {
      onRefuse(button.dataset.popoverBody ?? "");
      return;
    }
    const rule = stepRule(current);
    void send(ruleParams(shown.count + delta, rule.strategy, rule.alpha));
  }

  /** @param {KnotRule} strategy @param {number} alpha */
  function replace(strategy, alpha) {
    const current = term();
    const shown = current ? shownKnots(current) : null;
    if (shown) void send(ruleParams(shown.count, strategy, alpha));
  }

  const onFewer = () => step(nodes.fewer, -1);
  const onMore = () => step(nodes.more, 1);
  const onRule = () => {
    const strategy = nodes.rule.value;
    if (!RULES.has(strategy)) return;
    replace(/** @type {KnotRule} */ (strategy), alphaValue(nodes.alpha, term()));
  };
  const onAlpha = () => {
    const value = Number(nodes.alpha.value);
    if (nodes.alpha.value.trim() === "" || !Number.isFinite(value)) {
      onSettled();
      return;
    }
    const alpha = Math.min(1, Math.max(0, value));
    nodes.alpha.value = String(alpha);
    replace("quantile_tempered", alpha);
  };
  const onReset = () => {
    if (nodes.reset.getAttribute("aria-disabled") === "true") {
      onRefuse(nodes.reset.dataset.popoverBody ?? "");
      return;
    }
    void send({ reset: true });
  };

  nodes.fewer.addEventListener("click", onFewer);
  nodes.more.addEventListener("click", onMore);
  nodes.rule.addEventListener("change", onRule);
  nodes.alpha.addEventListener("change", onAlpha);
  nodes.reset.addEventListener("click", onReset);
  return Object.freeze({
    destroy() {
      nodes.fewer.removeEventListener("click", onFewer);
      nodes.more.removeEventListener("click", onMore);
      nodes.rule.removeEventListener("change", onRule);
      nodes.alpha.removeEventListener("change", onAlpha);
      nodes.reset.removeEventListener("click", onReset);
    }
  });
}

/**
 * The alpha a rule change takes: the one shown, for the tempered quantiles.
 * @param {HTMLInputElement} input @param {TermPayload|null} term
 */
function alphaValue(input, term) {
  const typed = Number(input.value);
  if (input.value.trim() !== "" && Number.isFinite(typed)) return Math.min(1, Math.max(0, typed));
  return (term && shownKnots(term)?.alpha) ?? DEFAULT_ALPHA;
}

/**
 * The context chip naming the term's knots, in every mode: "10 knots · even
 * spacing", tinted while a knot change waits for Refit.
 * @param {HTMLElement} node @param {TermPayload|null} term
 */
export function renderKnotChip(node, term) {
  const chip = term ? knotChip(term) : null;
  node.hidden = chip === null;
  node.textContent = chip ? chip.text : "";
  node.dataset.waiting = String(Boolean(chip?.waiting));
}

/**
 * The status line while Knots mode is on: what each gesture does, or why the
 * last one did nothing, after the changes waiting for refit.
 * @param {HTMLElement} statusNode
 * @param {{pendingCount?:number, message?:string|null}} state
 */
export function renderKnotStatus(statusNode, { pendingCount = 0, message = null }) {
  const doc = statusNode.ownerDocument;
  /** @param {string} tag @param {string} content @param {string} [className] */
  const node = (tag, content, className) => {
    const element = doc.createElement(tag);
    element.textContent = content;
    if (className) element.className = className;
    return element;
  };
  /** @type {(Node|string)[]} */
  const parts = pendingCount > 0
    ? [node("strong", waitingLabel(pendingCount), "status-waiting"), " · "]
    : [];
  if (message) {
    parts.push(message);
  } else {
    parts.push(
      node("strong", "Knots."),
      " Drag a knot to move it · click the axis to add one · drag one below the axis to remove"
        + " it · arrow keys nudge the selected knot, ",
      node("kbd", "Delete"),
      " removes it"
    );
  }
  statusNode.replaceChildren(...parts);
}
