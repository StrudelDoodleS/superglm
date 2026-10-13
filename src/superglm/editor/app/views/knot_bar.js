// @ts-check
// Knots mode's controls in the toolbar: the count stepper, the rule that
// places the knots, the tempered quantiles' alpha, Reset knots, and the
// spline's basis: its Kind and the Shrink switch. Also the context chip that
// names a term's knots in every mode, and the status line's sentence while
// Knots mode is on.

import {
  DEFAULT_ALPHA,
  knotAxis,
  knotChip,
  knotPenaltyTitle,
  ruleParams,
  shownKnots,
  stepRule,
  stepperState
} from "../knots.js";

/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */
/** @typedef {import('../api/contracts.js').KnotParams} KnotParams */
/** @typedef {import('../api/contracts.js').KnotRule} KnotRule */
/** @typedef {import('../api/contracts.js').BasisKind} BasisKind */
/** @typedef {import('../api/contracts.js').BasisParams} BasisParams */
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
 * @property {HTMLSelectElement} kind
 * @property {HTMLButtonElement} shrink the Shrink switch, ``aria-pressed`` while it is on
 */

const RULES = new Set(["uniform", "quantile", "quantile_rows", "quantile_tempered"]);
const RESET_BODY = "Back to the knots declared in code. It waits for Refit.";
const RESET_NOTHING = "The knots are the ones declared in code.";
const KINDS = new Set(["ps", "bs", "cr", "ns"]);
export const SHRINK_BODY =
  "A second penalty, on the term's straight-line part, so the fit can shrink the term towards a"
  + " straight line and, where the data do not support it, out of the model. The change waits"
  + " for Refit.";

/**
 * The basis a term shows: the one its waiting changes put in force, else
 * the one in force, with which of its two parts a waiting change sets.
 * @param {TermPayload} term
 * @returns {{kind:BasisKind, select:boolean, kindWaiting:boolean, selectWaiting:boolean}|null}
 */
export function shownBasis(term) {
  const knots = term.knots;
  if (!knots || !knots.kind) return null;
  const waiting = term.pending?.basis ?? null;
  const kind = waiting ? waiting.kind : knots.kind;
  const select = waiting ? waiting.select : Boolean(knots.select);
  return {
    kind,
    select,
    kindWaiting: kind !== knots.kind,
    selectWaiting: select !== Boolean(knots.select),
  };
}

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
  // A term that takes evenly spaced knots only is offered even spacing alone;
  // a rule it shows in force stays named, but cannot be chosen.
  for (const option of Array.from(nodes.rule.options)) {
    if (option === nodes.hand) continue;
    const closed = axis.evenOnly !== null && option.value !== "uniform";
    option.disabled = closed;
    option.hidden = closed && option.value !== shown.strategy;
  }
  nodes.rule.value = shown.strategy;
  nodes.alphaWrap.hidden = shown.strategy !== "quantile_tempered";
  // An alpha being typed is left alone until it is sent.
  if (nodes.alpha.ownerDocument.activeElement !== nodes.alpha) {
    nodes.alpha.value = String(shown.alpha ?? DEFAULT_ALPHA);
  }
  const resettable = Boolean(term.knots?.resettable);
  renderAction(nodes.reset, "Reset knots", resettable, resettable ? RESET_BODY : RESET_NOTHING);
  renderBasis(nodes, term);
}

/**
 * The Kind dropdown offers the kinds the term can be switched to; a kind in
 * force it does not offer, the cardinal cubic regression spline, stays named
 * but cannot be chosen. Shrink is pressed while it is on, and says why when
 * it cannot change. Either takes the waiting tint while a change sets it.
 * @param {KnotBarNodes} nodes @param {TermPayload} term
 */
function renderBasis(nodes, term) {
  const basis = shownBasis(term);
  /** @type {Set<string>} */
  const offered = new Set(term.knots?.kinds ?? []);
  for (const option of Array.from(nodes.kind.options)) {
    option.disabled = !offered.has(option.value);
    option.hidden = option.disabled && option.value !== basis?.kind;
  }
  nodes.kind.value = basis?.kind ?? "";
  nodes.kind.dataset.waiting = String(Boolean(basis?.kindWaiting));
  nodes.shrink.setAttribute("aria-pressed", String(Boolean(basis?.select)));
  nodes.shrink.dataset.waiting = String(Boolean(basis?.selectWaiting));
  const available = Boolean(term.knots?.select_available);
  const reason = term.knots?.select_reason ?? null;
  renderAction(nodes.shrink, "Shrink", available, available || !reason ? SHRINK_BODY : reason);
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
 * @param {(params:BasisParams)=>unknown} options.onBasis stages a basis change
 * @param {(message:string)=>void} options.onRefuse
 * @param {()=>void} options.onSettled redraws the controls once a change is
 *   staged or refused, so they show what is in force
 */
export function bindKnotBar(nodes, { term, onChange, onBasis, onRefuse, onSettled }) {
  /** @param {KnotParams} params */
  async function send(params) {
    try {
      await onChange(params);
    } finally {
      onSettled();
    }
  }

  /** @param {BasisParams} params */
  async function sendBasis(params) {
    try {
      await onBasis(params);
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

  const onKind = () => {
    const current = term();
    const shown = current ? shownBasis(current) : null;
    const kind = nodes.kind.value;
    if (!shown || kind === shown.kind || !KINDS.has(kind)) {
      onSettled();
      return;
    }
    void sendBasis({ kind: /** @type {Exclude<BasisKind, "cr_cardinal">} */ (kind) });
  };
  const onShrink = () => {
    const current = term();
    const shown = current ? shownBasis(current) : null;
    if (!shown) return;
    if (nodes.shrink.getAttribute("aria-disabled") === "true") {
      onRefuse(nodes.shrink.dataset.popoverBody ?? "");
      return;
    }
    void sendBasis({ select: !shown.select });
  };

  nodes.fewer.addEventListener("click", onFewer);
  nodes.more.addEventListener("click", onMore);
  nodes.rule.addEventListener("change", onRule);
  nodes.alpha.addEventListener("change", onAlpha);
  nodes.reset.addEventListener("click", onReset);
  nodes.kind.addEventListener("change", onKind);
  nodes.shrink.addEventListener("click", onShrink);
  return Object.freeze({
    destroy() {
      nodes.fewer.removeEventListener("click", onFewer);
      nodes.more.removeEventListener("click", onMore);
      nodes.rule.removeEventListener("change", onRule);
      nodes.alpha.removeEventListener("change", onAlpha);
      nodes.reset.removeEventListener("click", onReset);
      nodes.kind.removeEventListener("change", onKind);
      nodes.shrink.removeEventListener("click", onShrink);
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
 * spacing", tinted while a knot change waits for Refit. Its hover text names
 * the smoothing penalty when it is not the standard one.
 * @param {HTMLElement} node @param {TermPayload|null} term
 */
export function renderKnotChip(node, term) {
  const chip = term ? knotChip(term) : null;
  node.hidden = chip === null;
  node.textContent = chip ? chip.text : "";
  node.dataset.waiting = String(Boolean(chip?.waiting));
  // An empty title shows no hover text.
  node.title = (term && chip ? knotPenaltyTitle(term) : null) ?? "";
}

/**
 * The status line while Knots mode is on: what each gesture does, or why the
 * last one did nothing; on a term that takes evenly spaced knots only, what
 * can change them instead. It leaves out the changes waiting for refit, which
 * the Refit button's count and the knots chip's tint already show, so the
 * gestures fit the line.
 * @param {HTMLElement} statusNode
 * @param {{message?:string|null, evenOnly?:boolean}} state
 */
export function renderKnotStatus(statusNode, { message = null, evenOnly = false }) {
  const doc = statusNode.ownerDocument;
  /** @param {string} tag @param {string} content */
  const node = (tag, content) => {
    const element = doc.createElement(tag);
    element.textContent = content;
    return element;
  };
  /** @type {(Node|string)[]} */
  const parts = [];
  if (message) {
    parts.push(message);
  } else if (evenOnly) {
    parts.push(
      node("strong", "Knots."),
      " This term takes evenly spaced knots only; change their count above the chart."
    );
  } else {
    parts.push(
      node("strong", "Knots."),
      " Drag to move · click the axis to add · drag below the axis to remove · arrow keys nudge"
        + " the selected knot, ",
      node("kbd", "Delete"),
      " removes it"
    );
  }
  statusNode.replaceChildren(...parts);
}
