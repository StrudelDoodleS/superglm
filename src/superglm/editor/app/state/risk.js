// @ts-check
/**
 * The Explain tab's request and its risks: which risks it asked the server for, and what the tab
 * shows of the answer. A risk is a value for each feature; a feature it does not name sits at the
 * reference risk's value. The tab holds up to eight, lettered A to H, and one of them is selected.
 */
import { defineMachine } from "./machine.js";

/** @typedef {import('../api/contracts.js').RiskResponse} RiskResponse */
/** @typedef {import('../api/contracts.js').RiskValues} RiskValues */
/** @typedef {import('../api/contracts.js').RiskProfiles} RiskProfiles */
/** @typedef {import('../api/contracts.js').RiskId} RiskId */
/** @typedef {'idle'|'loading'|'shown'|'failed'} RiskStatus */
/**
 * `key` names the request the view last asked: the model, the chart and the risks. `reply` is
 * the last answer, kept while a newer one loads so the plots do not blank, and after a failure
 * so the risks keep their controls. `message` and `retry` belong to `failed` alone.
 * @typedef {{status:RiskStatus, key:string|null, reply:RiskResponse|null, message:string|null,
 *   retry:boolean}} RiskRequest
 */
/**
 * @typedef {{type:'ask', key:string}
 *   |{type:'answer', key:string, reply:RiskResponse}
 *   |{type:'fail', key:string, message:string, retry:boolean}
 *   |{type:'close'}} RiskEvent
 */

/**
 * The request's lifecycle. `ask` starts a request and replaces one still loading; `answer` and
 * `fail` end the request they name, and one for an older request changes nothing; `close` is
 * the view put away, or a model that has no such view.
 */
export const RISK_REQUEST = defineMachine({
  name: "risk request",
  initial: /** @type {RiskStatus} */ ("idle"),
  moves: {
    idle: { ask: "loading" },
    loading: { ask: "loading", answer: "shown", fail: "failed", close: "idle" },
    shown: { ask: "loading", close: "idle" },
    failed: { ask: "loading", close: "idle" }
  }
});

/** @returns {RiskRequest} */
export function createRiskRequest() {
  return { status: "idle", key: null, reply: null, message: null, retry: false };
}

/**
 * Make one move of the request.
 * @param {RiskRequest} request @param {RiskEvent} event
 * @returns {RiskRequest} `request` itself where its state does not take the event, or the event
 *   answers a request the view has moved on from
 */
export function moveRiskRequest(request, event) {
  if (RISK_REQUEST.next(request.status, event.type) === null) return request;
  switch (event.type) {
    case "ask":
      if (request.status === "loading" && request.key === event.key) return request;
      return { status: "loading", key: event.key, reply: request.reply, message: null, retry: false };
    case "answer":
      if (event.key !== request.key) return request;
      return { status: "shown", key: event.key, reply: event.reply, message: null, retry: false };
    case "fail":
      if (event.key !== request.key) return request;
      return {
        status: "failed", key: event.key, reply: request.reply, message: event.message,
        retry: event.retry
      };
    case "close":
      return createRiskRequest();
  }
}

/** The risks the view holds at most: one colour each that stays apart from the others. */
export const RISK_IDS = /** @type {const} */ (["a", "b", "c", "d", "e", "f", "g", "h"]);

/**
 * The request a view of these risks asks, as one string: a new model, a redrawn chart or a
 * changed risk asks again. Which risk is selected is not part of it.
 * @param {number} generation @param {number} revision @param {RiskProfiles} profiles
 */
export function riskRequestKey(generation, revision, profiles) {
  return `${generation}:${revision}:${JSON.stringify(profiles.risks)}`;
}

/** @returns {RiskProfiles} risk A at the reference risk, selected */
export function createRiskProfiles() {
  return { risks: [{ id: "a", values: {} }], selected: "a" };
}

/**
 * @param {RiskProfiles} profiles @param {RiskId} id @param {(values: RiskValues) => RiskValues} change
 * @returns {RiskProfiles} `profiles` itself where it holds no such risk or nothing changes
 */
function withRisk(profiles, id, change) {
  const at = profiles.risks.findIndex((risk) => risk.id === id);
  if (at < 0) return profiles;
  const values = change(profiles.risks[at].values);
  if (values === profiles.risks[at].values) return profiles;
  return { ...profiles, risks: profiles.risks.map((risk, i) => (i === at ? { id, values } : risk)) };
}

/**
 * @param {RiskProfiles} profiles @param {RiskId} id @param {string} column
 * @param {number|string} value
 * @returns {RiskProfiles}
 */
export function withRiskValue(profiles, id, column, value) {
  return withRisk(profiles, id, (values) => (values[column] === value ? values : { ...values, [column]: value }));
}

/**
 * A risk given every value at once: a row from the book, or `{}` to go back to the reference risk.
 * @param {RiskProfiles} profiles @param {RiskId} id @param {RiskValues} values
 * @returns {RiskProfiles}
 */
export function withRiskValues(profiles, id, values) {
  return withRisk(profiles, id, () => values);
}

/**
 * One more risk, under the first free letter, and selected: with `values`, or a copy of the
 * selected one. A removed risk's letter and colour are free for the next one; the others keep
 * theirs.
 * @param {RiskProfiles} profiles @param {RiskValues} [values]
 * @returns {RiskProfiles} `profiles` itself when it is full
 */
export function withRiskAdded(profiles, values) {
  const id = RISK_IDS.find((letter) => !profiles.risks.some((risk) => risk.id === letter));
  if (id === undefined) return profiles;
  const from = profiles.risks.find((risk) => risk.id === profiles.selected) ?? profiles.risks[0];
  return { risks: [...profiles.risks, { id, values: { ...(values ?? from.values) } }], selected: id };
}

/**
 * A risk taken away; the last one stays. Where it was selected, the one before it is.
 * @param {RiskProfiles} profiles @param {RiskId} id
 * @returns {RiskProfiles}
 */
export function withRiskRemoved(profiles, id) {
  const at = profiles.risks.findIndex((risk) => risk.id === id);
  if (at < 0 || profiles.risks.length === 1) return profiles;
  const risks = profiles.risks.filter((risk) => risk.id !== id);
  return { risks, selected: profiles.selected === id ? risks[Math.max(0, at - 1)].id : profiles.selected };
}

/**
 * @param {RiskProfiles} profiles @param {RiskId} id
 * @returns {RiskProfiles}
 */
export function withRiskSelected(profiles, id) {
  return profiles.selected === id || !profiles.risks.some((risk) => risk.id === id)
    ? profiles
    : { ...profiles, selected: id };
}
