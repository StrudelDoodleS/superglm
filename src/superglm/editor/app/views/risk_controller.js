// @ts-check
/**
 * The Explain tab's requests: asks the server for the risks in view and keeps what the tab draws
 * (state/risk.js holds the request's lifecycle). The caller owns the risks themselves and draws
 * when `draw` is called.
 */
import { createRiskRequest, moveRiskRequest, riskRequestKey } from "../state/risk.js";
import { RISK_FAILED, RISK_SUPERSEDED } from "./risk_panel.js";

/** @typedef {import('../api/contracts.js').RiskResponse} RiskResponse */
/** @typedef {import('../api/contracts.js').RiskProfiles} RiskProfiles */
/** @typedef {import('../api/contracts.js').RiskValues} RiskValues */
/** @typedef {import('../state/risk.js').RiskRequest} RiskRequest */
/** @typedef {import('../state/risk.js').RiskEvent} RiskEvent */
/**
 * What the view wants drawn: nothing while `active` is false, else these risks under the model
 * and chart the generation and revision name.
 * @typedef {{active:boolean, generation:number, revision:number, profiles:RiskProfiles}} RiskWant
 */
/**
 * @typedef {object} RiskClient
 * @property {(risks: RiskValues[]) => Promise<unknown>} risk
 * @property {() => Promise<unknown>} riskRow
 */

// The client errors that say nothing about the request itself: wait, then ask again.
const RETRYABLE = new Set([408, 429]);

/**
 * What the view says when its request fails. A refusal (a 4xx reply) carries the server's own
 * sentence, which says what to change, so no Retry is offered; a failure to answer is offered one.
 * @param {unknown} error @returns {{message:string, retry:boolean}}
 */
export function riskFailure(error) {
  const status = error && typeof error === "object" && "status" in error ? Number(error.status) : NaN;
  const message = error instanceof Error ? error.message : "";
  const refusal = status >= 400 && status < 500 && !RETRYABLE.has(status);
  return refusal && message ? { message, retry: false } : { message: RISK_FAILED, retry: true };
}

/** @param {unknown} reply @returns {reply is RiskResponse} */
function isRiskResponse(reply) {
  return reply !== null && typeof reply === "object" && "risks" in reply &&
    Array.isArray(reply.risks) && "features" in reply && Array.isArray(reply.features);
}

/** @param {RiskWant} want */
export const wantKey = (want) => riskRequestKey(want.generation, want.revision, want.profiles);

/**
 * @param {object} options
 * @param {RiskClient} options.client
 * @param {() => void} options.draw called whenever what the view shows has changed
 */
export function createRiskController({ client, draw }) {
  let request = createRiskRequest();
  /** @type {RiskProfiles["risks"]} the risks of the request in `request.key` */
  let asked = [];

  /** @param {RiskEvent} event @returns {boolean} whether the request moved */
  function move(event) {
    const next = moveRiskRequest(request, event);
    const moved = next !== request;
    request = next;
    return moved;
  }

  /** @param {string} key @param {RiskProfiles["risks"]} risks */
  async function ask(key, risks) {
    asked = risks;
    if (!move({ type: "ask", key })) return;
    draw();
    /** @type {RiskEvent} */
    let outcome;
    try {
      const reply = await client.risk(risks.map((risk) => risk.values));
      // Each answered risk takes the letter it was asked under, so a curve keeps its colour.
      outcome = isRiskResponse(reply)
        ? { type: "answer", key, reply: { ...reply, risks: reply.risks.map((risk, i) => ({ ...risk, id: risks[i]?.id })) } }
        : { type: "fail", key, message: RISK_SUPERSEDED, retry: true };
    } catch (error) {
      outcome = { type: "fail", key, ...riskFailure(error) };
    }
    // An answer for a request the view has moved on from changes nothing.
    if (move(outcome)) draw();
  }

  return {
    /**
     * Ask for what the view wants, unless it is what the view last asked.
     * @param {RiskWant} want
     */
    sync(want) {
      if (!want.active) {
        if (move({ type: "close" })) draw();
        return;
      }
      const key = wantKey(want);
      if (key !== request.key) void ask(key, want.profiles.risks);
    },
    /** Ask again for the request that failed. */
    retry() {
      if (request.status === "failed" && request.key !== null) void ask(request.key, asked);
    },
    /**
     * One row of the training data, drawn at random.
     * @returns {Promise<RiskValues|null>} its values, or null where the server did not answer
     */
    async bookRow() {
      try {
        const reply = /** @type {{values?: RiskValues}|null} */ (await client.riskRow());
        return reply && typeof reply.values === "object" && reply.values !== null ? reply.values : null;
      } catch {
        return null;
      }
    },
    /** What the view draws now. */
    view() {
      return {
        reply: request.reply,
        loading: request.status === "loading",
        message: request.status === "failed" ? request.message : null,
        retry: request.status === "failed" && request.retry,
      };
    },
    /** @returns {RiskRequest} */
    request: () => request,
  };
}
