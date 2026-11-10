// @ts-check
import { defineMachine } from "./machine.js";

/** @typedef {import('../api/contracts.js').EditorRequestState} EditorRequestState */
/** @typedef {import('../api/contracts.js').MutationDescriptor} MutationDescriptor */
/** @typedef {import('../api/contracts.js').MutationStatus} MutationStatus */
/**
 * What happens to the editor's hold. `hold` starts a change that holds the editor, with the
 * busy overlay over the page when `blocking`; `pass` starts one that does not, a History note's
 * save; `settle` is a change answered, with a `notice` for the alert when it leaves something
 * to say; `fail` is a change that did not happen, with the alert's `message` and what Retry
 * would send again; `dismiss` closes the alert; `notice` puts a sentence in the alert and moves
 * nothing else.
 * @typedef {{type:'hold', operation:string, blocking:boolean, detail?:string}
 *   |{type:'pass'}
 *   |{type:'settle', notice?:string}
 *   |{type:'fail', operation:string, error:string, message:string, retry:MutationDescriptor|null}
 *   |{type:'dismiss'}
 *   |{type:'notice', message:string}} MutationEvent
 */

/**
 * The editor's hold: one change at a time. `running` takes no new change, so a second one is
 * not sent while the first is on its way; it ends when its change is answered or fails. A
 * failure stays in `error`, with its alert, until the next change or Dismiss. A change that
 * does not hold the editor leaves it `idle` while it saves.
 */
export const MUTATION = defineMachine({
  name: "mutation",
  initial: /** @type {MutationStatus} */ ("idle"),
  moves: {
    idle: {
      hold: "running", pass: "idle", settle: "idle", fail: "error", dismiss: "idle", notice: "idle"
    },
    running: { settle: "idle", fail: "error", dismiss: "running", notice: "running" },
    error: { hold: "running", pass: "idle", dismiss: "idle", notice: "error" }
  }
});

/**
 * Whether a change is running and the next must wait for it.
 * @param {EditorRequestState} request
 */
export function isHeld(request) {
  return MUTATION.next(request.mutation.status, "hold") === null;
}

/**
 * Make one move of the editor's hold and its alert.
 * @param {EditorRequestState} request
 * @param {MutationEvent} event
 * @returns {EditorRequestState} `request` itself where its state does not take the event, or
 *   the event changes nothing
 */
export function moveMutation(request, event) {
  if (MUTATION.next(request.mutation.status, event.type) === null) return request;
  switch (event.type) {
    case "hold":
      return {
        ...request,
        mutation: {
          status: "running",
          operation: event.operation,
          error: null,
          blocking: event.blocking,
          ...(event.detail === undefined ? {} : { detail: event.detail })
        },
        recovery: null
      };
    case "pass":
      return { ...request, mutation: { status: "idle", operation: null, error: null }, recovery: null };
    case "settle":
      return {
        ...request,
        mutation: { status: "idle", operation: null, error: null },
        recovery: event.notice === undefined ? null : { message: event.notice, retry: null }
      };
    case "fail":
      return {
        ...request,
        mutation: { status: "error", operation: event.operation, error: event.error },
        recovery: { message: event.message, retry: event.retry }
      };
    case "dismiss": {
      const failed = request.mutation.status === "error";
      if (request.recovery === null && !failed) return request;
      return {
        ...request,
        mutation: failed ? { status: "idle", operation: null, error: null } : request.mutation,
        recovery: null
      };
    }
    case "notice":
      return { ...request, recovery: { message: event.message, retry: null } };
  }
}
