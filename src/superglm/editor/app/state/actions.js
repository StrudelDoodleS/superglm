// @ts-check

import { commitRemote, patchView as patchViewState } from "./store.js";
import { selectModelRevision } from "./selectors.js";

/** @typedef {import('../api/contracts.js').ActionResult} ActionResult */
/** @typedef {import('../api/contracts.js').EditorSnapshot} EditorSnapshot */
/** @typedef {import('../api/contracts.js').EditorState} EditorState */
/** @typedef {import('../api/contracts.js').EvidencePanel} EvidencePanel */
/**
 * @typedef {Object} EditorStore
 * @property {()=>EditorState} getState
 * @property {(updater:(state:EditorState)=>EditorState)=>void} update
 */
/**
 * @typedef {Object} ActionClient
 * @property {(path:string, payload:Record<string, unknown>)=>Promise<unknown>} postJSON
 * @property {()=>Promise<unknown>} getState
 */
/**
 * @typedef {Object} MutationDescriptor
 * @property {string} name
 * @property {string} path
 * @property {Record<string, unknown>} payload
 */
/**
 * @typedef {Object} EditorActionOptions
 * @property {EditorStore} store
 * @property {ActionClient} client
 * @property {(revision:number)=>void|Promise<void>} [scheduleEvidence]
 */

/** @param {unknown} value @returns {Error} */
function normalizeError(value) {
  if (typeof value === "string" && value) return new Error(value);
  try {
    if (value instanceof Error) return value;
    if (value && typeof value === "object" && "message" in value) {
      const message = value.message;
      if (typeof message === "string" && message) return new Error(message);
    }
  } catch {
    // Hostile proxies and null-prototype values still receive a safe message.
  }
  return new Error("Editor request failed.");
}

/** @param {Record<string, unknown>} payload @returns {Record<string, unknown>} */
function snapshotPayload(payload) {
  return structuredClone(payload);
}

/** @param {string} message @returns {ActionResult} */
function skippedMutation(message) {
  return {
    ok: false,
    skipped: true,
    error: new Error(message)
  };
}

/** @param {EditorActionOptions} options */
export function createEditorActions({ store, client, scheduleEvidence = () => {} }) {
  /** @returns {Promise<EditorSnapshot>} */
  async function initialize() {
    const snapshot = /** @type {EditorSnapshot} */ (await client.getState());
    store.update((state) => commitRemote(state, snapshot));
    return snapshot;
  }

  /** @param {MutationDescriptor} descriptor @returns {Promise<ActionResult>} */
  async function executeStateMutation({ name, path, payload }) {
    if (store.getState().request.mutation.status === "running") {
      return skippedMutation("An editor mutation is already running.");
    }

    const confirmed = store.getState().remote.snapshot;
    const previousRevision = confirmed?.model_revision ?? -1;
    const requestPayload = snapshotPayload(payload);
    store.update((state) => ({
      ...state,
      request: {
        ...state.request,
        mutation: { status: "running", operation: name, error: null },
        recovery: null
      }
    }));

    /** @type {EditorSnapshot} */
    let snapshot;
    try {
      snapshot = /** @type {EditorSnapshot} */ (await client.postJSON(path, requestPayload));
    } catch (value) {
      const error = normalizeError(value);
      /** @type {EditorSnapshot|null} */
      let recovered = null;
      try {
        recovered = /** @type {EditorSnapshot} */ (await client.getState());
      } catch {
        // The exact last-confirmed snapshot remains authoritative while offline.
      }
      store.update((state) => {
        const restored = recovered
          ? commitRemote(state, recovered)
          : {
              ...state,
              view: { ...state.view, preview: null }
            };
        return {
          ...restored,
          request: {
            ...restored.request,
            mutation: { status: "error", operation: name, error: error.message },
            recovery: {
              message: error.message,
              retry: { name, path, payload: requestPayload }
            }
          }
        };
      });
      return { ok: false, error };
    }

    store.update((state) => {
      const committed = commitRemote(state, snapshot);
      return {
        ...committed,
        request: {
          ...committed.request,
          mutation: { status: "idle", operation: null, error: null },
          recovery: null
        }
      };
    });
    if (snapshot.model_revision !== previousRevision) {
      try {
        void Promise.resolve(scheduleEvidence(snapshot.model_revision)).catch(() => {});
      } catch {
        // Evidence refresh is independent of the already-confirmed mutation.
      }
    }
    return { ok: true, snapshot };
  }

  /**
   * Refresh one evidence panel. Returns true only when the latest success is accepted.
   * Errors and stale responses resolve false and never throw.
   *
   * @param {EvidencePanel} panel
   * @param {string} path
   * @param {Record<string, unknown>} payload
   * @returns {Promise<boolean>}
   */
  async function refreshEvidence(panel, path, payload) {
    const requestPayload = snapshotPayload(payload);
    let revision = -1;
    let sequence = 0;
    store.update((state) => {
      revision = selectModelRevision(state);
      sequence = state.request.nextSequence;
      const current = state.request.evidence[panel];
      return {
        ...state,
        request: {
          ...state.request,
          nextSequence: sequence + 1,
          evidence: {
            ...state.request.evidence,
            [panel]: {
              ...current,
              status: "updating",
              revision,
              sequence,
              error: null,
              retry: { path, payload: requestPayload }
            }
          }
        }
      };
    });

    try {
      const response = await client.postJSON(path, requestPayload);
      let accepted = false;
      store.update((state) => {
        const current = state.request.evidence[panel];
        if (current.sequence !== sequence) {
          return state;
        }
        if (selectModelRevision(state) !== revision) {
          return {
            ...state,
            request: {
              ...state.request,
              evidence: {
                ...state.request.evidence,
                [panel]: { ...current, status: "stale" }
              }
            }
          };
        }
        accepted = true;
        return {
          ...state,
          request: {
            ...state.request,
            evidence: {
              ...state.request.evidence,
              [panel]: {
                ...current,
                status: "current",
                payload: response,
                error: null,
                retry: null
              }
            }
          }
        };
      });
      return accepted;
    } catch (value) {
      const error = normalizeError(value);
      store.update((state) => {
        const current = state.request.evidence[panel];
        if (current.sequence !== sequence) {
          return state;
        }
        if (selectModelRevision(state) !== revision) {
          return {
            ...state,
            request: {
              ...state.request,
              evidence: {
                ...state.request.evidence,
                [panel]: { ...current, status: "stale" }
              }
            }
          };
        }
        return {
          ...state,
          request: {
            ...state.request,
            evidence: {
              ...state.request.evidence,
              [panel]: {
                ...current,
                status: "error",
                error: error.message,
                retry: { path, payload: requestPayload }
              }
            }
          }
        };
      });
      return false;
    }
  }

  /** @returns {Promise<ActionResult>} */
  function retryMutation() {
    const retry = store.getState().request.recovery?.retry;
    return retry
      ? executeStateMutation(retry)
      : Promise.resolve(skippedMutation("No failed editor mutation is available to retry."));
  }

  /** @param {EvidencePanel} panel @returns {Promise<boolean>} */
  function retryEvidence(panel) {
    const retry = store.getState().request.evidence[panel].retry;
    return retry ? refreshEvidence(panel, retry.path, retry.payload) : Promise.resolve(false);
  }

  /** @returns {void} */
  function dismissRecovery() {
    store.update((state) => {
      const mutation = state.request.mutation.status === "error"
        ? { status: /** @type {const} */ ("idle"), operation: null, error: null }
        : state.request.mutation;
      if (state.request.recovery === null && mutation === state.request.mutation) return state;
      return {
        ...state,
        request: { ...state.request, mutation, recovery: null }
      };
    });
  }

  /** @param {Partial<EditorState['view']>} patch @returns {void} */
  function patchView(patch) {
    store.update((state) => patchViewState(state, patch));
  }

  return {
    initialize,
    executeStateMutation,
    refreshEvidence,
    retryMutation,
    retryEvidence,
    dismissRecovery,
    patchView
  };
}
