// @ts-check

import {
  commitRemote,
  commitStructuralTransition,
  patchView as patchViewState
} from "./store.js";
import { selectModelRevision } from "./selectors.js";

/** @typedef {import('../api/contracts.js').ActionResult} ActionResult */
/** @typedef {import('../api/contracts.js').EditorSnapshot} EditorSnapshot */
/** @typedef {import('../api/contracts.js').EditorState} EditorState */
/** @typedef {import('../api/contracts.js').EvidencePanel} EvidencePanel */
/** @typedef {import('../api/contracts.js').MutationDescriptor} MutationDescriptor */
/** @typedef {import('../api/contracts.js').StructuralActionResult} StructuralActionResult */
/** @typedef {import('../api/contracts.js').StructuralMutationDescriptor} StructuralMutationDescriptor */
/** @typedef {import('../api/contracts.js').StructuralTransitionEnvelope} StructuralTransitionEnvelope */
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
 * @typedef {Object} EditorActionOptions
 * @property {EditorStore} store
 * @property {ActionClient} client
 * @property {(revision:number)=>void|Promise<void>} [scheduleEvidence]
 * @property {()=>void|Promise<void>} [waitForPaint]
 */

/** @returns {Promise<void>} */
export function nextPaint() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });
}

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

/** @param {string} message @returns {{ok:false, skipped:true, error:Error}} */
function skippedMutation(message) {
  return {
    ok: false,
    skipped: true,
    error: new Error(message)
  };
}

/** @param {unknown} value @returns {value is Record<string, unknown>} */
function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** @param {unknown} value @returns {value is number} */
function isNonnegativeFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

/** @param {unknown} value @returns {boolean} */
function isEditorSnapshot(value) {
  if (!isRecord(value) || !Number.isInteger(value.model_revision)) return false;
  if (typeof value.selected_term !== "string") return false;
  if (!isRecord(value.terms) || !isRecord(value.selection)) return false;
  if (typeof value.can_uncollapse_levels !== "boolean") return false;
  if (value.last_collapse !== null && !isRecord(value.last_collapse)) return false;
  return isRecord(value.history) &&
    Array.isArray(value.history.active) &&
    Array.isArray(value.history.redo);
}

/** @param {unknown} value @returns {boolean} */
function isStructuralTiming(value) {
  return isRecord(value) &&
    typeof value.operation === "string" &&
    isNonnegativeFiniteNumber(value.fit_ms) &&
    isNonnegativeFiniteNumber(value.summary_ms) &&
    isNonnegativeFiniteNumber(value.state_ms) &&
    isNonnegativeFiniteNumber(value.server_total_ms);
}

/** @param {unknown} value @returns {StructuralTransitionEnvelope} */
function structuralEnvelope(value) {
  if (
    !isRecord(value) ||
    !isEditorSnapshot(value.state) ||
    !isRecord(value.summary) ||
    typeof value.summary.available !== "boolean" ||
    !isStructuralTiming(value.timing)
  ) {
    throw new Error("Structural transition response is malformed.");
  }
  return /** @type {StructuralTransitionEnvelope} */ (value);
}

/** @param {EditorActionOptions} options */
export function createEditorActions({
  store,
  client,
  scheduleEvidence = () => {},
  waitForPaint = nextPaint
}) {
  /**
   * Restores the last confirmed remote pair after a failed mutation, while retaining the existing
   * authoritative-state recovery request and retry behavior.
   *
   * @param {unknown} error
   * @param {MutationDescriptor|StructuralMutationDescriptor} descriptor
   * @param {EditorState['remote']} confirmed
   * @returns {Promise<{ok:false, error:Error}>}
   */
  async function recoverMutation(error, descriptor, confirmed) {
    const normalizedError = normalizeError(error);
    /** @type {EditorSnapshot|null} */
    let recovered = null;
    try {
      recovered = /** @type {EditorSnapshot} */ (await client.getState());
    } catch {
      // The exact last-confirmed remote pair remains authoritative while offline.
    }
    const isStructural = "waitForSecondary" in descriptor;
    store.update((state) => {
      const currentRemote = isStructural || state.remote === confirmed ? confirmed : state.remote;
      const restoredBase = {
        ...state,
        remote: currentRemote,
        view: { ...state.view, preview: null }
      };
      const restored = recovered ? commitRemote(restoredBase, recovered) : restoredBase;
      return {
        ...restored,
        request: {
          ...restored.request,
          mutation: {
            status: "error",
            operation: descriptor.name,
            error: normalizedError.message
          },
          recovery: {
            message: normalizedError.message,
            retry: descriptor
          }
        }
      };
    });
    return { ok: false, error: normalizedError };
  }

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

    const confirmed = store.getState().remote;
    const previousRevision = confirmed.snapshot?.model_revision ?? -1;
    const descriptor = { name, path, payload: snapshotPayload(payload) };
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
      snapshot = /** @type {EditorSnapshot} */ (
        await client.postJSON(path, descriptor.payload)
      );
    } catch (value) {
      return recoverMutation(value, descriptor, confirmed);
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
   * @param {StructuralMutationDescriptor} descriptor
   * @returns {Promise<StructuralActionResult>}
   */
  async function executeStructuralMutation({
    name,
    path,
    payload,
    waitForSecondary = async () => {}
  }) {
    if (store.getState().request.mutation.status === "running") {
      return skippedMutation("An editor mutation is already running.");
    }

    const confirmed = store.getState().remote;
    const descriptor = {
      name,
      path,
      payload: snapshotPayload(payload),
      waitForSecondary
    };
    store.update((state) => ({
      ...state,
      request: {
        ...state.request,
        mutation: { status: "running", operation: name, error: null },
        recovery: null
      }
    }));

    /** @type {StructuralTransitionEnvelope} */
    let envelope;
    try {
      envelope = structuralEnvelope(await client.postJSON(path, descriptor.payload));
      store.update((state) => commitStructuralTransition(state, envelope));
      await waitForPaint();
      await waitForSecondary(envelope.state.model_revision);
    } catch (value) {
      return recoverMutation(value, descriptor, confirmed);
    }
    store.update((state) => ({
      ...state,
      request: {
        ...state.request,
        mutation: { status: "idle", operation: null, error: null },
        recovery: null
      }
    }));
    return { ok: true, envelope };
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

  /** @returns {Promise<ActionResult|StructuralActionResult>} */
  function retryMutation() {
    const retry = store.getState().request.recovery?.retry;
    if (!retry) {
      return Promise.resolve(skippedMutation("No failed editor mutation is available to retry."));
    }
    return "waitForSecondary" in retry
      ? executeStructuralMutation(retry)
      : executeStateMutation(retry);
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
    executeStructuralMutation,
    refreshEvidence,
    retryMutation,
    retryEvidence,
    dismissRecovery,
    patchView
  };
}
