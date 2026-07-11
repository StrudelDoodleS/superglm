// @ts-check

import {
  beginEvidence,
  commitRemote,
  commitSelectionRemote,
  commitStructuralTransition,
  completeEvidence,
  failEvidence,
  normalizeSelectionIndices,
  patchView as patchViewState,
  selectionIndicesEqual,
  setSelectionPreview
} from "./store.js";
import { selectModelRevision } from "./selectors.js";

/** @typedef {import('../api/contracts.js').ActionResult} ActionResult */
/** @typedef {import('../api/contracts.js').EditorSnapshot} EditorSnapshot */
/** @typedef {import('../api/contracts.js').EditorState} EditorState */
/** @typedef {import('../api/contracts.js').EvidencePanel} EvidencePanel */
/** @typedef {import('../api/contracts.js').MutationDescriptor} MutationDescriptor */
/** @typedef {import('../api/contracts.js').RecoveryRequestState} RecoveryRequestState */
/** @typedef {import('../api/contracts.js').SummaryPayload} SummaryPayload */
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
 * @property {(revision:number, options?:{immediate?:boolean, summaryCommitted?:boolean})=>void|Promise<void>} [scheduleVisibleEvidence]
 * @property {()=>void|Promise<void>} [waitForPaint]
 * @property {(callback:()=>void, delay:number)=>any} [setTimer]
 * @property {(timer:any)=>void} [clearTimer]
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

const STRUCTURAL_OUTCOME_UNCERTAIN =
  "The model change outcome is uncertain. The operation was not retried.";
const STRUCTURAL_REFRESH_INCOMPLETE =
  "The model change completed, but browser refresh was incomplete.";
const EVIDENCE_DEBOUNCE_MS = 150;

/** @param {()=>void|Promise<void>} hook @returns {Promise<void>} */
async function notifyTimingHook(hook) {
  try {
    await hook();
  } catch {
    // Timing hooks cannot change the mutation outcome.
  }
}

/** @returns {SummaryPayload} */
function reconciledSummaryPayload() {
  return {
    available: false,
    label: "Summary unavailable",
    error: "The model state was reconciled, but its summary is stale. Refresh the summary."
  };
}

/** @param {EditorActionOptions} options */
export function createEditorActions({
  store,
  client,
  scheduleVisibleEvidence = () => {},
  waitForPaint = nextPaint,
  setTimer = globalThis.setTimeout.bind(globalThis),
  clearTimer = globalThis.clearTimeout.bind(globalThis)
}) {
  const evidenceTimers = new Map();
  /** @param {RecoveryRequestState|null} recovery */
  function finishStructuralMutation(recovery) {
    try {
      store.update((state) => ({
        ...state,
        request: {
          ...state.request,
          mutation: { status: "idle", operation: null, error: null },
          recovery
        }
      }));
    } catch {
      // The store installs the final request state before reporting listener failures.
    }
  }

  /**
   * Reconciles one state-only recovery response against the current remote revision. Structural
   * recovery advances only to a newer revision. Ordinary recovery also accepts an equal revision
   * because UI-only state, such as the selected term, does not increment the model revision.
   *
   * @param {unknown} error
   * @param {string} operation
   * @param {MutationDescriptor|null} retry
   * @param {(state:EditorState, snapshot:EditorSnapshot)=>EditorState} [commitRecovered]
   * @returns {Promise<{ok:false, error:Error}>}
   */
  async function recoverMutation(error, operation, retry, commitRecovered = commitRemote) {
    const normalizedError = normalizeError(error);
    /** @type {EditorSnapshot|null} */
    let recovered = null;
    try {
      const candidate = await client.getState();
      if (isEditorSnapshot(candidate)) {
        recovered = /** @type {EditorSnapshot} */ (candidate);
      }
    } catch {
      // The current remote pair remains authoritative while offline.
    }
    store.update((state) => {
      const currentRevision = state.remote.snapshot?.model_revision ?? -1;
      let restored;
      if (recovered && recovered.model_revision > currentRevision) {
        restored = commitRecovered(
          {
            ...state,
            remote: {
              snapshot: state.remote.snapshot,
              summary: reconciledSummaryPayload()
            }
          },
          recovered
        );
      } else if (
        recovered && retry !== null && recovered.model_revision === currentRevision
      ) {
        restored = commitRecovered(state, recovered);
      } else {
        restored = {
          ...state,
          view: { ...state.view, preview: null, selectionPreview: null }
        };
      }
      return {
        ...restored,
        request: {
          ...restored.request,
          mutation: {
            status: "error",
            operation,
            error: normalizedError.message
          },
          recovery: {
            message: retry ? normalizedError.message : STRUCTURAL_OUTCOME_UNCERTAIN,
            retry
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

    const previousRevision = store.getState().remote.snapshot?.model_revision ?? -1;
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
      return recoverMutation(value, name, descriptor);
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
        void Promise.resolve(scheduleVisibleEvidence(snapshot.model_revision)).catch(() => {});
      } catch {
        // Evidence refresh is independent of the already-confirmed mutation.
      }
    }
    return { ok: true, snapshot };
  }

  /**
   * @param {{term:string, indices:number[]}} selection
   * @returns {Promise<ActionResult>}
   */
  async function executeSelectionMutation({ term, indices }) {
    if (store.getState().request.mutation.status === "running") {
      return skippedMutation("An editor mutation is already running.");
    }

    const normalized = normalizeSelectionIndices(indices);
    const currentSelection = normalizeSelectionIndices(
      store.getState().remote.snapshot?.selection[term] ?? []
    );
    if (selectionIndicesEqual(currentSelection, normalized)) {
      return skippedMutation("The editor selection is already current.");
    }

    /** @type {MutationDescriptor} */
    const descriptor = {
      name: "select",
      path: "/select",
      payload: snapshotPayload({ term, indices: normalized })
    };
    store.update((state) => {
      const previewed = setSelectionPreview(state, term, normalized);
      return {
        ...previewed,
        request: {
          ...previewed.request,
          mutation: { status: "running", operation: "select", error: null },
          recovery: null
        }
      };
    });

    /** @type {EditorSnapshot} */
    let snapshot;
    try {
      snapshot = /** @type {EditorSnapshot} */ (
        await client.postJSON(descriptor.path, descriptor.payload)
      );
    } catch (value) {
      return recoverMutation(
        value,
        descriptor.name,
        descriptor,
        commitSelectionRemote
      );
    }

    store.update((state) => {
      const committed = commitSelectionRemote(state, snapshot);
      return {
        ...committed,
        request: {
          ...committed.request,
          mutation: { status: "idle", operation: null, error: null },
          recovery: null
        }
      };
    });
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
    onRequestSettled = () => {},
    onPrimaryCommitted = () => {},
    onPaintSettled = () => {}
  }) {
    if (store.getState().request.mutation.status === "running") {
      return skippedMutation("An editor mutation is already running.");
    }

    const requestPayload = snapshotPayload(payload);
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
    /** @type {unknown} */
    let response;
    try {
      response = await client.postJSON(path, requestPayload);
    } catch (value) {
      await notifyTimingHook(onRequestSettled);
      return recoverMutation(value, name, null);
    }
    await notifyTimingHook(onRequestSettled);
    try {
      envelope = structuralEnvelope(response);
    } catch (value) {
      return recoverMutation(value, name, null);
    }

    const stateBeforeCommit = store.getState();
    try {
      store.update((state) => commitStructuralTransition(state, envelope));
    } catch (value) {
      if (store.getState() === stateBeforeCommit) return recoverMutation(value, name, null);
      finishStructuralMutation({ message: STRUCTURAL_REFRESH_INCOMPLETE, retry: null });
      return { ok: true, envelope };
    }
    await notifyTimingHook(onPrimaryCommitted);
    try {
      await waitForPaint();
    } catch {
      finishStructuralMutation({ message: STRUCTURAL_REFRESH_INCOMPLETE, retry: null });
      return { ok: true, envelope };
    }
    await notifyTimingHook(onPaintSettled);
    finishStructuralMutation(null);
    try {
      void Promise.resolve(scheduleVisibleEvidence(envelope.state.model_revision, {
        immediate: true,
        summaryCommitted: true
      })).catch(() => {});
    } catch {
      // Evidence refresh cannot change an authoritative structural success.
    }
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
      const begun = beginEvidence(
        state,
        panel,
        revision,
        sequence,
        { path, payload: requestPayload }
      );
      return {
        ...begun,
        request: {
          ...begun.request,
          nextSequence: sequence + 1
        }
      };
    });
    const requestBody = {
      ...requestPayload,
      model_revision: revision,
      request_sequence: sequence
    };

    try {
      const response = await client.postJSON(path, requestBody);
      if (
        !isRecord(response) ||
        response.status === "superseded" ||
        Number(response.model_revision) !== revision ||
        Number(response.request_sequence) !== sequence
      ) {
        return false;
      }
      let accepted = false;
      store.update((state) => {
        const completed = completeEvidence(state, panel, revision, sequence, response);
        accepted = completed !== state;
        return completed;
      });
      return accepted;
    } catch (value) {
      const error = normalizeError(value);
      store.update((state) => failEvidence(state, panel, revision, sequence, error.message));
      return false;
    }
  }

  /**
   * Debounce cacheable evidence independently for each panel.
   *
   * @param {EvidencePanel} panel
   * @param {string} path
   * @param {Record<string, unknown>} payload
   * @param {{immediate?:boolean}} [options]
   */
  function schedulePanelEvidence(panel, path, payload, { immediate = false } = {}) {
    const requestPayload = snapshotPayload(payload);
    if (evidenceTimers.has(panel)) {
      clearTimer(evidenceTimers.get(panel));
      evidenceTimers.delete(panel);
    }
    const invoke = () => {
      evidenceTimers.delete(panel);
      void refreshEvidence(panel, path, requestPayload);
    };
    if (immediate) {
      invoke();
      return;
    }
    evidenceTimers.set(panel, setTimer(invoke, EVIDENCE_DEBOUNCE_MS));
  }

  /** @returns {Promise<ActionResult>} */
  function retryMutation() {
    const retry = store.getState().request.recovery?.retry;
    if (!retry) {
      return Promise.resolve(skippedMutation("No failed editor mutation is available to retry."));
    }
    if (retry.name === "select") {
      return executeSelectionMutation({
        term: /** @type {string} */ (retry.payload.term),
        indices: /** @type {number[]} */ (retry.payload.indices)
      });
    }
    return executeStateMutation(retry);
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
    executeSelectionMutation,
    executeStateMutation,
    executeStructuralMutation,
    refreshEvidence,
    schedulePanelEvidence,
    retryMutation,
    retryEvidence,
    dismissRecovery,
    patchView
  };
}
