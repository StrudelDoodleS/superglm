// @ts-check
import { createEmptyEvidenceState } from "../api/contracts.js";
import { selectActiveTermName } from "./selectors.js";

/** @typedef {import('../api/contracts.js').EditorSnapshot} EditorSnapshot */
/** @typedef {import('../api/contracts.js').EditorState} EditorState */
/** @typedef {import('../api/contracts.js').EvidencePanel} EvidencePanel */
/** @typedef {import('../api/contracts.js').StructuralTransitionEnvelope} StructuralTransitionEnvelope */
/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */

/** @param {EditorSnapshot|null} snapshot @returns {EditorState} */
export function createInitialEditorState(snapshot = null) {
  return {
    remote: { snapshot, summary: null, chartEpoch: 0 },
    view: {
      activeTerm: snapshot?.selected_term || "",
      activeView: "editor",
      mode: "select",
      showCi: false,
      showContrib: false,
      summaryLevelDisplay: "expanded",
      zoomByTerm: {},
      groupModeByTerm: {},
      inspectorPane: "summary",
      inspectorOpen: true,
      preview: null,
      selectionPreview: null
    },
    request: {
      mutation: { status: "idle", operation: null, error: null },
      evidence: {
        metrics: createEmptyEvidenceState(),
        summary: createEmptyEvidenceState(),
        report: createEmptyEvidenceState()
      },
      recovery: null,
      nextSequence: 1
    }
  };
}

/**
 * Creates a synchronous store for immutable editor state. Update callbacks must treat their
 * input as immutable and return that same reference for a no-op. State commits, selector
 * comparisons, and listener notifications all finish before `update` returns. Listener-driven
 * nested updates run immediately, after the triggering subscription's cached value is updated.
 * A listener failure does not starve later subscriptions; the first failure is rethrown after
 * all remaining notifications have been attempted.
 *
 * @param {EditorState} initialState
 */
export function createEditorStore(initialState) {
  let state = initialState;
  /** @type {Set<any>} Internal erasure; the public subscribe method remains generic. */
  const subscriptions = new Set();

  /**
   * Applies an immutable state update and synchronously notifies changed subscriptions.
   *
   * @param {(state: EditorState) => EditorState} updater Treat state as immutable and return it
   * unchanged when the update is a no-op.
   */
  function update(updater) {
    const previous = state;
    const next = updater(previous);
    if (next === previous) return;
    state = next;
    let firstListenerError;
    let listenerFailed = false;
    for (const subscription of subscriptions) {
      const selected = subscription.selector(state);
      if (!subscription.equals(selected, subscription.value)) {
        const oldValue = subscription.value;
        subscription.value = selected;
        try {
          subscription.listener(selected, oldValue);
        } catch (error) {
          if (!listenerFailed) {
            listenerFailed = true;
            firstListenerError = error;
          }
        }
      }
    }
    if (listenerFailed) throw firstListenerError;
  }

  /**
   * Subscribes synchronously to one selected value. Equality receives the next selected value
   * followed by the previous one. The cached value is replaced before the listener runs, so a
   * listener-triggered nested update runs immediately against current state without replaying the
   * outer value. Subscriptions added or removed by a listener take effect immediately.
   *
   * @template T
   * @param {(state: EditorState) => T} selector
   * @param {(nextSelected: T, previousSelected: T) => void} listener
   * @param {(nextSelected: T, previousSelected: T) => boolean} [equals]
   * @returns {() => void}
   */
  function subscribe(selector, listener, equals = Object.is) {
    const subscription = { selector, listener, equals, value: selector(state) };
    subscriptions.add(subscription);
    return () => {
      subscriptions.delete(subscription);
    };
  }

  return {
    getState: () => state,
    update,
    subscribe
  };
}

/** @param {EditorState} state @param {Partial<EditorState['view']>} patch */
export function patchView(state, patch) {
  return { ...state, view: { ...state.view, ...patch } };
}

/**
 * @param {EditorState} state
 * @param {string} term
 * @param {TermPayload} payload
 * @param {number[]} [selection]
 */
export function setPreviewTerm(state, term, payload, selection = []) {
  return patchView(state, { preview: { term, payload, selection: selection.slice() } });
}

/** @param {number[]} indices @returns {number[]} */
export function normalizeSelectionIndices(indices) {
  return [...new Set(indices)].sort((left, right) => left - right);
}

/** @param {number[]} left @param {number[]} right @returns {boolean} */
export function selectionIndicesEqual(left, right) {
  if (left.length !== right.length) return false;
  return left.every((value, index) => value === right[index]);
}

/** @param {EditorState} state @param {string} term @param {number[]} indices */
export function setSelectionPreview(state, term, indices) {
  const normalized = normalizeSelectionIndices(indices);
  const current = state.view.selectionPreview;
  if (
    current?.term === term &&
    selectionIndicesEqual(current.indices, normalized)
  ) {
    return state;
  }
  return patchView(state, { selectionPreview: { term, indices: normalized } });
}

/** @param {EditorState} state */
export function clearSelectionPreview(state) {
  if (state.view.selectionPreview === null) return state;
  return patchView(state, { selectionPreview: null });
}

/** @param {EditorSnapshot|null} current @param {EditorSnapshot} incoming */
function isOlderGeneratedSnapshot(current, incoming) {
  const currentGeneration = current?.state_generation;
  const incomingGeneration = incoming.state_generation;
  return typeof currentGeneration === "number" &&
    typeof incomingGeneration === "number" &&
    Number.isInteger(currentGeneration) &&
    Number.isInteger(incomingGeneration) &&
    incomingGeneration < currentGeneration;
}

/** @param {EditorState} state */
function clearRemotePreviews(state) {
  return {
    ...state,
    view: { ...state.view, preview: null, selectionPreview: null }
  };
}

/** @param {EditorState} state @param {number} revision */
function invalidatePriorEvidence(state, revision) {
  const evidence = { ...state.request.evidence };
  let changed = false;
  for (const panel of /** @type {EvidencePanel[]} */ (["metrics", "summary", "report"])) {
    const current = evidence[panel];
    if (
      current.revision === revision ||
      (current.status === "idle" && current.payload === null)
    ) {
      continue;
    }
    evidence[panel] = { ...current, status: "stale", error: null };
    changed = true;
  }
  if (!changed) return state.request;
  return { ...state.request, evidence };
}

/** @param {EditorState} state @param {EditorSnapshot} snapshot */
export function commitRemote(state, snapshot) {
  if (isOlderGeneratedSnapshot(state.remote.snapshot, snapshot)) {
    return clearRemotePreviews(state);
  }
  const previousRevision = state.remote.snapshot?.model_revision;
  const request = previousRevision !== undefined && previousRevision !== snapshot.model_revision
    ? invalidatePriorEvidence(state, snapshot.model_revision)
    : state.request;
  /** @type {EditorState} */
  const candidate = {
    ...state,
    remote: {
      snapshot,
      summary: state.remote.summary,
      chartEpoch: state.remote.chartEpoch + 1
    },
    view: { ...state.view, preview: null, selectionPreview: null },
    request
  };
  return {
    ...candidate,
    view: { ...candidate.view, activeTerm: selectActiveTermName(candidate) }
  };
}

/** @param {EditorState} state @param {EditorSnapshot} snapshot */
export function commitSelectionRemote(state, snapshot) {
  const current = state.remote.snapshot;
  if (isOlderGeneratedSnapshot(current, snapshot)) {
    return clearRemotePreviews(state);
  }
  const hasChartGenerations = Number.isInteger(current?.chart_generation) &&
    Number.isInteger(snapshot.chart_generation);
  const sameChart = hasChartGenerations
    ? current?.chart_generation === snapshot.chart_generation
    : current?.model_revision === snapshot.model_revision;
  if (
    current &&
    sameChart &&
    current.selected_term === snapshot.selected_term
  ) {
    return {
      ...state,
      remote: {
        ...state.remote,
        snapshot
      },
      view: { ...state.view, selectionPreview: null }
    };
  }
  return commitRemote(state, snapshot);
}

/** @param {EditorState} state @param {StructuralTransitionEnvelope} envelope */
export function commitStructuralTransition(state, envelope) {
  const committed = commitRemote(state, envelope.state);
  if (committed.remote.snapshot !== envelope.state) return committed;
  const summary = committed.request.evidence.summary;
  return {
    ...committed,
    remote: { ...committed.remote, snapshot: envelope.state, summary: envelope.summary },
    request: {
      ...committed.request,
      evidence: {
        ...committed.request.evidence,
        summary: {
          ...summary,
          status: /** @type {const} */ ("current"),
          revision: envelope.state.model_revision,
          payload: envelope.summary,
          error: null,
          retry: null
        }
      }
    }
  };
}

/**
 * @param {EditorState} state
 * @param {EvidencePanel} panel
 * @param {EditorState['request']['evidence'][EvidencePanel]} evidence
 */
function replaceEvidence(state, panel, evidence) {
  return {
    ...state,
    request: {
      ...state.request,
      evidence: { ...state.request.evidence, [panel]: evidence }
    }
  };
}

/**
 * @param {EditorState} state
 * @param {EvidencePanel} panel
 * @param {number} revision
 * @param {number} sequence
 * @param {{path:string, payload:Record<string, unknown>}} retry
 */
export function beginEvidence(state, panel, revision, sequence, retry) {
  const previous = state.request.evidence[panel];
  return replaceEvidence(state, panel, {
    ...previous,
    status: "updating",
    revision,
    sequence,
    error: null,
    retry
  });
}

/**
 * @param {EditorState} state
 * @param {EvidencePanel} panel
 * @param {number} revision
 * @param {number} sequence
 * @param {unknown} payload
 */
export function completeEvidence(state, panel, revision, sequence, payload) {
  const current = state.request.evidence[panel];
  if (revision !== state.remote.snapshot?.model_revision || sequence !== current.sequence) {
    return state;
  }
  return replaceEvidence(state, panel, {
    ...current,
    status: "current",
    payload,
    error: null,
    retry: null
  });
}

/**
 * @param {EditorState} state
 * @param {EvidencePanel} panel
 * @param {number} revision
 * @param {number} sequence
 * @param {unknown} error
 */
export function failEvidence(state, panel, revision, sequence, error) {
  const current = state.request.evidence[panel];
  if (revision !== state.remote.snapshot?.model_revision || sequence !== current.sequence) {
    return state;
  }
  return replaceEvidence(state, panel, {
    ...current,
    status: current.payload === null ? "error" : "stale",
    error: String(error)
  });
}
