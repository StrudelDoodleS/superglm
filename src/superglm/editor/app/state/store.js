// @ts-check
import { createEmptyEvidenceState } from "../api/contracts.js";
import { selectActiveTermName } from "./selectors.js";

/** @typedef {import('../api/contracts.js').EditorSnapshot} EditorSnapshot */
/** @typedef {import('../api/contracts.js').EditorState} EditorState */
/** @typedef {import('../api/contracts.js').TermPayload} TermPayload */

/** @param {EditorSnapshot|null} snapshot @returns {EditorState} */
export function createInitialEditorState(snapshot = null) {
  return {
    remote: { snapshot },
    view: {
      activeTerm: snapshot?.selected_term || "",
      activeView: "editor",
      mode: "select",
      showCi: false,
      showContrib: false,
      zoomByTerm: {},
      groupModeByTerm: {},
      inspectorPane: "summary",
      inspectorOpen: true,
      preview: null
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
    for (const subscription of subscriptions) {
      const selected = subscription.selector(state);
      if (!subscription.equals(selected, subscription.value)) {
        const oldValue = subscription.value;
        subscription.value = selected;
        subscription.listener(selected, oldValue);
      }
    }
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

/** @param {EditorState} state @param {EditorSnapshot} snapshot */
export function commitRemote(state, snapshot) {
  /** @type {EditorState} */
  const candidate = {
    ...state,
    remote: { snapshot },
    view: { ...state.view, preview: null }
  };
  return {
    ...candidate,
    view: { ...candidate.view, activeTerm: selectActiveTermName(candidate) }
  };
}
