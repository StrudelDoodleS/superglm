// @ts-check
import { createEmptyEvidenceState } from "../api/contracts.js";
import { defineMachine } from "./machine.js";
import { createRiskProfiles } from "./risk.js";
import { selectActiveTermName } from "./selectors.js";

/** @typedef {import('../api/contracts.js').EditorSnapshot} EditorSnapshot */
/** @typedef {import('../api/contracts.js').EditorState} EditorState */
/** @typedef {import('../api/contracts.js').EvidencePanel} EvidencePanel */
/** @typedef {import('../api/contracts.js').EvidenceState} EvidenceState */
/** @typedef {import('../api/contracts.js').EvidenceStatus} EvidenceStatus */
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
      termView: "chart",
      showCi: false,
      showContrib: false,
      freeLevels: null,
      showFreeLevels: false,
      showUnsmoothed: false,
      risks: createRiskProfiles(),
      unsmoothed: {},
      summaryLevelDisplay: "expanded",
      zoomByTerm: {},
      groupModeByTerm: {},
      inspectorPane: "summary",
      inspectorOpen: true,
      preview: null,
      selectionAnchor: null,
      selectionSpan: null
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

/**
 * The share of a term's weight that a selection holds, as Python computes it: from the
 * term's weights where any is positive, else from one per point.
 * @param {TermPayload} term @param {number} total @param {number[]} indices
 */
function selectedWeightShare(term, total, indices) {
  if (!(total > 0)) return 0;
  const weights = term.weights;
  const usable = weights?.length === term.n_points && weights.some((weight) => weight > 0);
  let selected = 0;
  for (const index of indices) selected += usable ? weights[index] : 1;
  return selected / total;
}

/** @param {TermPayload} term @param {number[]} indices @returns {TermPayload} */
function withSelectedShare(term, indices) {
  const { impact } = term;
  if (!impact) return term;
  const share = selectedWeightShare(term, impact.total_weight ?? 0, indices);
  if (impact.selected_weight_share === share) return term;
  return { ...term, impact: { ...impact, selected_weight_share: share } };
}

/**
 * Show a term and its selection. These are the page's to decide and need no reply: the
 * chart keeps its epoch, and createEditorActions tells Python in the background.
 * @param {EditorState} state @param {string} term @param {number[]} indices
 */
export function commitLocalView(state, term, indices) {
  const snapshot = state.remote.snapshot;
  const payload = snapshot?.terms[term];
  if (!snapshot || !payload) return state;
  const selection = normalizeSelectionIndices(indices);
  if (
    snapshot.selected_term === term &&
    state.view.activeTerm === term &&
    selectionIndicesEqual(snapshot.selection[term] ?? [], selection)
  ) {
    return state;
  }
  return {
    ...state,
    remote: {
      ...state.remote,
      snapshot: {
        ...snapshot,
        selected_term: term,
        selection: { ...snapshot.selection, [term]: selection },
        terms: { ...snapshot.terms, [term]: withSelectedShare(payload, selection) }
      }
    },
    view: { ...state.view, activeTerm: term }
  };
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
    view: { ...state.view, preview: null }
  };
}

/** @param {EditorState} state @param {number} revision */
function invalidatePriorEvidence(state, revision) {
  const evidence = { ...state.request.evidence };
  let changed = false;
  for (const panel of /** @type {EvidencePanel[]} */ (["metrics", "summary", "report"])) {
    const current = evidence[panel];
    evidence[panel] = moveEvidence(current, { type: "lapse", revision }, state.remote.snapshot);
    changed ||= evidence[panel] !== current;
  }
  if (!changed) return state.request;
  return { ...state.request, evidence };
}

/**
 * Whether two snapshots wait for different changes. Undo and Redo of a
 * waiting change keep the model revision, so this is how a report that shows
 * the waiting list (the Cross-validation tab's chip and Run CV's reason)
 * learns it is out of date.
 *
 * @param {EditorSnapshot|null} previous @param {EditorSnapshot} next
 */
export function waitingListChanged(previous, next) {
  /** @param {EditorSnapshot|null} snapshot */
  const ids = (snapshot) => (snapshot?.pending ?? []).map((step) => step.id).join(" ");
  return previous !== null && ids(previous) !== ids(next);
}

/** @param {EditorState} state */
function invalidateReportEvidence(state) {
  const current = state.request.evidence.report;
  const report = moveEvidence(current, { type: "lapse" }, state.remote.snapshot);
  if (report === current) return state.request;
  return { ...state.request, evidence: { ...state.request.evidence, report } };
}

/**
 * The anchor is a source index, and a level reorder (a drag, its Undo or Redo,
 * Reset order) renumbers its term's levels. The anchor follows its level, as
 * the selection does, and goes when that level is gone.
 *
 * @param {EditorState['view']['selectionAnchor']} anchor
 * @param {EditorSnapshot|null} previous
 * @param {EditorSnapshot} next
 */
function anchorFollowingItsLevel(anchor, previous, next) {
  if (!anchor) return anchor;
  const before = previous?.terms?.[anchor.term]?.levels;
  const after = next.terms?.[anchor.term]?.levels;
  if (!Array.isArray(before) || !Array.isArray(after)) return anchor;
  const level = before[anchor.index];
  if (after[anchor.index] === level) return anchor;
  const index = after.indexOf(level);
  return index >= 0 ? { term: anchor.term, index } : null;
}

/** @param {EditorState} state @param {EditorSnapshot} snapshot */
export function commitRemote(state, snapshot) {
  if (isOlderGeneratedSnapshot(state.remote.snapshot, snapshot)) {
    return clearRemotePreviews(state);
  }
  const previousRevision = state.remote.snapshot?.model_revision;
  const request = previousRevision !== undefined && previousRevision !== snapshot.model_revision
    ? invalidatePriorEvidence(state, snapshot.model_revision)
    : waitingListChanged(state.remote.snapshot, snapshot)
      ? invalidateReportEvidence(state)
      : state.request;
  const selectionAnchor = anchorFollowingItsLevel(
    state.view.selectionAnchor, state.remote.snapshot, snapshot
  );
  /** @type {EditorState} */
  const candidate = {
    ...state,
    remote: {
      snapshot,
      summary: state.remote.summary,
      chartEpoch: state.remote.chartEpoch + 1
    },
    view: { ...state.view, preview: null, selectionAnchor },
    request
  };
  return {
    ...candidate,
    view: { ...candidate.view, activeTerm: selectActiveTermName(candidate) }
  };
}

/** @param {EditorState} state @param {StructuralTransitionEnvelope} envelope */
export function commitStructuralTransition(state, envelope) {
  const committed = commitRemote(state, envelope.state);
  if (committed.remote.snapshot !== envelope.state) return committed;
  const installed = movePanel(committed, "summary", {
    type: "install",
    revision: envelope.state.model_revision,
    inForce: envelope.state.in_force ?? null,
    payload: envelope.summary
  });
  return {
    ...installed,
    remote: { ...installed.remote, snapshot: envelope.state, summary: envelope.summary }
  };
}

/**
 * Whether evidence asked for at `revision`, about the model `inForce` names, describes `snapshot`.
 * Where the server names the model, the model decides: an undo gives back the token its edit
 * replaced, so evidence asked for before the edit describes the snapshot again. A snapshot
 * without a token is matched by its revision.
 *
 * @param {EditorSnapshot|null} snapshot @param {number} revision @param {string|null} inForce
 */
function describesSnapshot(snapshot, revision, inForce) {
  if (inForce !== null && typeof snapshot?.in_force === "string") {
    return inForce === snapshot.in_force;
  }
  return revision === snapshot?.model_revision;
}

/**
 * What happens to an evidence panel. `ask` sends a request, which the panel then waits for;
 * `answer` and `fail` end the request they name by its `sequence` and by what it asked about;
 * `lapse` is a snapshot the panel no longer describes, by a new `revision` (a panel already at
 * that revision keeps its state) or, with none, by a changed waiting list; `install` puts in a
 * payload that came with the snapshot, as a structural change's summary does.
 * @typedef {{type:'ask', revision:number, inForce:string|null, sequence:number,
 *     retry:{path:string, payload:Record<string, unknown>}}
 *   |{type:'answer', revision:number, inForce:string|null, sequence:number, payload:unknown}
 *   |{type:'fail', revision:number, sequence:number, error:unknown}
 *   |{type:'lapse', revision?:number}
 *   |{type:'install', revision:number, inForce:string|null, payload:unknown}} EvidenceEvent
 */

/**
 * An evidence panel's lifecycle. A failure leaves a panel that shows nothing in `error`, and one
 * that shows an earlier payload `stale` with that payload kept (`fail_kept`). An answer lands
 * only while the panel waits for it, `updating`, or after the snapshot moved under it, `stale`:
 * an edit and its undo put back the model it was asked about. A panel `current` from a
 * structural change's summary takes no answer from the request that change overtook.
 */
export const EVIDENCE = defineMachine({
  name: "evidence",
  initial: /** @type {EvidenceStatus} */ ("idle"),
  moves: {
    idle: { ask: "updating", install: "current" },
    updating: {
      ask: "updating", answer: "current", fail: "error", fail_kept: "stale", lapse: "stale",
      install: "current"
    },
    current: { ask: "updating", lapse: "stale", install: "current" },
    stale: {
      ask: "updating", answer: "current", fail: "error", fail_kept: "stale", lapse: "stale",
      install: "current"
    },
    error: { ask: "updating", lapse: "stale", install: "current" }
  }
});

/**
 * The table's name for `event` on `evidence`: a failure over a payload the panel keeps is
 * `fail_kept`.
 * @param {EvidenceState} evidence @param {EvidenceEvent} event
 */
function evidenceMove(evidence, event) {
  return event.type === "fail" && evidence.payload !== null ? "fail_kept" : event.type;
}

/**
 * Make one move of an evidence panel.
 *
 * An answer lands only for the newest request (`sequence`) and only while the model that
 * request asked about, its own `inForce` captured when it was sent, describes `snapshot`. A
 * failure lands only for the newest request and only at the revision it was asked at: a
 * request overtaken by an edit, even one since undone, is asked again rather than shown as
 * failed.
 *
 * @param {EvidenceState} evidence @param {EvidenceEvent} event @param {EditorSnapshot|null} snapshot
 * @returns {EvidenceState} `evidence` itself where its state does not take the event, or the
 *   event is about another request or model
 */
export function moveEvidence(evidence, event, snapshot) {
  const target = EVIDENCE.next(evidence.status, evidenceMove(evidence, event));
  if (target === null) return evidence;
  switch (event.type) {
    case "ask":
      return {
        ...evidence,
        status: target,
        revision: event.revision,
        inForce: event.inForce,
        sequence: event.sequence,
        error: null,
        retry: event.retry
      };
    case "answer":
      if (
        event.sequence !== evidence.sequence ||
        !describesSnapshot(snapshot, event.revision, event.inForce)
      ) {
        return evidence;
      }
      return {
        ...evidence,
        status: target,
        // An answer asked for before an edit and its undo is current for the revision now held.
        revision: snapshot?.model_revision ?? event.revision,
        inForce: event.inForce,
        payload: event.payload,
        error: null,
        retry: null
      };
    case "fail":
      if (event.sequence !== evidence.sequence || event.revision !== snapshot?.model_revision) {
        return evidence;
      }
      return { ...evidence, status: target, error: String(event.error) };
    case "lapse":
      if (event.revision !== undefined && evidence.revision === event.revision) return evidence;
      return { ...evidence, status: target, error: null };
    case "install":
      return {
        ...evidence,
        status: target,
        revision: event.revision,
        inForce: event.inForce,
        payload: event.payload,
        error: null,
        retry: null
      };
  }
}

/**
 * @param {EditorState} state @param {EvidencePanel} panel @param {EvidenceEvent} event
 * @returns {EditorState} `state` itself where the panel does not move
 */
function movePanel(state, panel, event) {
  const current = state.request.evidence[panel];
  const next = moveEvidence(current, event, state.remote.snapshot);
  if (next === current) return state;
  return {
    ...state,
    request: {
      ...state.request,
      evidence: { ...state.request.evidence, [panel]: next }
    }
  };
}

/**
 * @param {EditorState} state
 * @param {EvidencePanel} panel
 * @param {number} revision
 * @param {number} sequence
 * @param {{path:string, payload:Record<string, unknown>}} retry
 * @param {string|null} [inForce] the snapshot's `in_force` when the request was made
 */
export function beginEvidence(state, panel, revision, sequence, retry, inForce = null) {
  return movePanel(state, panel, { type: "ask", revision, inForce, sequence, retry });
}

/**
 * @param {EditorState} state
 * @param {EvidencePanel} panel
 * @param {number} revision the revision the request was made at
 * @param {number} sequence
 * @param {unknown} payload
 * @param {string|null} [inForce] the `in_force` the request was made about
 */
export function completeEvidence(state, panel, revision, sequence, payload, inForce = null) {
  return movePanel(state, panel, { type: "answer", revision, inForce, sequence, payload });
}

/**
 * @param {EditorState} state
 * @param {EvidencePanel} panel
 * @param {number} revision the revision the request was made at
 * @param {number} sequence
 * @param {unknown} error
 */
export function failEvidence(state, panel, revision, sequence, error) {
  return movePanel(state, panel, { type: "fail", revision, sequence, error });
}
