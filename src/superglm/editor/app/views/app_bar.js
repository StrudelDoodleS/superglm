// @ts-check

/** @typedef {import('../api/contracts.js').EditorSnapshot} EditorSnapshot */

const NOTHING_WAITING =
  "Nothing is waiting. Collapse, Ungroup, Set reference and the shapes wait here for one refit.";

/**
 * @param {object} options
 * @param {HTMLElement} options.root
 * @param {HTMLButtonElement} options.undoButton
 * @param {HTMLButtonElement} options.redoButton
 * @param {HTMLButtonElement} options.revertButton
 * @param {HTMLButtonElement} options.refreshButton
 * @param {HTMLButtonElement} options.refitButton
 * @param {(view:string)=>unknown} options.onView
 * @param {()=>unknown} options.onUndo
 * @param {()=>unknown} options.onRedo
 * @param {()=>unknown} options.onRevert
 * @param {()=>unknown} options.onRefresh
 * @param {()=>unknown} options.onRefit
 */
export function bindAppBar({
  root, undoButton, redoButton, revertButton, refreshButton, refitButton,
  onView, onUndo, onRedo, onRevert, onRefresh, onRefit,
}) {
  const tabs = Array.from(root.querySelectorAll('[role="tab"]')).filter(
    (tab) => tab instanceof HTMLButtonElement,
  );

  /** @param {MouseEvent} event */
  function onClick(event) {
    const element = event.target instanceof Element ? event.target : null;
    const tab = element?.closest('[role="tab"]');
    if (!(tab instanceof HTMLButtonElement)) return;
    onView(tab.dataset.view || "editor");
  }

  /** @param {KeyboardEvent} event */
  function onTabKeyDown(event) {
    if (!(event.target instanceof HTMLButtonElement)) return;
    const index = tabs.indexOf(event.target);
    if (index < 0 || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      return;
    }
    event.preventDefault();
    const next = event.key === "Home"
      ? 0
      : event.key === "End"
        ? tabs.length - 1
        : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    tabs[next].focus();
    onView(tabs[next].dataset.view || "editor");
  }

  /** @param {KeyboardEvent} event */
  function onDocumentKeyDown(event) {
    if (isEditableTarget(event.target) || event.altKey || document.querySelector("dialog[open]")) {
      return;
    }
    const key = event.key.toLowerCase();
    if (!(event.ctrlKey || event.metaKey)) {
      // R refits what waits; with nothing waiting the button is disabled.
      if (key === "r" && !event.defaultPrevented && !refitButton.disabled) {
        event.preventDefault();
        onRefit();
      }
      return;
    }
    if (key === "z" && !event.shiftKey) {
      event.preventDefault();
      if (!undoButton.disabled) onUndo();
    } else if (key === "y" || (key === "z" && event.shiftKey)) {
      event.preventDefault();
      if (!redoButton.disabled) onRedo();
    }
  }

  root.addEventListener("click", onClick);
  root.addEventListener("keydown", onTabKeyDown);
  undoButton.addEventListener("click", onUndo);
  redoButton.addEventListener("click", onRedo);
  revertButton.addEventListener("click", onRevert);
  refreshButton.addEventListener("click", onRefresh);
  refitButton.addEventListener("click", onRefit);
  document.addEventListener("keydown", onDocumentKeyDown);

  return Object.freeze({
    destroy() {
      root.removeEventListener("click", onClick);
      root.removeEventListener("keydown", onTabKeyDown);
      undoButton.removeEventListener("click", onUndo);
      redoButton.removeEventListener("click", onRedo);
      revertButton.removeEventListener("click", onRevert);
      refreshButton.removeEventListener("click", onRefresh);
      refitButton.removeEventListener("click", onRefit);
      document.removeEventListener("keydown", onDocumentKeyDown);
    },
  });
}

/**
 * @param {object} options
 * @param {HTMLElement} options.root
 * @param {string} options.activeView
 * @param {HTMLButtonElement} options.undoButton
 * @param {HTMLButtonElement} options.redoButton
 * @param {HTMLButtonElement} options.revertButton
 * @param {HTMLButtonElement} options.refreshButton
 * @param {HTMLButtonElement} options.refitButton
 * @param {HTMLElement} options.refitCount the count badge inside Refit
 * @param {string|null} options.undoLabel what Undo would take back; null disables it
 * @param {string|null} options.redoLabel what Redo would put back; null disables it
 * @param {boolean} options.canRevert
 * @param {boolean} options.busy
 * @param {number} options.pendingCount how many structural changes wait for Refit
 */
export function renderAppBar({
  root, activeView, undoButton, redoButton, revertButton, refreshButton, refitButton, refitCount,
  undoLabel, redoLabel, canRevert, busy, pendingCount,
}) {
  for (const element of root.querySelectorAll('[role="tab"]')) {
    if (!(element instanceof HTMLButtonElement)) continue;
    const active = element.dataset.view === activeView;
    element.classList.toggle("active", active);
    element.setAttribute("aria-selected", String(active));
    element.tabIndex = active ? 0 : -1;
  }
  undoButton.disabled = undoLabel === null;
  undoButton.dataset.popoverBody = undoLabel === null ? "Nothing to undo." : `Undo: ${undoLabel}`;
  redoButton.disabled = redoLabel === null;
  redoButton.dataset.popoverBody = redoLabel === null ? "Nothing to redo." : `Redo: ${redoLabel}`;
  revertButton.disabled = !canRevert;
  refreshButton.disabled = busy;
  renderRefit(refitButton, refitCount, pendingCount, busy);
}

/**
 * Refit is quiet while nothing waits and the one filled action while changes
 * do. Its badge and its name carry the count.
 * @param {HTMLButtonElement} button @param {HTMLElement} count
 * @param {number} pending @param {boolean} busy
 */
function renderRefit(button, count, pending, busy) {
  const waiting = pending > 0;
  const changes = `${pending} ${pending === 1 ? "change" : "changes"}`;
  button.disabled = busy || !waiting;
  button.classList.toggle("has-pending", waiting);
  button.setAttribute("aria-label", waiting ? `Refit, ${changes} waiting` : "Refit, nothing waiting");
  button.dataset.popoverBody = waiting
    ? `Apply ${changes} in one fit. Hand edits on terms whose structure did not change are kept.`
    : NOTHING_WAITING;
  count.hidden = !waiting;
  count.textContent = String(pending);
}

/**
 * Whether anything differs from the opened model: a live manual edit, a
 * change waiting for Refit, or an in-force model a structural step or a
 * distribution re-profile put there. The live edits and waiting changes are
 * the run just before the timeline's marker, so one exists exactly when the
 * entry before the marker is an edit or a waiting change.
 * @param {EditorSnapshot} snapshot
 */
export function revertAvailable(snapshot) {
  const { timeline } = snapshot;
  const marker = timeline.findIndex((entry) => entry.kind === "marker");
  const last = timeline[marker - 1];
  return last?.kind === "edit" || last?.status === "waiting" || !snapshot.in_force_is_original;
}

/** @param {EventTarget | null} target */
function isEditableTarget(target) {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return target.isContentEditable || tag === "input" || tag === "select" || tag === "textarea";
}
