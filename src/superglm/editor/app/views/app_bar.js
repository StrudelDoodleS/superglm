// @ts-check

/**
 * @param {object} options
 * @param {HTMLElement} options.root
 * @param {HTMLButtonElement} options.undoButton
 * @param {HTMLButtonElement} options.redoButton
 * @param {(view:string)=>unknown} options.onView
 * @param {()=>unknown} options.onUndo
 * @param {()=>unknown} options.onRedo
 */
export function bindAppBar({ root, undoButton, redoButton, onView, onUndo, onRedo }) {
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
    if (isEditableTarget(event.target) || event.altKey) return;
    const primary = event.ctrlKey || event.metaKey;
    if (!primary) return;
    const key = event.key.toLowerCase();
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
  document.addEventListener("keydown", onDocumentKeyDown);

  return Object.freeze({
    destroy() {
      root.removeEventListener("click", onClick);
      root.removeEventListener("keydown", onTabKeyDown);
      undoButton.removeEventListener("click", onUndo);
      redoButton.removeEventListener("click", onRedo);
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
 * @param {boolean} options.canUndo
 * @param {boolean} options.canRedo
 */
export function renderAppBar({ root, activeView, undoButton, redoButton, canUndo, canRedo }) {
  for (const element of root.querySelectorAll('[role="tab"]')) {
    if (!(element instanceof HTMLButtonElement)) continue;
    const active = element.dataset.view === activeView;
    element.classList.toggle("active", active);
    element.setAttribute("aria-selected", String(active));
    element.tabIndex = active ? 0 : -1;
  }
  undoButton.disabled = !canUndo;
  redoButton.disabled = !canRedo;
}

/** @param {EventTarget | null} target */
function isEditableTarget(target) {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName.toLowerCase();
  return target.isContentEditable || tag === "input" || tag === "select" || tag === "textarea";
}
