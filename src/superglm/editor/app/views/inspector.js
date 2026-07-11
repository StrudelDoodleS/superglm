// @ts-check

/** @typedef {import('../api/contracts.js').EditorViewState['inspectorPane']} InspectorPane */

const ROVING_KEYS = new Set(["ArrowLeft", "ArrowRight", "Home", "End"]);

/**
 * Bind inspector controls while leaving panel and open state in the editor store.
 *
 * @param {object} options
 * @param {HTMLElement} options.root
 * @param {HTMLButtonElement} options.toggle
 * @param {HTMLButtonElement} options.closeButton
 * @param {HTMLButtonElement} options.scrim
 * @param {(panel:InspectorPane)=>unknown} options.onPanelChange
 * @param {(open:boolean)=>unknown} options.onOpenChange
 * @param {()=>boolean} options.isOpen
 * @param {()=>boolean} options.isNarrow
 * @returns {{
 *   open:(panel?:InspectorPane, source?:HTMLElement|null)=>void,
 *   close:(options?:{restoreFocus?:boolean})=>void,
 *   destroy:()=>void
 * }}
 */
export function bindInspector({
  root,
  toggle,
  closeButton,
  scrim,
  onPanelChange,
  onOpenChange,
  isOpen,
  isNarrow,
}) {
  /** @type {HTMLButtonElement[]} */
  const tabs = [];
  for (const element of root.querySelectorAll('[role="tab"]')) {
    if (element instanceof HTMLButtonElement) tabs.push(element);
  }

  /** @type {HTMLElement|null} */
  let opener = null;

  /** @param {InspectorPane} [panel] @param {HTMLElement|null} [source] */
  function open(panel = "summary", source = null) {
    const activeElement = document.activeElement;
    opener = source || (activeElement instanceof HTMLElement ? activeElement : null);
    onPanelChange(panel);
    onOpenChange(true);
  }

  /** @param {{restoreFocus?:boolean}} [options] */
  function close({ restoreFocus = true } = {}) {
    onOpenChange(false);
    const focusTarget = opener || toggle;
    if (restoreFocus) focusTarget.focus();
  }

  /** @param {MouseEvent} event */
  function onRootClick(event) {
    const target = event.target instanceof Element ? event.target : null;
    const tab = target?.closest("[data-inspector-tab]");
    if (!(tab instanceof HTMLButtonElement) || !root.contains(tab)) return;
    const panel = inspectorPane(tab.dataset.inspectorTab);
    if (panel) onPanelChange(panel);
  }

  /** @param {KeyboardEvent} event */
  function onRootKeyDown(event) {
    if (!(event.target instanceof HTMLButtonElement) || !ROVING_KEYS.has(event.key)) return;
    const index = tabs.indexOf(event.target);
    if (index < 0 || tabs.length === 0) return;
    event.preventDefault();
    const next = event.key === "Home"
      ? 0
      : event.key === "End"
        ? tabs.length - 1
        : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    const nextTab = tabs[next];
    const panel = inspectorPane(nextTab.dataset.inspectorTab);
    nextTab.focus();
    if (panel) onPanelChange(panel);
  }

  function onToggleClick() {
    if (isOpen()) {
      onOpenChange(false);
      return;
    }
    opener = toggle;
    onOpenChange(true);
  }

  function onCloseClick() {
    close();
  }

  function onScrimClick() {
    close();
  }

  /** @param {KeyboardEvent} event */
  function onDocumentKeyDown(event) {
    if (event.key !== "Escape" || !isOpen() || !isNarrow()) return;
    event.preventDefault();
    close();
  }

  root.addEventListener("click", onRootClick);
  root.addEventListener("keydown", onRootKeyDown);
  toggle.addEventListener("click", onToggleClick);
  closeButton.addEventListener("click", onCloseClick);
  scrim.addEventListener("click", onScrimClick);
  document.addEventListener("keydown", onDocumentKeyDown);

  return Object.freeze({
    open,
    close,
    destroy() {
      root.removeEventListener("click", onRootClick);
      root.removeEventListener("keydown", onRootKeyDown);
      toggle.removeEventListener("click", onToggleClick);
      closeButton.removeEventListener("click", onCloseClick);
      scrim.removeEventListener("click", onScrimClick);
      document.removeEventListener("keydown", onDocumentKeyDown);
    },
  });
}

/**
 * Render store-owned inspector state into its tabs, panes, toggle, and responsive scrim.
 *
 * @param {object} options
 * @param {HTMLElement} options.root
 * @param {HTMLButtonElement} options.toggle
 * @param {HTMLButtonElement} options.scrim
 * @param {InspectorPane} options.panel
 * @param {boolean} options.open
 * @param {boolean} options.narrow
 */
export function renderInspector({ root, toggle, scrim, panel, open, narrow }) {
  root.dataset.open = String(open);
  root.setAttribute("aria-hidden", String(!open));
  toggle.setAttribute("aria-expanded", String(open));
  scrim.hidden = !(open && narrow);

  for (const element of root.querySelectorAll('[role="tab"]')) {
    if (!(element instanceof HTMLButtonElement)) continue;
    const active = element.dataset.inspectorTab === panel;
    element.classList.toggle("active", active);
    element.setAttribute("aria-selected", String(active));
    element.tabIndex = active ? 0 : -1;
  }
  for (const element of root.querySelectorAll('[role="tabpanel"]')) {
    if (!(element instanceof HTMLElement)) continue;
    element.hidden = element.dataset.inspectorPane !== panel;
  }
}

/** @param {string|undefined} value @returns {InspectorPane|null} */
function inspectorPane(value) {
  return value === "summary" || value === "history" || value === "advanced" || value === "help"
    ? value
    : null;
}
