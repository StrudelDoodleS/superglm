// @ts-check

/** @typedef {import('../api/contracts.js').EditorSnapshot} EditorSnapshot */
/** @typedef {import('../api/contracts.js').MutationDescriptor} MutationDescriptor */
/**
 * @typedef {{requiresConfirmation:false}|{
 *   requiresConfirmation:true,
 *   historyCount:number,
 *   selectedTerm:string,
 *   selectedLabels:string[],
 *   operationTitle:string,
 *   message:string
 * }} StructuralImpact
 */

/** @type {Readonly<Record<string, string>>} */
const OPERATION_TITLES = Object.freeze({
  "collapse levels": "Collapse levels",
  "ungroup levels": "Ungroup levels",
  "restore collapsed levels": "Restore previous collapse",
});

/**
 * Copy the user-visible impact of a structural refit from an authoritative snapshot.
 *
 * @param {EditorSnapshot} snapshot
 * @param {MutationDescriptor} operation
 * @returns {StructuralImpact}
 */
export function structuralImpact(snapshot, operation) {
  const historyCount = snapshot.history.active.length + snapshot.history.redo.length;
  if (historyCount === 0) return { requiresConfirmation: false };

  const selectedTerm = snapshot.selected_term;
  const levels = snapshot.terms[selectedTerm]?.levels || [];
  const selectedLabels = (snapshot.selection[selectedTerm] || [])
    .map((index) => levels[index])
    .filter((label) => typeof label === "string");
  const operationTitle = Object.prototype.hasOwnProperty.call(OPERATION_TITLES, operation.name)
    ? OPERATION_TITLES[operation.name]
    : sentenceCase(operation.name);
  const historyNoun = historyCount === 1 ? "entry" : "entries";
  const labelCopy = selectedLabels.length > 0 ? ` ${selectedLabels.join(", ")}` : "";
  const question = operation.name === "restore collapsed levels"
    ? `Restore the previous collapse in ${selectedTerm}?`
    : `${operationTitle}${labelCopy} in ${selectedTerm}?`;

  return {
    requiresConfirmation: true,
    historyCount,
    selectedTerm,
    selectedLabels,
    operationTitle,
    message: `${question} This refit clears ${historyCount} manual edit history ${historyNoun}.`,
  };
}

/** @param {string} value */
function sentenceCase(value) {
  return value ? value[0].toUpperCase() + value.slice(1) : "Confirm structural refit";
}

/**
 * Bind the one shared structural confirmation dialog.
 *
 * @param {HTMLDialogElement} dialog
 * @returns {{confirm:(impact:StructuralImpact)=>Promise<boolean>}}
 */
export function bindStructuralConfirm(dialog) {
  const title = dialog.querySelector("#structuralConfirmTitle");
  const message = dialog.querySelector("#structuralConfirmMessage");
  if (!title || !message) {
    throw new Error("Structural confirmation dialog markup is incomplete");
  }

  /** @type {{resolve:(confirmed:boolean)=>void, launcher:HTMLElement|null}|null} */
  let pending = null;

  /** @param {HTMLElement|null} launcher */
  function restoreLauncher(launcher) {
    if (!launcher || !launcher.isConnected) return;
    try {
      launcher.focus({ preventScroll: true });
    } catch {
      launcher.focus();
    }
  }

  function onCancel() {
    dialog.returnValue = "cancel";
  }

  function onClose() {
    const confirmation = pending;
    if (!confirmation) return;
    pending = null;
    restoreLauncher(confirmation.launcher);
    confirmation.resolve(dialog.returnValue === "confirm");
  }

  dialog.addEventListener("cancel", onCancel);
  dialog.addEventListener("close", onClose);

  return Object.freeze({
    /** @param {StructuralImpact} impact */
    confirm(impact) {
      if (!impact.requiresConfirmation) return Promise.resolve(true);
      if (pending) return Promise.resolve(false);

      title.textContent = impact.operationTitle;
      message.textContent = impact.message;
      dialog.returnValue = "";
      const activeElement = document.activeElement;
      const launcher = activeElement instanceof HTMLElement ? activeElement : null;

      return new Promise((resolve, reject) => {
        pending = { resolve, launcher };
        try {
          dialog.showModal();
        } catch (error) {
          pending = null;
          restoreLauncher(launcher);
          reject(error);
        }
      });
    },
  });
}
