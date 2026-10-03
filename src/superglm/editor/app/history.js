// @ts-check
// The History pane: the session newest first, git-style. The changes waiting
// for Refit come first, then the applied ones, then what Redo would put back.
// Each step shows its short id, its time and an automatic message. A pencil
// writes a note in place, and notes are saved with the exported Python model.

import { escapeHTML } from "./format.js";

/** @typedef {import('./api/contracts.js').TimelineEntry} TimelineEntry */

const PENCIL = '<svg class="history-note-icon" viewBox="0 0 24 24" aria-hidden="true">'
  + '<path d="M4 20h4L19 9l-4-4L4 16z"></path><path d="m14 6 4 4"></path></svg>';
// S2 refuses a longer note with a fixed sentence.
const NOTE_MAX_LENGTH = 2000;
const STATUSES = new Set(["applied", "waiting", "edit"]);

/** @type {Readonly<Record<string, string>>} */
const OPERATION_WORDS = Object.freeze({
  collapse: "collapse",
  collapse_levels: "collapse",
  ungroup: "ungroup",
  ungroup_levels: "ungroup",
  set_reference: "reference",
  shape: "shape",
  shape_range: "shape",
  refit_pending: "refit",
  carry_edits: "edits carried over",
  revert_to_original: "revert",
});

/** The timeline each pane last drew, so a cancelled note can put it back. */
const drawn = new WeakMap();

/**
 * Render the session's timeline: the waiting changes, then the applied ones,
 * newest first, then the undone ones in the order Redo would put them back.
 * The newest step, which Undo takes, says so.
 * @param {TimelineEntry[]|undefined} timeline
 * @param {HTMLElement|null} node
 */
export function renderHistory(timeline, node) {
  if (!node) return;
  const entries = Array.isArray(timeline) ? timeline : [];
  drawn.set(node, entries);
  const marker = entries.findIndex((entry) => entry.kind === "marker");
  const done = marker < 0 ? entries : entries.slice(0, marker);
  const undone = marker < 0 ? [] : entries.slice(marker + 1);
  if (!done.length && !undone.length) {
    node.innerHTML = `<div class="history-empty">Nothing yet.</div>`;
    return;
  }
  const newest = done.at(-1) ?? null;
  const newestFirst = [...done].reverse();
  node.innerHTML = [
    historySection(
      "waiting", "Waiting for refit",
      newestFirst.filter((entry) => entryStatus(entry) === "waiting"), newest
    ),
    historySection(
      "applied", "Applied",
      newestFirst.filter((entry) => entryStatus(entry) !== "waiting"), newest
    ),
    historySection("undone", "Undone", undone, null, "Redo puts back the top one first."),
    `<p class="history-foot">Notes are saved with the exported Python model.</p>`
  ].join("");
}

/**
 * Wire the pane's pencils. A click opens the step's note in place. Enter, or
 * leaving the field, saves it, and Escape keeps the old one. The pane redraws
 * from the next confirmed timeline, and a note that did not save is put back
 * as it was.
 * @param {HTMLElement} node
 * @param {{onNote:(id:string, note:string)=>unknown}} handlers
 * @returns {{destroy:()=>void}}
 */
export function bindHistory(node, { onNote }) {
  /** @param {MouseEvent} event */
  function onClick(event) {
    const button = event.target instanceof Element
      ? event.target.closest(".history-note-edit")
      : null;
    const item = button?.closest("[data-step-id]");
    if (!(item instanceof HTMLElement) || !node.contains(item)) return;
    openNoteEditor(item);
  }

  /** @param {HTMLElement} item */
  function openNoteEditor(item) {
    const id = item.dataset.stepId || "";
    /** @type {TimelineEntry[]} */
    const entries = drawn.get(node) ?? [];
    const current = entries.find((entry) => entry.id === id)?.note ?? "";
    const doc = item.ownerDocument;
    const editor = doc.createElement("label");
    editor.className = "history-note-editor";
    editor.innerHTML = PENCIL;
    const input = doc.createElement("input");
    input.type = "text";
    input.className = "history-note-input";
    input.value = current;
    input.maxLength = NOTE_MAX_LENGTH;
    input.placeholder = "Why? Add a note to this step";
    input.setAttribute("aria-label", "Note for this step");
    editor.append(input);
    const shown = item.querySelector(".history-note");
    if (shown) shown.replaceWith(editor);
    else item.append(editor);
    input.focus();

    let settled = false;
    /** @param {boolean} save */
    const finish = (save) => {
      if (settled) return;
      settled = true;
      const note = input.value.trim();
      if (!save || note === current) {
        renderHistory(drawn.get(node), node);
        return;
      }
      void Promise.resolve(onNote(id, note)).finally(() => {
        if (editor.isConnected) renderHistory(drawn.get(node), node);
      });
    };
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        finish(true);
      } else if (event.key === "Escape") {
        // The note's own Escape: it must not also close the inspector or a popover.
        event.preventDefault();
        event.stopPropagation();
        finish(false);
      }
    });
    input.addEventListener("blur", () => finish(true));
  }

  node.addEventListener("click", onClick);
  return Object.freeze({
    destroy() {
      node.removeEventListener("click", onClick);
    },
  });
}

/**
 * @param {string} kind @param {string} title @param {TimelineEntry[]} entries
 * @param {TimelineEntry|null} newest @param {string} [hint]
 */
function historySection(kind, title, entries, newest, hint = "") {
  if (!entries.length) return "";
  const items = entries.map((entry) => historyItem(entry, entry === newest)).join("");
  const hintHTML = hint ? `<p class="history-section-hint">${hint}</p>` : "";
  return `<section class="history-section ${kind}" aria-label="${title}">`
    + `<h3 class="history-section-title">${title}</h3>${hintHTML}`
    + `<ol class="history-list">${items}</ol></section>`;
}

/** @param {TimelineEntry} entry @param {boolean} takesUndo */
function historyItem(entry, takesUndo) {
  const status = entryStatus(entry);
  const id = typeof entry.id === "string" ? entry.id : "";
  const shownId = id || (typeof entry.hash === "string" ? entry.hash : "");
  const note = typeof entry.note === "string" ? entry.note : "";
  const classes = ["history-item", status, entry.redo ? "redo" : ""].filter(Boolean).join(" ");
  const idAttribute = id ? ` data-step-id="${escapeHTML(id)}"` : "";
  const chip = takesUndo ? `<span class="history-undo-chip">Undo takes this</span>` : "";
  const stamp = `${shownId ? `<code class="history-id">${escapeHTML(shownId)}</code>` : ""}`
    + timeTag(entry.time);
  const noteHTML = note
    ? `<p class="history-note">${PENCIL}<span>${escapeHTML(note)}</span></p>`
    : "";
  return `<li class="${classes}"${idAttribute}><div class="history-row">`
    + `<div class="history-body"><div class="history-label">${escapeHTML(entry.label || "")}</div>`
    + `<div class="history-meta">${escapeHTML(entryMeta(entry, status))}</div></div>`
    + `${chip}<div class="history-stamp">${stamp}</div>${id ? noteButton(note) : ""}</div>`
    + `${noteHTML}</li>`;
}

/**
 * A timeline from before step ids has no status: its edits are edits and
 * its steps applied ones.
 * @param {TimelineEntry} entry @returns {"applied"|"waiting"|"edit"}
 */
function entryStatus(entry) {
  if (entry.status && STATUSES.has(entry.status)) return entry.status;
  return entry.kind === "edit" ? "edit" : "applied";
}

/** @param {TimelineEntry} entry @param {string} status */
function entryMeta(entry, status) {
  const operation = String(entry.operation ?? "");
  const word = status === "edit"
    ? "edit"
    : OPERATION_WORDS[operation] ?? (operation.replaceAll("_", " ") || "step");
  return [entry.term, word].filter(Boolean).join(" · ");
}

/** @param {unknown} seconds */
function timeTag(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "";
  const at = new Date(seconds * 1000);
  const clock = `${String(at.getHours()).padStart(2, "0")}:${String(at.getMinutes()).padStart(2, "0")}`;
  return `<time datetime="${at.toISOString()}">${clock}</time>`;
}

/** @param {string} note */
function noteButton(note) {
  const label = note ? "Edit note" : "Add a note";
  return `<button class="history-note-edit icon-button" type="button" aria-label="${label}"`
    + ` data-popover-title="${label}"`
    + ` data-popover-body="Say why. Notes are saved with the exported Python model.">`
    + `${PENCIL}</button>`;
}
