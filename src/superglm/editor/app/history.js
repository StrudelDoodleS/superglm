// @ts-check
// The History pane: the session newest first, git-style. The changes waiting
// for Refit come first, then the applied ones down to the opened model, then
// what Redo would put back. Each step shows its short id, its time and an
// automatic message. A pencil writes a note in place, and notes are saved with
// the exported Python model.

import { escapeHTML, fmt } from "./format.js";
import { SHAPE_NAMES } from "./shapes.js";
import { OPERATION_HELP } from "./views/help_content.js";

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

/**
 * The selection-menu action behind each edit whose name is not its own. Its
 * Help title, less "selection", names the edit: "Straighten", "Average".
 * @type {Readonly<Record<string, string>>}
 */
const EDIT_ACTIONS = Object.freeze({
  linear_interpolate: "linearise",
  weighted_average: "average",
});

/**
 * Each structural operation by the change it makes; a change refitted at once
 * is named by its route.
 * @type {Readonly<Record<string, string>>}
 */
const CHANGES = Object.freeze({
  collapse: "collapse",
  collapse_levels: "collapse",
  ungroup: "ungroup",
  ungroup_levels: "ungroup",
  set_reference: "set_reference",
  shape: "shape",
  shape_range: "shape",
});

// The session's root, below every applied step. The state payload carries no
// id or time for the session's start, so the root shows neither, and it takes
// no note.
const OPENED_ITEM = '<li class="history-item root"><div class="history-row">'
  + '<div class="history-body"><div class="history-label">Opened model</div></div></div></li>';

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
      newestFirst.filter((entry) => entryStatus(entry) !== "waiting"), newest, "", OPENED_ITEM
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
 * @param {string} [last] an item that ends the list, whatever it holds
 */
function historySection(kind, title, entries, newest, hint = "", last = "") {
  if (!entries.length && !last) return "";
  const items = entries.map((entry) => historyItem(entry, entry === newest)).join("") + last;
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
    + `<div class="history-body"><div class="history-label">${escapeHTML(entryMessage(entry, status))}</div>`
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

/**
 * The step's automatic message, sentence-cased and without its term, which the
 * meta line names: "Collapse B10 + B11", "Line 18 – 26", "Refit · 2 changes".
 * A structural change is told from its operation and parameters. A change
 * refitted at once on its own is listed as its step, which carries no
 * parameters, so its message comes from its label, the backend's fixed
 * sentence for that operation. An edit on a selection names its action and
 * the stretch of axis it changed, "Smooth 62.7 – 85", or the levels it changed
 * where they are no stretch, "Decrease B1, B5"; any other edit keeps its
 * label. The labels themselves stay as they are: the Undo popover reads them.
 * @param {TimelineEntry} entry @param {"applied"|"waiting"|"edit"} status
 */
function entryMessage(entry, status) {
  const label = String(entry.label ?? "");
  if (status === "edit") return editMessage(entry) ?? sentenceCase(label);
  const change = CHANGES[String(entry.operation ?? "")] ?? "";
  const term = typeof entry.term === "string" ? entry.term : "";
  return changeMessage(change, entry.params ?? {})
    ?? sentenceCase(spacedRange(change, withoutTerm(label, change, term)));
}

/**
 * The message for a structural change from its parameters, or null when they
 * do not say it.
 * @param {string} change @param {Record<string, unknown>} params @returns {string|null}
 */
function changeMessage(change, params) {
  const levels = Array.isArray(params.levels) && params.levels.length
    ? params.levels.map(String)
    : null;
  if (change === "collapse" && levels) return `Collapse ${levels.join(" + ")}`;
  if (change === "ungroup" && levels) return `Ungroup ${levels.join(", ")}`;
  if (change === "set_reference" && params.level !== undefined && params.level !== null) {
    return `Set reference ${String(params.level)}`;
  }
  const shape = typeof params.degree === "number" ? SHAPE_NAMES[params.degree] : undefined;
  if (change === "shape" && shape && params.lo !== undefined && params.hi !== undefined) {
    return `${shape} ${edgeText(params.lo)} – ${edgeText(params.hi)}`;
  }
  return null;
}

/**
 * An edit as the selection menu names its action, with the stretch of axis it
 * changed as the axis prints it: "Make increasing 18 – 30", "Decrease B10".
 * Levels with others between them are no stretch, so they are named instead:
 * "Decrease B1, B5". Null for an edit whose params carry no range, such as a
 * handle move.
 * @param {TimelineEntry} entry @returns {string|null}
 */
function editMessage(entry) {
  const params = entry.params ?? {};
  const operation = String(entry.operation ?? "");
  if (Array.isArray(params.levels) && params.levels.length) {
    return `${editAction(operation, params)} ${levelList(params.levels)}`;
  }
  if (params.lo === undefined || params.hi === undefined) return null;
  const lo = axisText(params.lo);
  const hi = axisText(params.hi);
  return `${editAction(operation, params)} ${lo === hi ? lo : `${lo} – ${hi}`}`;
}

/**
 * Levels named in axis order, up to three, then the first two and a count:
 * "B1, B5", "B1, B3, B5", "B1, B5 +2 more".
 * @param {unknown[]} levels
 */
function levelList(levels) {
  const names = levels.map(String);
  return names.length > 3
    ? `${names.slice(0, 2).join(", ")} +${names.length - 2} more`
    : names.join(", ");
}

/**
 * @param {string} operation @param {Record<string, unknown>} params
 */
function editAction(operation, params) {
  const action = menuAction(operation, params);
  return Object.hasOwn(OPERATION_HELP, action)
    ? OPERATION_HELP[action].title.replace(/ selection$/, "")
    : sentenceCase(operation.replaceAll("_", " "));
}

/**
 * The selection-menu action that made an edit: a shift up or down, a fit
 * increasing or decreasing, else the action of the edit's own name.
 * @param {string} operation @param {Record<string, unknown>} params
 */
function menuAction(operation, params) {
  if (operation === "shift") return Number(params.delta) < 0 ? "shift_down" : "shift_up";
  if (operation === "isotonic") return params.direction === "decreasing" ? "decreasing" : "increasing";
  return EDIT_ACTIONS[operation] ?? operation;
}

/**
 * An edit's edge as the axis prints it: a number in the axis's tick format, a
 * level as it is.
 * @param {unknown} edge
 */
function axisText(edge) {
  return typeof edge === "number" ? fmt(edge) : String(edge);
}

/**
 * The backend's label without its mention of the term: "collapse B10 + B11 in
 * VehBrand" reads "collapse B10 + B11", "set reference of VehBrand to B2" reads
 * "set reference B2".
 * @param {string} label @param {string} change @param {string} term
 */
function withoutTerm(label, change, term) {
  if (!term) return label;
  const reference = `set reference of ${term} to `;
  if (change === "set_reference" && label.startsWith(reference)) {
    return `set reference ${label.slice(reference.length)}`;
  }
  const suffix = ` in ${term}`;
  return label.endsWith(suffix) ? label.slice(0, -suffix.length) : label;
}

/**
 * A shape's range with its dash spaced, as a message built from its edges
 * writes it: "Line 18–26" reads "Line 18 – 26".
 * @param {string} change @param {string} text
 */
function spacedRange(change, text) {
  return change === "shape" ? text.replace(/(\S)–(\S)/u, "$1 – $2") : text;
}

/**
 * A range edge as the backend writes it: a band's label as it is, a number to
 * six significant figures.
 * @param {unknown} edge
 */
function edgeText(edge) {
  return typeof edge === "number" && Number.isFinite(edge)
    ? String(Number(edge.toPrecision(6)))
    : String(edge);
}

/** @param {string} text */
function sentenceCase(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
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
