// @ts-check

/**
 * The Chart / Table switch and the rating-table view: the current term's
 * block of the Excel rating table, with the number formats and the note the
 * workbook gives it.
 */

/** @typedef {import('../api/contracts.js').RatingTableResponse} RatingTableResponse */
/** @typedef {import('../api/contracts.js').TermView} TermView */

export const RATING_TABLE_LOADING = "Building the rating table...";
export const RATING_TABLE_FAILED = "The rating table could not be built.";

/**
 * What the table view shows: a message, or a header and formatted rows.
 * @typedef {Object} RatingTableModel
 * @property {string} term
 * @property {string|null} message
 * @property {string[]} header
 * @property {string[][]} body
 * @property {boolean[]} numeric
 * @property {string|null} note
 */

/**
 * @param {string} term
 * @param {string} message
 * @returns {RatingTableModel}
 */
export function ratingTableMessage(term, message) {
  return { term, message, header: [], body: [], numeric: [], note: null };
}

/**
 * @param {RatingTableResponse} response
 * @returns {RatingTableModel}
 */
export function ratingTableModel(response) {
  if (!response.available) {
    return ratingTableMessage(response.term, response.reason || RATING_TABLE_FAILED);
  }
  const numeric = response.columns.map((_, column) =>
    response.rows.some((row) => typeof row[column] === "number")
  );
  return {
    term: response.term,
    message: null,
    header: response.columns.slice(),
    body: response.rows.map((row) =>
      row.map((value, column) => formatRatingCell(value, response.formats[column] ?? null))
    ),
    numeric,
    note: response.note,
  };
}

/**
 * One cell as the workbook's number format shows it. The export applies four
 * formats: fixed places ("0.000000", "0.000000000000"), grouped with two
 * places ("#,##0.00") and scientific ("0.00000000000000E+00"); any other
 * cell is shown as it is.
 *
 * @param {string|number|boolean|null} value
 * @param {string|null} format
 * @returns {string}
 */
export function formatRatingCell(value, format) {
  if (value === null) return "";
  if (typeof value !== "number") return String(value);
  if (!Number.isFinite(value)) return "";
  if (format === "#,##0.00") {
    return value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  const fixed = format ? /^0\.(0+)$/.exec(format) : null;
  if (fixed) return value.toFixed(fixed[1].length);
  const scientific = format ? /^0\.(0+)E\+00$/.exec(format) : null;
  if (scientific) {
    const [mantissa, exponent] = value.toExponential(scientific[1].length).split("e");
    const power = Number(exponent);
    return `${mantissa}E${power < 0 ? "-" : "+"}${String(Math.abs(power)).padStart(2, "0")}`;
  }
  return String(value);
}

/**
 * @param {HTMLElement} frame
 * @param {RatingTableModel} model
 */
export function renderRatingTable(frame, model) {
  const doc = frame.ownerDocument;
  frame.dataset.term = model.term;
  if (model.message !== null) {
    const message = doc.createElement("p");
    message.className = "rating-table-message";
    message.textContent = model.message;
    frame.replaceChildren(message);
    return;
  }
  const table = doc.createElement("table");
  table.className = "rating-table";
  const caption = doc.createElement("caption");
  caption.textContent = `${model.term} · as written to the Excel rating table`;
  const headRow = doc.createElement("tr");
  model.header.forEach((label, column) => {
    const cell = doc.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    if (model.numeric[column]) cell.className = "numeric";
    headRow.append(cell);
  });
  const head = doc.createElement("thead");
  head.append(headRow);
  const body = doc.createElement("tbody");
  for (const values of model.body) {
    const row = doc.createElement("tr");
    values.forEach((text, column) => {
      const cell = doc.createElement("td");
      cell.textContent = text;
      if (model.numeric[column]) cell.className = "numeric";
      row.append(cell);
    });
    body.append(row);
  }
  table.append(caption, head, body);
  if (!model.note) {
    frame.replaceChildren(table);
    return;
  }
  // The workbook writes a block's note above its header.
  const note = doc.createElement("p");
  note.className = "rating-table-note";
  note.textContent = model.note;
  frame.replaceChildren(note, table);
}

/**
 * @param {HTMLElement} root
 * @param {TermView} view
 */
export function renderTermViewToggle(root, view) {
  for (const button of root.querySelectorAll("[data-term-view]")) {
    if (!(button instanceof HTMLButtonElement)) continue;
    const active = button.dataset.termView === view;
    button.setAttribute("aria-checked", String(active));
    button.tabIndex = active ? 0 : -1;
    button.classList.toggle("active", active);
  }
}

/**
 * @param {HTMLElement} root
 * @param {{onChange:(view:TermView)=>unknown}} options
 * @returns {{destroy:()=>void}}
 */
export function bindTermViewToggle(root, { onChange }) {
  /** @param {MouseEvent} event */
  function onClick(event) {
    const button = event.target instanceof Element ? event.target.closest("[data-term-view]") : null;
    if (!(button instanceof HTMLButtonElement) || !root.contains(button)) return;
    const view = button.dataset.termView;
    if (view === "chart" || view === "table") onChange(view);
  }

  /** @param {KeyboardEvent} event */
  function onKeyDown(event) {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    const current = root.querySelector('[data-term-view][aria-checked="true"]');
    /** @type {TermView} */
    const next = current instanceof HTMLElement && current.dataset.termView === "table"
      ? "chart"
      : "table";
    event.preventDefault();
    onChange(next);
    const button = root.querySelector(`[data-term-view="${next}"]`);
    if (button instanceof HTMLElement) button.focus();
  }

  root.addEventListener("click", onClick);
  root.addEventListener("keydown", onKeyDown);
  return Object.freeze({
    destroy() {
      root.removeEventListener("click", onClick);
      root.removeEventListener("keydown", onKeyDown);
    },
  });
}
