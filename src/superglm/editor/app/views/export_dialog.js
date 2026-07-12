// @ts-check

const EXPORTS = Object.freeze({
  joblib: Object.freeze({
    filename: "superglm_edited_model.joblib",
    description: "Python model",
    validationDescription: "Validated Python model",
    accept: Object.freeze({ "application/octet-stream": Object.freeze([".joblib"]) }),
  }),
  xlsx: Object.freeze({
    filename: "superglm_rating_tables.xlsx",
    description: "Excel rating workbook",
    validationDescription: "Excel rating workbook",
    accept: Object.freeze({
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": Object.freeze([
        ".xlsx",
      ]),
    }),
  }),
});

/** @typedef {keyof typeof EXPORTS} ExportFormat */

/**
 * @typedef {object} ExportDialogNodes
 * @property {HTMLElement} action
 * @property {HTMLDialogElement} dialog
 * @property {HTMLElement} close
 * @property {HTMLInputElement[]} formatInputs
 * @property {HTMLInputElement} filename
 * @property {HTMLInputElement} directory
 * @property {HTMLButtonElement} download
 * @property {HTMLButtonElement} saveToKernel
 * @property {HTMLButtonElement|null} [openDirectory]
 * @property {HTMLElement} status
 */

/**
 * @typedef {object} ExportDialogContext
 * @property {{requestBlob:(path:string)=>Promise<Response>, postJSON:(path:string,payload:Record<string,unknown>)=>Promise<unknown>}} client
 * @property {ExportDialogNodes} nodes
 * @property {(blob:Blob, filename:string, metadata:{description:string,accept:Readonly<Record<string,readonly string[]>>})=>Promise<string|null>} saveBlobToFile
 */

/** @param {unknown} error */
function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

/** @param {string|null} disposition */
function filenameFromDisposition(disposition) {
  if (!disposition) return "";
  const encoded = disposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (encoded) {
    try {
      return decodeURIComponent(encoded[1]);
    } catch {
      // Fall back to the ordinary filename or the requested name.
    }
  }
  const quoted = disposition.match(/filename="([^"]+)"/i);
  return quoted ? quoted[1] : "";
}

/** @param {unknown} value @returns {value is {path:string}} */
function hasPath(value) {
  return Boolean(value && typeof value === "object" && "path" in value &&
    typeof value.path === "string");
}

/** @param {unknown} value @returns {value is {validation_scope:string}} */
function hasValidationScope(value) {
  return Boolean(value && typeof value === "object" && "validation_scope" in value &&
    typeof value.validation_scope === "string");
}

/** @param {string} message @param {ExportFormat} format @param {string|null} validation */
function successMessage(message, format, validation) {
  if (format !== "joblib") return message;
  if (validation === "artifact+predictions") {
    return `${message} Round-trip validated; predictions validated.`;
  }
  if (validation === "artifact") return `${message} Round-trip validated.`;
  return message;
}

/**
 * Bind the self-contained model/workbook export dialog.
 *
 * @param {ExportDialogContext} context
 */
export function bindExportDialog({ client, nodes, saveBlobToFile }) {
  let pending = false;

  /** @returns {ExportFormat} */
  function selectedFormat() {
    const value = nodes.formatInputs.find((input) => input.checked)?.value;
    return value === "xlsx" ? "xlsx" : "joblib";
  }

  function normaliseFilename() {
    const format = selectedFormat();
    const current = nodes.filename.value.trim();
    const wasDefault = Object.values(EXPORTS).some((entry) => entry.filename === current);
    if (!current || wasDefault) nodes.filename.value = EXPORTS[format].filename;
  }

  function requestedFilename() {
    const format = selectedFormat();
    const current = nodes.filename.value.trim();
    if (current) return current;
    nodes.filename.value = EXPORTS[format].filename;
    return nodes.filename.value;
  }

  /** @param {boolean} value */
  function setPending(value) {
    pending = value;
    nodes.download.disabled = value;
    nodes.saveToKernel.disabled = value;
    if (nodes.openDirectory) nodes.openDirectory.disabled = value;
  }

  /** @param {()=>Promise<void>} operation */
  async function run(operation) {
    if (pending) return;
    setPending(true);
    try {
      await operation();
    } catch (error) {
      nodes.status.textContent = errorMessage(error);
    } finally {
      setPending(false);
    }
  }

  async function openDialog() {
    nodes.status.textContent = "";
    if (nodes.dialog.open) return;
    if (typeof nodes.dialog.showModal === "function") nodes.dialog.showModal();
    else nodes.dialog.setAttribute("open", "");
    if (nodes.directory.value) return;
    try {
      const payload = await client.postJSON("/save_directory", { path: "" });
      if (hasPath(payload)) nodes.directory.value = payload.path;
    } catch (error) {
      nodes.status.textContent = errorMessage(error);
    }
  }

  function closeDialog() {
    if (typeof nodes.dialog.close === "function") nodes.dialog.close();
    else nodes.dialog.removeAttribute("open");
  }

  async function downloadExport() {
    await run(async () => {
      const format = selectedFormat();
      const metadata = EXPORTS[format];
      const filename = requestedFilename();
      nodes.status.textContent = `Preparing ${metadata.description.toLowerCase()}...`;
      const query = `format=${encodeURIComponent(format)}&filename=${encodeURIComponent(filename)}`;
      const response = await client.requestBlob(`/download_export?${query}`);
      const blob = await response.blob();
      const responseFilename =
        filenameFromDisposition(response.headers.get("content-disposition")) || filename;
      const message = await saveBlobToFile(blob, responseFilename, {
        description: metadata.description,
        accept: metadata.accept,
      });
      if (message === null) {
        nodes.status.textContent = "Download cancelled.";
        return;
      }
      nodes.status.textContent = successMessage(
        message,
        format,
        response.headers.get("x-superglm-validation"),
      );
    });
  }

  async function saveExport() {
    await run(async () => {
      const format = selectedFormat();
      const filename = requestedFilename();
      nodes.status.textContent = "Saving to kernel path...";
      const payload = await client.postJSON("/export_file", {
        format,
        directory: nodes.directory.value || ".",
        filename,
      });
      if (!hasPath(payload)) throw new Error("Export response did not include a saved path.");
      const validation = hasValidationScope(payload) ? payload.validation_scope : null;
      nodes.status.textContent = successMessage(`Saved ${payload.path}`, format, validation);
    });
  }

  async function openDirectory() {
    if (!nodes.openDirectory) return;
    await run(async () => {
      nodes.status.textContent = "Opening folder...";
      const payload = await client.postJSON("/open_directory", {
        path: nodes.directory.value || ".",
      });
      if (!hasPath(payload)) throw new Error("Folder response did not include a path.");
      nodes.status.textContent = `Opened ${payload.path}`;
    });
  }

  /** @type {Array<[HTMLElement,string,EventListener]>} */
  const listeners = [
    [nodes.action, "click", openDialog],
    [nodes.close, "click", closeDialog],
    [nodes.download, "click", downloadExport],
    [nodes.saveToKernel, "click", saveExport],
  ];
  if (nodes.openDirectory) listeners.push([nodes.openDirectory, "click", openDirectory]);
  for (const input of nodes.formatInputs) listeners.push([input, "change", normaliseFilename]);
  for (const [node, event, listener] of listeners) node.addEventListener(event, listener);

  return {
    destroy() {
      for (const [node, event, listener] of listeners) {
        node.removeEventListener(event, listener);
      }
    },
  };
}
