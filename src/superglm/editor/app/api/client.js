// @ts-check

/** @typedef {(input:RequestInfo|URL, init?:RequestInit)=>Promise<Response>} FetchImpl */
/** @typedef {{token?:string, fetchImpl?:FetchImpl}} EditorClientOptions */

const KNOWN_STATE_HEADER = "X-SuperGLM-Editor-Known-State";
const NO_KNOWN_STATE = "none";

/**
 * @typedef {Object} CutState a state reply that names each term's version and may leave
 *   out the terms this client already holds at that version
 * @property {number} state_generation
 * @property {Record<string, unknown>} terms
 * @property {Record<string, string>} term_versions
 */

/** @param {unknown} value @returns {value is CutState} */
function isCutState(value) {
  if (value === null || typeof value !== "object") return false;
  const state = /** @type {Record<string, unknown>} */ (value);
  return Number.isInteger(state.state_generation) &&
    state.terms !== null && typeof state.terms === "object" &&
    state.term_versions !== null && typeof state.term_versions === "object";
}

export class EditorAPIError extends Error {
  /**
   * @param {string} message
   * @param {number} status
   * @param {unknown} payload
   */
  constructor(message, status, payload) {
    super(message);
    this.name = "EditorAPIError";
    this.status = status;
    this.payload = payload;
  }
}

/** @returns {string} */
function tokenFromLocation() {
  if (typeof window === "undefined") return "";
  return new URLSearchParams(window.location.search).get("token") || "";
}

/** @param {unknown} payload @param {Response} response */
function errorMessage(payload, response) {
  if (payload && typeof payload === "object" && "error" in payload) {
    const message = payload.error;
    if (typeof message === "string" && message) return message;
  }
  return response.statusText || `HTTP ${response.status}`;
}

/** @param {Response} response @returns {Promise<EditorAPIError>} */
async function responseError(response) {
  /** @type {unknown} */
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    // Error responses are not guaranteed to contain JSON.
  }
  return new EditorAPIError(errorMessage(payload, response), response.status, payload);
}

/**
 * The page's fetch: the notebook host's when the editor runs inside a
 * notebook cell, which carries requests over widget messages, else the
 * browser's, to the local server.
 * @returns {FetchImpl}
 */
function hostFetch() {
  const host = /** @type {{superglmEditorHost?: {fetch?: FetchImpl}}} */ (globalThis)
    .superglmEditorHost;
  return host?.fetch ?? globalThis.fetch;
}

/** @param {EditorClientOptions} [options] */
export function createEditorClient({
  token = tokenFromLocation(),
  fetchImpl = hostFetch()
} = {}) {
  /** @param {RequestInit} options */
  function withEditorToken(options) {
    const headers = new Headers(options.headers || {});
    if (token) headers.set("X-SuperGLM-Editor-Token", token);
    return { ...options, headers };
  }

  /** @param {string} path @param {RequestInit} options @returns {Promise<unknown>} */
  async function sendJSON(path, options) {
    const response = await fetchImpl(path, withEditorToken(options));
    if (!response.ok) throw await responseError(response);
    return response.json();
  }

  // The page's copy of every term, as of the newest state reply it has merged. A
  // reply leaves out the terms unchanged since the reply this names, so an unchanged
  // term keeps its object from one snapshot to the next. Terms are never written to
  // in place: a drag edits a clone.
  /** @type {{generation:number, terms:Record<string, unknown>, versions:Record<string, string>}|null} */
  let replica = null;

  /**
   * @param {CutState} state
   * @returns {CutState|null} the state with every term, or null where the copy holds
   *   another version of a term the reply left out
   */
  function mergedState(state) {
    /** @type {Record<string, unknown>} */
    const terms = {};
    for (const [name, version] of Object.entries(state.term_versions)) {
      if (name in state.terms) terms[name] = state.terms[name];
      else if (replica?.versions[name] === version) terms[name] = replica.terms[name];
      else return null;
    }
    if (!replica || state.state_generation >= replica.generation) {
      replica = { generation: state.state_generation, terms, versions: state.term_versions };
    }
    return { ...state, terms };
  }

  /** @param {CutState} state a copy that cannot complete it asks for every term */
  async function wholeState(state) {
    return mergedState(state) ?? mergedState(/** @type {CutState} */ (await sendJSON("/state", {
      headers: { [KNOWN_STATE_HEADER]: NO_KNOWN_STATE }
    })));
  }

  /** @param {string} path @param {RequestInit} [options] @returns {Promise<unknown>} */
  async function requestJSON(path, options = {}) {
    const headers = new Headers(options.headers || {});
    headers.set(KNOWN_STATE_HEADER, replica ? String(replica.generation) : NO_KNOWN_STATE);
    const reply = await sendJSON(path, { ...options, headers });
    if (isCutState(reply)) return wholeState(reply);
    const state = /** @type {{state?:unknown}|null} */ (reply)?.state;
    if (isCutState(state)) return { .../** @type {object} */ (reply), state: await wholeState(state) };
    return reply;
  }

  /** @param {string} path @param {Record<string, unknown>} payload */
  function postJSON(path, payload) {
    return requestJSON(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
  }

  /** @param {string} path @param {RequestInit} [options] @returns {Promise<Response>} */
  async function requestBlob(path, options = {}) {
    const response = await fetchImpl(path, withEditorToken(options));
    if (!response.ok) throw await responseError(response);
    return response;
  }

  /** @returns {Promise<unknown>} */
  function getState() {
    return requestJSON("/state");
  }

  /** @param {string} term @returns {Promise<unknown>} */
  function ratingTable(term) {
    return postJSON("/rating_table", { term });
  }

  /** @param {string} term an ordered term, refitted with its levels free @returns {Promise<unknown>} */
  function freeLevels(term) {
    return postJSON("/free_levels", { term });
  }

  /** @param {string} term refitted with its smoothing switched off @returns {Promise<unknown>} */
  function unsmoothed(term) {
    return postJSON("/unsmoothed", { term });
  }

  /**
   * The distribution panel's data for one feature of a SuperLSS session.
   * @param {string} feature a data column or a term key such as "scale:age"
   * @returns {Promise<unknown>} a DistributionPanelResponse, or a superseded reply
   */
  function distribution(feature) {
    return postJSON("/distribution", { feature });
  }

  /** @param {string} kind "cv" or "final_fit" */
  function jobStart(kind) {
    return postJSON("/job_start", { kind });
  }

  /** @param {string} jobId @param {boolean} [wait] wait up to 30 s for the job to stop running */
  function jobStatus(jobId, wait = false) {
    return postJSON("/job_status", { job_id: jobId, wait });
  }

  /** @param {string} jobId */
  function jobCancel(jobId) {
    return postJSON("/job_cancel", { job_id: jobId });
  }

  return {
    requestJSON, postJSON, requestBlob, getState, ratingTable, freeLevels, unsmoothed,
    distribution, jobStart, jobStatus, jobCancel
  };
}

export const editorClient = createEditorClient();

/** @param {string} path @param {RequestInit} [options] */
export function requestJSON(path, options = {}) {
  return editorClient.requestJSON(path, options);
}

/** @param {string} path @param {Record<string, unknown>} payload */
export function postJSON(path, payload) {
  return editorClient.postJSON(path, payload);
}

/** @param {string} path @param {RequestInit} [options] */
export function requestBlob(path, options = {}) {
  return editorClient.requestBlob(path, options);
}
