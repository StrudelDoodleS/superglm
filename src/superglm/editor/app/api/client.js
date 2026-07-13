// @ts-check

/** @typedef {(input:RequestInfo|URL, init?:RequestInit)=>Promise<Response>} FetchImpl */
/** @typedef {{token?:string, fetchImpl?:FetchImpl}} EditorClientOptions */

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

/** @param {EditorClientOptions} [options] */
export function createEditorClient({
  token = tokenFromLocation(),
  fetchImpl = globalThis.fetch
} = {}) {
  /** @param {RequestInit} options */
  function withEditorToken(options) {
    const headers = new Headers(options.headers || {});
    if (token) headers.set("X-SuperGLM-Editor-Token", token);
    return { ...options, headers };
  }

  /** @param {string} path @param {RequestInit} [options] @returns {Promise<unknown>} */
  async function requestJSON(path, options = {}) {
    const response = await fetchImpl(path, withEditorToken(options));
    if (!response.ok) throw await responseError(response);
    return response.json();
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

  return { requestJSON, postJSON, requestBlob, getState };
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
