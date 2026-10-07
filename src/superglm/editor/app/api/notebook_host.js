// @ts-check

// The editor inside a notebook cell. anywidget loads this one module: it
// writes the editor page into a frame in the cell, runs the page's modules
// from blob URLs, and carries the page's requests to Python over the
// widget's messages, where the local server's own request handlers answer
// them. The frame keeps the page's styles and element ids out of the
// notebook, and the notebook's out of the page.

export const REQUEST = "superglm.request";
export const RESPONSE = "superglm.response";
const MODULE_SPECIFIER = /(["'])superglm-module:([^"']+)\1/g;
const THEME_STORAGE_KEY = "superglm.editor.theme";
// A Response with one of these statuses must have no body.
const NULL_BODY_STATUSES = new Set([101, 103, 204, 205, 304]);

/**
 * @typedef {object} WidgetModel
 * @property {(name:string)=>any} get
 * @property {(content:unknown, callbacks?:unknown, buffers?:ArrayBuffer[])=>void} send
 * @property {(event:string, callback:(...args:any[])=>void)=>void} on
 * @property {(event:string, callback:(...args:any[])=>void)=>void} off
 */

/** @typedef {{Response: typeof Response, Headers: typeof Headers, Blob: typeof Blob}} Realm */

/**
 * @typedef {object} PendingRequest
 * @property {(response:Response)=>void} resolve
 * @property {(error:Error)=>void} reject
 * @property {BlobPart[]} parts
 * @property {number} received
 */

/** @returns {string} a random prefix that keeps one transport's request ids its own */
function transportPrefix() {
  const words = new Uint32Array(2);
  globalThis.crypto.getRandomValues(words);
  return Array.from(words, (word) => word.toString(36)).join("");
}

/**
 * A fetch whose requests go to Python as widget messages. Python answers each
 * in one or more parts, all carrying the request's id; the Response resolves
 * once every part has arrived. Responses are built in `realm`, the page's
 * window, so the page reads them as its own.
 *
 * Python's replies reach every view of the widget, and a rebuilt page gets a
 * new transport while its old requests may still be answered, so each
 * transport's ids carry their own random prefix.
 * @param {Pick<WidgetModel, "send">} model
 * @param {Realm} realm
 */
export function createMessageFetch(model, realm) {
  const prefix = transportPrefix();
  let nextId = 0;
  /** @type {Map<string, PendingRequest>} */
  const pending = new Map();

  /**
   * @param {RequestInfo|URL} input
   * @param {RequestInit} [init]
   * @returns {Promise<Response>}
   */
  function fetchOverMessages(input, init = {}) {
    const url = input instanceof URL ? input.pathname + input.search : String(input);
    if (init.body != null && typeof init.body !== "string") {
      return Promise.reject(new Error("The notebook editor sends text request bodies only."));
    }
    const id = `${prefix}-${nextId++}`;
    const headers = [...new realm.Headers(init.headers).entries()];
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject, parts: [], received: 0 });
      model.send({
        type: REQUEST,
        id,
        method: init.method || "GET",
        url,
        headers,
        body: init.body ?? null
      });
    });
  }

  /**
   * One part of a reply from Python.
   * @param {any} message
   * @param {(DataView<ArrayBuffer>|ArrayBuffer)[]} [buffers]
   */
  function receive(message, buffers = []) {
    if (!message || message.type !== RESPONSE) return;
    const entry = pending.get(message.id);
    if (!entry) return;
    entry.parts[message.part] = buffers[0] ?? new ArrayBuffer(0);
    entry.received += 1;
    if (entry.received < message.parts) return;
    pending.delete(message.id);
    const status = Number(message.status);
    const body = NULL_BODY_STATUSES.has(status) ? null : new realm.Blob(entry.parts);
    entry.resolve(new realm.Response(body, { status, headers: new realm.Headers(message.headers) }));
  }

  /** Fail every request still waiting: its page is gone. */
  function close() {
    for (const entry of pending.values()) entry.reject(new Error("The editor view was closed."));
    pending.clear();
  }

  return { fetch: fetchOverMessages, receive, close, pending };
}

/**
 * Each module's source with its imports pointing at the blob URLs of the
 * modules before it. Python lists the modules so each follows its imports.
 * @param {{path:string, source:string}[]} modules
 * @param {(source:string)=>string} urlFor makes a module URL from its final source
 * @returns {Record<string, string>} module path to URL
 */
export function linkModules(modules, urlFor) {
  /** @type {Record<string, string>} */
  const urls = {};
  for (const { path, source } of modules) {
    const linked = source.replace(MODULE_SPECIFIER, (_match, quote, target) => {
      const url = urls[target];
      if (!url) throw new Error(`Editor module ${path} imports ${target} before it is loaded.`);
      return `${quote}${url}${quote}`;
    });
    urls[path] = urlFor(linked);
  }
  return urls;
}

/**
 * The theme before first paint, as the page's own inline script sets it when
 * the local server serves the page: the remembered choice, else the browser's.
 * @param {Window} win
 */
function applyTheme(win) {
  let choice = null;
  try {
    choice = win.localStorage.getItem(THEME_STORAGE_KEY);
  } catch {
    // Storage can be blocked; the browser's setting decides.
  }
  const dark =
    choice === "dark" ||
    (choice !== "light" && win.matchMedia("(prefers-color-scheme: dark)").matches);
  win.document.documentElement.dataset.theme = dark ? "dark" : "light";
}

/** @param {{model: WidgetModel, el: HTMLElement}} context */
function render({ model, el }) {
  const bundle = model.get("bundle");
  const frame = el.ownerDocument.createElement("iframe");
  frame.title = "SuperGLM editor";
  frame.style.cssText =
    `display:block;width:100%;height:${Number(model.get("height")) || 720}px;` +
    "border:1px solid #d0d7de;border-radius:6px;background:white";
  /** @type {ReturnType<typeof createMessageFetch>|null} */
  let transport = null;
  /** @type {Window|null} */
  let builtFor = null;
  /** @type {string[]} */
  let moduleUrls = [];

  /** @param {any} message @param {(DataView<ArrayBuffer>|ArrayBuffer)[]} [buffers] */
  const onMessage = (message, buffers) => transport?.receive(message, buffers);
  model.on("msg:custom", onMessage);

  function revokeModules() {
    for (const url of moduleUrls) URL.revokeObjectURL(url);
    moduleUrls = [];
  }

  // A frame gets a fresh window whenever the notebook re-attaches the cell's
  // output, which leaves it blank; each new window gets the page again, and
  // the page reloads its state from Python as it does on a browser reload.
  function build() {
    const win = /** @type {(Window & typeof globalThis)|null} */ (frame.contentWindow);
    if (!win || win === builtFor) return;
    builtFor = win;
    transport?.close();
    revokeModules();
    const doc = win.document;
    doc.open();
    doc.write(bundle.html);
    doc.close();
    applyTheme(win);
    transport = createMessageFetch(model, win);
    Object.assign(win, { superglmEditorHost: { kind: "notebook", fetch: transport.fetch } });
    const urls = linkModules(bundle.modules, (source) =>
      win.URL.createObjectURL(new win.Blob([source], { type: "text/javascript" }))
    );
    moduleUrls = Object.values(urls);
    const script = doc.createElement("script");
    script.type = "module";
    script.src = urls[bundle.entry];
    doc.head.append(script);
  }

  frame.addEventListener("load", build);
  el.append(frame);
  build();
  return () => {
    frame.removeEventListener("load", build);
    model.off("msg:custom", onMessage);
    transport?.close();
    revokeModules();
    frame.remove();
  };
}

export default { render };
