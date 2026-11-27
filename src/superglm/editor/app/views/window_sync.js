// @ts-check

import { isReportName } from "../api/contracts.js";

/** @typedef {import('../api/contracts.js').EditorSnapshot} EditorSnapshot */
/** @typedef {import('../api/contracts.js').ReportName} ReportName */
/** @typedef {'assess'|ReportName} AssessView */
/**
 * A job's start or end, as one window tells the others of it.
 * @typedef {{id: string, phase: "started"|"settled"}} JobNotice
 */
/**
 * What a window tells the editor's other windows: its session key, with a job
 * notice, or `hello` from a page that has just loaded.
 * @typedef {{type: "superglm.changed", key: string, job?: JobNotice, hello?: true}} WindowMessage
 */
/**
 * @typedef {object} SyncChannel
 * @property {((event: {data: unknown}) => unknown)|null} onmessage
 * @property {(data: unknown) => void} postMessage
 * @property {() => void} close
 */
/** @typedef {new (name: string) => SyncChannel} SyncChannelConstructor */

const ASSESS_PARAM = "assess";

/**
 * The view an Assess window opens on, from the page's query, or null on an
 * ordinary page.
 * @param {string} search
 * @returns {AssessView|null}
 */
export function readAssessWindow(search) {
  const view = new URLSearchParams(search).get(ASSESS_PARAM);
  return view === "assess" || isReportName(view) ? view : null;
}

/**
 * The address of an Assess window on `view`: the page's own path and query,
 * so its token and test flag come along.
 * @param {string} href
 * @param {AssessView} view
 * @returns {string}
 */
export function assessWindowUrl(href, view) {
  const url = new URL(href);
  url.searchParams.set(ASSESS_PARAM, view);
  return url.pathname + url.search;
}

/**
 * The channel one editor's windows share: an editor on another port, or with
 * another token, never hears it.
 * @param {string} href
 * @returns {string}
 */
export function syncChannelName(href) {
  const url = new URL(href);
  return `superglm-editor:${url.port}:${url.searchParams.get("token") ?? ""}`;
}

/**
 * What two windows must agree on to show the same session: the model and the
 * one in force, what waits for Refit, the jobs running and the Final fit. A
 * selection is not in it, so moving through the Editor tells nobody anything.
 * @param {EditorSnapshot|null} snapshot
 * @returns {string}
 */
export function sessionKey(snapshot) {
  if (!snapshot) return "";
  return [
    snapshot.model_revision,
    snapshot.in_force ?? "",
    (snapshot.pending ?? []).map((step) => step.id).join(),
    (snapshot.jobs_running ?? []).map((job) => job.job_id).join(),
    snapshot.final_fit?.available ?? "",
    snapshot.final_fit?.stale ?? "",
  ].join("|");
}

/** @param {unknown} value @returns {value is JobNotice} */
function isJobNotice(value) {
  if (!value || typeof value !== "object") return false;
  const notice = /** @type {Record<string, unknown>} */ (value);
  return typeof notice.id === "string" && (notice.phase === "started" || notice.phase === "settled");
}

/** @param {unknown} data @returns {data is WindowMessage} */
function isWindowMessage(data) {
  if (!data || typeof data !== "object") return false;
  const message = /** @type {Record<string, unknown>} */ (data);
  return message.type === "superglm.changed"
    && typeof message.key === "string"
    && (message.job === undefined || isJobNotice(message.job))
    && (message.hello === undefined || message.hello === true);
}

/**
 * Tell the editor's other windows what changed here, and hear them. Without a
 * channel (an old browser) both do nothing. The page passes its
 * `BroadcastChannel`; there is no default, so a caller that passes none
 * opens nothing.
 * @param {object} options
 * @param {string} options.name
 * @param {(message: WindowMessage) => unknown} options.onMessage
 * @param {SyncChannelConstructor|undefined} options.Channel
 * @returns {{announce(message: WindowMessage): void, close(): void}}
 */
export function createWindowSync({ name, onMessage, Channel }) {
  if (!Channel) return Object.freeze({ announce() {}, close() {} });
  /** @type {SyncChannel|null} */
  let channel = new Channel(name);
  channel.onmessage = (event) => {
    if (isWindowMessage(event.data)) onMessage(event.data);
  };
  return Object.freeze({
    /** @param {WindowMessage} message */
    announce(message) {
      channel?.postMessage(message);
    },
    close() {
      channel?.close();
      channel = null;
    },
  });
}

/**
 * Run `run` once for a burst of asks: an ask while it runs runs it once more
 * after, however many came.
 * @param {() => Promise<unknown>} run
 * @returns {() => Promise<void>}
 */
export function coalesce(run) {
  let running = false;
  let again = false;
  return async () => {
    if (running) {
      again = true;
      return;
    }
    running = true;
    try {
      do {
        again = false;
        await run();
      } while (again);
    } finally {
      running = false;
    }
  };
}

/**
 * One window's part in following the session with the others.
 *
 * A window re-reads when another's key differs from its own, or when it hears
 * news: a job notice it has not handled, or a page's hello. News also lapses
 * the reports a job changes. Each job's start and end is handled once per
 * window, whether its own polling or a message brought it: `isNews` says
 * whether a notice is new here, and marks it handled. Two windows that agree
 * send nothing back, so one change costs each window one re-read.
 * @param {object} options
 * @param {string} options.name
 * @param {SyncChannelConstructor|undefined} options.Channel
 * @param {() => string} options.key this window's session key now
 * @param {() => unknown} options.onJobNews lapse the reports a job changes
 * @param {() => unknown} options.reread re-read the session
 */
export function createSessionPeer({ name, Channel, key, onJobNews, reread }) {
  /** @type {Set<string>} */
  const handled = new Set();
  /** @param {JobNotice} notice */
  function isNews(notice) {
    const id = `${notice.phase}:${notice.id}`;
    if (handled.has(id)) return false;
    handled.add(id);
    return true;
  }
  const sync = createWindowSync({
    name,
    Channel,
    onMessage: (message) => {
      const news = message.hello === true || (message.job !== undefined && isNews(message.job));
      if (news) onJobNews();
      if (news || message.key !== key()) void reread();
    },
  });
  /** @param {{job?: JobNotice, hello?: true}} extra */
  const post = (extra) => sync.announce({ type: "superglm.changed", key: key(), ...extra });
  return Object.freeze({
    isNews,
    /** Tell the others this window's key, with a notice it has just handled. @param {JobNotice} [job] */
    tell(job) {
      post(job ? { job } : {});
    },
    /** A page that has just loaded may have missed a job's end: every other window re-reads once. */
    hello() {
      post({ hello: true });
    },
    close: sync.close,
  });
}
