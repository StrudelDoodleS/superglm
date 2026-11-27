// @ts-check

import { isReportName } from "../api/contracts.js";

/** @typedef {import('../api/contracts.js').EditorSnapshot} EditorSnapshot */
/** @typedef {import('../api/contracts.js').ReportName} ReportName */
/** @typedef {'assess'|ReportName} AssessView */
/**
 * What a window tells the editor's other windows: its session key, and whether a job
 * it watches started or ended.
 * @typedef {{type: "superglm.changed", key: string, job: boolean}} WindowMessage
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

/** @param {unknown} data @returns {data is WindowMessage} */
function isWindowMessage(data) {
  if (!data || typeof data !== "object") return false;
  const message = /** @type {Record<string, unknown>} */ (data);
  return message.type === "superglm.changed"
    && typeof message.key === "string"
    && typeof message.job === "boolean";
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
 * The ids of running jobs not in `known`, which are added to it: the jobs that
 * started since the last look.
 * @param {Readonly<Record<string, {job_id: string, status: string}|null>>} jobs
 * @param {Set<string>} known
 * @returns {string[]}
 */
export function startedJobs(jobs, known) {
  const started = Object.values(jobs).flatMap(
    (job) => (job && job.status === "running" && !known.has(job.job_id) ? [job.job_id] : []),
  );
  for (const id of started) known.add(id);
  return started;
}
