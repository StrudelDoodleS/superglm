// @ts-check

import { clockText, foldColour, foldProgress } from "./cv_tab.js";

/** @typedef {import('../api/contracts.js').JobStatus} JobStatus */
/** @typedef {{fill:"done"|"now"|"waiting", colour:string}} PillTick */
/**
 * @typedef {object} PillState
 * @property {string} text
 * @property {string} time
 * @property {"running"|"finished"|"failed"} tone
 * @property {PillTick[]} ticks
 * @property {"cv"|"final"} view the tab a click opens
 */

// Past this many folds the ticks stop being countable, and the text says it all.
const MAX_TICKS = 12;

/** @param {JobStatus} job @returns {PillTick[]} */
function foldTicks(job) {
  const { done, nFolds } = foldProgress(job);
  if (!nFolds || nFolds > MAX_TICKS) return [];
  return Array.from({ length: nFolds }, (_, index) => ({
    fill: index < done.length ? "done" : index === done.length && job.status === "running" ? "now" : "waiting",
    colour: foldColour(index),
  }));
}

/** @param {JobStatus} job */
function runningText(job) {
  const name = job.kind === "cv" ? "Run CV" : "Final fit";
  if (job.cancel_requested) return `${name}: cancelling`;
  if (job.kind === "final_fit") return "Final fit: fitting all rows";
  const { done, nFolds } = foldProgress(job);
  if (!nFolds) return "Run CV: starting";
  return done.length >= nFolds ? "Run CV: reading the fold curves" : `Run CV: fold ${done.length + 1} of ${nFolds}`;
}

/** @param {JobStatus} job */
function finishedText(job) {
  if (job.kind === "final_fit") return "Final fit finished";
  const headline = /** @type {{label?:string, value?:number}|null|undefined} */ (job.result?.headline);
  if (!headline || typeof headline.value !== "number") return "Run CV finished";
  return `Run CV finished: ${headline.label} ${headline.value.toFixed(Math.abs(headline.value) >= 100 ? 2 : 4)}`;
}

/**
 * What the pill shows for ``job``, or null when it shows nothing: no job, one
 * that was cancelled, or one that ended and has been seen on its tab.
 * @param {JobStatus|null|undefined} job @param {boolean} seen
 * @returns {PillState|null}
 */
export function jobPillState(job, seen) {
  if (!job) return null;
  const view = job.kind === "cv" ? "cv" : "final";
  const time = clockText(job.elapsed_s);
  const ticks = foldTicks(job);
  if (job.status === "running") return { text: runningText(job), time, tone: "running", ticks, view };
  if (seen || job.status === "cancelled") return null;
  if (job.status === "failed") {
    return { text: `${job.kind === "cv" ? "Run CV" : "Final fit"} failed`, time: "", tone: "failed", ticks: [], view: "cv" };
  }
  return { text: finishedText(job), time, tone: "finished", ticks, view };
}

/**
 * The job the pill follows: a running one first, Run CV before Final fit, else
 * the one that started last.
 * @param {Record<string, JobStatus|null>} jobs
 * @returns {JobStatus|null}
 */
export function pillJob(jobs) {
  const known = Object.values(jobs).filter((job) => job !== null);
  const running = known.filter((job) => job.status === "running");
  const pool = running.length ? running : known;
  return pool.reduce((best, job) => {
    if (!best) return job;
    return (job.started_at ?? 0) > (best.started_at ?? 0) ? job : best;
  }, /** @type {JobStatus|null} */ (null));
}

/**
 * Draw ``pill`` from ``state``, touching the page only where something changed,
 * so a poll that changes nothing costs nothing.
 * @param {HTMLElement} pill @param {PillState|null} state
 */
export function renderJobPill(pill, state) {
  pill.hidden = state === null;
  if (state === null) return;
  pill.dataset.tone = state.tone;
  pill.dataset.view = state.view;
  const signature = state.ticks.map((tick) => tick.fill).join();
  if (pill.dataset.ticks !== signature) {
    pill.dataset.ticks = signature;
    const ticks = /** @type {HTMLElement} */ (pill.querySelector("[data-pill-ticks]"));
    ticks.innerHTML = state.ticks
      .map((tick) => `<i data-fill="${tick.fill}"${tick.fill === "done" ? ` style="background: ${tick.colour}"` : ""}></i>`)
      .join("");
  }
  const text = /** @type {HTMLElement} */ (pill.querySelector("[data-pill-text]"));
  const time = /** @type {HTMLElement} */ (pill.querySelector("[data-pill-time]"));
  if (text.textContent !== state.text) text.textContent = state.text;
  if (time.textContent !== state.time) time.textContent = state.time;
}
