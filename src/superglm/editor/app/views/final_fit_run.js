// @ts-check

/** @typedef {NonNullable<import('../api/contracts.js').EditorSnapshot['final_fit_plan']>} FinalFitPlan */
/** @typedef {import('../api/contracts.js').JobStatus} JobStatus */

export const FINAL_FIT_RUN_HINT =
  "Refit the current structure on train and validation rows and put your hand edits back. "
  + "The test split stays held out. Export offers the result as Final fit model.";

/**
 * What the Final fit panel's own button shows, where there is no
 * Cross-validation tab to run it from. The button follows the same plan as
 * that tab's: disabled with the server's reason when the fit cannot run, and
 * while one runs. A failure to start is told beside it, not in a popup.
 *
 * @param {{final:FinalFitPlan|null|undefined, running?:boolean, error?:string}} state
 *   `final` is the snapshot's `final_fit_plan`, absent until a snapshot arrives
 * @returns {{disabled:boolean, hint:string, note:string}}
 */
export function finalFitRunState({ final, running = false, error = "" }) {
  const blocked = Boolean(final) && !final?.available;
  return {
    disabled: running || blocked,
    hint: blocked && final?.reason ? final.reason : FINAL_FIT_RUN_HINT,
    note: error ? `Final fit could not run. ${sentence(error)} Change the model if needed, then run it again.` : "",
  };
}

/**
 * Why a settled job did not produce a Final fit model, or "" when it did or
 * is still running.
 * @param {Pick<JobStatus, 'status'> & {error?:string|null}} job
 */
export function finalFitFailure(job) {
  if (job.status === "failed") return job.error || "The editor hit an internal error.";
  if (job.status === "cancelled") return "The job was cancelled and nothing was kept.";
  return "";
}

/** @param {string} text */
function sentence(text) {
  const trimmed = text.trim();
  return /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}
