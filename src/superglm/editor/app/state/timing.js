// @ts-check

/** @param {number} end @param {number} start */
function elapsed(end, start) {
  const duration = Number(end) - Number(start);
  return Number.isFinite(duration) ? Math.max(0, duration) : 0;
}

/**
 * Keep browser phases separate from server timings in one diagnostic payload.
 *
 * @param {{timing?:Record<string, unknown>}} envelope
 * @param {{operationStart:number, requestStart:number, requestEnd:number,
 *   commitEnd:number, paintEnd:number}} milestones
 */
export function clientTransitionTiming(envelope, milestones) {
  return {
    ...(envelope?.timing || {}),
    client_request_ms: elapsed(milestones.requestEnd, milestones.requestStart),
    client_commit_ms: elapsed(milestones.commitEnd, milestones.requestEnd),
    client_paint_ms: elapsed(milestones.paintEnd, milestones.commitEnd),
    client_primary_ms: elapsed(milestones.paintEnd, milestones.requestEnd),
    client_total_ms: elapsed(milestones.paintEnd, milestones.operationStart)
  };
}

/**
 * Track the latest completed request duration independently for each evidence panel.
 *
 * @param {{now?:()=>number, onComplete?:(panel:string, duration:number)=>void}} [options]
 */
export function createEvidenceTimingTracker({
  now = () => performance.now(),
  onComplete = () => {}
} = {}) {
  /** @type {Map<string, {sequence:number, startedAt:number}>} */
  const starts = new Map();
  /** @type {Map<string, number>} */
  const completed = new Map();

  return {
    /**
     * @param {string} panel
     * @param {{status:string, sequence:number}} next
     * @param {{status:string, sequence:number}} previous
     */
    observe(panel, next, previous) {
      if (
        next.status === "updating" &&
        (previous.status !== "updating" || next.sequence !== previous.sequence)
      ) {
        starts.set(panel, { sequence: next.sequence, startedAt: now() });
        return;
      }
      const start = starts.get(panel);
      const terminal = next.status === "current" || next.status === "stale" ||
        next.status === "error";
      if (!terminal || start === undefined || start.sequence !== next.sequence) return;
      const duration = elapsed(now(), start.startedAt);
      starts.delete(panel);
      completed.set(panel, duration);
      onComplete(panel, duration);
    },

    /** @returns {Record<string, number>} */
    durations() {
      return Object.fromEntries(completed);
    }
  };
}
