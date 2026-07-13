import assert from "node:assert/strict";
import test from "node:test";

import {
  clientTransitionTiming,
  createEvidenceTimingTracker
} from "../../src/superglm/editor/app/state/timing.js";

test("client transition timing keeps request commit and paint phases separate", () => {
  const timing = clientTransitionTiming(
    { timing: { operation: "collapse_levels", fit_ms: 20, server_total_ms: 30 } },
    {
      operationStart: 10,
      requestStart: 12,
      requestEnd: 52,
      commitEnd: 60,
      paintEnd: 76
    }
  );

  assert.deepEqual(timing, {
    operation: "collapse_levels",
    fit_ms: 20,
    server_total_ms: 30,
    client_request_ms: 40,
    client_commit_ms: 8,
    client_paint_ms: 16,
    client_primary_ms: 24,
    client_total_ms: 66
  });
  assert.equal("client_recovery_ms" in timing, false);
});

test("evidence timing follows the active sequence and reports panel completion", () => {
  let clock = 100;
  /** @type {Array<[string, number]>} */
  const completed = [];
  const tracker = createEvidenceTimingTracker({
    now: () => clock,
    onComplete: (panel, duration) => completed.push([panel, duration])
  });
  const idle = { status: "idle", sequence: 0 };
  const metricsOne = { status: "updating", sequence: 1 };

  tracker.observe("metrics", metricsOne, idle);
  clock = 145;
  tracker.observe("metrics", { status: "current", sequence: 1 }, metricsOne);

  clock = 200;
  const summaryTwo = { status: "updating", sequence: 2 };
  tracker.observe("summary", summaryTwo, idle);
  clock = 210;
  const summaryThree = { status: "updating", sequence: 3 };
  tracker.observe("summary", summaryThree, summaryTwo);
  clock = 250;
  tracker.observe("summary", { status: "stale", sequence: 3 }, summaryThree);
  clock = 270;
  tracker.observe("summary", { status: "current", sequence: 2 }, summaryThree);

  assert.deepEqual(completed, [["metrics", 45], ["summary", 40]]);
  assert.deepEqual(tracker.durations(), { metrics: 45, summary: 40 });
});
