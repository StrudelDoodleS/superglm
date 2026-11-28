// @ts-nocheck

import { readFileSync } from "node:fs";

import { CHECK_REPORTS } from "../../src/superglm/editor/app/api/contracts.js";

/** Each check's /report payload for a calm fit and a wrong one, as Python wrote them. */
export const CHECK_FIXTURE = JSON.parse(readFileSync(new URL("./fixtures/assess_checks.json", import.meta.url)));

/** `evidence` with every check current on its payload in `reports`, the rest as it was. */
export function withCurrentChecks(evidence, reports) {
  const checks = Object.fromEntries(CHECK_REPORTS.map((name) =>
    [name, { ...evidence[name], status: "current", payload: reports[name] }]));
  return { ...evidence, ...checks };
}
