import assert from "node:assert/strict";
import test from "node:test";

import { createEditorActions } from "../../src/superglm/editor/app/state/actions.js";
import {
  createEditorStore,
  createInitialEditorState
} from "../../src/superglm/editor/app/state/store.js";

const summaryModulePath = "../../src/superglm/editor/app/summary.js";
const {
  collapseTransition,
  renderSummary,
  runDistributionProfile,
  uncollapseTransition,
  ungroupTransition
} = await import(summaryModulePath);

/** @param {number} revision */
function snapshot(revision) {
  return {
    model_revision: revision,
    selected_term: "age",
    terms: {
      age: {
        kind: "spline", term_type: "spline", x: [1], y: [1], original_y: [1],
        previous_y: null, levels: null, n_points: 1, controls: null,
        group_display: null, impact: {}
      }
    },
    selection: { age: [0] },
    can_uncollapse_levels: false,
    last_collapse: null,
    history: { active: [], redo: [] }
  };
}

function profileTraceNodes() {
  return {
    summaryStatus: { textContent: "" },
    summaryNote: { textContent: "" },
    summaryFrame: { innerHTML: "", setAttribute: () => {} },
    profileProgress: {
      hidden: true,
      classList: { toggle: () => {} }
    },
    profileTraceStatus: { textContent: "" },
    profileTraceLegend: { innerHTML: "" },
    profileTracePlot: { innerHTML: "" },
    profileTraceTable: { innerHTML: "" }
  };
}

/** @param {string} tweedieMethod */
function compactTweedieSummaryMarkup(tweedieMethod) {
  const nodes = {
    summaryStatus: { textContent: "" },
    summaryNote: { textContent: "" },
    summaryFrame: { innerHTML: "" }
  };
  renderSummary({
    available: true,
    label: "Summary",
    html: "",
    compact: {
      model: {
        family: "Tweedie",
        link: "Log",
        method: "PIRLS",
        tweedie_p: 1.55,
        tweedie_p_ci: null,
        tweedie_p_ci_status: "not computed",
        tweedie_p_method: tweedieMethod
      },
      rows: []
    }
  }, nodes);
  return nodes.summaryFrame.innerHTML;
}

/** @param {string | undefined} ciStatus @param {string} [parameter] */
async function completedProfileLegend(ciStatus, parameter = "tweedie_p") {
  const nodes = profileTraceNodes();
  const result = { available: false, label: "Profiled model", error: "No compact summary" };
  const isNb2 = parameter === "nb2_theta";
  let requestCount = 0;
  const request = async () => {
    requestCount += 1;
    if (requestCount === 1) {
      return { status: "running", phase: "profile_ci", job_id: "job-ci", trace: [] };
    }
    return {
      status: "complete",
      parameter,
      trace: [{ [isNb2 ? "theta" : "p"]: 1.55, nll: 0.1 }],
      profile_estimate: {
        parameter: isNb2 ? "theta" : "p",
        label: isNb2 ? "theta_hat" : "p_hat",
        value: 1.55,
        ci_low: null,
        ci_high: null,
        ci_status: ciStatus
      },
      result
    };
  };

  await runDistributionProfile(
    nodes,
    parameter,
    async () => {},
    { request, pause: async () => {} }
  );
  return nodes;
}

test("structural transition descriptors are pure route descriptions", () => {
  assert.deepEqual(collapseTransition("region"), {
    name: "collapse levels",
    path: "/collapse_levels",
    payload: { term: "region", method: "auto" }
  });
  assert.deepEqual(ungroupTransition("region"), {
    name: "ungroup levels",
    path: "/ungroup_levels",
    payload: { term: "region", method: "auto" }
  });
  assert.deepEqual(uncollapseTransition(), {
    name: "restore collapsed levels",
    path: "/uncollapse_levels",
    payload: {}
  });
});

test("transition descriptor payloads are independent caller-owned values", () => {
  const first = collapseTransition("region");
  first.payload.term = "mutated";

  assert.deepEqual(collapseTransition("region").payload, {
    term: "region",
    method: "auto"
  });
});

test("rendering unchanged summary markup preserves the existing table DOM", () => {
  let writes = 0;
  let markup = "";
  const summaryFrame = {
    get innerHTML() { return markup; },
    set innerHTML(value) {
      writes += 1;
      markup = value;
    }
  };
  const nodes = {
    summaryStatus: { textContent: "" },
    summaryNote: { textContent: "" },
    summaryFrame
  };
  const payload = { available: false, label: "Summary", error: "Unavailable" };

  renderSummary(payload, nodes);
  renderSummary(payload, nodes);

  assert.equal(writes, 1);

  markup = "";
  renderSummary(payload, nodes);

  assert.equal(writes, 2);
  assert.match(markup, /Unavailable/);
});

for (const method of [
  "Profile MLE (Brent)",
  "Approximate profile (Brent; Pearson plug-in)",
  "Profile MLE (Brent; density approximation)"
]) {
  test(`compact Tweedie summary renders the profile method: ${method}`, () => {
    const markup = compactTweedieSummaryMarkup(method);

    assert.match(markup, /Tweedie p method/);
    assert.ok(markup.includes(method));
  });
}

test("profile completion is accepted before the caller schedules new-revision evidence", async () => {
  /** @type {string[]} */
  const events = [];
  const result = { available: false, label: "Profiled model", error: "No compact summary" };
  const nodes = {
    summaryStatus: { textContent: "" },
    summaryNote: { textContent: "" },
    summaryFrame: {
      innerHTML: "",
      /** @param {string} name @param {string} value */
      setAttribute: (name, value) => events.push(`frame:${name}:${value}`)
    }
  };
  /** @param {string} path */
  const request = async (path) => {
    events.push(`request:${path}`);
    if (path.endsWith("/start")) return { status: "running", job_id: "job-7" };
    return { status: "complete", result };
  };

  const payload = await runDistributionProfile(
    nodes,
    "tweedie_p",
    async (/** @type {typeof result} */ accepted) => {
      events.push(`accepted:${accepted.label}`);
    },
    { request, pause: async () => {} }
  );

  assert.strictEqual(payload, result);
  assert.deepEqual(events.filter((event) => event.startsWith("request:")), [
    "request:/profile_distribution/start",
    "request:/profile_distribution/status/job-7"
  ]);
  assert.ok(events.indexOf("accepted:Profiled model") > events.indexOf(
    "request:/profile_distribution/status/job-7"
  ));
  assert.equal(events.some((event) => event.includes("/metrics")), false);
});

test("uncached Tweedie MLE profile reports that its CI was not computed", async () => {
  const nodes = await completedProfileLegend("not computed");

  assert.match(nodes.profileTraceLegend.innerHTML, /CI not computed/);
  assert.doesNotMatch(nodes.profileTraceLegend.innerHTML, /CI pending/);
  assert.doesNotMatch(nodes.profileTraceStatus.textContent, /profile CI/i);
});

test("Pearson plug-in profile reports that a likelihood-ratio CI is unavailable", async () => {
  const nodes = await completedProfileLegend("unavailable for Pearson plug-in");

  assert.match(nodes.profileTraceLegend.innerHTML, /CI unavailable/);
  assert.doesNotMatch(nodes.profileTraceLegend.innerHTML, /CI pending/);
});

test("NB2 profile keeps its pending CI wording until its interval is available", async () => {
  const nodes = await completedProfileLegend(undefined, "nb2_theta");

  assert.match(nodes.profileTraceLegend.innerHTML, /CI pending/);
  assert.doesNotMatch(nodes.profileTraceLegend.innerHTML, /CI not computed/);
});

test("state-only recovery publishes a stale summary payload when remote summary is null", async () => {
  const nodes = {
    summaryStatus: { textContent: "Old manual summary" },
    summaryNote: { textContent: "old note" },
    summaryFrame: { innerHTML: "<p>old manual summary html</p>" }
  };
  const store = createEditorStore(createInitialEditorState(snapshot(2)));
  let summaryRenders = 0;
  store.subscribe((state) => state.remote.summary, (summary) => {
    summaryRenders += 1;
    assert.ok(summary);
    renderSummary(summary, nodes);
  });
  const actions = createEditorActions({
    store,
    client: {
      postJSON: async () => { throw new Error("response lost"); },
      getState: async () => snapshot(3)
    }
  });

  const result = await actions.executeStateMutation({ name: "drag", path: "/drag", payload: {} });

  assert.equal(result.ok, false);
  assert.equal(store.getState().remote.snapshot?.model_revision, 3);
  assert.equal(store.getState().remote.summary?.available, false);
  assert.doesNotThrow(() => JSON.stringify(store.getState().remote.summary));
  assert.equal(summaryRenders, 1);
  assert.equal(nodes.summaryStatus.textContent, "Summary unavailable");
  assert.match(nodes.summaryFrame.innerHTML, /reconciled/i);
  assert.match(nodes.summaryFrame.innerHTML, /refresh/i);
  assert.doesNotMatch(nodes.summaryFrame.innerHTML, /old manual summary html/);
});
