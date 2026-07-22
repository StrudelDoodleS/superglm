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
  refreshSummary,
  renderSummary,
  runDistributionProfile,
  runOffsetRefit,
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

function compactSummaryNodes() {
  return {
    summaryStatus: { textContent: "" },
    summaryNote: { textContent: "" },
    summaryFrame: { innerHTML: "", setAttribute: () => {} }
  };
}

/**
 * @param {"expanded"|"grouped"} levelDisplay
 * @param {{hasLevelGroups?:boolean}} [options]
 */
function compactLevelSummary(levelDisplay, { hasLevelGroups = true } = {}) {
  const expanded = levelDisplay === "expanded";
  return {
    available: true,
    label: "Summary",
    level_display: levelDisplay,
    html: `<p>Full ${levelDisplay}</p>`,
    compact: {
      model: {},
      level_display: levelDisplay,
      has_level_groups: hasLevelGroups,
      level_groups: hasLevelGroups
        ? [{
          feature: "territory",
          group_id: "G1",
          members: ["B", "C", '<img src=x onerror="alert(1)">']
        }]
        : [],
      rows: expanded
        ? ["B", "C"].map((member) => ({
          name: `territory[${member}]`,
          group: "territory",
          level_group: "G1",
          kind: "coef",
          coef: 0.2,
          se: 0.1,
          p_value: 0.04,
          sig_code: "*",
          sig_class: "sig-standard"
        }))
        : [{
          name: "territory",
          group: "territory",
          level_group: "G1",
          kind: "coef",
          coef: 0.2,
          se: 0.1,
          p_value: 0.04,
          sig_code: "*",
          sig_class: "sig-standard"
        }]
    }
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
  /** @type {object | null} */
  let firstElementChild = null;
  const summaryFrame = {
    // Browsers serialize parsed HTML rather than returning the exact source
    // string. Character references are one common source of differences.
    get innerHTML() { return markup.replaceAll("&quot;", '"'); },
    set innerHTML(value) {
      writes += 1;
      markup = value;
      firstElementChild = value ? {} : null;
    },
    get firstElementChild() { return firstElementChild; }
  };
  const nodes = {
    summaryStatus: { textContent: "" },
    summaryNote: { textContent: "" },
    summaryFrame
  };
  const payload = { available: false, label: "Summary", error: 'Unavailable "now"' };

  renderSummary(payload, nodes);
  renderSummary(payload, nodes);

  assert.equal(writes, 1);

  markup = "";
  firstElementChild = null;
  renderSummary(payload, nodes);

  assert.equal(writes, 2);
  assert.match(markup, /Unavailable/);
});

test("expanded compact summary shows group indicators without a membership legend", () => {
  const nodes = compactSummaryNodes();

  renderSummary(compactLevelSummary("expanded"), nodes);

  assert.match(nodes.summaryFrame.innerHTML, /Level group/);
  assert.match(nodes.summaryFrame.innerHTML, /territory\[B\]/);
  assert.match(nodes.summaryFrame.innerHTML, /territory\[C\]/);
  assert.match(nodes.summaryFrame.innerHTML, />G1</);
  assert.doesNotMatch(nodes.summaryFrame.innerHTML, /Level groups \(territory\)/);
});

test("reference summary rows retain their emitted significance class", () => {
  const nodes = compactSummaryNodes();
  const payload = compactLevelSummary("expanded", { hasLevelGroups: false });
  payload.compact.rows[0].sig_class = "sig-reference";

  renderSummary(payload, nodes);

  assert.match(nodes.summaryFrame.innerHTML, /summary-row sig-reference/);
  assert.match(nodes.summaryFrame.innerHTML, /summary-se se-cell sig-reference/);
  assert.doesNotMatch(nodes.summaryFrame.innerHTML, /summary-row sig-unknown/);
});

test("grouped compact summary renders one row and escaped membership legend", () => {
  const nodes = compactSummaryNodes();

  renderSummary(compactLevelSummary("grouped"), nodes);

  assert.equal((nodes.summaryFrame.innerHTML.match(/<tr class="summary-row/g) || []).length, 1);
  assert.match(nodes.summaryFrame.innerHTML, /Level groups \(territory\):/);
  assert.match(nodes.summaryFrame.innerHTML, /G1/);
  assert.match(nodes.summaryFrame.innerHTML, /B, C/);
  assert.match(nodes.summaryFrame.innerHTML, /&lt;img src=x onerror=&quot;alert\(1\)&quot;&gt;/);
  assert.doesNotMatch(nodes.summaryFrame.innerHTML, /<img/);
  assert.match(nodes.summaryFrame.innerHTML, /aria-label="Level group membership"/);
});

test("ordinary compact summary remains a six-column table", () => {
  const nodes = compactSummaryNodes();
  const payload = compactLevelSummary("expanded", { hasLevelGroups: false });

  renderSummary(payload, nodes);

  assert.equal((nodes.summaryFrame.innerHTML.match(/<th(?:\s|>)/g) || []).length, 6);
  assert.doesNotMatch(nodes.summaryFrame.innerHTML, /Level group/);
  assert.doesNotMatch(nodes.summaryFrame.innerHTML, /summary-level-group/);
});

test("direct summary helpers include the grouped level display", async () => {
  /** @type {Array<{path:string, body:Record<string, unknown>}>} */
  const calls = [];
  const response = { available: false, label: "Summary", error: "Unavailable" };
  /** @param {string} path @param {{body?:string}} [options] */
  const request = async (path, options = {}) => {
    calls.push({
      path,
      body: options.body ? JSON.parse(options.body) : {}
    });
    if (path === "/profile_distribution/start") {
      return { status: "complete", job_id: "job-grouped", result: response };
    }
    return response;
  };
  const nodes = {
    ...profileTraceNodes(),
    summarySource: { value: "selected" },
    summaryLevelDisplay: "grouped",
    refitOffset: { disabled: false }
  };

  await refreshSummary(nodes, { request });
  await runOffsetRefit(nodes, async () => {}, { request });
  await runDistributionProfile(
    nodes,
    "tweedie_p",
    async () => {},
    { request, pause: async () => {} }
  );

  assert.deepEqual(calls.map(({ path, body }) => [path, body.level_display]), [
    ["/summary", "grouped"],
    ["/refit_offset", "grouped"],
    ["/profile_distribution/start", "grouped"]
  ]);
});

test("offset refit does not render a response for an obsolete level display", async () => {
  let levelDisplay = "expanded";
  const nodes = {
    ...compactSummaryNodes(),
    summarySource: { value: "selected" },
    get summaryLevelDisplay() {
      return levelDisplay;
    },
    refitOffset: { disabled: false }
  };
  /** @param {string} _path @param {{body:string}} options */
  const request = async (_path, options) => {
    assert.equal(JSON.parse(options.body).level_display, "expanded");
    levelDisplay = "grouped";
    nodes.summaryFrame.innerHTML = "<p>Current grouped summary</p>";
    return compactLevelSummary("expanded");
  };

  await runOffsetRefit(nodes, async () => {}, { request });

  assert.equal(nodes.summarySource.value, "refit");
  assert.equal(nodes.summaryFrame.innerHTML, "<p>Current grouped summary</p>");
});

test("direct summary helpers default legacy callers to expanded", async () => {
  /** @type {Array<Record<string, unknown>>} */
  const calls = [];
  const nodes = {
    ...compactSummaryNodes(),
    summarySource: { value: "selected" }
  };
  /** @param {string} _path @param {{body:string}} options */
  const request = async (_path, options) => {
    calls.push(JSON.parse(options.body));
    return { available: false, label: "Summary", error: "Unavailable" };
  };

  await refreshSummary(nodes, { request });

  assert.equal(calls[0].level_display, "expanded");
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
