# Narrative Reference Pages for SuperGLM and SuperLSS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `api/model` and `api/distributional` from grouped phone books into guided references: a lifecycle strip at the top, a short paragraph per group that names the members a reader actually reaches for, short names in the tables, and a sidebar that lists reference pages rather than every generated member.

**Architecture:** The pages keep their autosummary tables (they generate the member pages and the coverage test reads them) but each table follows a paragraph of prose with inline cross-references. A sphinx-design grid of five small cards at the top links to the page's own sections. The theme's sidebar depth drops so generated member pages stop flooding the left navigation.

**Tech Stack:** MyST, autosummary (tilde-prefixed entries), sphinx-design grid cards, pydata-sphinx-theme options.

**Spec:** `docs/superpowers/specs/2026-09-13-docs-rebuild-design.md` §8 (API reference) and §9 (look and feel). Max, 2026-09-14, on the grouped page: "still kind of a phone book".

## Global Constraints

- Worktree `/home/max/projects/superglm/.claude/worktrees/docs-rebuild-sphinx`, branch `docs/api-model-narrative` (based on the structural PR head). One simple shell command per Bash call; no `cd`, no heredocs, no bare `git stash`.
- Every claim in the prose is checked against the member's docstring (`uv run python -c "import superglm, inspect; print(inspect.getdoc(superglm.SuperGLM.<name>))"`); never describe a member from its name alone.
- Cross-references in prose use `` {py:meth}`~superglm.SuperGLM.fit_reml` `` and `` {py:attr}`~superglm.SuperGLM.result` `` so they render as short names and fail the build if wrong. `{py:meth}` for methods, `{py:attr}` for properties.
- Strict build: `SUPERGLM_DOCS_EXECUTE=off uv run sphinx-build -b html -n -W --keep-going -d docs/_build/doctrees docs docs/_build/html`, zero warnings. `uv run pytest tests/docs -m "not docs" -q` must pass, including the member-coverage test.
- No raw HTML on these pages (spec portability rule); the lifecycle strip is a sphinx-design grid.
- Commit messages carry the attribution trailer lines from the session's system reminder.

---

### Task 1: Short names and the coverage test

**Files:**
- Modify: `docs/api/model.md`, `docs/api/distributional.md` (every member entry)
- Modify: `tests/docs/test_api_reference.py`

- [ ] **Step 1:** Prefix every `superglm.SuperGLM.<member>` and `superglm.SuperLSS.<member>` line inside the autosummary blocks with `~` so the tables show `fit_reml`, not the dotted path. Leave the top-level class entries and the "Related objects" / "Declarations and families" entries unprefixed (they are different objects and their module path is informative).
- [ ] **Step 2:** In `tests/docs/test_api_reference.py`, both `documented_names()` and `listed_members()` strip a leading `~` before matching: `line = raw.strip().lstrip("~")`.
- [ ] **Step 3:** `uv run pytest tests/docs/test_api_reference.py -q` passes (3 tests). Build strictly. Commit: `Docs: short member names in the grouped reference tables`.

### Task 2: The narrative on the SuperGLM page

**Files:**
- Modify: `docs/api/model.md`

- [ ] **Step 1: Lifecycle strip.** Directly under the opening paragraph and the class entry, before "## Fit":

````markdown
::::{grid} 2 3 5 5
:gutter: 2

:::{grid-item-card} 1 · Construct
:link: #configuration-and-fitted-attributes
:link-type: url
A family, a feature spec, a penalty policy.
:::
:::{grid-item-card} 2 · Fit
:link: #fit
:link-type: url
`fit_reml` for REML; `fit` for a fixed penalty.
:::
:::{grid-item-card} 3 · Read the fit
:link: #read-the-fit
:link-type: url
Summary, per-term curves, diagnostics.
:::
:::{grid-item-card} 4 · Predict
:link: #predict
:link-type: url
Means, relativities, reconstructed effects.
:::
:::{grid-item-card} 5 · Deploy
:link: #export-for-deployment
:link-type: url
Rating tables and the payload behind them.
:::
::::
````

Check the heading anchors MyST generates (`myst_heading_anchors = 3`): the slug of "Configuration and fitted attributes" is `configuration-and-fitted-attributes`; confirm each `#anchor` against the built HTML ids and fix any that differ.

- [ ] **Step 2: A paragraph per group, before its table.** Write each from the docstrings. The shape: two to four sentences; name the members a reader reaches for first, with inline `{py:meth}`/`{py:attr}` references; say when the others matter. The content each paragraph must carry:

  - **Fit.** `fit_reml` is the normal path and estimates the smoothing parameters; `fit` holds them fixed at `spline_penalty` and is also the path for sparse selection; `fit_path` walks a regularisation path; `refit_unpenalised` refits on the active features without the selection penalty; `estimate_p` and `estimate_theta` profile the Tweedie power and the NB2 dispersion and refit; `bind_levels` fixes the categorical level universe before fitting; `clone_unfitted` copies the configuration without the fit.
  - **Predict.** `predict` returns the mean on the response scale; `relativities` gives the per-feature multiplicative tables a rating engine wants; `reconstruct_feature` returns one feature's fitted curve on its original scale.
  - **Read the fit.** Start with `summary`; `term_inference` is the object behind every curve and band; `simultaneous_bands` widens them to hold jointly; `metrics` and `drop1` for fit statistics and drop-one deviance; `term_importance` and `term_drop_diagnostics` rank terms; `random_effects` and `factor_smooth` report credibility terms; `knot_summary`, `design_summary` and the `result` property expose what was actually built.
  - **Plot.** `plot` draws; `plot_data` returns the numbers behind it for your own figures; `plot_diagnostics` is the residual figure with the simulated Q-Q envelope.
  - **Diagnose.** `diagnostics` is the audit dictionary; `spline_redundancy` and `discretization_impact` check basis and binning choices; `iteration_diagnostics`, `reml_diagnostics` and `training_telemetry` are the solver's own records, dependency-free for logging.
  - **Constrain shapes after fitting.** `apply_shape_postfit` repairs monotone and curvature constraints declared with `Constraint.postfit`; `monotonize` is the monotone-only form; `apply_monotone_postfit` is its compatibility alias. Say in one sentence that fit-time constraints (`Constraint.fit`) need none of this.
  - **Screen interactions.** `screen_interactions` ranks candidate pairs with PSST before you add any to the spec; link the how-to.
  - **Export for deployment.** `export_rating_tables` writes the tables; `rating_table_payload` is the renderer-independent object behind them.
  - **Configuration and fitted attributes.** The properties: `family`, `link`, `features`, `penalty`, `lambda2`, `selection_penalty` echo the configuration; `selection_penalty_`, `distribution_`, `theta_` are resolved by the fit (trailing underscore, scikit-learn convention).
  - **Results and records** (rename "Related objects"). `PathResult` from `fit_path`; `REMLResult` behind `fit_reml`; `LambdaPolicy` and `warmup` for controlling and warming the REML search; `ModelSummary`, `ModelMetrics`, `FitDiagnosticReport`, `DiscretizationResult` are what the reading and diagnosing methods return.

  Every sentence that describes behaviour must match the docstring; where a docstring is thinner than the sentence you want, say less rather than invent.

- [ ] **Step 3:** Build strictly, `uv run pytest tests/docs -m "not docs" -q`, screenshot `api/model.html` at 1440 light, read it: the strip renders as five cards in one row, every paragraph's links are live (blue, not plain text), tables show short names. Commit: `Docs: the SuperGLM reference reads as a guide, with the tables as its index`.

### Task 3: The same on the SuperLSS page

**Files:**
- Modify: `docs/api/distributional.md`

- [ ] **Step 1:** Lifecycle strip with five cards: Declare (`family.location(...)` and friends; link `#declare-and-fit`), Fit (`#declare-and-fit`), Predict (`#predict`), Check (`#check-the-fit`), Price (`#price-and-portfolio-views`).
- [ ] **Step 2:** A paragraph per group, from the docstrings: declare and fit (`fit_reml` joint estimation; `fit` fixed; `diagnose` explains stopping; `predictors` and `family` echo the declaration); predict (`predict` mean; `predict_parameters` every parameter; `predict_link` the linear predictors; `predict_cdf` and `predict_quantile`; the three `posterior_*` methods for uncertainty by simulation); read the fit (`summary`, `term_inference`, `term_test`, the fitted attributes); check the fit (`residuals` and `residual_set`; `check` and `check_2d` binned residual moments; `actual_expected`; `calibration`; `scores` proper scoring rules; `compare` against another fit); price and portfolio views (`risk_curves`, `density_fan`, `parameter_spread`, `portfolio`); plot; smoothing certification and telemetry (what `smoothing_certified_` means, when the `smoothing_unresolved_upper_bound_` list is non-empty, `training_telemetry` for audit); save and load (`to_bytes`/`from_bytes`, trusted sources only, as the docstring says); configuration (`discrete`, `n_bins`, `separation`, `weight_semantics`); declarations and families (the helper functions and the nine family classes, one sentence each on what parameter set they carry).
- [ ] **Step 3:** Build, tests, screenshot, commit: `Docs: the SuperLSS reference reads as a guide, with the tables as its index`.

### Task 4: Sidebar depth

**Files:**
- Modify: `docs/conf.py` (`html_theme_options`)

- [ ] **Step 1:** Add `"navigation_depth": 1` to the pydata theme options so the left sidebar lists a section's pages, not the generated member pages beneath them. Build and check three sidebars in the built HTML: `api/model.html` lists the thirteen reference pages and no `superglm.SuperGLM.*` entries; `how-to/index.html` still lists every how-to; `development/index.html` still lists releases, cost-and-timing, the migrations and `internals/index`. If depth 1 hides the migrations, use `"navigation_depth": 2` and instead exclude the generated pages from the sidebar with `html_sidebars = {"index": [], "api/generated/*": ["sidebar-nav-bs.html"]}` only if that keeps them off `api/model.html`'s sidebar; report which setting worked.
- [ ] **Step 2:** Commit: `Docs: the reference sidebar lists pages, not every generated member`.

### Task 5: Verify

The critic: clean strict build, `pytest tests/docs -m "not docs"`, screenshots of `api/model.html`, `api/distributional.html`, one generated method page and `how-to/index.html` in light and dark; checks that every prose cross-reference resolved (no unresolved-reference warnings, links in the HTML for each backticked member name), that each group paragraph's claims match the docstrings for at least three members it spot-checks per page, that the strip's five links land on the right headings, and that the sidebars are as Task 4 says. No push.
