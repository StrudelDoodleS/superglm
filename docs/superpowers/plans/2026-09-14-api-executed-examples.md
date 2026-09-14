# Executed Examples on the Reference Group Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each reference group page that matters to a pricing actuary ends with a short, executed example: real calls on a small simulated book, with the returned tables and figures rendered on the page. Inference first.

**Architecture:** The group pages under `docs/api/model/` and `docs/api/distributional/` become MyST notebooks (jupytext front matter, `{code-cell}` blocks) executed by MyST-NB at build time and by `tests/docs/test_notebooks.py` on a fresh kernel. Each page is self-contained: one visible setup cell simulates the data and fits the model, then three to six cells call the members the page's paragraph names and show what comes back. No shared helper module; no data download.

**Tech Stack:** MyST-NB (`nb_execution_mode` cache/force), jupytext front matter, matplotlib with `docs/_static/superglm.mplstyle`, pandas HTML repr.

**Spec:** `docs/superpowers/specs/2026-09-13-docs-rebuild-design.md` §7 (executable docs) and §8. Max, 2026-09-14: "some examples of method calls that actually do something would be nice, like risk curves, density fan, parameter spread or the terribly named 'portfolio'", then "so inference function especially".

## Global Constraints

- Branch `docs/api-examples`, stacked on `docs/api-internals` (#396). One simple shell command per Bash call; no `cd`, no heredocs, no bare `git stash`, never `git switch`/`checkout` (the branch is set).
- Public API only in code cells: `tests/docs/test_conventions.py` fails any `._` attribute access. Inspect result objects with `dataclasses.fields(...)`, `vars(...)` or their public attributes; never reach into private modules.
- Every cell runs clean on a fresh kernel: no warnings on stderr (silence known-benign ones in the hidden style cell only if they are matplotlib font noise; a model warning is a sign to change the data or the call, not to hide it), no exceptions, fixed seeds, runtime under 40 s per page.
- Outputs are trimmed: `.head()`, `.round(3)`, at most two figures per page, at most six code cells in an example. The example section is titled `## Example` and sits after the autosummary table.
- Prose in the example cells' surrounding markdown states what the output shows; every claim about a number must be true of the rendered output (check after execution, not before).
- Page front matter and the hidden style cell are exactly:

````markdown
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# <existing title>

```{code-cell} ipython3
:tags: [remove-cell]

import logging
from pathlib import Path

import matplotlib.pyplot as plt

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
for candidate in (
    Path("../../_static/superglm.mplstyle"),
    Path("docs/_static/superglm.mplstyle"),
):
    if candidate.exists():
        plt.style.use(str(candidate))
        break
else:
    raise FileNotFoundError("superglm.mplstyle: run this page from its own directory")
```
````

  The existing `# Title`, paragraph and `{eval-rst}` autosummary block stay exactly as they are; the front matter goes above the title and the style cell directly under it.
- Verification per page: `uv run pytest tests/docs/test_notebooks.py -m docs -k "<page stem>" -q` (executes on a fresh kernel) and, once per task, `SUPERGLM_DOCS_EXECUTE=off uv run sphinx-build -b html -n -W --keep-going -d <own tmp dir>/doctrees docs <own tmp dir>/html` at zero warnings. Two writers work at once, so each uses its own output directory under the scratchpad; `docs/api/generated` is shared and identical.
- Commit messages carry the attribution trailer lines from the session's system reminder. No push.

---

### Task 1: SuperGLM pages (writer A)

**Files:** `docs/api/model/inference.md` (first), `docs/api/model/fit.md`, `docs/api/model/predict.md`, `docs/api/model/deploy.md`, `docs/api/model/plot.md`.

- [ ] **Data and fit, repeated per page as a visible setup cell** (10 to 15 lines): a simulated motor book, `rng = np.random.default_rng(0)`, n = 4000 rows with `age` (18 to 80), `region` (four levels with different base rates), `veh_power` (numeric), `exposure` in (0.1, 1], Poisson claim counts whose log rate is a smooth non-linear function of age plus the region effect plus a mild linear power effect, times exposure. Fit `SuperGLM(family="poisson", features={"age": Spline(kind="cr", k=10), "region": Categorical(), "veh_power": Numeric()})` (adapt names to the real constructor: check `inspect.signature(SuperGLM.__init__)` and the feature specs' signatures before writing) with `fit_reml(X, y, offset=np.log(exposure))` or the documented exposure argument. Verify the fit converges cleanly before writing any prose.
- [ ] **inference.md example**: `model.summary()` (show the printed summary); `ti = model.term_inference("age")` then a small DataFrame from its public fields (`x` grid, `relativity`, `ci_lower`, `ci_upper`) `.head()`, plus `ti.edf`; `model.simultaneous_bands("age").head()`; `model.drop1(X, y)`; `model.term_importance(X)`; `model.metrics(X, y)` shown as a short table of its public fields. One figure: the age curve with pointwise and simultaneous bands drawn from those frames (matplotlib), so the reader sees what the bands mean.
- [ ] **fit.md example**: `fit_reml` then `model.reml_diagnostics()` reduced to the estimated lambdas and iteration count; one `fit(spline_penalty=<value>)` on a clone for contrast, comparing `term_inference("age").edf` between the two.
- [ ] **predict.md example**: `model.predict(X_new)` on three hand-written rows, `model.relativities()["region"]`, `model.reconstruct_feature("age")` head.
- [ ] **deploy.md example**: `model.export_rating_tables(<temp dir>)` (use `tempfile.mkdtemp()`), list the files written and show the head of one table read back with pandas; or `rating_table_payload()` if the export needs arguments the simulated model cannot satisfy. Say which.
- [ ] **plot.md example**: `model.plot("age", engine="matplotlib")` as the one figure; mention `plot_data` returns the numbers.
- [ ] Run the per-page notebook test after each page and the strict build once at the end; commit per page: `Docs: executed example on the SuperGLM <group> page`.

### Task 2: SuperLSS pages (writer B)

**Files:** `docs/api/distributional/inference.md` (first), `docs/api/distributional/price.md`, `docs/api/distributional/predict.md`, `docs/api/distributional/check-the-fit.md`.

- [ ] **Data and fit, repeated per page**: a simulated severity book, `rng = np.random.default_rng(1)`, n = 3000 rows with `age` and `region`; claim amounts log-normal whose location depends smoothly on age and on region and whose scale depends on age, so both parameters have real structure. Fit `SuperLSS(family.location(s("age", kind="cr", k=8), cat("region")), family.scale(s("age", kind="cr", k=6)))` with `family = LogNormalLS()` or `GaussianLS()` on the log response, following `docs/tutorials/distributional-model.md` for the exact declaration syntax, then `fit_reml(X, y)`. Verify it converges and that `smoothing_certified_` is not `False` before writing prose.
- [ ] **inference.md example**: `model.summary()` (DataFrame head); `model.term_inference("location", "age")` shown as its public fields in a small table plus one figure of the effect with its band; `model.term_test("scale", "age")` and what its statistic and p-value say.
- [ ] **price.md example**: a `reference` row (a typical policy); `model.risk_curves(reference, "age")` drawn as quantile curves with bands; `model.density_fan(reference, "age")` drawn as a fan; `model.parameter_spread(X, threshold=<a large claim amount>)` reduced to the two or three numbers that answer "how far do identically priced rows differ in tail risk"; `model.portfolio(X)` reduced to the total's mean and a 5 to 95 percent interval. Use `dataclasses.fields` on an instance to learn the payload fields; never guess names.
- [ ] **predict.md example**: `model.predict_parameters(X).head()`, `model.predict_quantile(X, 0.9)` alongside `predict` for the first rows, `model.posterior_predictive(...)` only if it runs under ten seconds.
- [ ] **check-the-fit.md example**: `model.check(X, y, "age")` as a table or figure, `model.actual_expected(X, y, "region")`, `model.scores(X, y).mean()`, and `model.compare` against a location-only fit if that fits in the cell budget.
- [ ] Same verification and commit pattern as Task 1: `Docs: executed example on the SuperLSS <group> page`.

### Task 3: Style sheet, test scope, and the tutorial's style cell

- [ ] `docs/_static/superglm.mplstyle` is the byte-identical copy from branch `docs/reml-smoothness` (already in place); `tests/docs/test_notebooks.py` no longer skips `api` (already in place). Both are committed with the first example page.

### Task 4: Verify (critic)

Clean executed strict build (`rm -rf docs/_build docs/api/generated && SUPERGLM_DOCS_EXECUTE=force uv run sphinx-build -b html -n -W --keep-going -d docs/_build/doctrees docs docs/_build/html`) at zero warnings; `uv run pytest tests/docs -q` including the executing notebook tests; `uv run pytest tests/docs/test_conventions.py`; total execution time per page under 40 s (read the cache's timings or time the notebook tests); in the built HTML every example cell has an output, no output block is a warning or traceback, figures are present and styled (white background, the site palette); screenshots at 1440 light of the two Inference pages and the Price page; prose claims about numbers checked against the rendered outputs; the group paragraph and table above the example are unchanged from the branch base; no `._` in any code cell. No push.
