# How REML Chooses Smoothness: Page and Landing Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An executed explanation page that shows, in pictures, what the smoothing penalty does and how REML picks it, and a landing-page figure glued from that page so it can never go stale.

**Architecture:** One MyST Markdown notebook under `docs/explanation/` fits a spline to simulated data with a known truth at fixed penalties and by REML, draws four figures with a shared matplotlib style, and glues the two-panel "no penalty versus REML" figure under a key that `docs/index.md` pastes with MyST-NB's cross-document glue. The existing notebook test executes the page in CI.

**Tech Stack:** superglm public API only (`SuperGLM`, `Spline`, `fit`, `fit_reml`, `term_inference`, `metrics`, `reml_diagnostics`, `cross_validate`), matplotlib, MyST-NB glue, the dataviz skill for the figures.

**Spec:** `docs/superpowers/specs/2026-09-13-docs-rebuild-design.md` (§7 executable docs, §9 look and feel, landing bullet)

## Global Constraints

- Worktree `/home/max/projects/superglm/.claude/worktrees/docs-rebuild-sphinx`, branch `docs/reml-smoothness`, stacked on `worktree-docs-rebuild-sphinx` (PR #392). One simple shell command per Bash call; no `cd`, no heredocs, no bare `git stash`.
- Executed pages use the public API only; `tests/docs/test_conventions.py` enforces it. Every executed page is self-contained: its own imports and data, no shared helper module.
- Strict build: `SUPERGLM_DOCS_EXECUTE=force uv run sphinx-build -b html -n -W --keep-going -d docs/_build/doctrees docs docs/_build/html`, zero warnings.
- Never run the full test suite; `uv run pytest tests/docs -q` is the test command.
- The page must execute in under 60 seconds on a laptop (measure with `time`).
- The message, in Max's words: "here is what happens if lambda is zero, super wiggly; else it is reasonable." Every figure serves that sentence. No derivations on this page; link to the solvers essay for them.
- Figures: opaque white background, the site palette (ink `#15171C` for the fitted curve, red `#D6402B` for the second series, yellow `#F4B942` for fills and histograms, muted grey `#6B7280` for the truth and axes), light grid, no chart junk. Load the dataviz skill before writing any plotting code and follow it.
- Commit with the attribution trailer lines from the session's system reminder.

---

### Task 1: Shared matplotlib style

**Files:**
- Create: `docs/_static/superglm.mplstyle`

**Interfaces:**
- Produces: a style sheet every executed page applies with a guarded `plt.style.use` (guarded because the file does not exist in Colab).

- [ ] **Step 1: Write the style sheet**

```
# superglm documentation figure style. Palette from spec §9.
figure.facecolor: white
figure.dpi: 130
figure.figsize: 8.0, 4.0
savefig.dpi: 160
savefig.bbox: tight
axes.facecolor: white
axes.edgecolor: 6B7280
axes.linewidth: 0.8
axes.grid: True
axes.grid.axis: y
grid.color: E5E3DC
grid.linewidth: 0.6
axes.spines.top: False
axes.spines.right: False
axes.titlesize: 12
axes.titleweight: 600
axes.titlelocation: left
axes.labelsize: 10
axes.labelcolor: 3A3F4A
axes.prop_cycle: cycler('color', ['15171C', 'D6402B', 'F4B942', '6B7280', '2E5AAC'])
xtick.color: 6B7280
ytick.color: 6B7280
xtick.labelsize: 9
ytick.labelsize: 9
legend.frameon: False
legend.fontsize: 9
lines.linewidth: 2.0
font.family: sans-serif
font.sans-serif: Source Sans 3, DejaVu Sans, Arial, sans-serif
```

- [ ] **Step 2: Check matplotlib parses it**

Run: `uv run python -c "import matplotlib.pyplot as plt; plt.style.use('docs/_static/superglm.mplstyle'); print('ok')"`
Expected: `ok`. A `KeyError` names a misspelt rc key; fix the key.

- [ ] **Step 3: Commit**

```bash
git add docs/_static/superglm.mplstyle
git commit -m "Docs: one matplotlib style for every figure on the site"
```

### Task 2: The explanation page

**Files:**
- Create: `docs/explanation/how-reml-chooses-smoothness.md`
- Modify: `docs/explanation/index.md` (add the page first in the toctree)

**Interfaces:**
- Produces: glue keys `lambda-zero-vs-reml` (two-panel figure for the landing page), `lambda-triptych`, `edf-and-holdout`, `reml-path`, `select-shrinkage`; all consumed only by this page except the first, which Task 3 pastes onto `index.md`.

- [ ] **Step 1: Write the page skeleton with front matter and setup**

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

# How REML chooses smoothness

A spline can follow the data as closely as you let it. The smoothing penalty
decides how closely, and REML decides the penalty. This page shows what that
means on data where the true curve is known, so every fit can be judged
against it.

```{code-cell} ipython3
:tags: [remove-cell]

import warnings
from pathlib import Path

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
for candidate in (Path("../_static/superglm.mplstyle"), Path("docs/_static/superglm.mplstyle")):
    if candidate.exists():
        plt.style.use(str(candidate))
        break
```

```{code-cell} ipython3
import numpy as np
import pandas as pd
from myst_nb import glue

from superglm import Spline, SuperGLM, cross_validate

rng = np.random.default_rng(20260913)
n = 400
x = np.sort(rng.uniform(0.0, 1.0, n))
truth = np.sin(2 * np.pi * x) + 0.6 * x
y = truth + rng.normal(0.0, 0.35, n)
X = pd.DataFrame({"x": x})
grid = pd.DataFrame({"x": np.linspace(0.0, 1.0, 300)})
```
````

`from myst_nb import glue` is the documented import for gluing; it is available in the docs environment and in the notebook test because `myst-nb` is in the `docs` group, which the `docs-notebooks` CI job installs. In Colab it is not installed: wrap the import as

```python
try:
    from myst_nb import glue
except ImportError:  # running outside the docs build, e.g. Colab
    def glue(name, obj, display=True):
        return obj
```

- [ ] **Step 2: Figure 1, no penalty versus REML, plus the triptych**

Fit three models on the same data: `fit(spline_penalty=0.0)`, `fit_reml()`, and `fit(spline_penalty=1e4)`, all with `Spline(kind="ps", k=30)`. Draw the fitted curve on `grid` from `predict`, the truth dashed in muted grey, the data as small light points, and the per-fit EDF in the panel title from `term_inference("x").edf`. Two figures: `fig_pair` with the first two panels (landing), `fig_three` with all three (this page).

```python
def fit_at(lam):
    model = SuperGLM(family="gaussian", selection_penalty=0.0, spline_penalty=lam,
                     features={"x": Spline(kind="ps", k=30)})
    return model.fit(X, y)

reml = SuperGLM(family="gaussian", selection_penalty=0.0,
                features={"x": Spline(kind="ps", k=30)}).fit_reml(X, y)
fits = [("No penalty, lambda = 0", fit_at(0.0)), ("REML", reml), ("Lambda = 10,000", fit_at(1e4))]

def draw(ax, title, model):
    ax.scatter(x, y, s=6, color="#C9CCD3", zorder=1)
    ax.plot(grid["x"], truth_on(grid), color="#6B7280", linestyle="--", linewidth=1.4, label="truth", zorder=2)
    ax.plot(grid["x"], model.predict(grid), color="#15171C", linewidth=2.2, label="fit", zorder=3)
    edf = model.term_inference("x").edf
    ax.set_title(f"{title}   EDF {edf:.1f}")
    ax.set_xlabel("x")

def truth_on(frame):
    return np.sin(2 * np.pi * frame["x"].to_numpy()) + 0.6 * frame["x"].to_numpy()

fig_pair, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, (title, model) in zip(axes, fits[:2]):
    draw(ax, title, model)
axes[0].set_ylabel("y")
axes[0].legend(loc="upper right")
glue("lambda-zero-vs-reml", fig_pair, display=False)

fig_three, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, (title, model) in zip(axes, fits):
    draw(ax, title, model)
axes[0].set_ylabel("y")
glue("lambda-triptych", fig_three, display=False)
```

Then paste `fig_three` in the page with

````markdown
```{glue:figure} lambda-triptych
:name: fig-lambda-triptych

The same 400 points three times. With no penalty the spline chases every point. REML picks a penalty that follows the truth. A penalty far too large flattens the curve to a line.
```
````

If the no-penalty fit is not visibly wiggly at `k=30`, raise `k` to 40 and the noise to 0.45 until it is; record the values you settled on in the prose.

- [ ] **Step 3: Figure 2, EDF and held-out deviance across lambda**

A log grid of twelve fixed penalties from `1e-4` to `1e4`. For each: EDF from `term_inference("x").edf` and mean held-out deviance from `cross_validate(model, X, y, cv=KFold(5, shuffle=True, random_state=1), scoring=("deviance",))`, reading `cv.mean_scores["deviance"]`. Two panels sharing the x axis (log lambda): EDF, and held-out deviance; a vertical red line at the REML lambda read from `reml.reml_diagnostics()["lambdas"]["x"]`. Glue as `edf-and-holdout`. Caption: "What the penalty buys. Left: how many effective parameters the spline keeps. Right: held-out deviance. The red line is the lambda REML chose without ever seeing a holdout."

- [ ] **Step 4: Figure 3, the optimiser's path**

From `reml.reml_diagnostics()`: `lambda_history` (list of dicts keyed by term) and `objective_history`. Two small panels: lambda per iteration on a log axis, and the objective per iteration; the final point in red. Glue as `reml-path`. Caption: "REML is a maximisation, not a search: seven steps from the starting value to the optimum on this example."

- [ ] **Step 5: Figure 4, `select=True`**

Add a second column `z = rng.uniform(0, 1, n)` that has no effect on `y`. Fit `{"x": Spline(kind="ps", k=12, select=sel), "z": Spline(kind="ps", k=12, select=sel)}` with `fit_reml` for `sel` in `(False, True)`. A 2×2 grid: rows are `select=False` and `select=True`, columns are the `x` term and the `z` term, each drawn from `term_inference(name)` (`x` values and `log_relativity` or `smooth_curve`, whichever the object exposes on the response scale for a Gaussian identity fit; read the field names with `dir`). Per-panel title carries the EDF. Glue as `select-shrinkage`. Caption: "A term with no signal. Without the double penalty it keeps a wobble and a couple of degrees of freedom; with `select=True` REML shrinks it to flat."

- [ ] **Step 6: Prose, takeaways, next steps**

Between the figures, at most three short paragraphs each, in the spec's voice: no derivations, terms defined on first use (penalty, lambda, EDF, REML in one sentence each, linked to the glossary once it exists, otherwise to `solvers-and-internals`). End with:

```markdown
## Main takeaways

- No penalty means the spline reproduces the noise; the effective degrees of freedom climb towards the basis size.
- REML picks the penalty from the data alone, and on this example it lands where held-out deviance is lowest.
- The optimiser reaches that value in a handful of steps; it is a maximisation, not a grid search.
- `select=True` lets REML remove a term that carries no signal instead of leaving it a little wiggly.

## Next steps

- [Choose a fitting path](../how-to/choose-a-fitting-path.md)
- [Solvers and internals](solvers-and-internals.md) for the REML criterion itself
- [`SuperGLM.fit_reml`](../api/generated/superglm.SuperGLM.fit_reml.rst)
```

Use the cross-reference form the build accepts for the API link (a MyST `{py:meth}` role, `` {py:meth}`superglm.SuperGLM.fit_reml` ``, is the reliable one).

- [ ] **Step 7: Add to the explanation index**

`docs/explanation/index.md` toctree gains `how-reml-chooses-smoothness` as the first entry.

- [ ] **Step 8: Execute and time**

Run: `time uv run pytest tests/docs/test_notebooks.py -m docs -q -k how-reml`
Expected: PASS in under 60 seconds. If the cross-validation grid is the cost, drop the grid to eight lambdas.

- [ ] **Step 9: Conventions test**

Run: `uv run pytest tests/docs/test_conventions.py -q`
Expected: PASS (no private attribute access).

- [ ] **Step 10: Commit**

```bash
git add docs/explanation/how-reml-chooses-smoothness.md docs/explanation/index.md
git commit -m "Docs: how REML chooses smoothness, four executed figures"
```

### Task 3: Glue the pair onto the landing page

**Files:**
- Modify: `docs/index.md` (after the twelve-line example, before the closing paragraph)

- [ ] **Step 1: Paste the figure**

Insert after the code block of "Twelve lines to a fitted model":

````markdown
```{rst-class} sg-section-title
```

## What REML does

```{glue:figure} lambda-zero-vs-reml
:doc: explanation/how-reml-chooses-smoothness
:name: fig-landing-lambda

The same data twice. Left, no penalty: the spline chases every point. Right, the penalty REML chose. [How it chooses](explanation/how-reml-chooses-smoothness.md).
```
````

MyST-NB's cross-document glue takes the source document path relative to the docs root in the `:doc:` option. If the build reports the key as unknown, read <https://myst-nb.readthedocs.io/en/latest/render/glue.html#pasting-from-other-documents> for the accepted spelling (the option may require the `.md` suffix) and adjust.

- [ ] **Step 2: Strict executed build**

Run: `rm -rf docs/_build docs/api/generated` then `SUPERGLM_DOCS_EXECUTE=force uv run sphinx-build -b html -n -W --keep-going -d docs/_build/doctrees docs docs/_build/html`
Expected: `build succeeded.`, zero warnings; `docs/_build/html/index.html` contains an `<img` inside a figure with the caption above.

- [ ] **Step 3: Look once**

Screenshot `index.html` and `explanation/how-reml-chooses-smoothness.html` at 1440 wide in light and dark with Playwright into the scratchpad `reml-page/` directory. Fix only what is broken (a figure not rendering, a caption missing).

- [ ] **Step 4: Commit and open the stacked pull request**

```bash
git add docs/index.md
git commit -m "Docs: landing figure glued from the smoothness page"
git push -u origin docs/reml-smoothness
```

Open a PR with base `worktree-docs-rebuild-sphinx`, title `Docs: how REML chooses smoothness, with the landing figure glued from it`, body: what the page shows (four figures), the glue mechanism and why (never stale), execution time, and the spec bullet it closes. End with the attribution lines.

---

## Self-review

- Spec §7: executed page, self-contained, public API, tested by the existing notebook test. §9: palette, no committed figure, landing bullet closed. The `select=True` visual and the lambda demonstration Max asked for are Task 2 steps 2 and 5. The optimiser path (step 4) is the one figure not asked for; it is cheap and answers "how does it pick it", keep it unless it pushes execution past 60 seconds.
