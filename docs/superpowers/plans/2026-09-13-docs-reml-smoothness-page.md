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

---

## Revision 2026-09-13, evening: the page becomes an explanation, the landing loses the section

Max, after seeing the first build: the REML material should not be on the front page. It should be its own page that explains the REML idea, cites the papers, writes out the maths in LaTeX, explains what the terms are, what they do and why they matter in plain words, and then displays the effects; on brand, comic-book theme still. He also asked for an interactive slider instead of two static figures, with a panel beside it showing how EDF and lambda change as REML iterates, and for `selection_penalty=0.0` to disappear from examples because it confuses people.

Tasks 1 to 3 are done on the branch (three commits plus one repair). Tasks 4 to 8 below reshape the result. The final critic's open findings are folded in: the `select-shrinkage` caption must say what the yellow band is, the `reml-path` caption must say what the red marker is, the triptych must render at the same effective size as the other figures.

Verified references (DOIs read from the publishers on 2026-09-13):

- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *Journal of the Royal Statistical Society: Series B*, 73(1), 3–36. https://doi.org/10.1111/j.1467-9868.2010.00749.x
- Wood, S. N., Pya, N., and Säfken, B. (2016). Smoothing parameter and model selection for general smooth models. *Journal of the American Statistical Association*, 111(516), 1548–1563. https://doi.org/10.1080/01621459.2016.1180986 (preprint arXiv:1511.03864; author PDF at https://webhomes.maths.ed.ac.uk/~swood34/gsm.pdf)
- Marra, G., and Wood, S. N. (2011). Practical variable selection for generalized additive models. *Computational Statistics & Data Analysis*, 55(7), 2372–2387. https://doi.org/10.1016/j.csda.2011.02.004
- Reiss, P. T., and Ogden, R. T. (2009). Smoothing parameter selection for a class of semiparametric linear models. *Journal of the Royal Statistical Society: Series B*, 71(2), 505–523. https://doi.org/10.1111/j.1467-9868.2008.00695.x

Two more may be cited only after the implementer confirms the DOI resolves (`curl -s https://api.crossref.org/works/<doi>` returns the matching title): Wood, S. N. (2017), *Generalized Additive Models: An Introduction with R*, 2nd edition, Chapman and Hall/CRC, doi 10.1201/9781315370279; Wahba, G. (1985), A comparison of GCV and GML for choosing the smoothing parameter in the generalized spline smoothing problem, *Annals of Statistics*, 13(4), 1378–1402, doi 10.1214/aos/1176349743.

### Task 4: Restructure the page as an explanation

**Files:**
- Modify: `docs/explanation/how-reml-chooses-smoothness.md`
- Modify: `docs/_static/custom.css` (two callout classes, under 30 lines)

**Interfaces:**
- Produces: the section order and heading texts below, which Task 5 inserts into; the CSS classes `sg-plain` and `sg-pricing` on admonitions.

- [ ] **Step 1: Hide the code**

Every code cell that is not already `remove-cell` gets `:tags: [hide-input]`, so the page reads as an explanation and the code is one click away ("Show code cell source"). The notebook stays executable and downloadable; nothing else changes about execution.

- [ ] **Step 2: Drop `selection_penalty=0.0`**

Remove the argument from every `SuperGLM(...)` call on the page. `None`, the default, already disables selection.

- [ ] **Step 3: Rewrite the top of the page in this order**

Keep the title. Replace everything between the title and the "Data with a known truth" heading with the sections below, and rename "Data with a known truth" to "See it happen" (the data cell and Task 5's widget open that section; the four existing figures follow it).

````markdown
A spline can follow the data as closely as you let it. The smoothing penalty
decides how closely, and REML decides the penalty. This page says what that
means, first in plain words, then in the maths, then in pictures on data
where the true curve is known.

```{admonition} In plain words
:class: tip sg-plain

A spline with forty pieces can copy the noise in the data as easily as the
signal. The penalty charges the fit for wiggliness, and lambda is the price
per unit of wiggle. REML sets the price by asking one question of the data:
which amount of smoothness makes what we observed most probable, once the
curve's own uncertainty has been averaged out? No holdout set, no grid
search; one criterion, maximised.
```

## The maths

The model is an additive predictor with one smooth per covariate:

$$
g(\mu_i) = \eta_i = \beta_0 + \sum_j f_j(x_{ij}), \qquad
f_j(x) = \sum_{k=1}^{K_j} \beta_{jk}\, b_{jk}(x),
$$

where the $b_{jk}$ are basis functions (here B-splines) and $K_j$ is the
basis size, the `k` you pass to `Spline`. Fitting maximises a penalised
log-likelihood,

$$
\ell_p(\boldsymbol\beta; \boldsymbol\lambda)
= \ell(\boldsymbol\beta)
- \tfrac{1}{2} \sum_j \lambda_j\, \boldsymbol\beta^\top \mathbf S_j\, \boldsymbol\beta ,
$$

in which $\ell$ is the ordinary log-likelihood of the family and each
$\mathbf S_j$ is a penalty matrix measuring the wiggliness of $f_j$. For a
P-spline the penalty is the sum of squared second differences of
neighbouring coefficients,

$$
\boldsymbol\beta^\top \mathbf S\, \boldsymbol\beta
= \sum_{k=3}^{K} \left(\beta_k - 2\beta_{k-1} + \beta_{k-2}\right)^2 ,
$$

a discrete stand-in for $\int f''(x)^2\,\mathrm{d}x$. A straight line has
zero second differences, so the penalty cannot charge for one: lines are the
penalty's *null space*.

How much of the basis the fit actually uses is the **effective degrees of
freedom**,

$$
\tau(\boldsymbol\lambda) = \operatorname{tr}(\mathbf F), \qquad
\mathbf F = \left(\mathbf X^\top \mathbf W \mathbf X + \mathbf S_{\boldsymbol\lambda}\right)^{-1}
\mathbf X^\top \mathbf W \mathbf X, \qquad
\mathbf S_{\boldsymbol\lambda} = \sum_j \lambda_j \mathbf S_j ,
$$

where $\mathbf X$ holds the basis functions evaluated at the data and
$\mathbf W$ the working weights of the fit. With $\lambda = 0$ the trace is
the full basis size; as $\lambda \to \infty$ it falls to the size of the null
space, one line's worth. The EDF in every figure title below is this number.

REML chooses $\boldsymbol\lambda$ by maximising the criterion Wood (2011)
writes, for a fitted $\hat{\boldsymbol\beta}$ at the given
$\boldsymbol\lambda$, as

$$
\mathcal V(\boldsymbol\lambda)
= \ell(\hat{\boldsymbol\beta})
- \tfrac{1}{2}\hat{\boldsymbol\beta}^\top \mathbf S_{\boldsymbol\lambda}\hat{\boldsymbol\beta}
+ \tfrac{1}{2}\log\left|\mathbf S_{\boldsymbol\lambda}\right|_+
- \tfrac{1}{2}\log\left|\mathbf H + \mathbf S_{\boldsymbol\lambda}\right|
+ \tfrac{M_p}{2}\log(2\pi),
$$

with $\mathbf H$ the negative Hessian of $\ell$ at $\hat{\boldsymbol\beta}$
(for a GLM, $\mathbf X^\top \mathbf W \mathbf X$), $|\cdot|_+$ the product
of the non-zero eigenvalues, and $M_p$ the dimension of the null space. For
a Gaussian response this is exactly the restricted likelihood; for every
other family it is the Laplace approximation to it, which is what
`fit_reml` maximises.

| Term | What it does | Why it matters |
|---|---|---|
| $\ell(\hat{\boldsymbol\beta})$ | Rewards a fit that follows the data. | On its own it would always choose $\lambda = 0$. |
| $-\tfrac12 \hat{\boldsymbol\beta}^\top \mathbf S_{\boldsymbol\lambda} \hat{\boldsymbol\beta}$ | Charges the fitted curve for its wiggle at the current price. | The price is what is being chosen. |
| $+\tfrac12 \log\lvert\mathbf S_{\boldsymbol\lambda}\rvert_+$ | Grows with $\lambda$: the volume of curves the penalty considers plausible shrinks as the price rises. | This is the term that rewards simplicity. |
| $-\tfrac12 \log\lvert\mathbf H + \mathbf S_{\boldsymbol\lambda}\rvert$ | Falls with $\lambda$: the volume of curves the data leave plausible. | Together with the previous term it is the Occam factor: complexity is paid for automatically. |

The Bayesian reading makes the balance intuitive. The penalty is a prior
$\boldsymbol\beta \sim N(\mathbf 0, \mathbf S_{\boldsymbol\lambda}^{-})$ that
prefers smooth curves, and $\mathcal V$ is the log probability of the data
with the curve integrated out. Maximising it is asking which smoothness
makes the observed data most probable.

## Why REML and not a holdout

Cross-validation needs many refits and a holdout that is not always
available. Generalised cross-validation is a single criterion but Reiss and
Ogden (2009) showed it has more local optima than REML and tends to
under-smooth. Wood (2011) gave a stable Newton method for $\mathcal V$ that
also yields the smoothing-parameter uncertainty, and Wood, Pya and Säfken
(2016) extended it to any regular likelihood. The holdout curve in the
figures below is the check an actuary still trusts; on this data REML lands
where it bottoms out.

## Removing a term altogether

The penalty cannot shrink its own null space, so an ordinary smooth can
never disappear: at most it becomes a straight line. Marra and Wood (2011)
add a second penalty on that null space, $\lambda_j^{*} \mathbf S_j^{*}$
with $\mathbf S_j^{*} = \mathbf U_j \mathbf U_j^\top$ built from the
null-space eigenvectors of $\mathbf S_j$, and let REML estimate both
prices. That is `select=True`: a term with no signal can then be shrunk to
zero, which the last figure shows.

```{admonition} What this means for a tariff
:class: note sg-pricing

A curve that follows noise is a price that follows noise, and a price that
follows noise is one a competitor can pick off. REML gives a reproducible,
defensible choice of smoothness that a reviewer can read off the summary:
the penalty, the effective degrees of freedom, and the criterion value at
the optimum.
```

## Words used above

| Word | Meaning here |
|---|---|
| Basis size, `k` | How many pieces the spline is built from; the most flexible the curve can be. |
| Penalty | A number that grows with the wiggliness of the curve. |
| Lambda | The price per unit of penalty; large means smooth. |
| EDF | Effective degrees of freedom: how many of the `k` pieces the fit really uses. |
| Null space | The shapes the penalty cannot charge for: straight lines. |
| REML | The criterion that chooses lambda from the data; for non-Gaussian families its Laplace approximation, sometimes written LAML. |
| Holdout deviance | The model's error on rows it never saw; lower is better. |

## See it happen
````

Then the data cell, Task 5's widget, and the four existing figures in their current order, each under its existing heading demoted to `###`.

- [ ] **Step 4: Fix the captions and the triptych size**

`select-shrinkage` caption opens with "A term with no signal. The black curve is the fitted effect and the yellow band its 95% confidence interval." The `reml-path` caption adds "the red marker is the value REML settled on". The triptych's `figsize=(12, 3.8)` becomes `figsize=(9, 3.0)`, then check its tick labels at 1440 wide match the other figures.

- [ ] **Step 5: References section before Main takeaways**

````markdown
## References

- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *Journal of the Royal Statistical Society: Series B*, 73(1), 3–36. [doi:10.1111/j.1467-9868.2010.00749.x](https://doi.org/10.1111/j.1467-9868.2010.00749.x)
- Wood, S. N., Pya, N., and Säfken, B. (2016). Smoothing parameter and model selection for general smooth models. *Journal of the American Statistical Association*, 111(516), 1548–1563. [doi:10.1080/01621459.2016.1180986](https://doi.org/10.1080/01621459.2016.1180986)
- Marra, G., and Wood, S. N. (2011). Practical variable selection for generalized additive models. *Computational Statistics & Data Analysis*, 55(7), 2372–2387. [doi:10.1016/j.csda.2011.02.004](https://doi.org/10.1016/j.csda.2011.02.004)
- Reiss, P. T., and Ogden, R. T. (2009). Smoothing parameter selection for a class of semiparametric linear models. *Journal of the Royal Statistical Society: Series B*, 71(2), 505–523. [doi:10.1111/j.1467-9868.2008.00695.x](https://doi.org/10.1111/j.1467-9868.2008.00695.x)
````

Before writing the criterion, fetch https://webhomes.maths.ed.ac.uk/~swood34/gsm.pdf and confirm the Laplace-approximate REML expression there has the same four terms as the display above (log-likelihood at the penalised optimum, the penalty, the positive log-determinant of the penalty, the log-determinant of Hessian plus penalty). If the paper's form differs, follow the paper and say so in the deviation report; do not invent a variant. Add the Wood (2017) book and Wahba (1985) only if their DOIs resolve on Crossref.

- [ ] **Step 6: Callout styles**

Append to `docs/_static/custom.css`:

```css
/* Plain-words and tariff callouts on explanation pages. */
div.admonition.sg-plain > .admonition-title,
div.admonition.sg-pricing > .admonition-title {
  font-family: var(--sg-display);
  font-weight: 400;
  letter-spacing: 0.04em;
  font-size: 1.2rem;
}
div.admonition.sg-plain { border-color: var(--sg-ink); }
div.admonition.sg-plain > .admonition-title { background: var(--sg-yellow); color: #15171c; }
div.admonition.sg-pricing { border-color: var(--sg-red); box-shadow: 4px 4px 0 var(--sg-red); }
div.admonition.sg-pricing > .admonition-title { background: var(--sg-red); color: #ffffff; }
```

If the pydata theme's own title icon collides with the display face, hide it for these two classes with `div.admonition.sg-plain > .admonition-title::before { display: none; }` and the same for `sg-pricing`.

- [ ] **Step 7: Build, look, commit**

Clean strict executed build. Confirm every display equation renders (search the built HTML for `class="math notranslate"` blocks, at least five) and the two callouts carry the display font. Commit: `Docs: the smoothness page explains the REML idea, maths and terms before the pictures`.

### Task 5: The lambda slider with the REML-path panel

**Files:**
- Modify: `docs/explanation/how-reml-chooses-smoothness.md` (one new visible-output cell at the top of "See it happen", after the data cell)
- Modify: `docs/_static/custom.css` (widget styles, under 50 lines)

**Interfaces:**
- Produces: an `IPython.display.HTML` output rendered inline; no glue, no landing page use.

Max's ask: "an animation or interactive slider for the effect of lambda on a spline instead of two static figures", and "alongside it show how EDF and lambda change on a graph next to it as REML iterates it out". Decision: one widget with two panels. Left, the fit at the selected lambda. Right, EDF against log lambda: a thin curve across the whole grid (what the slider sweeps), a moving marker for the slider's current lambda, and REML's own iterates as numbered red dots joined in order, ending on the yellow REML tick. Two play modes: "Sweep" runs the slider across the grid; "Run REML" steps through the iterates, moving the left fit to each iterate's lambda and lighting its dot, with the objective value in the readout. A GIF cannot be paused at the lambda the reader cares about; the matplotlib JS animation embeds a PNG per frame and weighs megabytes; a live kernel is impossible because numba and tabmat have no WebAssembly build.

- [ ] **Step 1: Precompute the curves and the path**

```python
import json

lambdas = np.concatenate(([0.0], np.logspace(-4, 4, 60)))
curves, edfs = [], []
for lam in lambdas:
    model = fit_at(lam)
    curves.append(np.round(model.predict(grid), 3).tolist())
    edfs.append(round(float(model.term_inference("x").edf), 2))
history = reml.reml_diagnostics()
reml_lambda = float(history["lambdas"]["x"])
reml_index = int(np.argmin(np.abs(np.log10(np.maximum(lambdas, 1e-12)) - np.log10(reml_lambda))))
path_lambdas = [float(step["x"]) for step in history["lambda_history"]]
path_edf = [round(float(fit_at(lam).term_inference("x").edf), 2) for lam in path_lambdas]
payload = {
    "x": np.round(grid["x"].to_numpy(), 4).tolist(),
    "truth": np.round(truth_on(grid), 3).tolist(),
    "points": [np.round(x, 4).tolist(), np.round(y, 3).tolist()],
    "lambdas": [float(v) for v in lambdas],
    "edf": edfs,
    "curves": curves,
    "reml_index": reml_index,
    "reml_lambda": reml_lambda,
    "path": {"lambdas": path_lambdas, "edf": path_edf,
             "objective": [round(float(v), 3) for v in history["objective_history"]]},
}
```

Sixty-one fits plus one per REML iterate, one smooth on 400 rows; the page must stay under 60 seconds.

- [ ] **Step 2: Build the widget HTML**

Same cell, below the payload, then `display(HTML(slider_html(payload, "reml")))`. One `<figure class="sg-slider" id="sg-slider-reml">` containing two inline SVGs side by side (`viewBox="0 0 640 300"` each; the container stacks them at phone width), a control bar (buttons "Sweep" and "Run REML", the range input, a readout span, a "Jump to REML" button), a `<noscript>` line pointing at the static figures below, and one `<script>` that:

- draws the left panel exactly as the static pair: data as small muted points, truth dashed, the fit path in ink, red when the slider sits on the REML index;
- draws the right panel: x axis is $\log_{10}\lambda$ from −4 to 4 with $\lambda = 0$ at the left edge as its own tick labelled "0"; y axis EDF from 0 to the grid maximum; the grid EDF curve as a thin ink polyline; the REML iterates as red circles with their iteration number beside each and a red polyline joining them in order; a filled ink marker on the grid curve at the slider's position; a yellow vertical tick at the REML lambda on both the right panel and the slider track;
- readout: `λ = <value>   EDF <edf>` and, while running REML, `iteration k: λ = …, EDF …, objective …`;
- "Sweep": one step every 90 ms across the grid, button reads "Pause" while running; "Run REML": one iterate every 700 ms, moving the slider to the nearest grid index of that iterate's lambda and lighting its dot; "Jump to REML" sets the slider to `reml_index`.

Number formatting: `0` for zero, one decimal above one, `toExponential(1)` below one. All colours from CSS classes (`sg-slider__pt`, `sg-slider__truth`, `sg-slider__fit`, `sg-slider__grid`, `sg-slider__path`, `sg-slider__iter`, `sg-slider__marker`, `sg-slider__axis`, `sg-slider__tick`), never inline, so dark mode follows the theme variables.

- [ ] **Step 3: Styles**

Append to `docs/_static/custom.css`, using the theme variables `--pst-color-text-base`, `--pst-color-text-muted`, `--pst-color-background`, and the site's `--sg-ink`, `--sg-red`, `--sg-yellow`, `--sg-shadow`: the figure as a comic panel (2px border, 4px offset shadow), SVGs at `width: 100%; height: auto`, a two-column grid that stacks under 700 px, the range input with `accent-color: var(--sg-red)`, the buttons in the site's button style, the readout in the monospace face.

- [ ] **Step 4: Verify and commit**

Clean strict executed build. With Playwright, open the page, screenshot at slider position 0, after `page.click("button.sg-slider__reml")`, and after "Run REML" completes (wait 6 s), light and dark, 1440 and 420 wide, into the scratchpad `reml-page/slider/` directory; the fit path must differ between the first two, the right panel must show numbered red dots ending on the yellow tick. `uv run pytest tests/docs -q` passes; the page stays under 60 seconds. Commit: `Docs: a lambda slider with the REML path beside it, precomputed on every build`.

### Task 6: Take the section off the landing page

**Files:**
- Modify: `docs/index.md`
- Modify: `docs/explanation/how-reml-chooses-smoothness.md` (delete the hidden glue cell that builds the data-URI pair and the `fig_pair` figure; keep the triptych)
- Modify: `docs/_static/custom.css` (delete the `.sg-figure` rules if nothing else uses them)

- [ ] **Step 1: Remove the section**

Delete the "What REML does" heading, its `{rst-class}` line, the `{container}` block with the `{glue:any}` directive and the caption paragraph. Replace them with one sentence after the twelve-line code block:

```markdown
Why the fitted curves look the way they do: [How REML chooses smoothness](explanation/how-reml-chooses-smoothness.md).
```

- [ ] **Step 2: Remove the glue plumbing**

In the page, delete the `remove-cell` cell that renders `fig_pair` to a data URI and glues `lambda-zero-vs-reml`, and the `fig_pair` construction if nothing else uses it. `grep -rn 'lambda-zero-vs-reml\|sg-figure' docs --include='*.md' --include='*.css' | grep -v superpowers` must print nothing afterwards.

- [ ] **Step 3: Build and commit**

Clean strict executed build, zero warnings; `docs/_build/html/index.html` contains no `data:image` and one link to the smoothness page under the code block. Commit: `Docs: the landing page links to the smoothness page instead of embedding it`.

### Task 7: The style sheet on the other executed page

**Files:**
- Modify: `docs/tutorials/distributional-model.md`

- [ ] **Step 1:** Insert the same guarded `remove-cell` style cell used by the smoothness page directly after the `%pip` cell. Re-run `uv run jupytext --sync docs/tutorials/distributional-model.md` so the paired `.ipynb` follows; confirm it still has zero outputs. Build; commit: `Docs: the distributional tutorial uses the site figure style`.

### Task 8: Verify

The critic runs: clean strict executed build; `uv run pytest tests/docs -q`; page execution time; screenshots of the page (top, the widget, the maths) and of the landing page in light and dark; checks that every display equation renders, that the two callouts carry the display font, that no `selection_penalty=0.0` remains on the page or the landing page, that the code cells are collapsed by default, and that the widget behaves (fit path changes with the slider, REML iterates drawn). No push; the coordinator opens the pull request.

## Post-fold note (2026-09-14)

The stacked PR #393 never ran the dev-ci `docs` job (stacked PRs get seven checks), so the first strict build of this page without execution was the one on #392 after the fold, and it went red: the four `glue:figure` pastes find no glue data when nothing executes. Fix at 096bc558 on the branch of #392: `docs/conf.py` suppresses `mystnb.glue` only when `SUPERGLM_DOCS_EXECUTE=off`, and `tests/docs/test_conventions.py` pins every glue paste to a glue call on the same page. Run the off-mode CI command locally before pushing any change to this page.
