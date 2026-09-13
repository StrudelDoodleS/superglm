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

Four words carry the page. The **penalty** is a charge on how much the fitted
curve bends. **Lambda** is the size of that charge: zero lets the curve do
what it likes, a large value forces it towards a straight line. The
**effective degrees of freedom** (EDF) count how many parameters a fit is
really using once the penalty has done its work; it runs from the basis size
at lambda zero down to the null space at lambda infinity. **REML** is the
criterion that picks lambda from the data alone, treating the spline
coefficients as random effects and choosing the lambda that makes the
observed data most probable; the criterion itself is in
[Solvers and internals](solvers-and-internals.md).

```{code-cell} ipython3
:tags: [remove-cell]

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
for candidate in (Path("../_static/superglm.mplstyle"), Path("docs/_static/superglm.mplstyle")):
    if candidate.exists():
        plt.style.use(str(candidate))
        break
```

## Data with a known truth

Four hundred points on a sine wave with a gentle slope, plus Gaussian noise
with standard deviation 0.45. The basis is a P-spline with 40 functions,
deliberately generous, so that an unpenalised fit has room to misbehave.

```{code-cell} ipython3
import base64
import io

import numpy as np
import pandas as pd
from IPython.display import HTML
from sklearn.model_selection import KFold

from superglm import Spline, SuperGLM, cross_validate

try:
    from myst_nb import glue
except ImportError:  # running outside the docs build, e.g. Colab

    def glue(name, obj, display=True):
        return obj


rng = np.random.default_rng(20260913)
n = 400
k = 40
noise_sd = 0.45
x = np.sort(rng.uniform(0.0, 1.0, n))
truth = np.sin(2 * np.pi * x) + 0.6 * x
y = truth + rng.normal(0.0, noise_sd, n)
X = pd.DataFrame({"x": x})
grid = pd.DataFrame({"x": np.linspace(0.0, 1.0, 300)})


def truth_on(frame):
    xs = frame["x"].to_numpy()
    return np.sin(2 * np.pi * xs) + 0.6 * xs
```

## No penalty, REML, far too much penalty

Three fits of the same model. The first fixes lambda at zero, the second lets
`fit_reml` choose it, the third fixes it at ten thousand.

```{code-cell} ipython3
def fit_at(lam):
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=lam,
        features={"x": Spline(kind="ps", k=k)},
    )
    return model.fit(X, y)


reml = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={"x": Spline(kind="ps", k=k)},
).fit_reml(X, y)

fits = [
    ("No penalty, lambda = 0", fit_at(0.0)),
    ("REML", reml),
    ("Lambda = 10,000", fit_at(1e4)),
]
for title, model in fits:
    print(f"{title:<24} EDF {model.term_inference('x').edf:5.1f}")
```

```{code-cell} ipython3
:tags: [remove-output]

def draw(ax, title, model):
    ax.scatter(x, y, s=6, color="#C9CCD3", label="data", zorder=1)
    ax.plot(
        grid["x"], truth_on(grid),
        color="#6B7280", linestyle="--", linewidth=1.4, label="truth", zorder=2,
    )
    ax.plot(grid["x"], model.predict(grid), color="#15171C", linewidth=2.2, label="fit", zorder=3)
    edf = model.term_inference("x").edf
    ax.set_title(f"{title}\nEDF {edf:.1f}")
    ax.set_xlabel("x")


fig_pair, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, (title, model) in zip(axes, fits[:2]):
    draw(ax, title, model)
axes[0].set_ylabel("y")
axes[0].legend(loc="upper right")
fig_pair.tight_layout()

# The landing page pastes this figure from here with MyST-NB's cross-document
# glue, which carries HTML across documents but not image files, so the figure
# travels as an inline data URI.
buffer = io.BytesIO()
fig_pair.savefig(buffer, format="png")
encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
glue(
    "lambda-zero-vs-reml",
    HTML(
        f'<img src="data:image/png;base64,{encoded}" '
        'alt="Two fits of the same data: no penalty on the left, the REML penalty on the right" '
        'style="max-width: 100%; height: auto;">'
    ),
    display=False,
)
plt.close(fig_pair)

fig_three, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, (title, model) in zip(axes, fits):
    draw(ax, title, model)
axes[0].set_ylabel("y")
axes[0].legend(loc="upper right")
fig_three.tight_layout()
glue("lambda-triptych", fig_three, display=False)
```

```{glue:figure} lambda-triptych
:name: fig-lambda-triptych

The same 400 points three times. With no penalty the spline chases every point. REML picks a penalty that follows the truth. A penalty far too large flattens the curve to a line.
```

With no penalty the fit spends all 39 of its free degrees of freedom on the
noise. With the REML lambda it spends about eight, and the curve sits on the
truth. At lambda ten thousand it has two left, enough for a slope and a
little bend, and misses the peaks entirely.

## What the penalty buys

Now sweep lambda over a log grid of twelve fixed values and ask two things of
each fit: how many effective degrees of freedom it keeps, and how well it
predicts rows it did not see, measured by mean deviance over five folds.

```{code-cell} ipython3
lambdas = np.logspace(-4, 4, 12)
edf = []
holdout = []
for lam in lambdas:
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        spline_penalty=lam,
        features={"x": Spline(kind="ps", k=k)},
    )
    edf.append(model.fit(X, y).term_inference("x").edf)
    cv = cross_validate(
        model, X, y,
        cv=KFold(5, shuffle=True, random_state=1),
        scoring=("deviance",),
    )
    holdout.append(cv.mean_scores["deviance"])

reml_lambda = reml.reml_diagnostics()["lambdas"]["x"]
print(f"REML chose lambda = {reml_lambda:.1f}")
```

```{code-cell} ipython3
:tags: [remove-output]

fig_sweep, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, values, label in zip(axes, (edf, holdout), ("EDF", "Held-out deviance")):
    ax.plot(lambdas, values, color="#15171C", marker="o", markersize=4)
    ax.axvline(reml_lambda, color="#D6402B", linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.set_title(label)
axes[0].text(
    reml_lambda * 1.5, max(edf) * 0.95, "REML", color="#D6402B", fontsize=9, va="top",
)
fig_sweep.tight_layout()
glue("edf-and-holdout", fig_sweep, display=False)
```

```{glue:figure} edf-and-holdout
:name: fig-edf-and-holdout

What the penalty buys. Left: how many effective parameters the spline keeps. Right: held-out deviance. The red line is the lambda REML chose without ever seeing a holdout.
```

The left panel is the dial: each factor of ten in lambda takes away a few
degrees of freedom. The right panel is the reason the dial matters. Held-out
deviance is flat and high on the left, where every fit reproduces the noise,
falls to a minimum, and rises steeply on the right, where the fits are too
stiff to reach the peaks. REML never touched a fold and still landed at the
bottom.

## How REML gets there

REML is a maximisation, not a search. `reml_diagnostics` keeps the path the
optimiser took, one lambda and one objective value per outer step.

```{code-cell} ipython3
:tags: [remove-output]

diag = reml.reml_diagnostics()
path = [step["x"] for step in diag["lambda_history"]]
objective = list(diag["objective_history"])

fig_path, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].plot(range(len(path)), path, color="#15171C", marker="o", markersize=4)
axes[0].plot(len(path) - 1, path[-1], color="#D6402B", marker="o", markersize=7)
axes[0].set_yscale("log")
axes[0].set_title(f"lambda per step ({diag['n_reml_iter']} steps)")
axes[0].set_xlabel("step")
axes[1].plot(range(1, len(objective) + 1), objective, color="#15171C", marker="o", markersize=4)
axes[1].plot(len(objective), objective[-1], color="#D6402B", marker="o", markersize=7)
axes[1].set_title("REML objective per step")
axes[1].set_xlabel("step")
fig_path.tight_layout()
glue("reml-path", fig_path, display=False)
```

```{glue:figure} reml-path
:name: fig-reml-path

REML is a maximisation, not a search: a handful of steps from the starting value to the optimum on this example, and the objective settles after the first two.
```

The first step moves lambda by orders of magnitude and does almost all of the
work on the objective; the remaining steps are refinement. A grid over twelve
values, as in the previous figure, costs twelve fits and five folds each; the
optimiser costs a handful of fits and no folds.

## Removing a term that carries no signal

Add a second column `z` that has nothing to do with `y`, and fit both columns
as splines. The ordinary penalty charges for bending, so a term it cannot
justify is shrunk to a straight line, and a straight line costs nothing under
that penalty, so it stays. `select=True` adds a second penalty on the straight
part as well, and REML can then take the term out altogether.

```{code-cell} ipython3
z = rng.uniform(0.0, 1.0, n)
X2 = pd.DataFrame({"x": x, "z": z})

selected = {}
for sel in (False, True):
    selected[sel] = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": Spline(kind="ps", k=12, select=sel),
            "z": Spline(kind="ps", k=12, select=sel),
        },
    ).fit_reml(X2, y)
    for name in ("x", "z"):
        print(f"select={sel!s:<5} {name}  EDF {selected[sel].term_inference(name).edf:6.3f}")
```

```{code-cell} ipython3
:tags: [remove-output]

fig_select, axes = plt.subplots(2, 2, figsize=(9, 6), sharex="col", sharey="col")
for row, sel in enumerate((False, True)):
    for col, name in enumerate(("x", "z")):
        ax = axes[row, col]
        term = selected[sel].term_inference(name)
        ax.fill_between(term.x, term.ci_lower, term.ci_upper, color="#F4B942", alpha=0.35, linewidth=0)
        ax.plot(term.x, term.log_relativity, color="#15171C")
        ax.axhline(0.0, color="#6B7280", linewidth=0.8)
        ax.set_title(f"select={sel}, term {name}\nEDF {term.edf:.2f}")
        if row == 1:
            ax.set_xlabel(name)
        if col == 0:
            ax.set_ylabel("effect on y")
fig_select.tight_layout()
glue("select-shrinkage", fig_select, display=False)
```

```{glue:figure} select-shrinkage
:name: fig-select-shrinkage

A term with no signal. Without the double penalty the `z` term keeps a slope and one degree of freedom, because the ordinary penalty cannot charge for a straight line; with `select=True` REML shrinks it to flat and its EDF to zero. The `x` term is untouched either way.
```

## Main takeaways

- No penalty means the spline reproduces the noise; the effective degrees of freedom climb towards the basis size.
- REML picks the penalty from the data alone, and on this example it lands where held-out deviance is lowest.
- The optimiser reaches that value in a handful of steps; it is a maximisation, not a grid search.
- `select=True` lets REML remove a term that carries no signal instead of leaving it a straight line.

## Next steps

- [Choose a fitting path](../how-to/choose-a-fitting-path.md)
- [Solvers and internals](solvers-and-internals.md) for the REML criterion itself
- {py:meth}`superglm.SuperGLM.fit_reml`
