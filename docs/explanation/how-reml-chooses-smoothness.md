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

Four hundred points on a sine wave with a gentle slope, plus Gaussian noise
with standard deviation 0.45. The basis is a P-spline with 40 functions,
deliberately generous, so that an unpenalised fit has room to misbehave.

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import pandas as pd
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

### No penalty, REML, far too much penalty

Three fits of the same model. The first fixes lambda at zero, the second lets
`fit_reml` choose it, the third fixes it at ten thousand.

```{code-cell} ipython3
:tags: [hide-input]

def fit_at(lam):
    model = SuperGLM(
        family="gaussian",
        spline_penalty=lam,
        features={"x": Spline(kind="ps", k=k)},
    )
    return model.fit(X, y)


reml = SuperGLM(
    family="gaussian",
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
:tags: [hide-input, remove-output]

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

fig_three, axes = plt.subplots(1, 3, figsize=(9, 3.0), sharey=True)
for ax, (title, model) in zip(axes, fits):
    draw(ax, title, model)
axes[0].set_ylabel("y")
axes[0].legend(loc="upper right")
fig_three.tight_layout()
glue("lambda-triptych", fig_three, display=False)
```

```{code-cell} ipython3
:tags: [remove-cell]

# The landing page pastes fig_pair from here with MyST-NB's cross-document
# glue, which carries HTML across documents but not image files, so the figure
# travels as an inline data URI.
import base64
import io

from IPython.display import HTML

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
```

```{glue:figure} lambda-triptych
:name: fig-lambda-triptych
:alt: Three fits of the same 400 points: no penalty, the REML penalty, and lambda ten thousand.

The same 400 points three times. With no penalty the spline chases every point. REML picks a penalty that follows the truth. A penalty far too large leaves the curve a slope and one gentle bend, and it misses the peaks.
```

With no penalty the fit spends all 39 of its free degrees of freedom on the
noise. With the REML lambda it spends about eight, and the curve sits on the
truth. At lambda ten thousand it has two left, enough for a slope and a
little bend, and misses the peaks entirely.

### What the penalty buys

Now sweep lambda over a log grid of twelve fixed values and ask two things of
each fit: how many effective degrees of freedom it keeps, and how well it
predicts rows it did not see, measured by mean deviance over five folds.

```{code-cell} ipython3
:tags: [hide-input]

lambdas = np.logspace(-4, 4, 12)
edf = []
holdout = []
for lam in lambdas:
    model = SuperGLM(
        family="gaussian",
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
:tags: [hide-input, remove-output]

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
:alt: Effective degrees of freedom and held-out deviance against lambda on a log axis, with a red line at the lambda REML chose.

What the penalty buys. Left: how many effective parameters the spline keeps. Right: held-out deviance. The red line is the lambda REML chose without ever seeing a holdout.
```

The left panel is the dial: each factor of ten in lambda takes away a few
degrees of freedom. The right panel is the reason the dial matters. Held-out
deviance is flat and high on the left, where every fit reproduces the noise,
falls to a minimum, and rises steeply on the right, where the fits are too
stiff to reach the peaks. REML never touched a fold and still landed at the
bottom.

### How REML gets there

REML is an optimisation the solver drives directly, not a grid search.
superglm minimises the negative REML criterion, so lower is better and the
right-hand curve below falls. `reml_diagnostics` keeps the path the optimiser
took: one lambda per outer step plus the starting value, and one criterion
value per step.

```{code-cell} ipython3
:tags: [hide-input, remove-output]

diag = reml.reml_diagnostics()
path = [step["x"] for step in diag["lambda_history"]]
objective = list(diag["objective_history"])

fig_path, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].plot(range(len(path)), path, color="#15171C", marker="o", markersize=4)
axes[0].plot(len(path) - 1, path[-1], color="#D6402B", marker="o", markersize=7)
axes[0].set_yscale("log")
axes[0].set_title(f"lambda after each step ({diag['n_reml_iter']} steps)")
axes[0].set_xlabel("step (0 = starting value)")
axes[0].set_ylabel("lambda")
axes[1].plot(range(1, len(objective) + 1), objective, color="#15171C", marker="o", markersize=4)
axes[1].plot(len(objective), objective[-1], color="#D6402B", marker="o", markersize=7)
axes[1].set_title("REML criterion per step (minimised)")
axes[1].set_xlabel("step")
axes[1].set_ylabel("REML criterion (lower is better)")
fig_path.tight_layout()
glue("reml-path", fig_path, display=False)
```

```{glue:figure} reml-path
:name: fig-reml-path
:alt: Two panels: lambda after each outer step on a log axis, and the REML criterion per step, both ending in a red marker.

REML is an optimisation the solver drives directly, not a grid search: a handful of steps from the starting value to the optimum on this example. Left, lambda after each step, where step 0 is the starting value, so the left panel carries one point more than the right. Right, the criterion superglm minimises, which is why the curve falls. In both panels the red marker is the value REML settled on.
```

The path is not a steady climb. The optimiser probes downwards once, then
climbs three orders of magnitude in two steps, overshoots, and settles back to
a lambda near the bottom of the held-out curve in the previous figure. The
criterion is within a fraction of a point of its final value after four steps;
everything past that is refinement. A grid over twelve values, as in the
previous figure, costs twelve fits and five folds each; the optimiser costs a
handful of fits and no folds.

### Removing a term that carries no signal

Add a second column `z` that has nothing to do with `y`, and fit both columns
as splines. The ordinary penalty charges for bending, so a term it cannot
justify is shrunk to a straight line, and a straight line costs nothing under
that penalty, so it stays. `select=True` adds a second penalty on the straight
part as well, and REML can then take the term out altogether.

```{code-cell} ipython3
:tags: [hide-input]

z = rng.uniform(0.0, 1.0, n)
X2 = pd.DataFrame({"x": x, "z": z})

selected = {}
for sel in (False, True):
    selected[sel] = SuperGLM(
        family="gaussian",
        features={
            "x": Spline(kind="ps", k=12, select=sel),
            "z": Spline(kind="ps", k=12, select=sel),
        },
    ).fit_reml(X2, y)
    for name in ("x", "z"):
        print(f"select={sel!s:<5} {name}  EDF {selected[sel].term_inference(name).edf:6.3f}")
```

```{code-cell} ipython3
:tags: [hide-input, remove-output]

fig_select, axes = plt.subplots(2, 2, figsize=(9, 6), sharex="col", sharey=True)
for row, sel in enumerate((False, True)):
    for col, name in enumerate(("x", "z")):
        ax = axes[row, col]
        term = selected[sel].term_inference(name)
        # ci_lower and ci_upper are on the relativity scale; the curve is on the
        # linear-predictor scale, which for this Gaussian identity fit is the
        # effect on y. Take logs so the band and the curve share an axis.
        ax.fill_between(
            term.x, np.log(term.ci_lower), np.log(term.ci_upper),
            color="#F4B942", alpha=0.35, linewidth=0,
        )
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
:alt: A two-by-two grid of fitted effects with confidence bands: the x term and the z term, fitted with select off and on.

A term with no signal. The black curve is the fitted effect and the yellow band its 95% confidence interval. All four panels share one y axis, so the `z` term's effect can be compared with the `x` term's. Without the double penalty the `z` term keeps a slope and one degree of freedom, because the ordinary penalty cannot charge for a straight line; with `select=True` REML shrinks it to flat and its EDF to zero. The `x` term is untouched either way.
```

## References

- Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *Journal of the Royal Statistical Society: Series B*, 73(1), 3–36. [doi:10.1111/j.1467-9868.2010.00749.x](https://doi.org/10.1111/j.1467-9868.2010.00749.x)
- Wood, S. N., Pya, N., and Säfken, B. (2016). Smoothing parameter and model selection for general smooth models. *Journal of the American Statistical Association*, 111(516), 1548–1563. [doi:10.1080/01621459.2016.1180986](https://doi.org/10.1080/01621459.2016.1180986)
- Marra, G., and Wood, S. N. (2011). Practical variable selection for generalized additive models. *Computational Statistics & Data Analysis*, 55(7), 2372–2387. [doi:10.1016/j.csda.2011.02.004](https://doi.org/10.1016/j.csda.2011.02.004)
- Reiss, P. T., and Ogden, R. T. (2009). Smoothing parameter selection for a class of semiparametric linear models. *Journal of the Royal Statistical Society: Series B*, 71(2), 505–523. [doi:10.1111/j.1467-9868.2008.00695.x](https://doi.org/10.1111/j.1467-9868.2008.00695.x)
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*, 2nd edition. Chapman and Hall/CRC. [doi:10.1201/9781315370279](https://doi.org/10.1201/9781315370279)
- Wahba, G. (1985). A comparison of GCV and GML for choosing the smoothing parameter in the generalized spline smoothing problem. *Annals of Statistics*, 13(4), 1378–1402. [doi:10.1214/aos/1176349743](https://doi.org/10.1214/aos/1176349743)

## Main takeaways

- No penalty means the spline reproduces the noise; the effective degrees of freedom climb towards the basis size.
- REML picks the penalty from the data alone, and on this example it lands where held-out deviance is lowest.
- The optimiser reaches that value in a handful of steps; it is an optimisation the solver drives directly, not a grid search, and the criterion it drives down is the negative REML criterion, so lower is better.
- `select=True` lets REML remove a term that carries no signal instead of leaving it a straight line.

## Next steps

- [Choose a fitting path](../how-to/choose-a-fitting-path.md)
- [Solvers and internals](solvers-and-internals.md) for the REML criterion itself
- {py:meth}`superglm.SuperGLM.fit_reml`
