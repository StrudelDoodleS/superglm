"""Standard errors come from the penalised fit when its rank is certified.

A fit whose own record certifies the penalised Hessian ``X'WX + S`` as positive definite shows
``sqrt(phi diag V)`` for every coefficient (``state_ops.standard_error_mask``).  Master blanked
each coefficient the data alone leave undetermined, which on a penalised fit hides numbers the
fit determines: every level of a ``RandomEffect``, every coefficient of an ``fs`` smooth, a thin
level of a spline by a factor.

Every reference here is built outside the inference path: mgcv's ``Vp`` on superglm's own design
at pinned smoothing parameters and scale (a fixture, mgcv 1.9-3 run as a black box), the dense
inverse of ``[1 X]' W [1 X] + S`` formed from the design and the penalty, and the data's null
space from a dense SVD.  Each test fails on master on the output it names, or is a control that
says why it must not move.
"""

from __future__ import annotations

import gc
import json
import lzma
import pickle
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import superglm
from superglm import (
    Categorical,
    Constraint,
    FactorSmooth,
    LambdaPolicy,
    Numeric,
    OrderedCategorical,
    RandomEffect,
    Spline,
    SuperGLM,
)
from superglm.model.fit_state import fitted_lambda2
from superglm.reml.penalty_algebra import build_penalty_matrix

from ._traced_memory import own_traced_peak
from .test_saved_fs_models import assert_predicts_as_saved

FIXTURES = Path(__file__).parent / "fixtures"
SAVED = FIXTURES / "saved_master_bce2910e5"
EPS = np.finfo(np.float64).eps
REFIT_NOTE = "Refit the model to show them."


# ------------------------------------------------------------------ independent references
def _design(model) -> np.ndarray:
    """``[1, X]``: the intercept and every coefficient's column of the fit's design, dense."""
    dm = model._dm
    blocks = [
        np.asarray(m.toarray(), dtype=np.float64).reshape(dm.n, -1) for m in dm.group_matrices
    ]
    return np.column_stack([np.ones(dm.n), *blocks])


def _working_weights(model, frame) -> np.ndarray:
    """Fisher weights ``(dmu/deta)^2 / V(mu)`` at the published fit, unit prior weights."""
    mu = np.asarray(model.predict(frame), dtype=np.float64)
    slope = np.asarray(model._link.deriv_inverse(model._link.link(mu)), dtype=np.float64)
    return slope**2 / np.asarray(model._distribution.variance(mu), dtype=np.float64)


def dense_raw_standard_errors(model, frame) -> np.ndarray:
    """``sqrt(diag(([1 X]' W [1 X] + S)^-1))`` over the slopes: the scale-free penalised SEs.

    One dense inverse of the penalised Hessian assembled from the design, the fitted mean and
    the penalty at the fitted smoothing parameters.  No rank decomposition, structured factor or
    covariance accessor of the package is read.
    """
    dm, X = model._dm, _design(model)
    H = X.T @ (_working_weights(model, frame)[:, None] * X)
    H[1:, 1:] += build_penalty_matrix(
        list(dm.group_matrices),
        list(model._groups),
        getattr(model, "_reml_lambdas", None) or fitted_lambda2(model),
        dm.p,
        reml_penalties=getattr(model, "_reml_penalties", None),
    )
    return np.sqrt(np.diag(np.linalg.inv(H))[1:])


def undetermined_slopes(A: np.ndarray) -> np.ndarray:
    """The slopes a null vector of ``A = sqrt(W) [1, X]`` moves: what those rows alone leave open.

    A dense SVD of the column-equilibrated weighted design.  A singular value at or below
    ``max(n, p) eps`` of the largest is null (``dgesdd``'s backward error); a slope a null
    vector moves by more than ``sqrt(eps)`` is undetermined.  The fixtures here are deficient
    by construction (a level's indicator against the intercept, a copied column, basis
    functions with no row), so every other singular value is orders of magnitude clear.
    """
    A = A / np.maximum(np.linalg.norm(A, axis=0), np.finfo(float).tiny)
    _, singular, right = np.linalg.svd(A, full_matrices=False)
    null = right[singular <= max(A.shape) * EPS * singular[0]]
    return np.any(np.abs(null[:, 1:]) > np.sqrt(EPS), axis=0)


def data_undetermined(model, frame) -> np.ndarray:
    """``undetermined_slopes`` of the fit's own rows at its fitted weights."""
    return undetermined_slopes(np.sqrt(_working_weights(model, frame))[:, None] * _design(model))


def _flat(model, se: dict) -> np.ndarray:
    return np.concatenate(
        [np.atleast_1d(np.asarray(se[g.name], dtype=np.float64)) for g in model._groups]
    )


def _fit(model, frame, y, *, method="fit_reml", **kwargs) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(model, method)(frame, y, **kwargs)
    return caught


def _pearson_scale(model, frame, y) -> float:
    """``pearson_chi2 / (n - edf)``: the quasi-likelihood scale of ``coefficient_se`` at unit weights."""
    mu = np.asarray(model.predict(frame), dtype=np.float64)
    pearson = float(np.sum((y - mu) ** 2 / model._distribution.variance(mu)))
    return pearson / (len(y) - model.result.effective_df)


# ------------------------------------------------------------------------------- fixtures
def _thin_level():
    """The mgcv fixture's rows: a spline by a factor whose level E has 85 rows at 13 values of x."""
    data = np.load(FIXTURES / "mgcv_factor_by_spline_gamma_data.npz", allow_pickle=False)
    return pd.DataFrame({"x": data["x"], "f": data["f"].astype(str)}), data["y"]


def _thin_level_model(x=None, **options) -> SuperGLM:
    return SuperGLM(
        family=superglm.families.gamma(),
        link="log",
        features={"x": Spline(n_knots=8) if x is None else x, "f": Categorical()},
        interactions=[("x", "f")],
        **options,
    )


def _zero_claim_levels(n=6000, *, levels="ABCDEF"):
    """Poisson rows with a factor whose level F (2% of rows) has no claim."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "age": rng.uniform(18, 80, n),
            "veh": pd.Categorical(
                rng.choice(list("ABCDEF"), n, p=[0.3, 0.25, 0.2, 0.15, 0.08, 0.02]),
                categories=list(levels),
            ),
            "dens": rng.normal(size=n),
        }
    )
    y = rng.poisson(np.exp(-1.0 + 0.3 * np.sin(frame["age"] / 10) + 0.2 * frame["dens"]))
    y = y.astype(float)
    y[(frame["veh"] == "F").to_numpy()] = 0.0
    return frame, y


def _random_effect_model(**extra) -> SuperGLM:
    features = {"age": Spline(n_knots=6), "veh": RandomEffect(), "dens": Numeric(), **extra}
    return SuperGLM(family="poisson", features=features)


def _factor_smooth(levels=200, zero=5, *, one_row=0, seed=2):
    """Twenty Poisson rows a level on average; the first ``zero`` levels have no claim.

    The next ``one_row`` levels keep their first row only.
    """
    rng = np.random.default_rng(seed)
    n = 20 * levels
    g = rng.integers(0, levels, n)
    x = rng.uniform(size=n)
    frame = pd.DataFrame({"x": x, "z": rng.normal(size=n), "g": [f"g{v:04d}" for v in g]})
    y = rng.poisson(np.exp(-1.0 + 0.5 * np.sin(3 * x) + 0.2 * np.sin(5 * x + g % 7))).astype(float)
    y[g < zero] = 0.0
    keep = np.ones(n, dtype=bool)
    for level in range(zero, zero + one_row):
        keep[np.flatnonzero(g == level)[1:]] = False
    return frame[keep].reset_index(drop=True), y[keep]


def _separated_region():
    """Poisson rows in twelve branches; region B is branches 0 to 2 and has no claim."""
    rng = np.random.default_rng(51)
    branch = rng.integers(0, 12, 1800)
    frame = pd.DataFrame(
        {"region": np.where(branch < 3, "B", "A"), "branch": [f"b{code:02d}" for code in branch]}
    )
    y = rng.poisson(np.exp(0.5 + branch / 30)).astype(float)
    y[branch < 3] = 0.0
    return frame, y


def _unpriced_region(case: str):
    """The separated region in a ``fit_reml`` that cannot price it: rows, response, model, event.

    Region B's coefficient has no finite estimate in any of them, and the event is the warning
    by which the fit says so.  What differs is which of the fit's own records holds it:

    - ``separated_level``: Poisson, the region a factor: the build-time scan counts its claims;
    - ``weak_numeric_level``: Poisson, the region a 0/1 column, which the scan does not read;
    - ``weak_unscanned_level``: Poisson, the region a factor, the scan turned off;
    - ``weak_gaussian_level``: Gaussian under a log link, the region's response zero: no
      boundary for the scan to count.  In these three REML's diagnostics name the slope as
      weakly identified, and nothing else does;
    - ``not_converged``: binomial, the region a 0/1 column whose rows are all ones: no record
      names the region, and the fit ends without meeting its convergence test.
    """
    frame, y = _separated_region()
    in_b = (frame["region"] == "B").to_numpy()
    options, region, event = {"family": "poisson"}, Categorical(base="A"), "SeparationWarning"
    if case == "weak_gaussian_level":
        y, options = np.where(in_b, 0.0, 2.0 + 0.1 * y), {"family": "gaussian", "link": "log"}
    if case == "not_converged":
        y = np.where(in_b, 1.0, np.random.default_rng(7).uniform(size=len(y)) < 0.3)
        options, event = {"family": "binomial"}, "ConvergenceWarning"
    if case in ("weak_numeric_level", "not_converged"):
        frame, region = frame.assign(region=in_b.astype(float)), Numeric()
    if case == "weak_unscanned_level":
        options["separation"] = "ignore"
    if case.startswith("weak"):
        event = "WeakIdentificationWarning"
    model = SuperGLM(
        **options, features={"region": region, "branch": RandomEffect()}, selection_penalty=0
    )
    return frame, y, model, event


def _unpriced_level(case: str):
    """The thin-level rows beside a factor whose level R a plain ``fit`` cannot price.

    ``fit`` keeps no REML diagnostics and both fits converge, so each has one record of it:

    - ``scanned_level``: Poisson, R without a claim: the build-time scan names it;
    - ``runtime_finding``: Gaussian under a log link, R's response negative: no boundary for
      the scan to count, and the solver's backstop reports the walk.
    """
    frame, _ = _thin_level()
    rng = np.random.default_rng(5)
    veh = rng.choice(list("PQR"), len(frame), p=[0.5, 0.45, 0.05])
    mean = np.exp(0.3 + 0.5 * frame["x"].to_numpy() / 80.0)
    if case == "scanned_level":
        y, options = np.where(veh == "R", 0.0, rng.poisson(mean)), {"family": "poisson"}
    else:
        y = np.where(veh == "R", -1.0, mean + rng.normal(scale=0.2, size=len(frame)))
        options = {"family": "gaussian", "link": "log"}
    model = SuperGLM(
        **options,
        features={"x": Spline(n_knots=8), "f": Categorical(), "veh": Categorical(base="P")},
        interactions=[("x", "f")],
    )
    return frame.assign(veh=veh), y.astype(float), model


def _branches(seed: int, n: int, count: int):
    """``n`` rows in ``count`` branches: the generator, codes, names and a spline over their order."""
    rng = np.random.default_rng(seed)
    branch = rng.integers(0, count, n)
    order = [f"br{code:02d}" for code in range(count)]
    spline = OrderedCategorical(order=order, basis=Spline(kind="ps", n_knots=count - 1))
    return rng, branch, [order[code] for code in branch], spline


def _recorded_separation(case: str):
    """A separated level, cell or slope beside a penalised term its column aliases.

    Returns rows, response, model, fit method and the fit's keyword arguments.  The alias is
    exact in the design, so the data rule leaves the coefficient blank, and the penalty breaks
    it, so the penalised Hessian has full rank; what the walked coefficient then shows is where
    its walk stopped.  Each fit converges, and one record names it on every machine (two of
    the ``reference`` fits also end with the runtime finding under one OpenBLAS kernel):

    - ``reference_m20``, ``reference_m21``, ``reference_tol``: binomial under a log link, the
      reference level without a success: REML's separated set (``reml_laplace_separated``);
    - ``slope``, ``varying_slope``, ``curve``: a numeric by a factor, one level's slope
      separated: the term's ``_separated_ranges``;
    - ``cell``, ``crossed_cell``, ``base_cell``, ``crossed_cell_fit``: a factor by a factor, one
      cell without a claim: the crossed scan's cells (``_separated_cells_at_fit``).
    """
    options: dict = {}
    if case in ("reference_m20", "reference_m21"):
        rng = np.random.default_rng(124)
        region = np.repeat(["a", "b", "c"], 40)
        frame = pd.DataFrame(
            {"g": region, "h": np.char.add(region, rng.integers(0, 3, len(region)).astype(str))}
        )
        y = np.where(region == "a", 0.0, rng.uniform(size=len(region)) < 0.05).astype(float)
        options["offset"] = float(case[-2:]) * -1.0 * (region == "a")
        features = {
            "g": Categorical(base="a"),
            "h": RandomEffect(lambda_policy=LambdaPolicy.fixed(7.0)),
        }
        model = SuperGLM(
            family="binomial",
            link="log",
            selection_penalty=0,
            separation="ignore",
            features=features,
        )
        return frame, y, model, "fit_reml", options
    if case == "reference_tol":
        rng = np.random.default_rng(101)
        level = rng.integers(0, 12, 2000)
        a = np.where(level < 3, "A", np.where(level < 7, "B", "C"))
        frame = pd.DataFrame({"a": a, "branch": [f"b{v:02d}" for v in rng.integers(0, 12, 2000)]})
        y = rng.binomial(1, np.clip(np.exp(-1.5 + 0.05 * level), 0.01, 0.8)).astype(float)
        y[a == "A"] = 0.0
        model = SuperGLM(
            family="binomial",
            link="log",
            separation="ignore",
            features={"a": Categorical(base="A"), "branch": RandomEffect()},
            selection_penalty=0,
        )
        return frame, y, model, "fit_reml", {"pirls_tol": 1e-4}
    if case in ("slope", "cell"):
        rng = np.random.default_rng(541)
        g = np.repeat(["a", "b"], 300)
        x = rng.integers(0, 2, 600).astype(float)
        y = rng.poisson(1.2, 600).astype(float)
        y[(g == "b") & (x == 1)] = 0.0
        z = 2 * (g == "b") + x + 0.05 * rng.integers(0, 2, 600)
        frame = pd.DataFrame({"x": x, "g": g, "z": z, "a": np.where(x > 0, "y", "n")})
        term, first = ("x", Numeric()) if case == "slope" else ("a", Categorical(base="n"))
        model = SuperGLM(
            family="poisson",
            features={term: first, "g": Categorical(base="a"), "z": Spline(kind="ps", n_knots=8)},
            interactions=[(term, "g")],
            selection_penalty=0,
            spline_penalty=1,
        )
        return frame, y, model, "fit", options
    if case == "varying_slope":
        rng = np.random.default_rng(3)
        c = np.repeat(["A", "B"], 800)
        x = rng.uniform(size=1600)
        x[(c == "B") & (x < 0.33)] = 0.0
        y = rng.poisson(np.exp(0.5 + 0.2 * x)).astype(float)
        y[(c == "B") & (x > 0.0)] = 0.0
        frame = pd.DataFrame({"x": x, "c": c, "z": np.where(c == "B", x, 0.0)})
        model = SuperGLM(
            family="poisson",
            features={
                "x": Numeric(),
                "c": Categorical(base="A"),
                "z": Spline(kind="ps", n_knots=6, select=True),
            },
            interactions=[("x", "c")],
            selection_penalty=0,
            max_iter=200,
            direct_solve="gram",
        )
        return frame, y, model, "fit", options
    if case == "curve":
        rng, branch, names, spline = _branches(88, 2400, 12)
        cat = np.where(branch < 3, "C", np.where(branch < 8, "B", "A"))
        x = rng.uniform(-2.0, 2.0, len(branch))
        logit = 0.2 + 0.5 * x + 0.3 * (cat == "B") + 0.05 * branch
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit))).astype(float)
        y[cat == "C"] = x[cat == "C"] >= 0
        frame = pd.DataFrame({"x": x, "cat": cat, "branch": names})
        model = SuperGLM(
            family="binomial",
            features={"x": Numeric(), "cat": Categorical(base="A"), "branch": spline},
            interactions=[("x", "cat")],
        )
        return frame, y, model, "fit", options
    if case == "crossed_cell":
        frame, _ = _thin_level()
        rng = np.random.default_rng(5)
        y = rng.poisson(np.exp(0.3 + 0.5 * frame["x"].to_numpy() / 80.0)).astype(float)
        a, b = (rng.choice([f"{name}{i}" for i in range(3)], len(frame)) for name in "ab")
        y[(a == "a2") & (b == "b2")] = 0.0
        model = SuperGLM(
            family="poisson",
            features={
                "x": Spline(n_knots=8),
                "f": Categorical(),
                "a": Categorical(base="a0"),
                "b": Categorical(base="b0"),
            },
            interactions=[("x", "f"), ("a", "b")],
        )
        return frame.assign(a=a, b=b), y, model, "fit", options
    rng, branch, names, spline = _branches(77, 2700, 18)
    a = np.array(["A0", "A1", "A2"])[branch // 6]
    b = np.array(["B0", "B1", "B2"])[(branch // 2) % 3]
    mean = np.exp(0.5 + 0.2 * (a == "A1") + 0.2 * (b == "B1") + 0.03 * branch)
    y = rng.poisson(mean).astype(float)
    empty = ("A0", "B0") if case == "base_cell" else ("A2", "B2")
    if case == "crossed_cell_fit":
        y = rng.poisson(mean).astype(float)
    y[(a == empty[0]) & (b == empty[1])] = 0.0
    model = SuperGLM(
        family="poisson",
        features={
            "a": Categorical(base="A0"),
            "b": Categorical(base="B0"),
            "branch": RandomEffect() if case == "base_cell" else spline,
        },
        interactions=[("a", "b")],
    )
    method = "fit_reml" if case == "base_cell" else "fit"
    return pd.DataFrame({"a": a, "b": b, "branch": names}), y, model, method, options


# -------------------------------------------------------------------- row A: the mgcv oracle
def test_a_thin_level_shows_mgcvs_bayesian_standard_errors():
    """A spline by a factor with a thin level: every SE is printed, and is mgcv's ``Vp``.

    The fixture is mgcv 1.9-3 on superglm's own design and penalties at pinned smoothing
    parameters and scale (``paraPen``), so both engines solve one penalised problem; it reports
    all 50 coefficients and drops none.  Refit here at those lambdas, the raw standard errors
    are the dense inverse's (condition about 40, so ``kappa u`` ~ 1e-14; rtol 1e-10), and
    ``SE^2 / phi`` is mgcv's ``Vp / phi`` to 1e-6: the refit's SSP whitening is formed at the
    pinned lambdas, the export's at REML's start, so its slope columns differ by up to 2e-7,
    which moves their variances by up to 4e-8.  The data alone leave 18 coefficients open
    (level E's basis functions past its rows, and what they alias), which the SVD of the design
    and the fixture's record of master agree on.  Master blanked those 18 and the ``f[E]`` row.
    """
    fixture = json.loads((FIXTURES / "mgcv_factor_by_spline_gamma.json").read_text("utf-8"))
    frame, y = _thin_level()
    model = _thin_level_model()
    model.lambda2 = dict(fixture["lambda_solver"])
    _fit(model, frame, y, method="fit", tol=1e-12, max_iter=500)
    metrics = model.metrics(frame, y)
    se = np.r_[metrics.intercept_se, _flat(model, metrics.coefficient_se)]
    open_to_the_data = data_undetermined(model, frame)
    np.testing.assert_array_equal(open_to_the_data, ~np.asarray(fixture["master_estimable"])[1:])
    assert open_to_the_data.sum() == 18

    assert np.all(np.isfinite(se)), f"{int(np.isnan(se).sum())} standard errors blank"
    rows = {row.name: row.se for row in model.summary()._coef_rows if not row.is_spline}
    assert all(value is not None and np.isfinite(value) for value in rows.values())
    np.testing.assert_allclose(
        _flat(model, metrics.coefficient_se_raw),
        dense_raw_standard_errors(model, frame),
        rtol=1e-10,
    )
    phi = model.result.phi
    vp = np.asarray(fixture["mgcv_se_vp"]) ** 2 / fixture["phi"]
    np.testing.assert_allclose(se**2 / phi, vp, rtol=1e-6)
    beta = np.r_[model.result.intercept, model.result.beta]
    mgcv = np.asarray(fixture["mgcv_coef"])
    assert np.max(np.abs(beta - mgcv)) <= 1e-6 * np.max(np.abs(mgcv))
    assert model.result.effective_df == pytest.approx(fixture["measured"]["edf_mgcv"], rel=1e-8)


# ----------------------------------------------------------------- row B': a penalised factor
def test_a_penalized_factor_shows_a_standard_error_for_every_level():
    """A ``RandomEffect`` with a level that has no claims and a level that has no rows.

    The ridge determines every level: the data alone cannot (the indicators sum to the
    intercept), so master blanked all seven.  Raw standard errors are the dense inverse's
    (scaled condition about 60; rtol 1e-9), ``coefficient_se`` is that times the Pearson scale,
    and the level with no rows shows its prior standard deviation ``sqrt(1 / lambda)`` exactly,
    being reached by nothing but the penalty.  Among the levels with rows the one without
    claims is the least certain.  A copy of the training rows and held-out rows follow the same
    rule: a copy re-forms the same Hessian, and held-out rows have a positive definite one too.
    """
    frame, y = _zero_claim_levels(levels="ABCDEFZ")
    model = _random_effect_model()
    assert _fit(model, frame, y) == []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = model.metrics(frame, y)
        raw, se = metrics.coefficient_se_raw, metrics.coefficient_se
    assert np.all(np.isfinite(se["veh"])), "blank standard errors on a penalised factor"
    reference = dense_raw_standard_errors(model, frame)
    np.testing.assert_allclose(_flat(model, raw), reference, rtol=1e-9)
    scale = _pearson_scale(model, frame, y)
    np.testing.assert_allclose(_flat(model, se), np.sqrt(scale) * reference, rtol=1e-9)

    component = next(c for c in model._reml_penalties if c.group_name == "veh")
    prior_sd = np.sqrt(1.0 / model._reml_lambdas[component.name])
    assert raw["veh"][6] == pytest.approx(prior_sd, rel=1e-12)
    assert np.argmax(raw["veh"][:6]) == 5

    copy = model.metrics(frame.copy(), y.copy()).coefficient_se
    np.testing.assert_allclose(_flat(model, copy), _flat(model, se), rtol=1e-9)
    held_out, y_held_out = _zero_claim_levels(4000, levels="ABCDEFZ")
    assert np.all(np.isfinite(_flat(model, model.metrics(held_out, y_held_out).coefficient_se)))


def test_a_reml_stopped_early_shows_the_standard_errors_of_the_penalty_it_reached():
    """``fit_reml`` out of iterations: its last coefficient solve converged, so the rule applies.

    One outer iteration leaves the smoothing parameters short of their optimum, and the fit
    says so (``ConvergenceWarning``; ``diagnostics`` reports it as not converged).  The
    coefficients are still the penalised optimum at the smoothing parameters reached, which is
    what the rule needs, so every level of the random effect shows the dense inverse's standard
    error at those parameters (rtol 1e-9); master blanked all seven.  The coefficient solve's
    own convergence is what gates the rule (``not_converged`` below).
    """
    frame, y = _zero_claim_levels(900, levels="ABCDEFZ")
    model = _random_effect_model()
    caught = {type(w.message).__name__ for w in _fit(model, frame, y, max_reml_iter=1)}
    assert "ConvergenceWarning" in caught and model.result.converged
    assert not model.diagnostics()["_model"]["converged"]
    raw = _flat(model, model.metrics(frame, y).coefficient_se_raw)
    np.testing.assert_allclose(raw, dense_raw_standard_errors(model, frame), rtol=1e-9)


# -------------------------------------------------------------- row C1: fully penalised smooths
@pytest.mark.parametrize(
    ("basis", "select", "blank_on_master"),
    [
        ("fs", False, 1209),
        # control: ``sz(select=True)`` penalises its level lines too, and its data are full rank,
        # so master showed every one of these already
        ("sz", True, 0),
    ],
)
def test_a_fully_penalized_factor_smooth_shows_every_standard_error(basis, select, blank_on_master):
    """``fs`` and ``sz(select=True)`` at 200 levels, five of them without a claim.

    Every coefficient is penalised, so no level can separate and the structured factor
    truncates nothing.  Master blanked the main spline and all 1,200 ``fs`` coefficients, whose
    level curves alias the main spline in the data.  Raw standard errors are the dense
    inverse's at ``p`` about 1,200 (scaled condition under 1e4; rtol 1e-8).
    """
    frame, y = _factor_smooth()
    model = SuperGLM(
        family="poisson",
        features={"z": Numeric(), "x": Spline(kind="ps", n_knots=6)},
        interactions=[FactorSmooth("x", group="g", basis=basis, k=6, select=select)],
        selection_penalty=0,
    )
    assert _fit(model, frame, y) == []
    assert model._fit_inference_info.get("structured_covariance")
    raw = _flat(model, model.metrics(frame, y).coefficient_se_raw)
    assert int(np.isnan(raw).sum()) == 0, (
        f"{int(np.isnan(raw).sum())} blank; master {blank_on_master}"
    )
    np.testing.assert_allclose(raw, dense_raw_standard_errors(model, frame), rtol=1e-8)


# ------------------------------------------------------- row K: fits the rule does not reach
_UNPRICED_REGION_CASES = (
    "separated_level",
    "weak_numeric_level",
    "weak_unscanned_level",
    "weak_gaussian_level",
    "not_converged",
)


_RECORDED_SEPARATION_CASES = (
    "reference_m20",
    "reference_m21",
    "reference_tol",
    "slope",
    "cell",
    "varying_slope",
    "curve",
    "crossed_cell",
    "base_cell",
    "crossed_cell_fit",
)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize(
    "case",
    [
        "monotone",
        "monotone_reml",
        "shape_repair",
        "selection_penalty",
        "exact_alias",
        "past_the_bar",
        "separated_level",
        "scanned_level",
        "runtime_finding",
        "weak_numeric_level",
        "weak_unscanned_level",
        "weak_gaussian_level",
        "not_converged",
        *_RECORDED_SEPARATION_CASES,
    ],
)
def test_a_fit_outside_the_rule_keeps_the_data_rule(case):
    """Controls: each fit shows a standard error exactly where the data alone determine one.

    They pass on master, and must after it: the rule that shows every standard error of a
    certified fit reaches none of them.  The reference is the data's null space from a dense
    SVD, not the fit's record.

    - ``monotone``, ``monotone_reml``: a shape-constrained fit, with a row binding and with
      none.  Its penalised Hessian is positive definite (Cholesky, condition about 100), but the
      published coefficients are the constrained optimum.
    - ``shape_repair``: an unconstrained fit of a rising curve that dips twice, which
      ``apply_shape_postfit`` then projects onto an increasing one.  Its groups carry no shape
      rule and the Gamma weights do not move, so the fit's rank record survives the repair,
      but the published coefficients are no longer the optimum that record describes.
    - ``selection_penalty``: the recorded Hessian carries the selection penalty's local
      curvature, which is not a quadratic penalty's.
    - ``exact_alias``: a copied column leaves the penalised Hessian singular.
    - ``past_the_bar``: a column within 1e-7 of another leaves it full rank at a scaled
      condition of 2e7, past the 1.19e7 a Gram is trusted to, so the fit's record is the row
      factor's and is not a certificate.
    - ``separated_level`` and the cases after it: a level or region with rows and no finite
      estimate (``_unpriced_region``, ``_unpriced_level``).  Its coefficient walks towards
      infinity and stops where the penalised Hessian is positive definite and well
      conditioned (the region's record has condition 3): no factorisation shows it, and each
      case is named by a different record of the fit.
    - ``reference_m20`` and the cases after it (``_recorded_separation``): the walked
      coefficient is one the data rule already left blank, behind an exact alias with a
      penalised term.  The penalty breaks the alias and holds every data-null direction, so
      neither the rank certificate nor the penalty's curvature sees the walk.  Where the walk
      runs to rounding (the ``reference`` cases and ``base_cell``) the rows' weights span
      thirteen orders, where an SVD and the fit's data rank can cut differently, so the
      reference is the fit's own record of its data rank, which this rule does not write.
    """
    open_to_the_data, options = None, {}
    if case in _RECORDED_SEPARATION_CASES:
        frame, y, model, method, options = _recorded_separation(case)
        _fit(model, frame, y, method=method, **options)
        assert model.result.converged
        if case.startswith("reference") or case == "base_cell":
            open_to_the_data = ~model.result.rank_info.coefficient_estimable()
        options = {key: value for key, value in options.items() if key == "offset"}
    elif case in _UNPRICED_REGION_CASES:
        frame, y, model, event = _unpriced_region(case)
        assert event in {type(w.message).__name__ for w in _fit(model, frame, y)}
        assert model.result.converged is (case != "not_converged")
    elif case in ("scanned_level", "runtime_finding"):
        frame, y, model = _unpriced_level(case)
        caught = {type(w.message).__name__ for w in _fit(model, frame, y, method="fit")}
        assert "SeparationWarning" in caught and model.result.converged
        # Every row has a positive weight, so the rows leave open what their design does at
        # unit weights; the walked level's fitted weights underflow to zero, which would add
        # its own column.
        open_to_the_data = undetermined_slopes(_design(model))
    elif case == "shape_repair":
        frame, _ = _thin_level()
        u = frame["x"].to_numpy() / 80.0
        y = np.exp(2.0 * u + 0.8 * np.sin(4.0 * np.pi * u))
        y = y * np.random.default_rng(11).gamma(5.0, 0.2, len(frame))
        model = _thin_level_model(
            Spline(kind="ps", n_knots=8, constraint=Constraint.postfit.increasing)
        )
        _fit(model, frame, y, method="fit")
        unconstrained = np.asarray(model.predict(frame), dtype=np.float64)
        model.apply_shape_postfit(frame)
        assert not np.allclose(model.predict(frame), unconstrained, rtol=0.1)
    elif case in ("exact_alias", "past_the_bar"):
        frame, y = _zero_claim_levels()
        noise = np.random.default_rng(5).normal(size=len(frame))
        near = 2.0 * frame["dens"] if case == "exact_alias" else frame["dens"] + 1e-7 * noise
        frame = frame.assign(near=near)
        model = _random_effect_model(near=Numeric())
        _fit(model, frame, y)
        decomposition = model.result.rank_info.augmented
        assert (decomposition.method, decomposition.rank < decomposition.width) == (
            "qr_svd",
            case == "exact_alias",
        )
    else:
        frame, y = _thin_level()
        monotone = Spline(kind="ps", n_knots=8, constraint=Constraint.fit.increasing)
        model = (
            _thin_level_model(selection_penalty=0.001)
            if case == "selection_penalty"
            else _thin_level_model(monotone)
        )
        _fit(model, frame, y, method="fit_reml" if case == "monotone_reml" else "fit")
    if open_to_the_data is None:
        open_to_the_data = data_undetermined(model, frame)
    assert open_to_the_data.any()
    for rows in (frame, frame.copy()):
        se = _flat(model, model.metrics(rows, y, **options).coefficient_se)
        np.testing.assert_array_equal(np.isnan(se), open_to_the_data)


@pytest.mark.pins_behaviour
@pytest.mark.parametrize(
    ("levels", "solve", "seed"),
    [
        *(pytest.param("no_claims", "auto", seed, id=f"no_claims-{seed}") for seed in range(1, 9)),
        pytest.param("one_row", "auto", 1, id="one_row"),
        pytest.param("five_levels", "gram", 5, id="five_levels-gram"),
    ],
)
def test_an_sz_level_the_data_cannot_price_keeps_its_term_blank(levels, solve, seed):
    """Controls: an ``sz`` term with unpenalised level lines and one level the data cannot price.

    They pass on master, and must after it.  A level with exposure and no claim (``no_claims``,
    forty levels; ``five_levels``, one level in five on the dense solver) has a line that walks
    towards minus infinity, and a level with one row (``one_row``) has a line its row cannot
    fix.  Either reaches the main spline and every other level's line through the sum-to-zero
    constraint, so the fit shows no standard error for the main spline or the term, and shows
    the one for ``z`` where the model has it.  The fit stops where the penalised Hessian is
    numerically positive definite: the structured factor truncates no direction in any of
    these, and the dense Gram of ``five_levels`` has full rank at a record condition of 7.9e6,
    under the 1.19e7 bar, with one OpenBLAS kernel and rank 32 of 33 with another.  What names
    the level is the count made when the design is built, which is the same on every machine.
    """
    if levels == "five_levels":
        rng = np.random.default_rng(seed)
        n = 4000
        g = rng.choice(list("ABCDE"), n, p=[0.3, 0.3, 0.2, 0.15, 0.05])
        x = rng.uniform(0, 10, n)
        y = rng.poisson(np.exp(-1.5 + 0.2 * np.sin(x))).astype(float)
        y[g == "E"] = 0.0
        frame, features = pd.DataFrame({"x": x, "g": g}), {}
    else:
        frame, y = _factor_smooth(
            40, int(levels == "no_claims"), one_row=int(levels == "one_row"), seed=seed
        )
        features = {"z": Numeric()}
    model = SuperGLM(
        family="poisson",
        features={**features, "x": Spline(kind="ps", n_knots=6)},
        interactions=[FactorSmooth("x", group="g", basis="sz", k=6)],
        selection_penalty=0,
        direct_solve=solve,
    )
    _fit(model, frame, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        se = model.metrics(frame, y).coefficient_se
    term = next(group.name for group in model._groups if group.name not in ("x", "z"))
    assert all(np.isfinite(se[name]).all() for name in features)
    assert np.isnan(se["x"]).all(), f"main spline: {np.isfinite(se['x']).sum()} shown"
    assert np.isnan(se[term]).all(), f"{term}: {np.isfinite(se[term]).sum()} shown"


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("rows", ["new_rows", "other_weights"])
def test_other_rows_do_not_certify_a_fit_its_own_rows_leave_undetermined(rows):
    """Controls: ``metrics`` on rows or weights that break an alias the training fit holds.

    They pass on master, and must after it.  ``dup`` is ``2 x`` on every training row that
    carries weight, so the fit cannot tell the two slopes apart and its penalised Hessian has
    rank 4 of 5.  The Hessian re-formed where the alias breaks is positive definite: on new
    rows where ``dup`` moves with a level of the random effect, and on the training rows at
    weights that reach the rows where ``dup`` differs, which the fit gave none.  That says
    nothing about the coefficients the fit published, so each reader keeps the data rule on
    its own rows.  The reference is the null space of those rows from a dense SVD.
    """
    rng = np.random.default_rng(912)
    n = 900
    g = rng.choice(list("ABC"), n)
    x = rng.normal(size=n)
    y = 1.0 + 0.4 * x + np.select([g == "B", g == "C"], [0.4, -0.4])
    y = y + rng.normal(scale=0.3, size=n)
    apart = rng.uniform(size=n) < 0.2
    if rows == "new_rows":
        frame = pd.DataFrame({"x": x, "dup": 2.0 * x, "g": g})
        weight = other_weight = None
        other = frame.assign(dup=2.0 * x + (g == "B"))
    else:
        frame = pd.DataFrame({"x": x, "dup": np.where(apart, rng.normal(size=n), 2.0 * x), "g": g})
        weight, other_weight = np.where(apart, 0.0, 1.0), np.where(apart, 2.0, 1.0)
        other = frame
    model = SuperGLM(
        family="gaussian",
        features={"x": Numeric(), "dup": Numeric(), "g": RandomEffect()},
    )
    _fit(model, frame, y, sample_weight=weight)
    own = model.metrics(frame, y, sample_weight=weight).coefficient_se
    assert np.isnan(own["x"]).all() and np.isnan(own["dup"]).all()

    columns = [np.ones(n), other["x"], other["dup"], *(g == level for level in "ABC")]
    root = np.ones(n) if other_weight is None else np.sqrt(other_weight)
    open_to_the_rows = undetermined_slopes(
        root[:, None] * np.column_stack(columns).astype(np.float64)
    )
    assert open_to_the_rows.sum() == {"new_rows": 5, "other_weights": 3}[rows]
    se = _flat(model, model.metrics(other, y, sample_weight=other_weight).coefficient_se)
    np.testing.assert_array_equal(np.isnan(se), open_to_the_rows)


@pytest.mark.parametrize("perturbation", [1.8e-7, 1e-7])
def test_other_rows_show_every_level_of_a_fit_whose_penalty_holds_what_its_data_leave(perturbation):
    """``metrics`` on rows that tell two training columns apart shows every level's standard error.

    ``dup`` is ``x + perturbation z`` on the training rows, which puts the fit's penalised Gram
    at the bar a Gram is trusted to: at 1.8e-7 one OpenBLAS kernel records the Gram (condition
    1.18e7) and another the row factor, and at 1e-7 every kernel records the row factor.  The
    training rows leave one direction open, the three levels against the intercept, and the
    ridge holds it, so the coefficients the fit published are determined whichever
    decomposition recorded them.  On rows where ``dup`` is a column of its own the re-formed
    Hessian is well conditioned, and each level's raw standard error is the dense inverse's of
    ``[1 X]'[1 X] + S`` at the unit ridge (rtol 1e-10).  A reader that asks the training fit
    for its Gram's certificate shows them with one kernel and blanks them with the other.
    """
    rng = np.random.default_rng(32)
    n = 420
    x, z = rng.normal(size=n), rng.normal(size=n)
    g = rng.choice(list("ABC"), n)
    y = x + 0.37 * z + rng.normal(size=n)
    model = SuperGLM(
        family="gaussian",
        features={
            "x": Numeric(),
            "dup": Numeric(),
            "g": RandomEffect(lambda_policy=LambdaPolicy.fixed(1.0)),
        },
    )
    _fit(model, pd.DataFrame({"x": x, "dup": x + perturbation * z, "g": g}), y)
    se = model.metrics(pd.DataFrame({"x": x, "dup": z, "g": g}), y).coefficient_se_raw["g"]

    design = np.column_stack((np.ones(n), x, z, *(g == level for level in "ABC")))
    hessian = design.T @ design
    hessian[-3:, -3:] += np.eye(3)
    np.testing.assert_allclose(se, np.sqrt(np.diag(np.linalg.inv(hessian))[-3:]), rtol=1e-10)


@pytest.mark.pins_behaviour
def test_metrics_on_other_rows_keeps_no_factor_for_the_saved_models_note():
    """Control: what a ``metrics`` object on other rows holds once its standard errors are read.

    It passes on master, and must after it.  Four hundred unpenalised levels on reordered rows,
    so the inverse is re-formed at ``p`` of 399.  The object keeps three ``p x p`` arrays for
    its readers (the inverse, the augmented covariance and the centred Gram), 3.05 times
    ``8 p^2`` bytes with its row vectors, as on master.  The decomposition they were formed
    from decides a saved model's note while it is in hand, and only that answer is kept:
    holding its Cholesky factor as well is a fourth array, 4.05.
    """
    levels = 400
    rng = np.random.default_rng(274)
    frame = pd.DataFrame({"g": np.tile(np.arange(levels).astype(str), 5)})
    y = rng.normal(size=len(frame))
    model = SuperGLM(family="gaussian", features={"g": Categorical(base="0")}, selection_penalty=0)
    model.fit(frame, y)
    other, y_other = frame.iloc[::-1].reset_index(drop=True), y[::-1].copy()

    def held_after_reading() -> tuple[int, int]:
        metrics = model.metrics(other, y_other)
        shown = int(np.isfinite(metrics.coefficient_se["g"]).sum())
        gc.collect()
        return tracemalloc.get_traced_memory()[0], shown

    _, (held, shown) = own_traced_peak(held_after_reading)
    assert shown == levels - 1
    assert held < 3.5 * 8 * (levels - 1) ** 2, f"{held / (8 * (levels - 1) ** 2):.2f} arrays held"


# ------------------------------------------------------------ row F: models saved by master
@pytest.mark.parametrize("solver", ["dense", "structured"])
@pytest.mark.parametrize("state", ["cold", "warm", "compact"])
def test_a_model_saved_by_master_keeps_its_blanks_and_says_a_refit_shows_them(solver, state):
    """Pickles written by master bce2910e5 (``scripts/make_saved_standard_error_fixtures.py``).

    Each is penalised to full rank, so a fit today shows every standard error; master blanked
    6 of 14 (dense) and 129 of 130 (structured).  Saved before any reader ran (``cold``), after
    ``summary()`` and ``metrics()`` cached their inference (``warm``) and with
    ``retain_fit_state=False`` (``compact``), the model predicts as saved and shows the blanks
    master showed, in ``metrics`` and in ``summary``, with the shown values master's within the
    covariance's rounding (rtol 1e-9).  Both summaries say a refit shows the rest, which is
    what fails on master.  Another save and load changes nothing, and a refit shows them all.
    """
    with lzma.open(SAVED / f"{solver}_{state}.pkl.xz", "rb") as handle:
        record = pickle.load(handle)
    assert record["commit"] == "bce2910e5" and record["state"] == state
    model, frame, y = record["model"], record["frame"], record["y"]
    saved = np.asarray(record["metrics_se"], dtype=np.float64)
    assert np.isnan(saved).sum() == {"dense": 6, "structured": 129}[solver]
    assert_predicts_as_saved(model, frame, record["prediction"])

    def blank_rows(summary) -> list[str]:
        return [
            row.name
            for row in summary._coef_rows
            if not row.is_spline
            and row.structured_kind is None
            and row.name != "Intercept"
            and (row.se is None or row.se != row.se)
        ]

    for loaded in (model, pickle.loads(pickle.dumps(model))):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = loaded.metrics(frame, y)
            se = _flat(loaded, metrics.coefficient_se)
            summaries = (loaded.summary(), metrics.summary())
        np.testing.assert_array_equal(np.isnan(se), np.isnan(saved))
        np.testing.assert_allclose(se[~np.isnan(saved)], saved[~np.isnan(saved)], rtol=1e-9)
        for summary in summaries:
            assert blank_rows(summary) == record["summary_blank_rows"]
            assert REFIT_NOTE in " ".join(str(summary).split())
            assert REFIT_NOTE in summary._repr_html_()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
        refit = _flat(model, model.metrics(frame, y).coefficient_se)
        text = str(model.summary())
    assert np.all(np.isfinite(refit)) and REFIT_NOTE not in " ".join(text.split())


def test_a_saved_models_note_counts_the_standard_errors_a_refit_shows():
    """A saved model's note is given where a refit shows every blank, and counts them.

    The dense fixture on three sets of rows, each read again after a refit on the training rows:

    - the training rows: its 6 blanks are named, and the refit shows them;
    - rows whose level F is relabelled to a level the fit never saw: the other levels no longer
      sum to the intercept and no row holds F, so F's standard error is the one blank.  The
      note is in the singular, and the refit shows it;
    - rows whose ``dens`` is constant: the Hessian re-formed there is singular, so the refit
      leaves the same 7 blank, and the note is not given.
    """
    with lzma.open(SAVED / "dense_cold.pkl.xz", "rb") as handle:
        record = pickle.load(handle)
    model, frame, y = record["model"], record["frame"], record["y"]
    without_f = frame.assign(veh=frame["veh"].where(frame["veh"] != "F", "Q"))
    constant = frame.assign(dens=0.0)

    def read(rows) -> tuple[int, str]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = model.metrics(rows, y)
            blank = int(np.isnan(_flat(model, metrics.coefficient_se)).sum())
            return blank, " ".join(str(metrics.summary()).split())

    blank, text = read(frame)
    assert blank == 6 and "6 standard errors are blank because" in text and REFIT_NOTE in text
    blank, text = read(without_f)
    assert blank == 1 and "1 standard error is blank because" in text
    assert "Refit the model to show it." in text
    blank, text = read(constant)
    assert blank == 7 and "Refit the model" not in text

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    assert [read(rows)[0] for rows in (frame, without_f, constant)] == [0, 0, 7]
