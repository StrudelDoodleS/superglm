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

import json
import lzma
import pickle
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
    Numeric,
    RandomEffect,
    Spline,
    SuperGLM,
)
from superglm.model.fit_state import fitted_lambda2
from superglm.reml.penalty_algebra import build_penalty_matrix

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


def data_undetermined(model, frame) -> np.ndarray:
    """The slopes a null vector of the data ``sqrt(W) [1, X]`` moves: what the data alone leave open.

    A dense SVD of the column-equilibrated weighted design.  A singular value at or below
    ``max(n, p) eps`` of the largest is null (``dgesdd``'s backward error); a slope a null
    vector moves by more than ``sqrt(eps)`` is undetermined.  The fixtures here are deficient
    by construction (a level's indicator against the intercept, a copied column, basis
    functions with no row), so every other singular value is orders of magnitude clear.
    """
    A = np.sqrt(_working_weights(model, frame))[:, None] * _design(model)
    A = A / np.maximum(np.linalg.norm(A, axis=0), np.finfo(float).tiny)
    _, singular, right = np.linalg.svd(A, full_matrices=False)
    null = right[singular <= max(A.shape) * EPS * singular[0]]
    return np.any(np.abs(null[:, 1:]) > np.sqrt(EPS), axis=0)


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


def _factor_smooth(levels=200, zero=5):
    """Twenty Poisson rows a level on average; the first ``zero`` levels have no claim."""
    rng = np.random.default_rng(2)
    n = 20 * levels
    g = rng.integers(0, levels, n)
    x = rng.uniform(size=n)
    frame = pd.DataFrame({"x": x, "z": rng.normal(size=n), "g": [f"g{v:04d}" for v in g]})
    y = rng.poisson(np.exp(-1.0 + 0.5 * np.sin(3 * x) + 0.2 * np.sin(5 * x + g % 7))).astype(float)
    y[g < zero] = 0.0
    return frame, y


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
@pytest.mark.parametrize(
    "case",
    ["monotone", "monotone_reml", "selection_penalty", "exact_alias", "past_the_bar"],
)
def test_a_fit_outside_the_rule_keeps_the_data_rule(case):
    """Controls: each fit shows a standard error exactly where the data alone determine one.

    They pass on master, and must after it.  The reference is the data's null space from a
    dense SVD, not the fit's record.

    - ``monotone``, ``monotone_reml``: a shape-constrained fit, with a row binding and with
      none.  Its penalised Hessian is positive definite (Cholesky, condition about 100), but the
      published coefficients are the constrained optimum.
    - ``selection_penalty``: the recorded Hessian carries the selection penalty's local
      curvature, which is not a quadratic penalty's.
    - ``exact_alias``: a copied column leaves the penalised Hessian singular.
    - ``past_the_bar``: a column within 1e-7 of another leaves it full rank at a scaled
      condition of 2e7, past the 1.19e7 a Gram is trusted to, so the fit's record is the row
      factor's and is not a certificate.
    """
    if case in ("exact_alias", "past_the_bar"):
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
    open_to_the_data = data_undetermined(model, frame)
    assert open_to_the_data.any()
    for rows in (frame, frame.copy()):
        se = _flat(model, model.metrics(rows, y).coefficient_se)
        np.testing.assert_array_equal(np.isnan(se), open_to_the_data)


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
