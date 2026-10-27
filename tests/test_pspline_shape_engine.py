"""A fit-time shape constraint on a P-spline fits on the QP; SCOP is an opt-in.

``shape_engine`` (``features/_spline_build.py``) decides the engine everywhere:
the QP, on first-difference (monotone) or exact curvature rows on the raw
B-spline coefficients, the rows ``bs`` uses; SCOP only for a ``PSpline``
declared with ``shape_engine="scop"`` or pickled before that option existed.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import Constraint, PSpline, Spline, SuperGLM, _roundoff, families
from superglm.features._spline_build import shape_engine
from superglm.features.spline import BSplineSmooth
from superglm.solvers import irls_direct

from .test_saved_fs_models import assert_predicts_as_saved

SAVED_SCOP = Path(__file__).parent / "fixtures" / "saved_v0_40_0" / "glm_poisson_scop.pkl"
FAMILIES = {
    "gaussian": families.gaussian,
    "poisson": families.poisson,
    "gamma": families.gamma,
    "binomial": families.binomial,
    "tweedie": lambda: families.tweedie(p=1.5),
}


def _book(family: str, n: int = 600, seed: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    """An increasing trend with a dip at x = 6, so an increasing fit holds rows active there."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    eta = 0.15 * x - 1.5 * np.exp(-((x - 6.0) ** 2) / 2.0)
    if family == "gaussian":
        y = eta + rng.normal(0.0, 0.3, n)
    elif family == "gamma":
        y = rng.gamma(3.0, np.exp(eta) / 3.0)
    elif family == "binomial":
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    elif family == "tweedie":
        claims = rng.poisson(0.5 * np.exp(eta))
        y = np.array([rng.gamma(2.0, 1.0, k).sum() for k in claims])
    else:
        y = rng.poisson(np.exp(eta)).astype(float)
    return pd.DataFrame({"x": x}), y


def _fit(family: str, spec, *, method: str = "fit_reml", discrete: bool = False) -> SuperGLM:
    frame, y = _book(family)
    model = SuperGLM(
        family=FAMILIES[family](), selection_penalty=0, discrete=discrete, features={"x": spec}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(model, method)(frame, y)
    return model


def _group(model: SuperGLM):
    return next(group for group in model._groups if group.feature_name == "x")


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_ps_monotone_rows_hold(family: str, discrete: bool) -> None:
    """The QP's rows hold on the published raw coefficients ``c = R_inv beta``.

    The QP certifies each row ``A_i beta >= 0`` to its feasibility tolerance
    plus ``gamma_(p+2)``, in row-action units; forming ``c`` and ``A c`` adds
    the rounding of ``K + p`` term products, so ``|A| |R_inv| |beta|`` bounds
    the action of every step. Fails when a P-spline routes to SCOP, whose fit
    carries no rows.
    """
    model = _fit(
        family, PSpline(n_knots=10, constraint=Constraint.fit.increasing), discrete=discrete
    )
    group, spec = _group(model), model._specs["x"]
    assert model.result.converged
    assert group.constraints is not None
    beta = model.result.beta[group.start : group.end]
    rows = spec._build_monotone_constraints_raw().A
    action = rows @ (spec._R_inv @ beta)
    reach = np.abs(rows) @ np.abs(spec._R_inv) @ np.abs(beta)
    width = rows.shape[1] + beta.size + 2
    bound = (irls_direct._QP_FEASIBILITY_TOL + _roundoff.gamma(width)) * reach
    assert np.all(action >= -bound), float(np.min(action / reach))
    assert np.any(action <= bound), "the dip holds no row active: the fixture tests nothing"


def test_ps_default_engine_is_qp() -> None:
    """A constrained P-spline, declared either way, fits on the QP and keeps no SCOP state."""
    for spec in (
        PSpline(n_knots=8, constraint=Constraint.fit.convex),
        Spline(kind="ps", k=12, constraint=Constraint.fit.convex),
    ):
        assert spec.shape_engine == "qp" and shape_engine(spec) == "qp"
        model = _fit("poisson", spec, method="fit")
        assert _group(model).monotone_engine == "qp"
        assert getattr(model._specs["x"], "_scop_Sigma", None) is None


# Master 973ca2cd's SCOP fit of ``_book("poisson")`` (``fit_reml``): deviance,
# EDF, and the prediction at x = 0.5, 2.75, 5, 7.25, 9.5. They agree between
# AVX-512 and Haswell kernels to 3e-15 relative; the same fit on the QP is
# 4e-3 away in deviance. Agreement to 1e-9 is therefore master's SCOP fit.
MASTER_SCOP = (
    787.1287586104952,
    4.449902735241759,
    [
        0.9994423789743592,
        1.1924052181169178,
        1.2016693636991418,
        1.726399582876236,
        3.5873923611520353,
    ],
)


@pytest.mark.parametrize("declare", ["PSpline", "Spline"])
def test_ps_scop_opt_in(declare: str) -> None:
    """``shape_engine="scop"`` fits on SCOP and reproduces master's SCOP fit."""
    constraint = Constraint.fit.increasing
    spec = (
        PSpline(n_knots=8, constraint=constraint, shape_engine="scop")
        if declare == "PSpline"
        else Spline(kind="ps", n_knots=8, constraint=constraint, shape_engine="scop")
    )
    assert shape_engine(spec) == "scop"
    model = _fit("poisson", spec)
    assert _group(model).monotone_engine == "scop"
    assert model._specs["x"]._scop_Sigma is not None
    deviance, edf, prediction = MASTER_SCOP
    grid = pd.DataFrame({"x": np.linspace(0.5, 9.5, 5)})
    np.testing.assert_allclose(model.result.deviance, deviance, rtol=1e-9)
    np.testing.assert_allclose(model.result.effective_df, edf, rtol=1e-9)
    np.testing.assert_allclose(model.predict(grid), prediction, rtol=1e-9)


def test_shape_engine_scop_needs_a_p_spline() -> None:
    with pytest.raises(ValueError, match="needs kind='ps'"):
        Spline(kind="bs", k=8, constraint=Constraint.fit.increasing, shape_engine="scop")
    with pytest.raises(ValueError, match="needs kind='ps'"):
        Spline(k=8, constraint=Constraint.fit.increasing, shape_engine="scop")
    with pytest.raises(ValueError, match="shape_engine must be one of"):
        PSpline(constraint=Constraint.fit.increasing, shape_engine="sco")


def test_saved_scop_model_predicts_and_refits_on_scop() -> None:
    """A P-spline saved before ``shape_engine`` existed predicts as saved and refits on SCOP.

    Its fitted spec and the unfitted declaration a refit starts from both lack
    the attribute, so both read the class default and stay on SCOP. Fails when
    ``shape_engine`` sends that default to the QP.
    """
    with open(SAVED_SCOP, "rb") as handle:
        record = pickle.load(handle)
    model, frame, y = record["model"], record["frame"], record["y"]
    fitted, declared = model._specs["x"], dict(model._config.feature_templates)["x"]
    assert "shape_engine" not in fitted.__dict__ and "shape_engine" not in declared.__dict__
    assert shape_engine(fitted) == shape_engine(declared) == "scop"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_predicts_as_saved(model, frame, record["prediction"])
        model.fit_reml(frame, y)
    assert _group(model).monotone_engine == "scop"
    assert model._specs["x"]._scop_Sigma is not None


def test_stale_scop_state_cleared() -> None:
    """A spec rebuilt on the QP after SCOP scores through its ``R_inv``, not the old SCOP basis.

    Fails when the build keeps the ``_scop_*`` state a SCOP build left.
    """
    x = np.random.default_rng(5).uniform(0.0, 10.0, 200)
    spec = PSpline(n_knots=6, constraint=Constraint.fit.increasing, shape_engine="scop")
    assert spec.build(x).monotone_engine == "scop" and spec._scop_Sigma is not None
    spec.shape_engine = "qp"
    info = spec.build(x)
    assert info.monotone_engine == "qp"
    r_inv = np.random.default_rng(6).standard_normal((spec._n_basis, info.n_cols))
    spec.set_reparametrisation(r_inv)
    np.testing.assert_array_equal(spec.transform(x), spec._basis_matrix(x).toarray() @ r_inv)


@pytest.mark.parametrize("kind", ["increasing", "decreasing", "convex", "concave"])
def test_ps_qp_matches_bs_qp_rows(kind: str) -> None:
    """P-spline and ``bs`` state identical rows on identical knots; only the penalty differs."""
    x = np.random.default_rng(7).uniform(-2.0, 3.0, 300)
    constraint = getattr(Constraint.fit, kind)
    ps, bs = (
        PSpline(n_knots=7, constraint=constraint),
        BSplineSmooth(n_knots=7, constraint=constraint),
    )
    ps_info, bs_info = ps.build(x), bs.build(x)
    np.testing.assert_array_equal(ps._knots, bs._knots)
    np.testing.assert_array_equal(
        ps._build_monotone_constraints_raw().A, bs._build_monotone_constraints_raw().A
    )
    np.testing.assert_array_equal(ps_info.constraints.A, bs_info.constraints.A)


def test_warm_qp_takes_one_step_per_changed_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each IRLS step's QP starts from the last active set.

    An active-set solve takes one step per row entering or leaving its working
    set, one on the set it starts from, and one that certifies it. Warm, that is
    the change from the previous solve plus two; cold, every active row enters
    again. Fails when every QP starts cold.
    """
    calls = []
    solve = irls_direct.solve_constrained_qp

    def counted(*args, **kwargs):
        result = solve(*args, **kwargs)
        calls.append((set(result.active_set), result.n_iter))
        return result

    monkeypatch.setattr(irls_direct, "solve_constrained_qp", counted)
    _fit("poisson", PSpline(n_knots=10, constraint=Constraint.fit.increasing), method="fit")
    assert len(calls) >= 2 and len(calls[0][0]) >= 3
    for (before, _), (after, steps) in zip(calls, calls[1:], strict=False):
        assert steps <= len(before ^ after) + 2, (sorted(before), sorted(after), steps)
