"""A fit-time shape constraint on a P-spline fits on the QP; SCOP is retired.

The QP fits first-difference (monotone) or exact curvature rows on the raw
B-spline coefficients, the rows ``bs`` uses. ``shape_engine="scop"`` names the
engine 0.41.0 used: it warns and fits on the QP. A model fitted on SCOP still
loads, predicts and reports from what it saved, and its refit goes to the QP.
"""

from __future__ import annotations

import copy
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Constraint,
    FeatureStructure,
    PSpline,
    Spline,
    Structure,
    SuperGLM,
    _roundoff,
    families,
)
from superglm.features._spline_build import shape_engine
from superglm.features._spline_config import SCOPRetiredWarning
from superglm.features.spline import BSplineSmooth
from superglm.solvers import irls_direct
from superglm.solvers.irls_direct import _qp as irls_direct_qp

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


_DECLARATIONS = {
    "PSpline": lambda **engine: PSpline(n_knots=8, constraint=Constraint.fit.increasing, **engine),
    "Spline-ps": lambda **engine: Spline(
        kind="ps", n_knots=8, constraint=Constraint.fit.increasing, **engine
    ),
    "Spline-bs": lambda **engine: Spline(
        kind="bs", k=10, constraint=Constraint.fit.increasing, **engine
    ),
}


def _numeric_sum(obj) -> float:
    """Sum every number in a nested payload: frames, arrays, dicts, lists, objects."""
    if isinstance(obj, dict):
        return sum(_numeric_sum(obj[k]) for k in sorted(obj, key=str))
    if isinstance(obj, (list, tuple)):
        return sum(_numeric_sum(v) for v in obj)
    if hasattr(obj, "select_dtypes"):
        return float(np.nansum(np.asarray(obj.select_dtypes("number"), dtype=float)))
    if isinstance(obj, np.ndarray) and obj.dtype.kind in "fiu":
        return float(np.nansum(obj.astype(float)))
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    if hasattr(obj, "__dict__"):
        return _numeric_sum(vars(obj))
    return 0.0


@pytest.mark.parametrize(
    "declare",
    [
        lambda: PSpline(n_knots=8, shape_engine="scop"),
        lambda: PSpline(n_knots=8, constraint=Constraint.postfit.increasing, shape_engine="scop"),
        lambda: Spline(kind="cr", k=10, constraint=Constraint.fit.increasing, shape_engine="scop"),
    ],
    ids=["ps-unconstrained", "ps-postfit", "cr-constrained"],
)
def test_shape_engine_scop_says_no_effect_where_0_41_never_used_scop(declare) -> None:
    """Only a P-spline with a fit-time constraint ever fitted on SCOP; elsewhere the fit is unchanged.

    Mutation check: with ``moves_fit`` forced True in ``warn_scop_retired``, the
    message claims the fit differs from 0.41.0 and this test fails.
    """
    with pytest.warns(SCOPRetiredWarning) as caught:
        declare()
    (message,) = [str(w.message) for w in caught if issubclass(w.category, SCOPRetiredWarning)]
    assert "no effect" in message and "differ" not in message, message


@pytest.mark.parametrize("declare", sorted(_DECLARATIONS))
def test_shape_engine_scop_warns_once_and_fits_the_qp_fit(declare: str) -> None:
    """``shape_engine="scop"`` is accepted with one visible warning and fits exactly the QP fit.

    The warning is a ``UserWarning`` (Python shows it by default, unlike a
    ``DeprecationWarning``), points at the declaring line, and says what
    happened and how to silence it. Mutation checks: with the call in
    ``warn_scop_retired`` removed, or its category a ``DeprecationWarning``,
    the warning assertions fail; with ``PSpline`` storing ``"scop"`` instead
    of ``"qp"``, the stored engine assertion fails.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = _DECLARATIONS[declare](shape_engine="scop")
    retired = [w for w in caught if issubclass(w.category, SCOPRetiredWarning)]
    assert len(retired) == 1, [str(w.message) for w in caught]
    warning = retired[0]
    assert issubclass(warning.category, UserWarning)
    assert not issubclass(warning.category, DeprecationWarning)
    assert warning.filename == __file__
    message = str(warning.message)
    moved = declare != "Spline-bs"  # 0.41.0 fitted only a constrained P-spline on SCOP
    parts = ("QP engine", "0.41.0") if moved else ("no effect",)
    for part in ("retired", "Drop shape_engine='scop'", *parts):
        assert part in message, message
    assert ("differ" in message) is moved, message
    assert getattr(spec, "shape_engine", "qp") == "qp" and shape_engine(spec) == "qp"

    model = _fit("poisson", spec)
    reference = _fit("poisson", _DECLARATIONS[declare]())
    assert _group(model).monotone_engine == "qp"
    assert getattr(model._specs["x"], "_scop_Sigma", None) is None
    grid = pd.DataFrame({"x": np.linspace(0.0, 10.0, 41)})
    np.testing.assert_array_equal(model.predict(grid), reference.predict(grid))
    assert model.result.effective_df == reference.result.effective_df
    assert model.reml_diagnostics()["lambdas"] == reference.reml_diagnostics()["lambdas"]


def test_saved_scop_model_predicts_and_reports_as_saved_and_refits_on_the_qp() -> None:
    """A model 0.41.0 or earlier fitted on SCOP loads, predicts and reports from what it saved.

    Prediction scores through the saved SCOP transform (``_scop_Sigma``), and
    the standard errors read the saved ``SCOPInferenceInfo``: the reference
    here forms them from that covariance and the term's basis alone. Its
    refit warns and goes to the QP, which drops the SCOP transform so the new
    coefficients score through ``R_inv``. Mutation checks: without the
    ``_scop_Sigma`` branch in ``_spline_runtime.transform`` the prediction
    fails; without the ``scop_inference`` branch in ``state_ops`` the standard
    errors fail; without ``_warn_if_saved_on_scop`` the warnings fail; without
    ``clear_scop_state`` the rebuilt fitted spec keeps the old transform.
    """
    with open(SAVED_SCOP, "rb") as handle:
        record = pickle.load(handle)
    model, frame, y = record["model"], record["frame"], record["y"]
    fitted, declared = model._specs["x"], dict(model._config.feature_templates)["x"]
    assert "shape_engine" not in fitted.__dict__ and "shape_engine" not in declared.__dict__
    assert fitted._scop_Sigma is not None
    with warnings.catch_warnings():
        warnings.simplefilter("error", SCOPRetiredWarning)  # loading and reading never warn
        assert_predicts_as_saved(model, frame, record["prediction"])
        inference = model.term_inference("x")
        model.summary()
        metrics = model.metrics(frame, y)
        leverage, cooks = np.asarray(metrics.leverage), np.asarray(metrics.cooks_distance)
        from superglm.export.rating_tables import build_rating_table_payload

        payload = build_rating_table_payload(model, frame, y)
    # Master 71b133b2 (the SCOP engine) on this fixture: the saved model's
    # diagnostics and rating tables read what it saved, so they do not move.
    np.testing.assert_allclose(leverage.sum(), 5.0197227810892695, rtol=1e-9)
    np.testing.assert_allclose(cooks.sum(), 1.1744743077217192, rtol=1e-9)
    np.testing.assert_allclose(_numeric_sum(payload), 9388.896825679873, rtol=1e-9)
    group = _group(model)
    covariance = np.asarray(model._solver_result.scop_inference.augmented_inverse)[1:, 1:]
    covariance = covariance[group.start : group.end, group.start : group.end]
    basis = fitted.transform(np.asarray(inference.x))
    expected = np.sqrt(np.einsum("ij,jk,ik->i", basis, covariance, basis) * model.result.phi)
    np.testing.assert_allclose(inference.se_log_relativity, expected, rtol=64 * np.finfo(float).eps)

    # The fitted spec itself, rebuilt (as an editor or frozen-model refit does),
    # warns, goes to the QP and scores through its new ``R_inv``.
    x = frame["x"].to_numpy()
    rebuilt = copy.deepcopy(fitted)
    with pytest.warns(SCOPRetiredWarning, match="fitted on the SCOP shape engine"):
        info = rebuilt.build(x)
    assert info.monotone_engine == "qp" and getattr(rebuilt, "_scop_Sigma", None) is None
    r_inv = np.random.default_rng(6).standard_normal((rebuilt._n_basis, info.n_cols))
    rebuilt.set_reparametrisation(r_inv)
    np.testing.assert_array_equal(rebuilt.transform(x), rebuilt._basis_matrix(x).toarray() @ r_inv)

    with pytest.warns(SCOPRetiredWarning, match="fitted on the SCOP shape engine"):
        model.fit_reml(frame, y)
    refitted, group = model._specs["x"], _group(model)
    assert group.monotone_engine == "qp" and group.constraints is not None
    assert getattr(refitted, "_scop_Sigma", None) is None
    np.testing.assert_array_equal(
        refitted.transform(x), refitted._basis_matrix(x).toarray() @ refitted._R_inv
    )


@pytest.mark.parametrize(
    ("entry", "warns"),
    [
        ({"basis": {"kind": "ps", "select": False}}, True),
        (
            {
                "knots": {
                    "n_knots": 6,
                    "strategy": "uniform",
                    "knot_alpha": 0.2,
                    "positions": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                }
            },
            True,
        ),
        # 0.41.0 fitted a ``bs`` on the QP too: the retirement moves nothing.
        ({"basis": {"kind": "bs", "select": False}}, False),
    ],
    ids=["ps-basis", "knots", "bs-basis"],
)
def test_a_structure_rebuilding_a_saved_scop_spline_warns_where_the_fit_moves(entry, warns) -> None:
    """A structure applied to a model saved on SCOP warns when its P-spline now fits on the QP.

    Mutation check: without the saved-SCOP mark in ``rebuild._carry_marks``
    the ``ps``-basis case fits silently.
    """
    with open(SAVED_SCOP, "rb") as handle:
        record = pickle.load(handle)
    structure = Structure(features={"x": FeatureStructure(kind="spline", **entry)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", SCOPRetiredWarning)  # applying never warns
        rebuilt = structure.apply(record["model"])
    if warns:
        with pytest.warns(SCOPRetiredWarning, match="fitted on the SCOP shape engine"):
            rebuilt.fit_reml(record["frame"], record["y"])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", SCOPRetiredWarning)
            rebuilt.fit_reml(record["frame"], record["y"])
    assert _group(rebuilt).monotone_engine == "qp"


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


def test_each_irls_qp_starts_from_an_earlier_active_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each IRLS step's QP is handed an active set an earlier solve returned.

    The last solve's set, or the committed state's after a rejected step. The
    test reads what each solve receives, not the steps it takes, which depend on
    the trajectory and on any cold retry. Fails when every QP starts cold.
    """
    calls = []
    solve = irls_direct_qp._solve_constrained_qp_with_cold_retry

    def recorded(H, g, A, b, active_set_init, **kwargs):
        result = solve(H, g, A, b, active_set_init, **kwargs)
        calls.append((active_set_init, list(result.active_set)))
        return result

    monkeypatch.setattr(irls_direct_qp, "_solve_constrained_qp_with_cold_retry", recorded)
    _fit("poisson", PSpline(n_knots=10, constraint=Constraint.fit.increasing), method="fit")
    assert len(calls) >= 2 and len(calls[0][1]) >= 3 and calls[0][0] is None
    for index, (handed, _) in enumerate(calls[1:], start=1):
        assert handed is not None
        assert any(list(handed) == returned for _, returned in calls[:index]), index
