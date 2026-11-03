import numpy as np
import pandas as pd
import pytest

from superglm import (
    BSplineSmooth,
    Constraint,
    FactorSmooth,
    PSpline,
    RandomEffect,
    Spline,
    SuperGLM,
)
from superglm.constraints import (
    shape_constraint_certificate,
    shape_constraint_is_roundoff_feasible,
)


def test_qp_convex_fit_reml_auto_lambda_constrained_refit():
    x = np.linspace(0.0, 1.0, 300)
    y = x + 1e-3 * np.random.default_rng(5).normal(size=x.size)
    df = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        features={"x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.convex)},
    ).fit_reml(df, y)

    group = next(group for group in model._groups if group.name == "x")
    beta = model.result.beta[group.sl]
    certificate = shape_constraint_certificate(model._specs["x"], beta, "convex")
    eps = np.finfo(np.float64).eps
    qp_tolerance = (
        100.0 * eps * (1.0 + np.linalg.norm(group.constraints.A, ord=np.inf) * np.linalg.norm(beta))
    )
    certificate_tolerance = 1000.0 * eps * (1.0 + beta.size)

    assert model._reml_lambdas["x"] > 0.0
    assert np.min(group.constraints.A @ beta) >= -qp_tolerance
    assert certificate.minimum_scaled_slack >= -certificate_tolerance


def test_scop_concave_fit_reml_discrete_estimates_lambda():
    x = np.linspace(0.0, 1.0, 300)
    y = 1.0 - (x - 0.4) ** 2
    df = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        discrete=True,
        features={"x": PSpline(n_knots=10, constraint=Constraint.fit.concave)},
    ).fit_reml(df, y)

    assert model._reml_lambdas["x"] > 0.0


@pytest.mark.parametrize(
    ("kind", "center", "sign"),
    [
        pytest.param("convex", -0.2, 1.0, id="convex-increasing"),
        pytest.param("convex", 1.2, 1.0, id="convex-decreasing"),
        pytest.param("concave", 1.2, -1.0, id="concave-increasing"),
        pytest.param("concave", -0.2, -1.0, id="concave-decreasing"),
    ],
)
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_scop_fit_reml_recovers_both_slope_orientations(
    kind,
    center,
    sign,
    discrete,
):
    x = np.linspace(0.0, 1.0, 300)
    y = sign * (x - center) ** 2
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=discrete,
        features={
            "x": PSpline(
                n_knots=10,
                constraint=getattr(Constraint.fit, kind),
            )
        },
    ).fit_reml(frame, y)

    fitted = model.predict(frame)
    r_squared = 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    curve = model.reconstruct_feature("x")["log_relativity"]
    group = next(group for group in model._groups if group.name == "x")
    certificate = shape_constraint_certificate(
        model._specs["x"],
        model.result.beta[group.sl],
        kind,
    )

    assert model.result.converged
    assert r_squared > 0.995
    assert np.ptp(curve) > 1.0
    assert model._reml_lambdas["x"] > 0.0
    assert certificate.minimum_scaled_slack >= -1e-10


@pytest.mark.parametrize("fit_method", ["fit", "fit_reml"])
@pytest.mark.parametrize(
    ("kind", "center", "sign", "knot_strategy", "discrete"),
    [
        pytest.param(
            "convex",
            0.38,
            1.0,
            "uniform",
            False,
            id="convex-default-dense",
        ),
        pytest.param(
            "convex",
            0.38,
            1.0,
            "quantile",
            True,
            id="convex-quantile-discrete",
        ),
        pytest.param(
            "concave",
            0.62,
            -1.0,
            "uniform",
            False,
            id="concave-default-dense",
        ),
        pytest.param(
            "concave",
            0.62,
            -1.0,
            "quantile",
            True,
            id="concave-quantile-discrete",
        ),
    ],
)
def test_cr_fit_and_fit_reml_recover_nonzero_curvature_constraint(
    fit_method,
    kind,
    center,
    sign,
    knot_strategy,
    discrete,
):
    rng = np.random.default_rng(217)
    x = np.sort(np.concatenate(([0.0, 1.0], rng.beta(0.7, 1.6, size=398))))
    y = sign * (x - center) ** 2
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=discrete,
        features={
            "x": Spline(
                kind="cr",
                n_knots=8,
                knot_strategy=knot_strategy,
                constraint=getattr(Constraint.fit, kind),
            )
        },
    )

    fitted_model = getattr(model, fit_method)(frame, y)
    fitted = fitted_model.predict(frame)
    r_squared = 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    reconstruction = fitted_model.reconstruct_feature("x")
    curve = reconstruction["log_relativity"]
    slope = np.gradient(curve, reconstruction["x"])
    group = next(group for group in fitted_model._groups if group.name == "x")
    beta = fitted_model.result.beta[group.sl]
    certificate = shape_constraint_certificate(fitted_model._specs["x"], beta, kind)
    eps = np.finfo(np.float64).eps
    qp_tolerance = (
        100.0 * eps * (1.0 + np.linalg.norm(group.constraints.A, ord=np.inf) * np.linalg.norm(beta))
    )
    certificate_tolerance = 1000.0 * eps * (1.0 + beta.size)

    assert fitted_model.result.converged
    assert r_squared > 0.95
    assert np.ptp(curve) > 0.25
    assert sign * slope[0] < -0.25
    assert sign * slope[-1] > 0.25
    assert np.min(group.constraints.A @ beta) >= -qp_tolerance
    assert certificate.minimum_scaled_slack >= -certificate_tolerance
    assert certificate.minimum_signed_derivative >= -1e-10


@pytest.mark.parametrize("fit_method", ["fit", "fit_reml"])
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize(
    ("kind", "sign"),
    [
        pytest.param("convex", 1.0, id="convex"),
        pytest.param("concave", -1.0, id="concave"),
    ],
)
def test_cr_clustered_quantile_large_response_completes_kkt_certificate(
    fit_method,
    discrete,
    kind,
    sign,
):
    rng = np.random.default_rng(217)
    x = np.sort(np.concatenate(([0.0, 1.0], rng.beta(0.3, 2.2, size=398))))
    y = 1e6 * sign * (x - 0.4) ** 2
    frame = pd.DataFrame({"x": x})
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        discrete=discrete,
        features={
            "x": Spline(
                kind="cr",
                n_knots=8,
                knot_strategy="quantile",
                constraint=getattr(Constraint.fit, kind),
            )
        },
    )

    fitted_model = getattr(model, fit_method)(frame, y)
    fitted = fitted_model.predict(frame)
    r_squared = 1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)
    group = next(group for group in fitted_model._groups if group.name == "x")
    beta = fitted_model.result.beta[group.sl]
    spec = fitted_model._specs["x"]
    group_index = next(
        index for index, fitted_group in enumerate(fitted_model._groups) if fitted_group is group
    )
    solver_map = fitted_model._dm.group_matrices[group_index].R_inv
    raw_constraints = spec._build_monotone_constraints_raw()
    expected_constraints = raw_constraints.compose(solver_map)

    assert fitted_model.result.converged
    assert fitted_model.result.termination_reason == "converged"
    assert r_squared > 0.95
    assert np.ptp(fitted) > 1e5
    assert group.constraints is not None
    np.testing.assert_array_equal(group.constraints.A, expected_constraints.A)
    np.testing.assert_array_equal(group.constraints.b, expected_constraints.b)
    assert shape_constraint_is_roundoff_feasible(spec, beta, kind)


def _scop_factor_smooth_frame():
    x = np.linspace(0.0, 1.0, 400)
    rng = np.random.default_rng(19)
    codes = np.arange(len(x)) % 20
    return pd.DataFrame(
        {
            "x": x,
            "z": rng.uniform(0.0, 1.0, len(x)),
            "group": np.array([f"g{code:02d}" for code in codes], dtype=object),
        }
    ), 0.3 + 0.7 * x + 0.2 * rng.normal(size=len(x))


@pytest.mark.parametrize(
    ("features_for", "interaction_for", "case"),
    [
        pytest.param("x", "x", "sz", id="sum-to-zero-same-variable"),
        pytest.param("x", "z", "fs", id="main-effect-basis-other-variable"),
        pytest.param("z", "x", "fs", id="constraint-on-the-other-variable"),
    ],
)
def test_scop_constraints_fit_beside_a_non_duplicating_factor_smooth(
    features_for: str,
    interaction_for: str,
    case: str,
) -> None:
    """Everything adjacent to the duplicated specification converges.

    ``sz`` excludes the main effect, so it states the constrained variable once.
    A factor smooth of any other variable states nothing twice. These are the
    cases the wider refusal was taking with it.
    """
    X, y = _scop_factor_smooth_frame()
    features = {features_for: PSpline(n_knots=7, constraint=Constraint.fit.increasing)}
    if interaction_for != features_for:
        features[interaction_for] = PSpline(n_knots=7)
    model = SuperGLM(
        family="gaussian",
        features=features,
        interactions=[FactorSmooth(interaction_for, group="group", basis=case, k=5)],
        direct_solve="auto",
    )
    model.fit_reml(X, y)
    assert model.result.converged


def test_scop_constraints_fit_alongside_a_random_effect() -> None:
    """A shape constraint and a variance component estimate together.

    To the extended Fellner-Schall update these are one kind of object: it
    estimates smoothing parameters and variance components by a single formula
    (Wood & Fasiolo, 2017, Biometrics 73(4):1071-1081), and the founding case in
    Fellner (1986) is a penalty assembled from identity blocks -- precisely what
    a ``RandomEffect`` contributes. Shape-constrained additive *mixed* models
    are the documented combination in the reference implementation (Pya
    Arnqvist, 2024, arXiv:2403.09438 section 3).

    This raised ``NotImplementedError`` until the EFS lambda step stopped
    reading ``PenaltyComponent.omega_ssp`` directly, which is ``None`` for an
    identity penalty and so reached a matmul as a 0-d operand.
    """
    rng = np.random.default_rng(11)
    n, n_levels = 900, 18
    x = np.sort(rng.uniform(0.0, 1.0, n))
    codes = rng.integers(0, n_levels, n)
    level_effects = rng.normal(0.0, 0.5, n_levels)
    # Non-monotone truth, so an honoured constraint has to visibly bite.
    y = np.sin(3.2 * x) + level_effects[codes] + rng.normal(0.0, 0.25, n)
    X = pd.DataFrame({"x": x, "group": [f"g{code:02d}" for code in codes]})

    model = SuperGLM(
        family="gaussian",
        features={
            "x": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
            "group": RandomEffect(),
        },
    )
    model.fit_reml(X, y)

    curve = model.plot_data("x")["terms"][0]["effect"].sort_values("x")
    steps = np.diff(curve["log_relativity"].to_numpy(dtype=float))
    assert np.all(steps >= -1e-9), f"constraint not honoured: worst step {steps.min():.3e}"
    assert steps.max() > 1e-3, "constrained curve is flat, so nothing was actually fitted"

    random_effect = model.random_effects("group")
    assert random_effect.variance_component > 0.0
    assert np.isfinite(random_effect.smoothing_lambda)


@pytest.mark.parametrize(
    ("spline_cls", "rel"),
    [
        pytest.param(BSplineSmooth, 1e-12, id="qp-engine"),
        pytest.param(PSpline, 1e-12, id="ps-qp-engine"),
    ],
)
def test_slack_constraint_with_a_random_effect_reproduces_the_free_fit(
    spline_cls,
    rel: float,
) -> None:
    """Equivalence control: a non-binding constraint must not move the fit.

    A fit that merely runs proves little. On a monotone truth the increasing
    constraint is inactive, so the constrained fit has to reproduce the
    unconstrained one -- variance component included, that being the quantity a
    shape constraint has no business perturbing.

    The QP engine, for a ``bs`` term or a ``PSpline``, reaches the free fit
    **exactly**. Its automatic-lambda path is an unconstrained REML pass followed by a constrained refit, so a slack
    constraint makes the refit a no-op: measured ``0`` on the variance
    component and ``2.1e-16`` on the deviance across five seeds, i.e. pure
    round-off. ``1e-12`` is that with room, and would catch any real drift.
    Note ``pytest.approx`` keeps its default ``abs=1e-12`` alongside ``rel``,
    so on a variance component of order 0.3 the effective bar is that absolute
    floor rather than ``3e-13`` -- still four orders inside the measurement.
    """
    rng = np.random.default_rng(5)
    n, n_levels = 1200, 20
    x = np.sort(rng.uniform(0.0, 1.0, n))
    codes = rng.integers(0, n_levels, n)
    level_effects = rng.normal(0.0, 0.5, n_levels)
    y = 2.5 * x + level_effects[codes] + rng.normal(0.0, 0.25, n)
    X = pd.DataFrame({"x": x, "group": [f"g{code:02d}" for code in codes]})

    def fit(constraint):
        kwargs = {"constraint": constraint} if constraint is not None else {}
        model = SuperGLM(
            family="gaussian",
            features={"x": spline_cls(n_knots=10, **kwargs), "group": RandomEffect()},
        )
        model.fit_reml(X, y)
        return model

    free = fit(None)
    constrained = fit(Constraint.fit.increasing)

    free_re = free.random_effects("group")
    constrained_re = constrained.random_effects("group")
    assert constrained_re.variance_component == pytest.approx(free_re.variance_component, rel=rel)
    assert constrained.result.deviance == pytest.approx(free.result.deviance, rel=rel)


_C = Constraint.fit
_U_RULES = _C.decreasing.below(25) & _C.flat.within(25, 40) & _C.increasing.above(40)


def _driver_ages(n: int = 20_000, seed: int = 43) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    age = rng.integers(17, 81, n).astype(np.float64)
    eta = np.where(age < 25, 0.08 * (25 - age), np.where(age > 40, 0.02 * (age - 40), 0.0))
    return pd.DataFrame({"age": age}), rng.poisson(np.exp(eta))


def _assert_u_rules_hold(model) -> None:
    from superglm.constraints import shape_derivative_matrix

    group = next(group for group in model._groups if group.name == "age")
    beta = model.result.beta[group.sl]
    grid = np.linspace(17.0, 80.0, 2000)
    f = shape_derivative_matrix(model._specs["age"], grid, 0) @ beta
    slope = shape_derivative_matrix(model._specs["age"], grid, 1) @ beta
    tol = 1e3 * 2.0**-53 * np.max(np.abs(f))
    assert np.max(slope[grid <= 25.0]) <= tol
    assert np.ptp(f[(grid >= 25.0) & (grid <= 40.0)]) <= tol
    assert np.min(slope[grid >= 40.0]) >= -tol


def _fit_recording_rows(monkeypatch, discrete: bool):
    """``fit_reml`` on the U rules, recording every raw row build and the strip."""
    from superglm.features import _spline_subclass_ops
    from superglm.model import fit_ops

    builds, stripped_after = [], []
    build_rows = _spline_subclass_ops.build_shape_constraints_raw
    strip = fit_ops.strip_qp_constraints

    def recording_build(spec):
        rows = build_rows(spec)
        builds.append(rows)
        return rows

    def recording_strip(groups):
        stripped_after.append(len(builds))
        return strip(groups)

    monkeypatch.setattr(_spline_subclass_ops, "build_shape_constraints_raw", recording_build)
    monkeypatch.setattr(fit_ops, "strip_qp_constraints", recording_strip)
    X, y = _driver_ages()
    model = SuperGLM(
        family="poisson",
        discrete=discrete,
        features={"age": Spline(kind="bs", k=12, constraint=_U_RULES)},
    )
    with pytest.warns(UserWarning, match="holds on") as seen:
        model.fit_reml(X, y)
    assert sum("holds on" in str(w.message) for w in seen) == 1
    return model, builds, stripped_after


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
def test_rules_survive_reml_strip_restore(monkeypatch, discrete):
    model, builds, stripped_after = _fit_recording_rows(monkeypatch, discrete)
    index = next(i for i, group in enumerate(model._groups) if group.name == "age")
    group = model._groups[index]

    assert model._reml_lambdas["age"] > 0.0
    # Rows were built before the strip and rebuilt after it, identically.
    assert len(stripped_after) == 1 and 0 < stripped_after[0] < len(builds)
    for rows in builds[1:]:
        assert np.array_equal(rows.A, builds[0].A)
        assert np.array_equal(rows.b, builds[0].b)
    restored = builds[0].compose(model._dm.group_matrices[index].R_inv)
    assert np.array_equal(group.constraints.A, restored.A)
    _assert_u_rules_hold(model)


def test_rules_discrete_path():
    X, y = _driver_ages()
    raw, models = {}, {}
    for discrete in (False, True):
        model = SuperGLM(
            family="poisson",
            selection_penalty=0.0,
            discrete=discrete,
            features={"age": Spline(kind="bs", k=12, constraint=_U_RULES)},
        )
        with pytest.warns(UserWarning, match="holds on"):
            model.fit(X, y)
        raw[discrete] = model._specs["age"]._build_monotone_constraints_raw()
        models[discrete] = model
    assert np.array_equal(raw[True].A, raw[False].A)
    assert np.array_equal(raw[True].b, raw[False].b)
    model = models[True]
    index = next(i for i, group in enumerate(model._groups) if group.name == "age")
    expected = raw[True].compose(model._dm.group_matrices[index].R_inv)
    assert np.array_equal(model._groups[index].constraints.A, expected.A)
    _assert_u_rules_hold(model)
