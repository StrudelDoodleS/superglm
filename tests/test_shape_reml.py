import numpy as np
import pandas as pd
import pytest

from superglm import (
    BSplineSmooth,
    Constraint,
    FactorSmooth,
    NegativeBinomial,
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


def test_scop_constraint_rejects_a_main_effect_factor_smooth_of_the_same_variable() -> None:
    """The one specification with no converging path -- and it is a spec, not a pair.

    A ``basis="fs"`` factor smooth carries its own main effect. Put one on a
    variable that also has a SCOP shape constraint and the model states that
    variable's effect twice: once confined to the shape cone, once free. The
    free copy absorbs what the constrained copy may not do, the two compensate,
    and no coefficient mode is reached -- measured 0/7 across seed, rows, level
    count and basis size, where every adjacent specification is 7/7.

    An earlier form of this refusal covered ``basis="sz"`` and factor smooths of
    other variables as well. That was generalised from a sweep which varied the
    data but never the specification, so all of its cases were this one.
    """
    X, y = _scop_factor_smooth_frame()
    model = SuperGLM(
        family="gaussian",
        features={"x": PSpline(n_knots=7, constraint=Constraint.fit.increasing)},
        interactions=[FactorSmooth("x", group="group", basis="fs", k=5)],
        direct_solve="auto",
    )

    with pytest.raises(NotImplementedError, match=r"stated twice"):
        model.fit_reml(X, y)


def test_scop_constraint_rejects_a_factor_smooth_grouped_by_the_constrained_column() -> None:
    """The duplication is reachable on the factor smooth's OTHER parent too.

    A factor smooth spans two columns. Along ``variable`` the free copy of the
    effect is the marginal smooth; along ``group`` it is the per-level
    null-space blocks, which are a per-level effect of the grouping column. So
    constraining the *grouping* column duplicates just as constraining the
    smoothed one does -- reachable when the grouping column carries an
    ``OrderedCategorical`` SCOP constraint, since that is what lets a
    categorical column be shape-constrained at all.

    Measured 0/6 across seed, rows, level count and basis size. A guard keyed on
    ``spec.variable`` alone passes this straight through to that failure, which
    is what an earlier form of this change did.
    """
    from superglm import OrderedCategorical

    rng = np.random.default_rng(0)
    n, n_levels = 1500, 10
    levels = [f"L{index:02d}" for index in range(n_levels)]
    codes = rng.integers(0, n_levels, n)
    x = rng.uniform(0.0, 1.0, n)
    y = 0.25 * codes + np.sin(3.0 * x) + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"x": x, "g": [levels[code] for code in codes]})

    model = SuperGLM(
        family="gaussian",
        features={
            "g": OrderedCategorical(
                order=levels,
                basis=Spline(kind="ps", n_knots=6, constraint=Constraint.fit.increasing),
            )
        },
        interactions=[FactorSmooth("x", group="g", basis="fs", k=5)],
    )

    with pytest.raises(NotImplementedError, match=r"stated twice"):
        model.fit_reml(X, y)


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


def test_structured_fit_constraint_preflight_runs_before_nb_theta_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superglm.model import fit_ops

    x = np.linspace(0.0, 1.0, 160)
    codes = np.arange(len(x)) % 40
    X = pd.DataFrame(
        {
            "x": x,
            "group": np.array([f"g{code}" for code in codes], dtype=object),
        }
    )
    y = np.resize(np.array([1.0, 2.0, 3.0, 4.0]), len(x))
    model = SuperGLM(
        family=NegativeBinomial(theta="auto"),
        features={
            "x": PSpline(
                n_knots=7,
                constraint=Constraint.fit.increasing,
            ),
        },
        interactions=[FactorSmooth("x", group="group", k=5)],
    )

    def unexpected_theta_profile(*_args, **_kwargs):
        raise AssertionError("NB theta profiling must not run before the shape preflight")

    monkeypatch.setattr(fit_ops, "_maybe_estimate_nb_theta", unexpected_theta_profile)

    with pytest.raises(
        NotImplementedError,
        match=r"fit-time SCOP shape constraints.*FactorSmooth",
    ):
        model.fit_reml(X, y)


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
    ("spline_cls", "engine", "rel"),
    [
        pytest.param(BSplineSmooth, "qp", 1e-12, id="qp-engine"),
        pytest.param(PSpline, "scop", 1e-3, id="scop-engine"),
    ],
)
def test_slack_constraint_with_a_random_effect_reproduces_the_free_fit(
    spline_cls,
    engine: str,
    rel: float,
) -> None:
    """Equivalence control: a non-binding constraint must not move the fit.

    A fit that merely runs proves little. On a monotone truth the increasing
    constraint is inactive, so the constrained fit has to reproduce the
    unconstrained one -- variance component included, that being the quantity a
    shape constraint has no business perturbing.

    The two engines earn different bars, and the difference is structural rather
    than a tolerance chosen for comfort:

    * QP reaches the free fit **exactly**. Its automatic-lambda path is an
      unconstrained REML pass followed by a constrained refit, so a slack
      constraint makes the refit a no-op: measured ``0`` on the variance
      component and ``2.1e-16`` on the deviance across five seeds, i.e. pure
      round-off. ``1e-12`` is that with room, and would catch any real drift.
    * SCOP reaches it only to ``~1e-4``, because the two arms are optimized by
      *different routines* -- the free arm by ``optimize_reml_best``, the
      constrained one by ``run_scop_efs_reml`` -- which stop at slightly
      different lambdas. Measured worst case over the same five seeds is
      ``1.13e-4`` on the variance component and ``8.4e-5`` on the deviance;
      ``1e-3`` is an order of magnitude above that and still an order below the
      perturbation this control exists to detect.
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
