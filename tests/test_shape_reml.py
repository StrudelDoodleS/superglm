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
from superglm.constraints import shape_constraint_certificate


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
    certificate = shape_constraint_certificate(fitted_model._specs["x"], beta, kind)

    assert fitted_model.result.converged
    assert fitted_model.result.termination_reason == "converged"
    assert r_squared > 0.95
    assert np.ptp(fitted) > 1e5
    assert certificate.minimum_scaled_slack >= -1e-10


@pytest.mark.parametrize(
    ("structured_kind", "expected_name"),
    [
        pytest.param("re", "RandomEffect", id="random-effect"),
        pytest.param("fs", "FactorSmooth", id="factor-smooth"),
        pytest.param("sz", "FactorSmooth", id="sum-to-zero"),
    ],
)
def test_structured_terms_reject_fit_time_shape_constraints_early(
    structured_kind: str,
    expected_name: str,
) -> None:
    x = np.linspace(0.0, 1.0, 160)
    codes = np.arange(len(x)) % 40
    X = pd.DataFrame(
        {
            "x": x,
            "group": np.array([f"g{code}" for code in codes], dtype=object),
        }
    )
    y = 0.3 + 0.7 * x
    features = {
        "x": PSpline(
            n_knots=7,
            constraint=Constraint.fit.increasing,
        )
    }
    interactions = []
    if structured_kind == "re":
        features["group"] = RandomEffect()
    else:
        interactions.append(
            FactorSmooth(
                "x",
                group="group",
                basis=structured_kind,
                k=5,
            )
        )
    model = SuperGLM(
        family="gaussian",
        features=features,
        interactions=interactions,
        direct_solve="auto",
    )

    with pytest.raises(
        NotImplementedError,
        match=rf"fit-time shape constraints.*{expected_name}",
    ):
        model.fit_reml(X, y)


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
            "group": RandomEffect(),
        },
    )

    def unexpected_theta_profile(*_args, **_kwargs):
        raise AssertionError("NB theta profiling must not run before the shape preflight")

    monkeypatch.setattr(fit_ops, "_maybe_estimate_nb_theta", unexpected_theta_profile)

    with pytest.raises(
        NotImplementedError,
        match=r"fit-time shape constraints.*RandomEffect",
    ):
        model.fit_reml(X, y)
