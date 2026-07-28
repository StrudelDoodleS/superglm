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
    SuperGLM,
)


def test_qp_convex_fit_reml_auto_lambda_constrained_refit():
    x = np.linspace(0.0, 1.0, 300)
    y = (x - 0.35) ** 2
    df = pd.DataFrame({"x": x})

    model = SuperGLM(
        family="gaussian",
        features={"x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.convex)},
    ).fit_reml(df, y)

    assert model._reml_lambdas["x"] > 0.0


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
