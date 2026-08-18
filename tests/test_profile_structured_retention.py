"""Compact structured reporting after profiled-family REML publication."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Constraint,
    FactorSmooth,
    LambdaPolicy,
    Numeric,
    PSpline,
    RandomEffect,
    Spline,
    SuperGLM,
)
from superglm.distributions import NegativeBinomial, Tweedie
from superglm.model import fit_ops as fit_ops_module
from superglm.model import reml_finalize as reml_finalize_module
from superglm.profiling.nb import NBProfileResult
from superglm.profiling.tweedie import TweedieProfileResult
from superglm.types import FeatureSpec


def _profile_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(20260726)
    n_levels = 5
    repeats = 16
    codes = np.repeat(np.arange(n_levels), repeats)
    rng.shuffle(codes)
    x = rng.uniform(-1.0, 1.0, len(codes))
    z = rng.normal(size=len(codes))
    shape = np.linspace(-1.0, 1.0, len(codes))
    level_effect = np.array([-0.35, -0.12, 0.08, 0.21, 0.37])
    mu = np.exp(0.45 + 0.18 * z + 0.25 * shape + level_effect[codes] * (1.0 + 0.4 * x))
    y = rng.poisson(mu).astype(np.float64)
    X = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "shape": shape,
            "group": np.array([f"group-{code}" for code in codes], dtype=object),
        }
    )
    return X, y


def _profile_model(
    *,
    family_name: str,
    term_kind: str,
    direct_solve: str,
    constrained_mode: str | None = None,
    retain_fit_state: bool = False,
) -> tuple[SuperGLM, str]:
    fixed = LambdaPolicy.fixed
    features: dict[str, FeatureSpec] = {"z": Numeric()}
    interactions: list[tuple[str, str] | object] = []
    if term_kind == "re":
        features["group"] = RandomEffect(lambda_policy=fixed(1.1))
        term_name = "group"
    elif term_kind == "fs":
        interactions.append(
            FactorSmooth(
                "x",
                group="group",
                k=5,
                lambda_policy={
                    "wiggle": fixed(1.2),
                    "null_0": fixed(0.8),
                    "null_1": fixed(0.9),
                },
            )
        )
        term_name = "x:group:fs"
    elif term_kind == "sz":
        features["x"] = Spline(n_knots=5, lambda_policy=fixed(1.0))
        interactions.append(
            FactorSmooth(
                "x",
                group="group",
                basis="sz",
                k=5,
                lambda_policy={"wiggle": fixed(1.2)},
            )
        )
        term_name = "x:group:sz"
    else:  # pragma: no cover - test parameter contract
        raise AssertionError(f"unsupported structured term {term_kind!r}")

    if constrained_mode is not None:
        constraint_policy = fixed(1.05) if constrained_mode == "fixed" else LambdaPolicy.estimate()
        features["shape"] = PSpline(
            n_knots=5,
            constraint=Constraint.fit.increasing,
            lambda_policy=constraint_policy,
        )

    if family_name == "nb":
        family: object = NegativeBinomial(theta="auto")
    elif family_name == "tweedie":
        family = Tweedie(p=1.5)
    elif family_name == "gaussian":
        family = "gaussian"
    else:  # pragma: no cover - test parameter contract
        raise AssertionError(f"unsupported family {family_name!r}")
    return (
        SuperGLM(
            family=family,
            features=features,
            interactions=interactions,
            selection_penalty=0.0,
            direct_solve=direct_solve,
            retain_fit_state=retain_fit_state,
            max_iter=60,
        ),
        term_name,
    )


def _install_deterministic_profile(monkeypatch: pytest.MonkeyPatch, family_name: str) -> None:
    if family_name == "nb":
        result = NBProfileResult(
            theta_hat=2.4,
            nll=0.0,
            n_evaluations=1,
            converged=True,
        )
        monkeypatch.setattr(
            "superglm.profiling.nb.estimate_nb_theta",
            lambda *args, **kwargs: result,
        )
        return

    result = TweedieProfileResult(
        p_hat=1.47,
        phi_hat=1.15,
        nll=0.0,
        n_evaluations=1,
        converged=True,
        method="brent",
        phi_method="mle",
        search_trace=pd.DataFrame({"p": [1.47], "phi": [1.15], "nll": [0.0]}),
    )
    monkeypatch.setattr(
        "superglm.profiling.tweedie.estimate_tweedie_p",
        lambda *args, **kwargs: result,
    )


def _assert_report_pickle_parity(model: SuperGLM, term_kind: str, term_name: str) -> None:
    restored = pickle.loads(pickle.dumps(model))
    if term_kind == "re":
        report = model.random_effects(term_name)
        restored_report = restored.random_effects(term_name)
        pd.testing.assert_frame_equal(restored_report.table, report.table)
        return

    report = model.factor_smooth(term_name, grid=9)
    restored_report = restored.factor_smooth(term_name, grid=9)
    pd.testing.assert_frame_equal(restored_report.table, report.table)
    pd.testing.assert_frame_equal(restored_report.curves, report.curves)


def _assert_profile_scratch_released(model: SuperGLM) -> None:
    assert getattr(model, "_reporting_support_state") is None
    assert getattr(model, "_dm") is None
    assert getattr(model, "_fit_weights") is None
    assert getattr(model, "_fit_mu") is None
    assert getattr(model, "_fit_X_ref") is None
    assert getattr(model, "_fit_state").retained is False


@pytest.mark.parametrize("family_name", ["nb", "tweedie"])
@pytest.mark.parametrize("term_kind", ["re", "fs", "sz"])
@pytest.mark.parametrize("direct_solve", ["gram", "auto", "structured"])
def test_profiled_reml_retains_compact_structured_reports(
    monkeypatch: pytest.MonkeyPatch,
    family_name: str,
    term_kind: str,
    direct_solve: str,
) -> None:
    X, y = _profile_data()
    model, term_name = _profile_model(
        family_name=family_name,
        term_kind=term_kind,
        direct_solve=direct_solve,
    )
    _install_deterministic_profile(monkeypatch, family_name)

    if family_name == "nb":
        model.estimate_theta(X, y, fit_mode="reml")
    else:
        model.estimate_p(X, y, fit_mode="reml", phi_method="mle")

    if direct_solve != "auto":
        assert model.result.direct_backend == direct_solve
    else:
        assert model.result.direct_backend in {"gram", "structured"}
    assert getattr(model, "_dm") is None
    assert getattr(model, "_fit_state").retained is False
    reporting_state = getattr(model, "_reporting_support_state")
    assert reporting_state is not None
    assert term_name in reporting_state.support_totals
    _assert_report_pickle_parity(model, term_kind, term_name)


@pytest.mark.parametrize("term_kind", ["re", "fs", "sz"])
@pytest.mark.parametrize("constrained_mode", ["fixed", "estimated"])
def test_constrained_reml_reaches_the_route_with_structured_terms(
    monkeypatch: pytest.MonkeyPatch,
    term_kind: str,
    constrained_mode: str,
) -> None:
    """Structured terms travel the constrained route instead of being refused.

    All three were refused here once. The shape constraint in this fixture sits
    on its own feature, so no factor smooth duplicates a constrained variable
    and nothing in this file meets the one specification still refused --
    ``tests/test_shape_reml.py`` owns that case, where the constraint and the
    ``fs`` factor smooth share a variable.
    """
    X, y = _profile_data()
    model, term_name = _profile_model(
        family_name="gaussian",
        term_kind=term_kind,
        direct_solve="auto",
        constrained_mode=constrained_mode,
    )
    route_name = "run_fixed_monotone_reml" if constrained_mode == "fixed" else "run_scop_efs_reml"
    real_route = getattr(fit_ops_module, route_name)
    route_calls = []

    def record_route(*args, **kwargs):
        route_calls.append(True)
        return real_route(*args, **kwargs)

    monkeypatch.setattr(fit_ops_module, route_name, record_route)
    model.fit_reml(X, y, max_reml_iter=2, runtime_validation="skip")

    assert route_calls, "the constrained route was never entered"
    assert model.result.converged
    if term_kind == "re":
        assert model.random_effects(term_name).variance_component > 0.0


def test_reporting_failure_preserves_installed_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _profile_data()
    model, term_name = _profile_model(
        family_name="gaussian",
        term_kind="re",
        direct_solve="auto",
    )
    model.fit_reml(X, y, runtime_validation="skip")
    revision = getattr(model, "_fit_revision")
    predictions = model.predict(X)
    report = model.random_effects(term_name)

    def fail_support(*args, **kwargs):
        raise RuntimeError("report support failed")

    monkeypatch.setattr(
        reml_finalize_module,
        "_build_reml_reporting_support_state",
        fail_support,
    )

    with pytest.raises(RuntimeError, match="report support failed"):
        model.fit_reml(X, y, runtime_validation="skip")

    assert getattr(model, "_fit_revision") == revision
    np.testing.assert_allclose(model.predict(X), predictions, rtol=0.0, atol=0.0)
    pd.testing.assert_frame_equal(model.random_effects(term_name).table, report.table)


@pytest.mark.parametrize("retain_fit_state", [False, True])
def test_tweedie_reml_profile_keeps_only_authoritative_public_row_state(
    monkeypatch: pytest.MonkeyPatch,
    retain_fit_state: bool,
) -> None:
    X, y = _profile_data()
    model, term_name = _profile_model(
        family_name="tweedie",
        term_kind="re",
        direct_solve="auto",
        retain_fit_state=retain_fit_state,
    )
    real_builder = reml_finalize_module.build_reporting_support_state
    builder_calls = []

    def record_builder(*args, **kwargs):
        builder_calls.append(True)
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(
        reml_finalize_module,
        "build_reporting_support_state",
        record_builder,
    )

    result = model.estimate_p(
        X,
        y,
        fit_mode="reml",
        phi_method="pearson",
        method="grid",
        grid=np.array([1.4, 1.6]),
    )

    context = result._objective.__self__
    scratch_model = context.model
    expected_builds = int(not retain_fit_state)
    assert len(builder_calls) == expected_builds
    _assert_profile_scratch_released(scratch_model)
    assert "_suppress_reporting_support" not in model.__dict__
    assert (getattr(model, "_dm") is not None) is retain_fit_state
    assert getattr(model, "_fit_state").retained is retain_fit_state
    if retain_fit_state:
        assert getattr(model, "_reporting_support_state") is None
    else:
        assert term_name in getattr(model, "_reporting_support_state").support_totals

    result._objective(1.5, source="ci_probe")

    assert len(builder_calls) == expected_builds
    _assert_profile_scratch_released(scratch_model)

    restored_result = pickle.loads(pickle.dumps(result))
    restored_result._objective(1.55, source="ci_probe")
    restored_scratch_model = restored_result._objective.__self__.model
    assert len(builder_calls) == expected_builds
    _assert_profile_scratch_released(restored_scratch_model)
    _assert_report_pickle_parity(model, "re", term_name)
