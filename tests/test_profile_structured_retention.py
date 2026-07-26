"""Compact structured reporting after profiled-family REML publication."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from superglm import (
    FactorSmooth,
    LambdaPolicy,
    Numeric,
    RandomEffect,
    Spline,
    SuperGLM,
)
from superglm.distributions import NegativeBinomial, Tweedie
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
    level_effect = np.array([-0.35, -0.12, 0.08, 0.21, 0.37])
    mu = np.exp(0.45 + 0.18 * z + level_effect[codes] * (1.0 + 0.4 * x))
    y = rng.poisson(mu).astype(np.float64)
    X = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "group": np.array([f"group-{code}" for code in codes], dtype=object),
        }
    )
    return X, y


def _profile_model(
    *,
    family_name: str,
    term_kind: str,
    direct_solve: str,
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

    family = NegativeBinomial(theta="auto") if family_name == "nb" else Tweedie(p=1.5)
    return (
        SuperGLM(
            family=family,
            features=features,
            interactions=interactions,
            selection_penalty=0.0,
            direct_solve=direct_solve,
            retain_fit_state=False,
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
