from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from benchmarks._constrained_fit_profile import (
    ProfileScenario,
    build_scenarios,
    make_synthetic_dataset,
    profile_callstack_and_memory,
    summarize_rows,
    write_profile_artifacts,
)

import superglm.distributions as distributions_module
import superglm.profiling.tweedie as tweedie_module
from superglm import Constraint, SuperGLM
from superglm.distributions import Tweedie
from superglm.features.numeric import Numeric
from superglm.features.spline import BSplineSmooth, PSpline, Spline
from superglm.types import LambdaPolicy


def _evaluate_constrained_profile_once(monkeypatch, feature):
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 24)})
    y = np.linspace(0.5, 2.0, len(X))
    direct_calls = []

    def fake_direct(**kwargs):
        direct_calls.append(kwargs)
        result = SimpleNamespace(
            beta=np.zeros(kwargs["X"].shape[1]),
            intercept=0.0,
            effective_df=1.0,
            n_iter=1,
            converged=True,
            iteration_log=[],
        )
        return result, None

    def fail_pirls(**kwargs):
        raise AssertionError("constrained profile incorrectly dispatched to PIRLS")

    monkeypatch.setattr(tweedie_module, "fit_irls_direct", fake_direct)
    monkeypatch.setattr(tweedie_module, "fit_pirls", fail_pirls)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0,
        features={"x": feature},
    )
    ctx = tweedie_module._build_profile_context(model, X, y, None, None, "pearson", False)
    ctx.evaluate(1.5, source="one_point")
    assert len(direct_calls) == 1
    return ctx, direct_calls[0]


def test_make_synthetic_dataset_repeated_support_has_fewer_unique_values():
    scenario = ProfileScenario(
        name="single_scop_repeated",
        engine="scop",
        n=1000,
        k=10,
        n_constrained=1,
        repeated_support=True,
        discrete=False,
        use_fremtpl=False,
    )
    X, y, w = make_synthetic_dataset(scenario, seed=42)

    assert len(X) == len(y) == len(w) == 1000
    assert X["x1"].nunique() < len(X)


def test_summarize_rows_emits_expected_columns():
    frame = summarize_rows(
        [
            {
                "scenario": "demo",
                "engine": "scop",
                "mode": "exact",
                "n": 1000,
                "k": 10,
                "n_constrained": 1,
                "runtime_s": 0.5,
                "n_reml_iter": 4,
                "n_pirls_iter": 2,
                "peak_mem_mb": 12.0,
            }
        ]
    )
    assert list(frame.columns) == [
        "scenario",
        "engine",
        "mode",
        "n",
        "k",
        "n_constrained",
        "runtime_s",
        "n_reml_iter",
        "n_pirls_iter",
        "peak_mem_mb",
    ]


def test_build_scenarios_includes_single_and_multi_feature_modes():
    scenarios = build_scenarios(max_n=100_000)

    assert any(s.engine == "scop" and s.n_constrained == 1 for s in scenarios)
    assert any(s.engine == "qp" and s.n_constrained == 1 for s in scenarios)
    assert any(s.n_constrained > 1 for s in scenarios)
    assert all(s.n <= 100_000 for s in scenarios)


def test_write_profile_artifacts_creates_expected_files(tmp_path: Path):
    result, stats_text, memory_stats = profile_callstack_and_memory(lambda: sum(range(10)))

    assert result == 45
    assert "sum" in stats_text
    assert memory_stats["peak_mb"] >= 0.0

    paths = write_profile_artifacts(
        base_dir=tmp_path,
        stem="demo",
        profile_stats={"stats": stats_text},
        memory_stats=memory_stats,
    )
    assert paths["cpu_txt"].exists()
    assert paths["memory_json"].exists()


@pytest.mark.parametrize(
    "feature, expected_engine",
    [
        (BSplineSmooth(n_knots=6, constraint=Constraint.fit.increasing), "qp"),
        (PSpline(n_knots=6, constraint=Constraint.fit.increasing), "scop"),
    ],
)
def test_tweedie_profile_constrained_terms_dispatch_to_direct(
    monkeypatch, feature, expected_engine
):
    ctx, call = _evaluate_constrained_profile_once(monkeypatch, feature)

    assert {group.monotone_engine for group in ctx.groups} == {expected_engine}
    assert call["groups"] is ctx.groups


def test_tweedie_profile_rejects_monotone_selection_penalty():
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 24)})
    y = np.linspace(0.5, 2.0, len(X))
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.1,
        features={
            "x": BSplineSmooth(n_knots=6, constraint=Constraint.fit.increasing),
        },
    )

    with pytest.raises(NotImplementedError, match="selection_penalty"):
        tweedie_module._build_profile_context(model, X, y, None, None, "pearson", False)


def test_tweedie_profile_rejects_mixed_scop_and_qp_engines():
    X = pd.DataFrame(
        {
            "x_qp": np.linspace(0.0, 1.0, 24),
            "x_scop": np.linspace(1.0, 0.0, 24),
        }
    )
    y = np.linspace(0.5, 2.0, len(X))
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0,
        features={
            "x_qp": BSplineSmooth(n_knots=6, constraint=Constraint.fit.increasing),
            "x_scop": PSpline(n_knots=6, constraint=Constraint.fit.increasing),
        },
    )

    with pytest.raises(NotImplementedError, match=r"SCOP \+ QP"):
        tweedie_module._build_profile_context(model, X, y, None, None, "pearson", False)


def test_tweedie_profile_rejects_fit_only_lambda_policy():
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 24)})
    y = np.linspace(0.5, 2.0, len(X))
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0,
        features={
            "x": Spline(
                n_knots=6,
                lambda_policy=LambdaPolicy.fixed(1.0),
            )
        },
    )

    with pytest.raises(NotImplementedError, match="lambda_policy"):
        tweedie_module._build_profile_context(model, X, y, None, None, "pearson", False)


def test_tweedie_profile_runs_the_same_response_validation_as_fit(monkeypatch):
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 6)})
    y = np.array([0.0, 1.0, 0.1, 2.0, 1.5, 0.5])
    calls = []

    def validate_response(y_arg, distribution):
        calls.append((y_arg.copy(), distribution))

    monkeypatch.setattr(distributions_module, "validate_response", validate_response)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0,
        features={"x": Numeric()},
    )

    tweedie_module._build_profile_context(model, X, y, None, None, "pearson", False)

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], y)
    assert isinstance(calls[0][1], Tweedie)
