"""Final compatibility and corruption gates for Tweedie profile installation."""

from __future__ import annotations

import copy
import re
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    LambdaPolicy,
    Numeric,
    Spline,
    SuperGLM,
    Tweedie,
    generate_tweedie_cpg,
)
from superglm.model import fit_ops, profile_ops
from superglm.profiling import tweedie as tweedie_module


def _numeric_problem(
    *,
    seed: int = 20260718,
    n: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Return a small identified problem with strict-positive weights and an offset."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    offset = 0.12 * np.sin(x)
    sample_weight = rng.uniform(0.6, 1.4, size=n)
    mu = np.exp(0.65 + 0.25 * x + offset)
    y = generate_tweedie_cpg(n, mu=mu, phi=0.8, p=1.5, rng=rng)
    return pd.DataFrame({"x": x}), y, sample_weight, offset


def _numeric_model(*, retain_fit_state: bool = True, discrete: bool = False) -> SuperGLM:
    """Build the common one-feature model used by validation tests."""
    return SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={"x": Numeric()},
        retain_fit_state=retain_fit_state,
        discrete=discrete,
    )


def _profile_once(
    model: SuperGLM,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    fit_mode: str = "fit",
    phi_method: str = "pearson",
):
    """Run one deterministic public profile transaction without warning noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return model.estimate_p(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            fit_mode=fit_mode,
            method="grid",
            grid=np.array([1.5]),
            phi_method=phi_method,
        )


def _prepared_result(
    *,
    seed: int,
) -> tuple[tweedie_module._PreparedTweedieProfileInputs, tweedie_module.TweedieProfileResult]:
    """Build a genuine token-stamped result from its exact prepared input graph."""
    X, y, sample_weight, offset = _numeric_problem(seed=seed, n=34)
    model = _numeric_model()
    prepared = tweedie_module._prepare_tweedie_profile_inputs(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
        method="grid",
        grid=np.array([1.5]),
        phi_method="mle",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = tweedie_module._estimate_tweedie_p_prepared(model, prepared)
    return prepared, result


def _live_result_copy(original):
    """Copy public fields without letting the serialization hook detach provenance."""
    candidate = copy.copy(original)
    for name in (
        "_evaluator",
        "_objective",
        "_evaluation_count",
        "_evaluation_record",
        "_frozen_evaluation_count",
    ):
        setattr(candidate, name, getattr(original, name))
    return candidate


@pytest.fixture(scope="module")
def certified_pair():
    """Return two independent authentic profile records with equal row counts."""
    return _prepared_result(seed=7181), _prepared_result(seed=7182)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("n_positive", -1),
        ("density_warning_severity", "high"),
        ("near_power_boundary", True),
        ("outer_boundary", "lower"),
        ("phi_n_evaluations", -1),
        ("phi_n_score_evaluations", -1),
        ("phi_score", np.nan),
        ("phi_optimizer", ""),
        ("warnings", [object()]),
    ],
)
def test_result_diagnostic_corruption_is_rejected(certified_pair, field, bad_value):
    """Every installed public diagnostic must remain certified, not merely finite."""
    (prepared, original), _ = certified_pair
    candidate = _live_result_copy(original)
    candidate.search_trace = original.search_trace.copy(deep=True)
    setattr(candidate, field, bad_value)

    with pytest.raises(RuntimeError, match="not installable"):
        profile_ops._validate_tweedie_profile_result_for_refit(candidate, prepared)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("n_positive", -1),
        ("phi_n_evaluations", -1),
        ("phi_optimizer", "wrong-optimizer"),
        ("phi_score", np.nan),
    ],
)
def test_winner_trace_diagnostic_corruption_is_rejected(certified_pair, field, bad_value):
    """Winner diagnostics must agree with the immutable top-level profile record."""
    (prepared, original), _ = certified_pair
    candidate = _live_result_copy(original)
    candidate.search_trace = original.search_trace.copy(deep=True)
    winner_index = candidate.search_trace.index[candidate.search_trace["p"] == candidate.p_hat][0]
    candidate.search_trace.at[winner_index, field] = bad_value

    with pytest.raises(RuntimeError, match="not installable"):
        profile_ops._validate_tweedie_profile_result_for_refit(candidate, prepared)


def test_foreign_live_evaluator_is_rejected(certified_pair):
    """A lazy evaluator from another dataset must never be installed."""
    (prepared, original), (_, foreign) = certified_pair
    candidate = _live_result_copy(original)
    candidate.search_trace = original.search_trace.copy(deep=True)
    candidate._evaluator = foreign._evaluator

    with pytest.raises(RuntimeError, match="evaluator|provenance|record"):
        profile_ops._validate_tweedie_profile_result_for_refit(candidate, prepared)


@pytest.fixture(scope="module")
def retained_stage():
    """Return one fully synchronized retained stage and its validator arguments."""
    X, y, sample_weight, offset = _numeric_problem(seed=7190)
    model = _numeric_model(retain_fit_state=True)
    result = _profile_once(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
    )
    args = (
        model,
        model._fit_X_ref,
        model._fit_y_ref,
        model._fit_sample_weight_ref,
        model._fit_offset_ref,
        result,
        "fit",
    )
    profile_ops._validate_tweedie_profile_stage(*args)
    return model, args


@pytest.mark.parametrize("target", ["_result", "_solver_result"])
@pytest.mark.parametrize("field", ["deviance", "effective_df", "n_iter"])
def test_public_private_solver_scalar_divergence_is_rejected(
    retained_stage,
    monkeypatch,
    target,
    field,
):
    """Canonical and private solver records must describe the same terminal fit."""
    model, args = retained_stage
    original = getattr(model, target)
    increment = 1 if field == "n_iter" else 1.0
    monkeypatch.setattr(
        model, target, replace(original, **{field: getattr(original, field) + increment})
    )

    with pytest.raises(RuntimeError, match="scalar coherence"):
        profile_ops._validate_tweedie_profile_stage(*args)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("_last_fit_meta", {"method": "fit_reml", "discrete": False}),
        ("_reml_result", SimpleNamespace(converged=True)),
        ("_reml_lambdas", {"x": 999.0}),
        ("_reml_penalties", []),
    ],
)
def test_terminal_fit_mode_provenance_is_rejected(retained_stage, monkeypatch, field, bad_value):
    """A plain final fit cannot carry REML metadata or claim REML provenance."""
    model, args = retained_stage
    monkeypatch.setattr(model, field, bad_value)

    with pytest.raises(RuntimeError, match="fit-mode|REML"):
        profile_ops._validate_tweedie_profile_stage(*args)


@pytest.fixture(scope="module")
def released_stage():
    """Release one valid synchronized stage while retaining its pre-release certificate."""
    X, y, sample_weight, offset = _numeric_problem(seed=7200)
    model = _numeric_model(retain_fit_state=True)
    result = _profile_once(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
    )
    model._retain_fit_state = False
    release_core = profile_ops._snapshot_tweedie_profile_release_core(model)
    fit_ops._maybe_release_fit_state(model)
    profile_ops._validate_released_tweedie_profile_stage(
        model,
        X,
        offset,
        result,
        release_core=release_core,
    )
    return model, X, offset, result, release_core


@pytest.mark.parametrize(
    "corruption",
    [
        "fit_stats",
        "family",
        "distribution",
        "public_beta",
        "solver_beta",
    ],
)
def test_released_core_semantic_mutation_is_rejected(released_stage, monkeypatch, corruption):
    """Row release may distill caches but may not alter the fitted model itself."""
    model, X, offset, result, release_core = released_stage
    if corruption == "fit_stats":
        monkeypatch.setattr(
            model,
            "_fit_stats",
            replace(model._fit_stats, log_likelihood=np.nan),
        )
    elif corruption == "family":
        monkeypatch.setattr(model, "family", Tweedie(p=1.7))
    elif corruption == "distribution":
        monkeypatch.setattr(model, "_distribution", Tweedie(p=1.7))
    elif corruption == "public_beta":
        monkeypatch.setattr(
            model,
            "_result",
            replace(model._result, beta=model._result.beta + 2.0),
        )
    else:
        monkeypatch.setattr(
            model,
            "_solver_result",
            replace(model._solver_result, beta=model._solver_result.beta + 2.0),
        )

    with pytest.raises(RuntimeError, match="released"):
        profile_ops._validate_released_tweedie_profile_stage(
            model,
            X,
            offset,
            result,
            release_core=release_core,
        )


@pytest.mark.parametrize("fit_mode", ["fit", "reml"])
@pytest.mark.parametrize("retain_fit_state", [True, False])
def test_discrete_profile_uses_fit_scale_prediction_contract(fit_mode, retain_fit_state):
    """Discrete final fits compare training means through the discrete scorer."""
    X, y, sample_weight, offset = _numeric_problem(seed=7210, n=72)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={"x": Spline(n_knots=5, penalty="ssp")},
        discrete=True,
        n_bins=16,
        retain_fit_state=retain_fit_state,
    )

    result = _profile_once(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
        fit_mode=fit_mode,
    )

    assert result.p_hat == 1.5
    assert model._last_fit_meta == {
        "method": "fit_reml" if fit_mode == "reml" else "fit",
        "discrete": True,
    }
    prediction = model.predict(X, offset=offset)
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction > 0.0)
    if retain_fit_state:
        np.testing.assert_allclose(
            model._predict_fast_discrete(X, offset=offset),
            model._fit_mu,
            rtol=1e-12,
            atol=1e-13,
        )
    else:
        assert model._dm is None
        assert model.__dict__["_fit_inference_info"]["W"].shape == (0,)


def _typed_categories(kind: str):
    """Return two valid immutable levels of the requested public category type."""
    if kind == "datetime":
        return pd.date_range("2026-01-01", periods=2, freq="D")
    if kind == "timedelta":
        return pd.timedelta_range("1D", periods=2, freq="2D")
    if kind == "interval":
        return pd.IntervalIndex.from_breaks([0, 1, 2])
    if kind == "python-datetime":
        return pd.Index([datetime(2026, 1, 1), datetime(2026, 1, 2)])
    if kind == "python-timedelta":
        return pd.Index([timedelta(days=1), timedelta(days=2)])
    return pd.Index([UUID(int=1), UUID(int=2)], dtype=object)


@pytest.mark.parametrize(
    "kind",
    [
        "datetime",
        "timedelta",
        "interval",
        "python-datetime",
        "python-timedelta",
        "uuid",
    ],
)
def test_fitted_typed_categorical_configuration_remains_profileable(kind):
    """Resolved immutable category levels remain safe to clone for a later profile."""
    rng = np.random.default_rng(7220)
    n = 32
    categories = _typed_categories(kind)
    X = pd.DataFrame(
        {
            "cat": pd.Categorical.from_codes(
                np.arange(n) % 2,
                categories=categories,
                ordered=True,
            )
        }
    )
    y = generate_tweedie_cpg(n, mu=np.full(n, 2.0), phi=0.8, p=1.5, rng=rng)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={"cat": Categorical(base="first")},
    )
    model.fit(X, y)

    result = _profile_once(model, X, y)

    assert result.p_hat == 1.5
    assert np.all(np.isfinite(model.predict(X)))


def test_lambda_policy_configuration_remains_profileable():
    """Known immutable LambdaPolicy objects are safe REML profile configuration."""
    X, y, sample_weight, offset = _numeric_problem(seed=7230, n=54)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={
            "x": Spline(
                kind="cr",
                n_knots=5,
                lambda_policy=LambdaPolicy.fixed(1.0),
            )
        },
    )

    result = _profile_once(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
        fit_mode="reml",
    )

    assert result.p_hat == 1.5
    assert model._reml_lambdas is not None
    assert set(model._reml_lambdas.values()) == {1.0}


def test_trivial_superglm_subclass_remains_profileable():
    """A state-compatible marker subclass preserves the public model extension seam."""

    class DerivedModel(SuperGLM):
        pass

    X, y, sample_weight, offset = _numeric_problem(seed=7231, n=42)
    model = DerivedModel(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={"x": Numeric()},
    )

    result = _profile_once(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
    )

    assert result.p_hat == 1.5
    assert type(model) is DerivedModel


def test_frozen_slots_link_remains_profileable():
    """Generated frozen-dataclass copy hooks are safe when every field is safe."""

    @dataclass(frozen=True, slots=True)
    class FrozenLogLink:
        marker: str = "safe"

        def link(self, mu):
            return np.log(mu)

        def inverse(self, eta):
            return np.exp(eta)

        def deriv(self, mu):
            return 1.0 / mu

        def deriv_inverse(self, eta):
            return np.exp(eta)

    X, y, sample_weight, offset = _numeric_problem(seed=7232, n=42)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        link=FrozenLogLink(),
        selection_penalty=0.0,
        features={"x": Numeric()},
    )
    model.fit(X, y, sample_weight=sample_weight, offset=offset)

    result = _profile_once(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
    )

    assert result.p_hat == 1.5
    assert isinstance(model.link, FrozenLogLink)


def test_immutable_registered_reducers_remain_profileable():
    """Standard immutable reducer-backed callables/configuration stay supported."""
    X, y, sample_weight, offset = _numeric_problem(seed=7233, n=42)
    model = _numeric_model()
    pattern = re.compile(r"^x$")
    model._specs["x"].audit_ufunc = np.square
    model._specs["x"].audit_pattern = pattern

    result = _profile_once(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
    )

    assert result.p_hat == 1.5
    assert model._specs["x"].audit_ufunc is np.square
    assert model._specs["x"].audit_pattern is pattern


@pytest.fixture(scope="module")
def released_spline():
    """Return a nontrivial released spline cache for EDF corruption checks."""
    X, y, sample_weight, offset = _numeric_problem(seed=7240, n=54)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={"x": Spline(n_knots=5, penalty="ssp")},
        retain_fit_state=False,
    )
    result = _profile_once(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
    )
    return model, result


@pytest.mark.parametrize("field", ["edf", "edf1"])
@pytest.mark.parametrize("bad_value", [-0.02, 1.02])
def test_released_component_edf_bounds_are_enforced(
    released_spline,
    monkeypatch,
    field,
    bad_value,
):
    """Per-coefficient EDF caches cannot contain impossible finite components."""
    model, result = released_spline
    inference = model.__dict__["_fit_inference_info"]
    baseline_failures: list[str] = []
    profile_ops._validate_released_tweedie_inference(model, result, baseline_failures)
    assert baseline_failures == []

    corrupted = inference[field].copy()
    original_sum = float(np.sum(corrupted))
    corrupted[0] = bad_value
    if field == "edf":
        corrupted[1] += original_sum - float(np.sum(corrupted))
    monkeypatch.setitem(inference, field, corrupted)

    failures: list[str] = []
    profile_ops._validate_released_tweedie_inference(model, result, failures)
    assert failures, f"released {field} accepted an impossible component {bad_value}"


@pytest.mark.parametrize(
    ("selection_penalty", "expected_active"),
    [(10.0, ["z"]), (100.0, [])],
)
def test_released_sparse_group_map_allows_zero_edf_inactive_groups(
    selection_penalty,
    expected_active,
):
    """Rank-aware released maps retain valid zero entries for inactive original groups."""
    rng = np.random.default_rng(18)
    n = 50
    X = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "cat": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        }
    )
    y = generate_tweedie_cpg(n, mu=np.full(n, 2.0), phi=0.8, p=1.5, rng=rng)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        features={
            "x": Numeric(),
            "z": Numeric(),
            "cat": Categorical(base="first"),
        },
        selection_penalty=selection_penalty,
        active_set=True,
        retain_fit_state=False,
    )

    _profile_once(model, X, y)

    inference = model.__dict__["_fit_inference_info"]
    assert [group.name for group in inference["active_groups"]] == expected_active
    inactive = set(inference["group_edf_map"]) - set(expected_active)
    assert all(abs(inference["group_edf_map"][name]) <= 1e-8 for name in inactive)
