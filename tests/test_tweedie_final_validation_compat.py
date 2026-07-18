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


@pytest.mark.parametrize("retain_fit_state", [True, False])
def test_profile_fit_guard_is_scoped_to_explicit_feature_columns(retain_fit_state):
    """Unused unhashable columns neither block profiling nor poison fit caches."""
    X, y, sample_weight, offset = _numeric_problem(seed=7211, n=36)
    X = X.rename(columns={"x": "used"})
    X["unused"] = [[index] if index % 2 else {"index": index} for index in range(len(X))]
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={"used": Numeric()},
        retain_fit_state=retain_fit_state,
    )

    _profile_once(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
    )

    if retain_fit_state:
        guard = model._fit_data_guard
        assert guard.x_columns == ("used",)
        assert guard.matches(
            X,
            y,
            sample_weight,
            offset,
            fit_weights=model._fit_weights,
            fit_offset=model._fit_offset,
        )
        first = model.metrics(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
        )
        X.at[X.index[0], "unused"] = {"changed": True}
        second = model.metrics(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
        )
        assert guard.matches(
            X,
            y,
            sample_weight,
            offset,
            fit_weights=model._fit_weights,
            fit_offset=model._fit_offset,
        )
        assert second.log_likelihood == pytest.approx(first.log_likelihood)
    else:
        assert model._fit_data_guard is None
        assert all(
            getattr(model, name) is None
            for name in (
                "_fit_X_ref",
                "_fit_y_ref",
                "_fit_sample_weight_ref",
                "_fit_offset_ref",
            )
        )
        metrics = model.metrics(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
        )
        assert np.isfinite(metrics.log_likelihood)
        assert model._fit_data_guard is None
        assert model._fit_X_ref is None


def test_explicit_column_projection_does_not_copy_unsafe_frame_metadata():
    """Column scoping validates DataFrame metadata before pandas can copy it."""

    class DeepcopyBomb:
        calls = 0

        def __deepcopy__(self, memo):
            type(self).calls += 1
            raise AssertionError("unsafe DataFrame metadata was copied")

    X, y, _sample_weight, _offset = _numeric_problem(seed=7216, n=20)
    X["unused"] = np.arange(len(X), dtype=np.float64)
    X.attrs["bomb"] = DeepcopyBomb()
    model = _numeric_model()

    with pytest.raises(TypeError, match="snapshot.*X"):
        tweedie_module._prepare_tweedie_profile_inputs(model, X, y)

    assert DeepcopyBomb.calls == 0


@pytest.mark.parametrize("retain_fit_state", [True, False])
def test_intercept_only_profile_ignores_all_caller_feature_values(retain_fit_state):
    """An intercept-only profile needs row identity, not unused cell contents."""
    _X, y, sample_weight, offset = _numeric_problem(seed=7214, n=30)
    X = pd.DataFrame(
        {"unused": [[index] if index % 2 else {"index": index} for index in range(len(y))]}
    )
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={},
        retain_fit_state=retain_fit_state,
    )

    _profile_once(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
    )

    assert model._feature_order == []
    assert model._groups == []
    assert np.all(np.isfinite(model.predict(X, offset=offset)))
    if retain_fit_state:
        assert model._fit_data_guard.x_columns == ()
        assert model._fit_data_guard.matches(
            X,
            y,
            sample_weight,
            offset,
            fit_weights=model._fit_weights,
            fit_offset=model._fit_offset,
        )
    else:
        assert model._fit_data_guard is None
        assert model._fit_X_ref is None


def test_profile_publication_accepts_stateless_slot_link_configuration():
    """Coherence validation supports safe slot-backed constructor objects."""

    class SlotLogLink:
        __slots__ = ()

        def link(self, mu):
            return np.log(mu)

        def inverse(self, eta):
            return np.exp(eta)

        def deriv(self, mu):
            return 1.0 / mu

        def deriv_inverse(self, eta):
            return np.exp(eta)

        def deriv2_inverse(self, eta):
            return np.exp(eta)

        def deriv3_inverse(self, eta):
            return np.exp(eta)

    X, y, sample_weight, offset = _numeric_problem(seed=7215, n=34)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        link=SlotLogLink(),
        selection_penalty=0.0,
        features={"x": Numeric()},
    )

    _profile_once(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
    )

    assert type(model._link_config) is SlotLogLink
    assert type(model._config.link) is SlotLogLink
    assert type(model._link) is SlotLogLink


@pytest.mark.parametrize("retain_fit_state", [True, False])
def test_profile_publication_resolves_constructor_and_postfit_interactions_once(
    retain_fit_state,
):
    """Published interaction intent rematerializes without requeueing duplicates."""
    rng = np.random.default_rng(7212)
    n = 36
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    mu = np.exp(0.55 + 0.15 * X["x1"] - 0.1 * X["x2"] + 0.08 * X["x3"])
    y = generate_tweedie_cpg(n, mu=mu, phi=0.75, p=1.5, rng=rng)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={name: Numeric() for name in X.columns},
        interactions=[("x1", "x2")],
        retain_fit_state=retain_fit_state,
    ).fit(X, y)
    model._add_interaction("x1", "x3", name="later")
    config_revision = model._config_revision
    fit_revision = model._fit_revision

    _profile_once(model, X, y)

    assert model._interaction_order == ["x1:x2", "later"]
    assert model._pending_interactions == ()
    assert model._config.interactions == model._pending_interactions
    assert model._config.interaction_order == tuple(model._interaction_order)
    assert tuple(name for name, _ in model._config.interaction_templates) == tuple(
        model._interaction_order
    )
    assert model._config_revision == config_revision + 1
    assert model._fit_revision == fit_revision + 1

    clone = model.clone_unfitted()
    assert clone._pending_interactions == ()
    assert clone._interaction_order == model._interaction_order
    assert tuple(clone._interaction_specs) == tuple(model._interaction_order)
    clone.fit(X, y)
    assert clone._interaction_order == model._interaction_order
    assert [group.name for group in clone._groups].count("x1:x2") == 1
    assert [group.name for group in clone._groups].count("later") == 1


@pytest.mark.parametrize("retain_fit_state", [True, False])
def test_shorthand_profile_publication_rematerializes_resolved_interaction(
    retain_fit_state,
):
    """Auto-detected spline parents survive profile publication and cloning."""
    rng = np.random.default_rng(7213)
    n = 32
    X = pd.DataFrame(
        {
            "x1": np.linspace(-1.0, 1.0, n),
            "x2": rng.normal(size=n),
        }
    )
    mu = np.exp(0.5 + 0.2 * X["x1"] - 0.1 * X["x2"])
    y = generate_tweedie_cpg(n, mu=mu, phi=0.7, p=1.5, rng=rng)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        splines=["x1", "x2"],
        n_knots=[5, 5],
        interactions=[("x1", "x2")],
        retain_fit_state=retain_fit_state,
    )

    _profile_once(model, X, y)

    assert model._feature_order == ["x1", "x2"]
    assert model._interaction_order == ["x1:x2"]
    assert model._pending_interactions == ()
    assert model._config.feature_templates == ()
    assert model._config.interactions == ()
    assert model._config.interaction_order == ("x1:x2",)
    configured_interaction = dict(model._config.interaction_templates)["x1:x2"]
    assert configured_interaction.parent_names == ("x1", "x2")

    clone = model.clone_unfitted()
    assert clone._feature_order == []
    assert clone._pending_interactions == ()
    assert clone._interaction_order == ["x1:x2"]
    clone.fit(X, y)
    assert clone._feature_order == ["x1", "x2"]
    assert clone._interaction_order == ["x1:x2"]
    assert [group.name for group in clone._groups].count("x1:x2") == 1


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
    pattern = re.compile(r"^x$")
    spec = Numeric()
    spec.audit_ufunc = np.square
    spec.audit_pattern = pattern
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={"x": spec},
    )

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
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_released_component_edf_finiteness_is_enforced(
    released_spline,
    monkeypatch,
    field,
    bad_value,
):
    """Per-coefficient EDF caches reject non-finite corruption."""
    model, result = released_spline
    inference = model.__dict__["_fit_inference_info"]
    baseline_failures: list[str] = []
    profile_ops._validate_released_tweedie_inference(model, result, baseline_failures)
    assert baseline_failures == []

    corrupted = inference[field].copy()
    corrupted[0] = bad_value
    monkeypatch.setitem(inference, field, corrupted)

    failures: list[str] = []
    profile_ops._validate_released_tweedie_inference(model, result, failures)
    assert failures, f"released {field} accepted a non-finite component {bad_value}"


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
