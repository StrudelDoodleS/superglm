"""Regression tests for the Tweedie profile evaluator lifecycle."""

import copy
import gc
import inspect
import pickle
import threading
import types
import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest

import superglm.profiling.tweedie as tweedie_module
from superglm import Numeric, SuperGLM, Tweedie, generate_tweedie_cpg

_EAGER_ALPHA = 0.20


@dataclass(frozen=True)
class _CachedProfile:
    model: SuperGLM
    X: pd.DataFrame
    result: tweedie_module.TweedieProfileResult
    interval: tuple[float, float]
    initial_total: int
    final_total: int
    initial_cache: dict[float, tuple[float, float]]
    progress_events: tuple[tuple[str, dict[str, object]], ...] = ()


class _UnpickleableTraceCallback:
    """Exercise callback cleanup without permitting accidental serialization."""

    def __call__(self, row) -> None:
        del row

    def __reduce__(self):
        raise TypeError("trace callback must not be serialized")


class _PickleableDeepcopyBomb:
    """Remain pickleable while detecting accidental deep-copy during pickling."""

    def __deepcopy__(self, memo):
        del memo
        raise AssertionError("pickle/copy.copy must not deep-copy arbitrary result attributes")


def _profile_problem(n: int = 60) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a deterministic, identified exact-profile problem."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=n)
    X = pd.DataFrame({"x": x})
    mu = np.exp(2.0 + 0.3 * x)
    y = generate_tweedie_cpg(n, mu=mu, phi=3.0, p=1.6, rng=rng)
    return X, y


def _profile_model(*, retain_fit_state: bool) -> SuperGLM:
    return SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0,
        features={"x": Numeric()},
        retain_fit_state=retain_fit_state,
    )


def _reachable_objects(root):
    """Yield the object graph while avoiding module and function globals."""
    queue = [root]
    seen: set[int] = set()
    excluded = (types.ModuleType, type, types.CodeType, types.FunctionType)
    scalar = (str, bytes, int, float, complex, bool, type(None))
    while queue:
        value = queue.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        yield value
        for child in gc.get_referents(value):
            if isinstance(child, excluded + scalar):
                continue
            queue.append(child)


@pytest.fixture(scope="module")
def retained_cached_profile() -> _CachedProfile:
    """Fit once, record the lazy state, then populate one real profile CI."""
    X, y = _profile_problem()
    model = _profile_model(retain_fit_state=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = model.estimate_p(
            X,
            y,
            method="brent",
            phi_method="mle",
            xatol=5e-4,
            maxiter=60,
        )
        initial_total = result.n_total_evaluations
        initial_cache = result._ci_cache.copy()
        interval = result.ci(alpha=_EAGER_ALPHA)
    return _CachedProfile(
        model=model,
        X=X,
        result=result,
        interval=interval,
        initial_total=initial_total,
        final_total=result.n_total_evaluations,
        initial_cache=initial_cache,
    )


@pytest.fixture(scope="module")
def released_eager_profile(retained_cached_profile: _CachedProfile) -> _CachedProfile:
    """Fit the same problem through the public released-state path."""
    X, y = _profile_problem()
    model = _profile_model(retain_fit_state=False)
    progress_events = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = model.estimate_p(
            X,
            y,
            method="grid",
            grid=np.array([retained_cached_profile.result.p_hat]),
            phi_method="mle",
            eager_ci_alpha=_EAGER_ALPHA,
            progress_callback=lambda event, payload: progress_events.append((event, payload)),
        )
    return _CachedProfile(
        model=model,
        X=X,
        result=result,
        interval=result._ci_cache[_EAGER_ALPHA],
        initial_total=result.n_evaluations,
        final_total=result.n_total_evaluations,
        initial_cache={},
        progress_events=tuple(progress_events),
    )


def test_public_estimate_p_exposes_keyword_only_eager_ci_alpha():
    parameter = inspect.signature(SuperGLM.estimate_p).parameters["eager_ci_alpha"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_retained_production_profile_keeps_lazy_evaluator(
    retained_cached_profile: _CachedProfile,
):
    profile = retained_cached_profile

    assert profile.initial_cache == {}
    assert profile.initial_total == profile.result.n_evaluations
    assert profile.final_total > profile.initial_total
    assert profile.result._evaluator is not None
    assert profile.result._ci_cache[_EAGER_ALPHA] is profile.interval
    assert profile.result.ci(_EAGER_ALPHA) is profile.interval


def test_released_profile_detaches_and_retains_no_profile_context(
    released_eager_profile: _CachedProfile,
):
    profile = released_eager_profile
    context_types = (tweedie_module._ProfileContext, tweedie_module._ProfileContextREML)

    assert profile.result._evaluator is None
    assert profile.result._objective is None
    assert profile.result._evaluation_count is None
    assert profile.result._evaluation_record is None
    assert not any(isinstance(value, context_types) for value in _reachable_objects(profile.model))


def test_eager_interval_survives_detach_and_other_alpha_has_guidance(
    released_eager_profile: _CachedProfile,
):
    profile = released_eager_profile

    assert profile.result.ci(_EAGER_ALPHA) is profile.interval
    with pytest.raises(RuntimeError, match="eager_ci_alpha.*retain_fit_state"):
        profile.result.ci(0.10)


def test_eager_ci_details_survive_detach_and_uncached_details_have_guidance(
    released_eager_profile: _CachedProfile,
):
    profile = released_eager_profile

    details = profile.result.ci_details(_EAGER_ALPHA)

    assert details is profile.result._ci_details_cache[_EAGER_ALPHA]
    assert details.interval is profile.interval
    with pytest.raises(RuntimeError, match="detached.*serialization"):
        profile.result.ci_details(0.10)


def test_installed_eager_ci_preserves_interval_identity(
    released_eager_profile: _CachedProfile,
):
    profile = released_eager_profile
    installed = profile.model._tweedie_profile_result

    interval = installed.ci(_EAGER_ALPHA)
    details = installed.ci_details(_EAGER_ALPHA)

    assert details.interval is interval
    assert installed._ci_cache[_EAGER_ALPHA] is interval
    assert installed._ci_details_cache[_EAGER_ALPHA] is details
    assert interval is not profile.interval


def test_eager_progress_payload_reports_requested_cached_interval(
    released_eager_profile: _CachedProfile,
):
    profile = released_eager_profile

    assert [event for event, _ in profile.progress_events] == ["best_found", "final_refit"]
    for _, payload in profile.progress_events:
        estimate = payload["profile_estimate"]
        assert estimate["ci_status"] == "available"
        assert estimate["ci_alpha"] == pytest.approx(_EAGER_ALPHA)
        assert (estimate["ci_low"], estimate["ci_high"]) == profile.interval


def test_detach_is_idempotent_and_freezes_evaluation_counts():
    X, y = _profile_problem(n=24)
    model = _profile_model(retain_fit_state=True)
    result = model.estimate_p(
        X,
        y,
        method="grid",
        grid=np.array([1.45]),
        phi_method="pearson",
    )
    evaluator = result._evaluator
    assert evaluator is not None
    assert np.isfinite(evaluator.evaluate(1.55, source="lifecycle_regression"))
    frozen_total = result.n_total_evaluations
    frozen_post_search = result.n_post_search_evaluations

    result.detach_evaluator()
    result.detach_evaluator()

    assert result._evaluator is None
    assert result.n_total_evaluations == frozen_total
    assert result.n_post_search_evaluations == frozen_post_search
    assert result._frozen_evaluation_count == frozen_total


@pytest.mark.parametrize("protocol", [4, pickle.HIGHEST_PROTOCOL])
def test_pickle_detaches_restored_copy_without_mutating_original(
    retained_cached_profile: _CachedProfile,
    protocol: int,
):
    profile = retained_cached_profile
    original_evaluator = profile.result._evaluator
    original_count = profile.result.n_total_evaluations
    original_interval = profile.result._ci_cache[_EAGER_ALPHA]

    restored = pickle.loads(pickle.dumps(profile.result, protocol=protocol))

    assert profile.result._evaluator is original_evaluator
    assert profile.result.n_total_evaluations == original_count
    assert profile.result._ci_cache[_EAGER_ALPHA] is original_interval
    assert restored._evaluator is None
    assert restored.n_total_evaluations == original_count
    assert restored.ci(_EAGER_ALPHA) == original_interval
    assert restored.ci(_EAGER_ALPHA) is restored._ci_cache[_EAGER_ALPHA]
    assert restored.ci_details(_EAGER_ALPHA).interval == original_interval
    context_types = (tweedie_module._ProfileContext, tweedie_module._ProfileContextREML)
    assert not any(isinstance(value, context_types) for value in _reachable_objects(restored))


def test_pickle_and_shallow_copy_detach_without_deepcopying_user_attributes(
    retained_cached_profile: _CachedProfile,
):
    """Snapshots detach library state without deep-copying arbitrary attributes."""
    original = retained_cached_profile.result
    original.self_reference_probe = original
    original.deepcopy_probe = _PickleableDeepcopyBomb()
    try:
        restored = pickle.loads(pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL))
        copied = copy.copy(original)
    finally:
        del original.self_reference_probe
        del original.deepcopy_probe

    assert restored.self_reference_probe is restored
    # Shallow-copy semantics preserve arbitrary user-attached references.
    assert copied.self_reference_probe is original
    for detached in (restored, copied):
        assert detached._evaluator is None
        assert detached._ci_cache == original._ci_cache
        assert detached._ci_cache is not original._ci_cache
        assert detached._ci_details_cache == original._ci_details_cache
        assert detached._ci_details_cache is not original._ci_details_cache


def test_deepcopy_detaches_and_preserves_a_result_self_cycle(
    retained_cached_profile: _CachedProfile,
):
    original = retained_cached_profile.result
    original.self_reference_probe = original
    try:
        copied = copy.deepcopy(original)
    finally:
        del original.self_reference_probe

    assert copied._evaluator is None
    assert copied.self_reference_probe is copied
    assert copied._ci_cache == original._ci_cache
    assert copied._ci_cache is not original._ci_cache


def test_pickle_waits_for_one_coherent_ci_cache_update(
    retained_cached_profile: _CachedProfile,
    monkeypatch,
):
    """Serialization cannot observe the tuple cache without matching details."""
    result = retained_cached_profile.result
    evaluator = result._evaluator
    assert evaluator is not None
    alpha = 0.23
    assert alpha not in result._ci_cache
    template = result.ci_details(_EAGER_ALPHA)
    entered = threading.Event()
    release = threading.Event()
    pickle_started = threading.Event()
    pickle_done = threading.Event()
    errors: list[BaseException] = []
    payloads: list[bytes] = []

    def blocked_details(*args, **kwargs):
        del args, kwargs
        entered.set()
        if not release.wait(timeout=5):
            raise AssertionError("timed out waiting to release the synthetic CI")
        return replace(template, alpha=alpha)

    monkeypatch.setattr(tweedie_module, "_profile_ci_p_detailed", blocked_details)

    def calculate_ci() -> None:
        try:
            result.ci(alpha)
        except BaseException as exc:  # pragma: no cover - surfaced in the main thread
            errors.append(exc)

    def serialize() -> None:
        pickle_started.set()
        try:
            payloads.append(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
        except BaseException as exc:  # pragma: no cover - surfaced in the main thread
            errors.append(exc)
        finally:
            pickle_done.set()

    ci_thread = threading.Thread(target=calculate_ci)
    pickle_thread = threading.Thread(target=serialize)
    ci_thread.start()
    assert entered.wait(timeout=5)
    pickle_thread.start()
    assert pickle_started.wait(timeout=5)
    assert not pickle_done.wait(timeout=0.1)
    release.set()
    ci_thread.join(timeout=5)
    pickle_thread.join(timeout=5)

    try:
        assert not ci_thread.is_alive()
        assert not pickle_thread.is_alive()
        assert errors == []
        assert len(payloads) == 1
        restored = pickle.loads(payloads[0])
        assert restored._evaluator is None
        assert restored._ci_cache[alpha] == template.interval
        assert restored._ci_details_cache[alpha].alpha == alpha
        assert restored._ci_details_cache[alpha].interval == restored._ci_cache[alpha]
        assert restored.n_total_evaluations == result.n_total_evaluations
    finally:
        with evaluator._lock:
            result._ci_cache.pop(alpha, None)
            result._ci_details_cache.pop(alpha, None)


def test_pickle_snapshot_does_not_follow_a_later_ci_cache_mutation(
    retained_cached_profile: _CachedProfile,
    monkeypatch,
):
    """A state snapshot owns cache containers before the pickler traverses them."""
    result = retained_cached_profile.result
    evaluator = result._evaluator
    assert evaluator is not None
    alpha = 0.24
    assert alpha not in result._ci_cache
    template = result.ci_details(_EAGER_ALPHA)
    initial_total = result.n_total_evaluations
    original_frozen_count = result._frozen_evaluation_count
    original_warnings = result.warnings
    barrier_entered = threading.Event()
    barrier_release = threading.Event()
    counter = [initial_total]
    payloads: list[bytes] = []
    errors: list[BaseException] = []

    class PickleBarrier:
        def __reduce__(self):
            barrier_entered.set()
            if not barrier_release.wait(timeout=5):
                raise AssertionError("timed out waiting to resume pickle traversal")
            return str, ("pickle-barrier",)

    result.warnings = [PickleBarrier(), *list(original_warnings)]
    state_keys = list(result.__dict__)
    assert state_keys.index("warnings") < state_keys.index("_ci_cache")
    monkeypatch.setattr(evaluator.context, "evaluation_count", lambda: counter[0])

    def synthetic_details(*args, **kwargs):
        del args, kwargs
        counter[0] = initial_total + 1
        return replace(template, alpha=alpha)

    monkeypatch.setattr(tweedie_module, "_profile_ci_p_detailed", synthetic_details)

    def serialize() -> None:
        try:
            payloads.append(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
        except BaseException as exc:  # pragma: no cover - surfaced in the main thread
            errors.append(exc)

    pickle_thread = threading.Thread(target=serialize)
    pickle_thread.start()
    try:
        assert barrier_entered.wait(timeout=5)
        interval = result.ci(alpha)
        assert result._ci_cache[alpha] is interval
        assert result.n_total_evaluations == initial_total + 1
    finally:
        barrier_release.set()
        pickle_thread.join(timeout=5)
        with evaluator._lock:
            result.warnings = original_warnings
            result._ci_cache.pop(alpha, None)
            result._ci_details_cache.pop(alpha, None)
            result._frozen_evaluation_count = original_frozen_count

    assert not pickle_thread.is_alive()
    assert errors == []
    assert len(payloads) == 1
    restored = pickle.loads(payloads[0])
    assert alpha not in restored._ci_cache
    assert alpha not in restored._ci_details_cache
    assert restored.n_total_evaluations == initial_total


def test_modern_count_high_water_cannot_be_overwritten_after_detach() -> None:
    """Counting and detachment share one lock through the high-water assignment."""
    assignment_entered = threading.Event()
    assignment_release = threading.Event()
    detach_done = threading.Event()
    reader_ident: list[int | None] = [None]
    errors: list[BaseException] = []
    totals: list[int] = []

    class BlockingAssignmentResult(tweedie_module.TweedieProfileResult):
        def __setattr__(self, name, value):
            if name == "_frozen_evaluation_count" and reader_ident[0] == threading.get_ident():
                assignment_entered.set()
                if not assignment_release.wait(timeout=5):
                    raise AssertionError("timed out waiting to store the observed count")
            super().__setattr__(name, value)

    class CountContext:
        value = 7

        def evaluation_count(self):
            return self.value

    context = CountContext()
    evaluator = tweedie_module._TweedieProfileEvaluator(context)
    result = BlockingAssignmentResult(
        p_hat=1.5,
        phi_hat=1.0,
        nll=0.0,
        n_evaluations=1,
        converged=True,
        method="grid",
        phi_method="pearson",
        search_trace=pd.DataFrame({"p": [1.5], "nll": [0.0]}),
        _evaluator=evaluator,
    )

    def read_total() -> None:
        reader_ident[0] = threading.get_ident()
        try:
            totals.append(result.n_total_evaluations)
        except BaseException as exc:  # pragma: no cover - surfaced in the main thread
            errors.append(exc)

    def detach() -> None:
        try:
            result.detach_evaluator()
        except BaseException as exc:  # pragma: no cover - surfaced in the main thread
            errors.append(exc)
        finally:
            detach_done.set()

    reader_thread = threading.Thread(target=read_total)
    detach_thread = threading.Thread(target=detach)
    reader_thread.start()
    assert assignment_entered.wait(timeout=5)
    context.value = 8
    detach_thread.start()
    assert not detach_done.wait(timeout=0.1)
    assignment_release.set()
    reader_thread.join(timeout=5)
    detach_thread.join(timeout=5)

    assert not reader_thread.is_alive()
    assert not detach_thread.is_alive()
    assert errors == []
    assert totals == [7]
    assert result._evaluator is None
    assert result._frozen_evaluation_count == 8
    assert result.n_total_evaluations == 8


def test_observed_legacy_evaluation_count_survives_detach():
    observed = [7]
    result = tweedie_module.TweedieProfileResult(
        p_hat=1.5,
        phi_hat=1.0,
        nll=0.0,
        n_evaluations=1,
        converged=True,
        method="grid",
        phi_method="pearson",
        search_trace=pd.DataFrame({"p": [1.5], "nll": [0.0]}),
        _objective=lambda p: float(p),
        _evaluation_count=lambda: observed[0],
    )

    assert result.n_total_evaluations == 7
    observed[0] = 2
    result.detach_evaluator()

    assert result.n_total_evaluations == 7
    assert result.n_post_search_evaluations == 6


@pytest.mark.parametrize(
    "profile_fixture",
    ["retained_cached_profile", "released_eager_profile"],
)
@pytest.mark.parametrize("protocol", [4, pickle.HIGHEST_PROTOCOL])
def test_model_pickle_round_trip_preserves_predictions_and_detaches_profile(
    request,
    profile_fixture: str,
    protocol: int,
):
    profile = request.getfixturevalue(profile_fixture)
    original_evaluator = profile.result._evaluator
    installed_result = profile.model._tweedie_profile_result
    installed_interval = installed_result.ci(_EAGER_ALPHA)
    installed_details = installed_result.ci_details(_EAGER_ALPHA)
    original_predictions = profile.model.predict(profile.X)

    restored_model = pickle.loads(pickle.dumps(profile.model, protocol=protocol))
    restored_result = restored_model._tweedie_profile_result
    restored_interval = restored_result.ci(_EAGER_ALPHA)
    restored_details = restored_result.ci_details(_EAGER_ALPHA)

    np.testing.assert_allclose(restored_model.predict(profile.X), original_predictions)
    assert profile.result._evaluator is original_evaluator
    assert installed_details.interval is installed_interval
    assert restored_result._evaluator is None
    assert restored_interval == pytest.approx(profile.interval)
    assert restored_details.interval is restored_interval


def test_detached_and_restored_results_keep_trace_plot_but_reject_dense_plot(
    released_eager_profile: _CachedProfile,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detached = released_eager_profile.result
    restored = pickle.loads(pickle.dumps(detached, protocol=pickle.HIGHEST_PROTOCOL))

    for result in (detached, restored):
        figure = result.trace_plot()
        assert figure.axes[0].get_title()
        plt.close(figure)
        with pytest.raises(RuntimeError, match="detached.*serialization.*trace_plot"):
            result.profile_plot(n_points=3)


def test_direct_legacy_state_ingestion_detaches_callbacks_and_preserves_tuple_cache(
    retained_cached_profile: _CachedProfile,
):
    legacy_state = retained_cached_profile.result.__dict__.copy()
    for name in ("_evaluator", "_frozen_evaluation_count", "_ci_details_cache"):
        legacy_state.pop(name, None)
    legacy_state["_objective"] = lambda p: float(p)
    legacy_state["_evaluation_count"] = lambda: 999
    legacy_state["_evaluation_record"] = lambda p: float(p)

    restored = object.__new__(tweedie_module.TweedieProfileResult)
    restored.__setstate__(legacy_state)

    assert restored._evaluator is None
    assert restored._objective is None
    assert restored._evaluation_count is None
    assert restored._evaluation_record is None
    assert restored._ci_details_cache == {}
    assert restored.ci(_EAGER_ALPHA) is legacy_state["_ci_cache"][_EAGER_ALPHA]
    with pytest.raises(RuntimeError, match="tuple-only cache"):
        restored.ci_details(_EAGER_ALPHA)


def test_unpickleable_callback_released_model_round_trip_is_uncached_and_detached():
    X, y = _profile_problem(n=24)
    model = _profile_model(retain_fit_state=False)
    result = model.estimate_p(
        X,
        y,
        method="grid",
        grid=np.array([1.5]),
        phi_method="mle",
        trace_callback=_UnpickleableTraceCallback(),
    )

    restored_model = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    restored_result = restored_model._tweedie_profile_result

    np.testing.assert_allclose(restored_model.predict(X), model.predict(X))
    assert result._ci_cache == {}
    assert restored_result._evaluator is None
    with pytest.raises(RuntimeError, match="detached.*serialization"):
        restored_result.ci(_EAGER_ALPHA)


def test_released_model_pickle_size_does_not_scale_with_training_rows():
    serialized_sizes = []
    for n_rows in (100, 1_000, 10_000):
        X = pd.DataFrame(index=pd.RangeIndex(n_rows))
        y = np.zeros(n_rows, dtype=np.float64)
        y[::100] = 1.0
        model = SuperGLM(
            family=Tweedie(p=1.5),
            selection_penalty=0,
            features={},
            retain_fit_state=False,
        )

        result = model.estimate_p(
            X,
            y,
            method="grid",
            grid=np.array([1.5]),
            phi_method="pearson",
        )
        reachable = tuple(_reachable_objects(model))

        assert result._evaluator is None
        assert not any(
            isinstance(value, pd.DataFrame) and len(value) == n_rows for value in reachable
        )
        assert not any(
            isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == n_rows
            for value in reachable
        )
        serialized_sizes.append(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)))

    assert serialized_sizes[-1] < 2.0 * serialized_sizes[0]


@pytest.mark.parametrize(
    "eager_ci_alpha",
    [
        -0.1,
        0.0,
        1.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        "0.05",
        np.array(0.05),
        np.ma.array(0.05, mask=True),
    ],
)
def test_invalid_eager_alpha_rejects_before_profile_preparation(monkeypatch, eager_ci_alpha):
    X, y = _profile_problem(n=8)
    model = _profile_model(retain_fit_state=True)

    def unexpected_profile_preparation(*args, **kwargs):
        raise AssertionError("profile preparation must not run")

    monkeypatch.setattr(
        tweedie_module,
        "_prepare_tweedie_profile_inputs",
        unexpected_profile_preparation,
    )

    with pytest.raises(ValueError, match="eager_ci_alpha"):
        model.estimate_p(X, y, eager_ci_alpha=eager_ci_alpha)


def test_eager_pearson_combination_rejects_before_profile_preparation(monkeypatch):
    X, y = _profile_problem(n=8)
    model = _profile_model(retain_fit_state=True)

    def unexpected_profile_preparation(*args, **kwargs):
        raise AssertionError("profile preparation must not run")

    monkeypatch.setattr(
        tweedie_module,
        "_prepare_tweedie_profile_inputs",
        unexpected_profile_preparation,
    )

    with pytest.raises(ValueError, match="requires phi_method='mle'"):
        model.estimate_p(X, y, phi_method="pearson", eager_ci_alpha=_EAGER_ALPHA)
