"""Adversarial regressions for atomic installation and evaluator release."""

from __future__ import annotations

import copyreg
import pickle
import threading
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest

import superglm.profiling.tweedie as tweedie_module
from superglm import Numeric, SuperGLM, Tweedie, generate_tweedie_cpg
from superglm.model import fit_ops, profile_ops


def _small_problem(n: int = 32) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a deterministic mixed zero/positive Tweedie sample."""
    rng = np.random.default_rng(20260718)
    x = np.linspace(-1.0, 1.0, n)
    mu = np.exp(0.7 + 0.25 * x)
    y = generate_tweedie_cpg(n, mu=mu, phi=0.8, p=1.5, rng=rng)
    return pd.DataFrame({"x": x}), y


def _new_model(*, retain_fit_state: bool = True) -> SuperGLM:
    return SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={"x": Numeric()},
        retain_fit_state=retain_fit_state,
    )


def _certified_result() -> tweedie_module.TweedieProfileResult:
    """Build a token-stamped result for transaction-boundary tests."""
    trace = pd.DataFrame(
        {
            "p": [1.5],
            "phi": [0.8],
            "nll": [0.0],
            "edf": [1.0],
            "fit_converged": [True],
            "solver_converged": [True],
            "reml_converged": [None],
            "phi_converged": [True],
            "objective_finite": [True],
            "n_saddlepoint": [0],
            "density_method": ["exact"],
            "density_exact": [True],
        }
    )
    return tweedie_module.TweedieProfileResult(
        p_hat=1.5,
        phi_hat=0.8,
        nll=0.0,
        n_evaluations=1,
        converged=True,
        method="brent",
        phi_method="mle",
        search_trace=trace,
        density_method="exact",
        density_exact=True,
        _objective=lambda p, source="": 0.0,
        _ll_scale=1.0,
        _evaluation_count=lambda: 1,
        _evaluation_record=lambda p: None,
        _validation_token=tweedie_module._TWEEDIE_PROFILE_RESULT_TOKEN,
    )


def _install_fake_profile(monkeypatch: pytest.MonkeyPatch, result=None) -> None:
    if result is None:
        result = _certified_result()
    monkeypatch.setattr(
        tweedie_module,
        "_estimate_tweedie_p_prepared",
        lambda *args, **kwargs: result,
    )


def _assert_identity_state(model: SuperGLM, expected: dict[str, object]) -> None:
    assert model.__dict__.keys() == expected.keys()
    for name, value in expected.items():
        assert model.__dict__[name] is value, name


def test_progress_callback_top_level_model_assignment_is_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback covers top-level model state, not arbitrary callback side effects."""
    X, y = _small_problem()
    model = _new_model()
    caller_mapping = vars(model)
    before = model.__dict__.copy()
    _install_fake_profile(monkeypatch)

    def mutate_then_fail(event, payload):
        if event == "best_found":
            model.lambda2 = 777.0
            raise RuntimeError("callback failed after top-level assignment")

    with pytest.raises(RuntimeError, match="top-level assignment"):
        model.estimate_p(X, y, progress_callback=mutate_then_fail)

    _assert_identity_state(model, before)
    assert vars(model) is caller_mapping


def test_early_validation_failure_preserves_caller_mapping_identity() -> None:
    """A rejected transaction must leave retained ``vars(model)`` views live."""
    X, y = _small_problem(n=8)
    model = _new_model()
    caller_mapping = vars(model)
    before = caller_mapping.copy()

    with pytest.raises(ValueError, match="eager_ci_alpha"):
        model.estimate_p(X, y, eager_ci_alpha=0.0)

    _assert_identity_state(model, before)
    assert vars(model) is caller_mapping
    marker = object()
    caller_mapping["mapping_identity_probe"] = marker
    assert model.mapping_identity_probe is marker
    del caller_mapping["mapping_identity_probe"]


class _NewHookPayload:
    armed = False
    calls = 0
    victim: SuperGLM | None = None

    def __new__(cls):
        if cls.armed:
            cls.calls += 1
            assert cls.victim is not None
            cls.victim.lambda2 = 901.0
            raise RuntimeError("custom __new__ executed")
        return super().__new__(cls)


class _SlotsSetattrPayload:
    __slots__ = ("value",)

    armed = False
    calls = 0
    victim: SuperGLM | None = None

    def __init__(self) -> None:
        object.__setattr__(self, "value", 1)

    def __setattr__(self, name, value) -> None:
        if type(self).armed:
            type(self).calls += 1
            assert type(self).victim is not None
            type(self).victim.lambda2 = 902.0
            raise RuntimeError("custom slot __setattr__ executed")
        object.__setattr__(self, name, value)


class _CopyregPayload:
    pass


class _BoundMethodOwner:
    calls = 0
    victim: SuperGLM | None = None

    def callback(self) -> None:
        return None

    def __deepcopy__(self, memo):
        type(self).calls += 1
        assert type(self).victim is not None
        type(self).victim.lambda2 = 904.0
        raise RuntimeError("bound-method owner deepcopy executed")


class _NumpyScalarPayload(np.float64):
    calls = 0
    victim: SuperGLM | None = None

    def __deepcopy__(self, memo):
        type(self).calls += 1
        assert type(self).victim is not None
        type(self).victim.lambda2 = 905.0
        raise RuntimeError("NumPy scalar subclass deepcopy executed")


class _SlotBasePayload:
    __slots__ = ("value",)

    def __init__(self) -> None:
        _SlotBasePayload.value.__set__(self, 1)


class _OverriddenSlotPayload(_SlotBasePayload):
    calls = 0
    victim: SuperGLM | None = None

    @property
    def value(self):
        type(self).calls += 1
        assert type(self).victim is not None
        type(self).victim.lambda2 = 906.0
        raise RuntimeError("overridden slot getter executed")


_TAIL_LOOKUP_CALLS: list[str] = []


class _TailLookupMixin:
    def __getattribute__(self, name):
        _TAIL_LOOKUP_CALLS.append(name)
        return super().__getattribute__(name)


def _assert_clone_payload_rejected(
    model: SuperGLM,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    calls,
) -> None:
    before = model.__dict__.copy()
    with pytest.raises(TypeError, match="cannot safely clone"):
        model.estimate_p(
            X,
            y,
            method="grid",
            grid=np.array([1.5]),
            phi_method="pearson",
        )
    assert calls() == 0
    _assert_identity_state(model, before)


def test_clone_preflight_rejects_custom_new_before_it_executes() -> None:
    X, y = _small_problem()
    model = _new_model()
    payload = _NewHookPayload()
    model._specs["x"].audit_payload = payload
    _NewHookPayload.victim = model
    _NewHookPayload.calls = 0
    _NewHookPayload.armed = True
    try:
        _assert_clone_payload_rejected(model, X, y, calls=lambda: _NewHookPayload.calls)
    finally:
        _NewHookPayload.armed = False
        _NewHookPayload.victim = None


def test_clone_preflight_rejects_slot_setattr_before_it_executes() -> None:
    X, y = _small_problem()
    model = _new_model()
    payload = _SlotsSetattrPayload()
    model._specs["x"].audit_payload = payload
    _SlotsSetattrPayload.victim = model
    _SlotsSetattrPayload.calls = 0
    _SlotsSetattrPayload.armed = True
    try:
        _assert_clone_payload_rejected(
            model,
            X,
            y,
            calls=lambda: _SlotsSetattrPayload.calls,
        )
    finally:
        _SlotsSetattrPayload.armed = False
        _SlotsSetattrPayload.victim = None


def test_clone_preflight_rejects_copyreg_reducer_before_it_executes() -> None:
    X, y = _small_problem()
    model = _new_model()
    model._specs["x"].audit_payload = _CopyregPayload()
    calls = 0

    def mutating_reducer(payload):
        nonlocal calls
        calls += 1
        model.lambda2 = 903.0
        raise RuntimeError("copyreg reducer executed")

    copyreg.pickle(_CopyregPayload, mutating_reducer)
    try:
        _assert_clone_payload_rejected(model, X, y, calls=lambda: calls)
    finally:
        copyreg.dispatch_table.pop(_CopyregPayload, None)


def test_clone_preflight_inspects_bound_method_owner() -> None:
    X, y = _small_problem()
    model = _new_model()
    owner = _BoundMethodOwner()
    model._specs["x"].audit_payload = owner.callback
    _BoundMethodOwner.victim = model
    _BoundMethodOwner.calls = 0
    try:
        _assert_clone_payload_rejected(model, X, y, calls=lambda: _BoundMethodOwner.calls)
    finally:
        _BoundMethodOwner.victim = None


def test_clone_preflight_rejects_numpy_scalar_subclass_hooks() -> None:
    X, y = _small_problem()
    model = _new_model()
    model._specs["x"].audit_payload = _NumpyScalarPayload(1.0)
    _NumpyScalarPayload.victim = model
    _NumpyScalarPayload.calls = 0
    try:
        _assert_clone_payload_rejected(model, X, y, calls=lambda: _NumpyScalarPayload.calls)
    finally:
        _NumpyScalarPayload.victim = None


def test_clone_preflight_covers_scalar_configuration_roots() -> None:
    X, y = _small_problem()
    model = _new_model()
    model._tol = _NumpyScalarPayload(1e-8)
    _NumpyScalarPayload.victim = model
    _NumpyScalarPayload.calls = 0
    try:
        _assert_clone_payload_rejected(model, X, y, calls=lambda: _NumpyScalarPayload.calls)
    finally:
        _NumpyScalarPayload.victim = None


def test_clone_preflight_rejects_overridden_slot_without_calling_getter() -> None:
    X, y = _small_problem()
    model = _new_model()
    model._specs["x"].audit_payload = _OverriddenSlotPayload()
    caller_mapping = vars(model)
    before = caller_mapping.copy()
    _OverriddenSlotPayload.victim = model
    _OverriddenSlotPayload.calls = 0
    try:
        with pytest.raises(TypeError, match="overridden slot"):
            model.estimate_p(
                X,
                y,
                method="grid",
                grid=np.array([1.5]),
                phi_method="pearson",
            )
    finally:
        _OverriddenSlotPayload.victim = None

    assert _OverriddenSlotPayload.calls == 0
    _assert_identity_state(model, before)
    assert vars(model) is caller_mapping


def test_model_preflight_rejects_tail_lookup_mixin_without_executing_it() -> None:
    """Class-only inspection must reject hooks hidden after the public model base."""

    class HookedModel(SuperGLM, _TailLookupMixin):
        pass

    model = HookedModel(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={"x": Numeric()},
    )
    _TAIL_LOOKUP_CALLS.clear()

    with pytest.raises(TypeError, match="custom behavior.*__getattribute__"):
        profile_ops._validate_tweedie_profile_model_storage(model)

    assert _TAIL_LOOKUP_CALLS == []


def test_model_preflight_rejects_dict_descriptor_without_executing_it() -> None:
    """Capturing rollback state bypasses a subclass's ``__dict__`` descriptor."""
    descriptor_calls: list[object] = []

    class DictDescriptorModel(SuperGLM):
        @property
        def __dict__(self):
            descriptor_calls.append(self)
            base_descriptor = type.__getattribute__(SuperGLM, "__dict__")["__dict__"]
            return base_descriptor.__get__(self, type(self))

    X, y = _small_problem(n=8)
    model = DictDescriptorModel(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={"x": Numeric()},
    )
    descriptor_calls.clear()

    with pytest.raises(TypeError, match="custom behavior.*__dict__"):
        model.estimate_p(X, y, method="grid", grid=np.array([1.5]))

    assert descriptor_calls == []


def test_clone_isolation_finds_mutable_owner_through_frozen_slots_link() -> None:
    """Slot-held bound methods cannot hide a caller-owned mutable from isolation."""

    @dataclass(frozen=True, slots=True)
    class FrozenCallbackLink:
        callback: object

        def link(self, mu):
            return np.log(mu)

        def inverse(self, eta):
            return np.exp(eta)

        def deriv(self, mu):
            return 1.0 / mu

        def deriv_inverse(self, eta):
            return np.exp(eta)

    X, y = _small_problem(n=12)
    owner: list[object] = []
    model = SuperGLM(
        family=Tweedie(p=1.5),
        link=FrozenCallbackLink(owner.append),
        selection_penalty=0.0,
        features={"x": Numeric()},
    )
    caller_mapping = vars(model)
    before = caller_mapping.copy()

    with pytest.raises(RuntimeError, match="shares mutable configuration"):
        model.estimate_p(
            X,
            y,
            method="grid",
            grid=np.array([1.5]),
            phi_method="pearson",
        )

    assert owner == []
    _assert_identity_state(model, before)
    assert vars(model) is caller_mapping


def test_nonraising_release_fit_stats_corruption_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _small_problem()
    model = _new_model(retain_fit_state=False)
    caller_mapping = vars(model)
    before = model.__dict__.copy()
    _install_fake_profile(monkeypatch)
    real_release = fit_ops._maybe_release_fit_state

    def corrupt_after_release(candidate):
        real_release(candidate)
        if not candidate._retain_fit_state:
            candidate._fit_stats = replace(
                candidate._fit_stats,
                log_likelihood=candidate._fit_stats.log_likelihood + 5.0,
            )

    monkeypatch.setattr(fit_ops, "_maybe_release_fit_state", corrupt_after_release)

    with pytest.raises(RuntimeError, match="released fit is not installable"):
        model.estimate_p(X, y)

    _assert_identity_state(model, before)
    assert vars(model) is caller_mapping


def test_released_transaction_certifies_detach_postcondition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _small_problem()
    model = _new_model(retain_fit_state=False)
    before = model.__dict__.copy()
    detach_calls = 0

    def ineffective_detach(result) -> None:
        nonlocal detach_calls
        detach_calls += 1

    monkeypatch.setattr(
        tweedie_module.TweedieProfileResult,
        "detach_evaluator",
        ineffective_detach,
    )

    with pytest.raises(RuntimeError, match="detach|released"):
        model.estimate_p(
            X,
            y,
            method="grid",
            grid=np.array([1.5]),
            phi_method="pearson",
        )

    assert detach_calls == 1
    _assert_identity_state(model, before)


class _BlockingEvaluatorContext:
    def __init__(self) -> None:
        self.first_entered = threading.Event()
        self.second_entered = threading.Event()
        self.release_first = threading.Event()
        self.trace_callback = None
        self._state_lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.count = 1

    def evaluate(self, p: float, source: str = "") -> float:
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if p == 1.5:
                self.first_entered.set()
                if not self.release_first.wait(timeout=5):
                    raise TimeoutError("first evaluator call was not released")
            else:
                self.second_entered.set()
        finally:
            with self._state_lock:
                self.active -= 1
                self.count += 1
        return float(p)

    def evaluation_count(self) -> int:
        return self.count

    def evaluation_record(self, p: float):
        return None


def _thread_call(target, errors: list[BaseException]) -> None:
    try:
        target()
    except BaseException as exc:
        errors.append(exc)


def test_retained_evaluator_serializes_concurrent_evaluations() -> None:
    context = _BlockingEvaluatorContext()
    evaluator = tweedie_module._TweedieProfileEvaluator(context)
    errors: list[BaseException] = []
    second_started = threading.Event()

    first = threading.Thread(
        target=_thread_call,
        args=(lambda: evaluator.evaluate(1.5, source="first"), errors),
    )

    def run_second() -> None:
        second_started.set()
        evaluator.evaluate(1.6, source="second")

    second = threading.Thread(target=_thread_call, args=(run_second, errors))
    first.start()
    assert context.first_entered.wait(timeout=2)
    second.start()
    assert second_started.wait(timeout=2)
    overlapped = context.second_entered.wait(timeout=0.5)
    context.release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert not overlapped
    assert context.max_active == 1


@pytest.mark.parametrize("operation", ["detach", "pickle"])
def test_evaluator_lifecycle_serializes_with_inflight_evaluation(operation: str) -> None:
    context = _BlockingEvaluatorContext()
    evaluator = tweedie_module._TweedieProfileEvaluator(context)
    result = _certified_result()
    result._objective = None
    result._evaluation_count = None
    result._evaluation_record = None
    result._evaluator = evaluator
    errors: list[BaseException] = []
    lifecycle_started = threading.Event()
    lifecycle_finished = threading.Event()
    serialized: list[bytes] = []

    evaluation = threading.Thread(
        target=_thread_call,
        args=(lambda: evaluator.evaluate(1.5, source="inflight"), errors),
    )

    def run_lifecycle() -> None:
        lifecycle_started.set()
        if operation == "detach":
            result.detach_evaluator()
        else:
            serialized.append(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
        lifecycle_finished.set()

    lifecycle = threading.Thread(target=_thread_call, args=(run_lifecycle, errors))
    evaluation.start()
    assert context.first_entered.wait(timeout=2)
    lifecycle.start()
    assert lifecycle_started.wait(timeout=2)
    finished_while_active = lifecycle_finished.wait(timeout=0.5)
    context.release_first.set()
    evaluation.join(timeout=2)
    lifecycle.join(timeout=2)

    assert not evaluation.is_alive()
    assert not lifecycle.is_alive()
    assert errors == []
    assert not finished_while_active
    if operation == "detach":
        assert result._evaluator is None
        assert result._frozen_evaluation_count == 2
    else:
        restored = pickle.loads(serialized[0])
        assert restored._evaluator is None
        assert restored.n_total_evaluations == 2
