"""Concurrency regressions for the Tweedie profile transaction guard."""

from __future__ import annotations

import pickle
import threading
from types import SimpleNamespace

import pytest

from superglm.model import profile_ops as profile_ops_module

_ACTIVE_CALL_ERROR = "Tweedie power estimation is already active"


def _estimate(model, **kwargs):
    """Enter the public profile wrapper with inputs owned by the test seam."""
    return profile_ops_module.estimate_p(model, object(), object(), **kwargs)


@pytest.mark.parametrize("callback_name", ["trace_callback", "progress_callback"])
def test_same_thread_callback_reentry_fails_fast_and_preserves_outer_guard(
    monkeypatch, callback_name
):
    model = SimpleNamespace(label=callback_name)
    completed = object()
    transaction_calls = []

    def fake_transaction(candidate, *args, progress_callback=None, **kwargs):
        transaction_calls.append(candidate)
        if callback_name == "trace_callback":
            callback = kwargs.get("trace_callback")
            if callback is not None:
                callback({"p": 1.5})
        elif progress_callback is not None:
            progress_callback("best_found", {"p": 1.5})
        return completed

    monkeypatch.setattr(profile_ops_module, "_estimate_p_transaction", fake_transaction)

    reentry_errors = []

    def reenter(*_args):
        # The first rejection must not accidentally clear the outer call's guard.
        for _ in range(2):
            with pytest.raises(RuntimeError, match=_ACTIVE_CALL_ERROR) as caught:
                _estimate(model)
            reentry_errors.append(caught.value)

    result = _estimate(model, **{callback_name: reenter})

    assert result is completed
    assert len(reentry_errors) == 2
    assert transaction_calls == [model]

    # Normal outer completion releases the model for a later profile call.
    assert _estimate(model) is completed
    assert transaction_calls == [model, model]


def test_failed_transaction_releases_guard(monkeypatch):
    model = SimpleNamespace(label="failure-cleanup")
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise LookupError("profile failed")
        return "recovered"

    monkeypatch.setattr(profile_ops_module, "_estimate_p_transaction", fail_once)

    with pytest.raises(LookupError, match="profile failed"):
        _estimate(model)

    assert _estimate(model) == "recovered"
    assert attempts == 2


def test_two_threads_reject_same_model_overlap_without_deadlock(monkeypatch):
    model = SimpleNamespace(label="shared")
    owner_entered = threading.Event()
    release_owner = threading.Event()
    outcomes = {}

    def blocking_transaction(*args, **kwargs):
        owner_entered.set()
        if not release_owner.wait(timeout=5):
            raise TimeoutError("test did not release the profile owner")
        return "owner-complete"

    monkeypatch.setattr(profile_ops_module, "_estimate_p_transaction", blocking_transaction)

    def invoke(label):
        try:
            outcomes[label] = _estimate(model)
        except BaseException as exc:  # capture thread failures for the main test
            outcomes[label] = exc

    owner = threading.Thread(target=invoke, args=("owner",), name="tweedie-profile-owner")
    contender = threading.Thread(
        target=invoke,
        args=("contender",),
        name="tweedie-profile-contender",
    )
    owner.start()
    try:
        assert owner_entered.wait(timeout=2), "owner never entered the profile transaction"
        contender.start()
        contender.join(timeout=2)

        assert not contender.is_alive(), "same-model rejection deadlocked"
        assert isinstance(outcomes.get("contender"), RuntimeError)
        assert _ACTIVE_CALL_ERROR in str(outcomes["contender"])
        assert owner.is_alive()
    finally:
        release_owner.set()
        owner.join(timeout=2)
        if contender.ident is not None:
            contender.join(timeout=2)

    assert not owner.is_alive()
    assert not contender.is_alive()
    assert outcomes.get("owner") == "owner-complete"

    # The rejected contender must not remove the owner's registry entry early,
    # and owner completion must remove it for subsequent calls.
    assert _estimate(model) == "owner-complete"


def test_different_models_can_profile_concurrently(monkeypatch):
    models = [SimpleNamespace(label="left"), SimpleNamespace(label="right")]
    entered = {id(model): threading.Event() for model in models}
    release_calls = threading.Event()
    outcomes = {}

    def blocking_transaction(candidate, *args, **kwargs):
        entered[id(candidate)].set()
        if not release_calls.wait(timeout=5):
            raise TimeoutError("test did not release concurrent profile calls")
        return candidate.label

    monkeypatch.setattr(profile_ops_module, "_estimate_p_transaction", blocking_transaction)

    def invoke(model):
        try:
            outcomes[model.label] = _estimate(model)
        except BaseException as exc:  # capture thread failures for the main test
            outcomes[model.label] = exc

    threads = [
        threading.Thread(target=invoke, args=(model,), name=f"tweedie-{model.label}")
        for model in models
    ]
    for thread in threads:
        thread.start()
    try:
        for model in models:
            assert entered[id(model)].wait(timeout=2), (
                f"{model.label} model was serialized behind an unrelated model"
            )
    finally:
        release_calls.set()
        for thread in threads:
            thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert outcomes == {"left": "left", "right": "right"}


def test_active_guard_never_enters_model_or_pickle_state(monkeypatch):
    model = SimpleNamespace(label="pickle", payload=(1, 2, 3))
    baseline_state = vars(model).copy()
    baseline_pickle = pickle.dumps(model)
    transaction_entered = threading.Event()
    release_transaction = threading.Event()
    outcomes = []

    def blocking_transaction(*args, **kwargs):
        transaction_entered.set()
        if not release_transaction.wait(timeout=5):
            raise TimeoutError("test did not release pickle probe")
        return "complete"

    monkeypatch.setattr(profile_ops_module, "_estimate_p_transaction", blocking_transaction)

    def invoke():
        try:
            outcomes.append(_estimate(model))
        except BaseException as exc:  # capture thread failures for the main test
            outcomes.append(exc)

    thread = threading.Thread(target=invoke, name="tweedie-pickle-probe")
    thread.start()
    try:
        assert transaction_entered.wait(timeout=2), "profile transaction never became active"

        active_pickle = pickle.dumps(model)
        restored = pickle.loads(active_pickle)
        assert vars(model) == baseline_state
        assert vars(restored) == baseline_state
        assert active_pickle == baseline_pickle
        assert not any("active" in name.lower() or "guard" in name.lower() for name in vars(model))
    finally:
        release_transaction.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert outcomes == ["complete"]
    assert vars(model) == baseline_state
    assert pickle.dumps(model) == baseline_pickle
