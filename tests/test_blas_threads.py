"""Scoped BLAS thread capping around solver entry points."""

from __future__ import annotations

import pytest
from threadpoolctl import ThreadpoolController

from superglm._blas_threads import _resolve_limit, solver_blas_threads


def _blas_thread_counts() -> list[int]:
    return [
        info["num_threads"]
        for info in ThreadpoolController().info()
        if info.get("user_api") == "blas"
    ]


def test_resolver_policy(monkeypatch):
    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    assert _resolve_limit() == 1
    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "auto")
    assert _resolve_limit() == 1
    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "4")
    assert _resolve_limit() == 4
    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "native")
    assert _resolve_limit() is None
    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "0")
    assert _resolve_limit() is None
    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "not-a-number")
    assert _resolve_limit() == 1


def test_context_caps_and_restores(monkeypatch):
    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _blas_thread_counts()
    with solver_blas_threads():
        inside = _blas_thread_counts()
        assert inside and all(count == 1 for count in inside)
    assert _blas_thread_counts() == before


def test_native_disables_capping(monkeypatch):
    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "native")
    before = _blas_thread_counts()
    with solver_blas_threads():
        assert _blas_thread_counts() == before


def test_off_synonyms_disable_capping(monkeypatch):
    for token in ("off", "none", "false"):
        monkeypatch.setenv("SUPERGLM_BLAS_THREADS", token)
        assert _resolve_limit() is None


def test_unparseable_value_warns_and_caps(monkeypatch):
    import pytest

    from superglm._blas_threads import _auto_policy

    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "fastest")
    with pytest.warns(UserWarning, match="SUPERGLM_BLAS_THREADS"):
        assert _resolve_limit() == 1
    # A typo falls back to the automatic cap, so widening must stay active
    # too -- capped-without-release would be a third, undocumented mode.
    assert _auto_policy() is True


def test_wide_design_releases_auto_cap(monkeypatch):
    from superglm._blas_threads import allow_wide_design

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _blas_thread_counts()
    with solver_blas_threads():
        allow_wide_design(500)
        assert all(count == 1 for count in _blas_thread_counts())
        allow_wide_design(1_500)
        assert _blas_thread_counts() == before
    assert _blas_thread_counts() == before


def test_wide_design_respects_explicit_cap(monkeypatch):
    from superglm._blas_threads import allow_wide_design

    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "2")
    before = _blas_thread_counts()
    with solver_blas_threads():
        allow_wide_design(5_000)
        assert all(count == 2 for count in _blas_thread_counts())
    assert _blas_thread_counts() == before


def test_wide_design_without_an_owning_scope_is_a_noop(monkeypatch):
    import superglm._blas_threads as blas

    class ForeignRegistration:
        def unregister(self):
            raise AssertionError("an unscoped fit released another fit's BLAS cap")

    monkeypatch.setattr(blas, "_registration", ForeignRegistration())
    blas.allow_wide_design(5_000)


def test_overlapping_scopes_restore_native_state(monkeypatch):
    """Concurrent fits must not leave the process pinned at the cap."""
    import threading
    import time

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _blas_thread_counts()
    both_inside = threading.Barrier(2)

    def worker(hold_seconds):
        with solver_blas_threads():
            both_inside.wait(timeout=10)
            time.sleep(hold_seconds)

    threads = [
        threading.Thread(target=worker, args=(0.0,)),
        threading.Thread(target=worker, args=(0.15,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert _blas_thread_counts() == before


def test_nested_wide_scope_rearms_cap_for_outer(monkeypatch):
    """Review finding: an inner wide fit released the cap for its enclosing
    narrow fit permanently. The release must end with the wide scope."""
    from superglm._blas_threads import allow_wide_design

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _blas_thread_counts()
    with solver_blas_threads():
        assert all(count == 1 for count in _blas_thread_counts())
        with solver_blas_threads():
            allow_wide_design(10_000)
            assert _blas_thread_counts() == before  # wide overlap: uncapped
        # inner wide scope exited: the outer narrow fit is capped again
        assert all(count == 1 for count in _blas_thread_counts())
    assert _blas_thread_counts() == before


def test_entrant_during_wide_overlap_gets_capped_after_wide_exits(monkeypatch):
    """Review finding: a narrow fit starting mid-wide-overlap stayed uncapped
    for its whole duration, not just the overlap."""
    import threading

    from superglm._blas_threads import allow_wide_design

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _blas_thread_counts()
    wide_entered = threading.Event()
    narrow_ready = threading.Event()
    release_wide = threading.Event()
    seen = {}

    def wide_fit():
        with solver_blas_threads():
            allow_wide_design(10_000)
            wide_entered.set()
            release_wide.wait(timeout=10)

    def narrow_fit():
        wide_entered.wait(timeout=10)
        with solver_blas_threads():
            seen["during"] = _blas_thread_counts()
            narrow_ready.set()
            release_wide.wait(timeout=10)
            # wide thread exits its scope below; give it a moment
            wide_thread.join(timeout=10)
            seen["after"] = _blas_thread_counts()

    wide_thread = threading.Thread(target=wide_fit)
    narrow_thread = threading.Thread(target=narrow_fit)
    wide_thread.start()
    narrow_thread.start()
    narrow_ready.wait(timeout=10)
    release_wide.set()
    narrow_thread.join(timeout=10)

    assert seen["during"] == before  # uncapped during the wide overlap
    assert all(count == 1 for count in seen["after"])  # re-armed after it
    assert _blas_thread_counts() == before


def test_enter_failure_does_not_leak_scope_counter(monkeypatch):
    """Review finding: a threadpool_limits failure on entry left the refcount
    stuck above zero, disabling capping process-wide forever."""
    import pytest
    import threadpoolctl

    import superglm._blas_threads as blas

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)

    def boom(*args, **kwargs):
        raise RuntimeError("no pools for you")

    monkeypatch.setattr(threadpoolctl, "threadpool_limits", boom)
    with pytest.raises(RuntimeError, match="no pools"):
        with solver_blas_threads():
            pass  # pragma: no cover
    assert blas._active_scopes == 0
    assert blas._wide_scopes == 0
    assert blas._registration is None

    monkeypatch.undo()
    before = _blas_thread_counts()
    with solver_blas_threads():
        assert all(count == 1 for count in _blas_thread_counts())
    assert _blas_thread_counts() == before


def test_assembly_work_counts_packed_channels() -> None:
    from superglm._blas_threads import assembly_work

    # k=3 predictors of width 71 at n=100,000: six packed channels of 2*n*71*71.
    assert assembly_work(100_000, [71, 71, 71]) == pytest.approx(6 * 2.0 * 100_000 * 71 * 71)
    assert assembly_work(5_000, [40]) == pytest.approx(2.0 * 5_000 * 40 * 40)


def test_row_space_work_releases_the_auto_cap_above_threshold(monkeypatch):
    from superglm._blas_threads import (
        _ROW_SPACE_WORK_THRESHOLD,
        allow_row_space_work,
        solver_blas_threads,
    )

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _blas_thread_counts()
    if not before or max(before) < 2:
        pytest.skip("BLAS pool has a single thread; a release is unobservable")
    with solver_blas_threads():
        allow_row_space_work(1_000, [10])  # 2e5 << threshold: stays capped
        assert all(count == 1 for count in _blas_thread_counts())
        allow_row_space_work(200_000, [71, 71, 71])  # 1.2e10 >= threshold: released
        assert _blas_thread_counts() == before
    assert _blas_thread_counts() == before
    assert _ROW_SPACE_WORK_THRESHOLD == 1.0e9


def test_row_space_work_without_an_owning_scope_is_a_noop(monkeypatch):
    from superglm._blas_threads import allow_row_space_work

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _blas_thread_counts()
    allow_row_space_work(10**7, [500, 500])
    assert _blas_thread_counts() == before


def test_explicit_integer_cap_is_not_released_by_row_space_work(monkeypatch):
    from superglm._blas_threads import allow_row_space_work, solver_blas_threads

    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "1")
    with solver_blas_threads():
        allow_row_space_work(10**7, [500, 500])
        assert all(count == 1 for count in _blas_thread_counts())
