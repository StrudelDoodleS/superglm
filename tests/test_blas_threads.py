"""Scoped BLAS thread capping around solver entry points."""

from __future__ import annotations

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
