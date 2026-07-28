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
