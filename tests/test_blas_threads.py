"""Scoped BLAS thread capping around solver entry points."""

from __future__ import annotations

import numpy as np
import pytest
from threadpoolctl import ThreadpoolController

from superglm._blas_threads import _resolve_limit, solver_blas_threads

# Every test here compares BLAS pool sizes with the process's own, so each needs
# the default multi-thread pools: scripts/run_test_suite.py runs them unpinned.
pytestmark = pytest.mark.threads


def _blas_thread_counts() -> list[int]:
    return [
        info["num_threads"]
        for info in ThreadpoolController().info()
        if info.get("user_api") == "blas"
    ]


def _visible_blas_counts() -> list[int]:
    counts = _blas_thread_counts()
    if not counts:
        pytest.skip("threadpoolctl exposes no BLAS pools; thread counts cannot be verified")
    return counts


def _native_blas_counts() -> list[int]:
    """The process's own BLAS pool sizes, or a skip when a cap would be invisible."""
    counts = _visible_blas_counts()
    if max(counts) < 2:
        pytest.skip("BLAS pool has a single thread; a cap or its release is unobservable")
    return counts


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
    with pytest.warns(UserWarning, match="SUPERGLM_BLAS_THREADS"):
        assert _resolve_limit() == 1


def test_context_caps_and_restores(monkeypatch):
    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _native_blas_counts()
    with solver_blas_threads():
        inside = _blas_thread_counts()
        assert inside and all(count == 1 for count in inside)
    assert _blas_thread_counts() == before


def test_native_disables_capping(monkeypatch):
    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "native")
    before = _native_blas_counts()
    with solver_blas_threads():
        assert _blas_thread_counts() == before


def test_off_synonyms_disable_capping(monkeypatch):
    for token in ("off", "none", "false"):
        monkeypatch.setenv("SUPERGLM_BLAS_THREADS", token)
        assert _resolve_limit() is None


def test_unparseable_value_warns_and_caps(monkeypatch):
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
    before = _native_blas_counts()
    with solver_blas_threads():
        allow_wide_design(500)
        assert all(count == 1 for count in _blas_thread_counts())
        allow_wide_design(1_500)
        assert _blas_thread_counts() == before
    assert _blas_thread_counts() == before


def test_a_narrow_kernel_inside_a_wide_fit_runs_on_one_thread(monkeypatch):
    """A wide design's released cap is re-armed around a narrower dense kernel.

    The structured border's factorization runs on its own width: below the
    break-even it takes one thread, and the released pool comes back after
    it.  At or past the break-even, outside a released scope, and under
    'native' the kernel leaves the pools alone.
    """
    from superglm._blas_threads import allow_wide_design, narrow_kernel_blas_threads

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _native_blas_counts()
    with narrow_kernel_blas_threads(100):
        assert _blas_thread_counts() == before
    with solver_blas_threads():
        with narrow_kernel_blas_threads(100):
            assert all(count == 1 for count in _blas_thread_counts())
        allow_wide_design(5_000)
        assert _blas_thread_counts() == before
        with narrow_kernel_blas_threads(1_045):
            assert all(count == 1 for count in _blas_thread_counts())
        assert _blas_thread_counts() == before
        with narrow_kernel_blas_threads(1_500):
            assert _blas_thread_counts() == before
    assert _blas_thread_counts() == before
    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "native")
    with solver_blas_threads():
        with narrow_kernel_blas_threads(100):
            assert _blas_thread_counts() == before


def test_wide_design_respects_explicit_cap(monkeypatch):
    from superglm._blas_threads import allow_wide_design

    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "2")
    before = _visible_blas_counts()
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
    before = _native_blas_counts()
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
    before = _native_blas_counts()
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
    before = _native_blas_counts()
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


def test_overlapping_narrow_kernels_keep_the_cap_and_restore_native_state(monkeypatch):
    """Review finding (PR #425): two wide fits whose narrow kernels overlap.

    Each kernel's limiter recorded the pools it saw on entry: the first
    restored the native pools while the second was still inside its kernel,
    and the second then restored one thread, which nothing undid after both
    fits returned.  The kernels must run on one thread until the last exits,
    and the pools come back to the native state after it.
    """
    import threading

    from superglm._blas_threads import allow_wide_design, narrow_kernel_blas_threads

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _native_blas_counts()
    both_inside = threading.Barrier(2)
    first_left = threading.Event()
    seen: dict[str, list[int]] = {}

    def wide_fit(name):
        with solver_blas_threads():
            allow_wide_design(5_000)
            with narrow_kernel_blas_threads(100):
                both_inside.wait(timeout=10)
                if name == "second":
                    first_left.wait(timeout=10)
                    seen["second_after_first_left"] = _blas_thread_counts()
            if name == "first":
                first_left.set()
            both_inside.wait(timeout=10)
            seen[f"{name}_after_both"] = _blas_thread_counts()
            both_inside.wait(timeout=10)

    threads = [threading.Thread(target=wide_fit, args=(name,)) for name in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(count == 1 for count in seen["second_after_first_left"])
    assert seen["first_after_both"] == before
    assert seen["second_after_both"] == before
    assert _blas_thread_counts() == before


def test_a_narrow_kernel_takes_one_thread_while_another_fit_is_wide(monkeypatch):
    """Review finding (PR #425): a fit re-capped by ``keep_narrow_cap`` beside a wide fit.

    The other fit's open wide scope keeps the pools released for the overlap,
    and the re-capped fit's own scope is no longer wide, so its narrow kernels
    ran on the released pools: their results depended on the thread count.
    """
    import threading

    from superglm._blas_threads import (
        allow_wide_design,
        keep_narrow_cap,
        narrow_kernel_blas_threads,
    )

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _native_blas_counts()
    wide_entered = threading.Event()
    done = threading.Event()
    seen: dict[str, list[int]] = {}

    def wide_fit():
        with solver_blas_threads():
            allow_wide_design(10_000)
            wide_entered.set()
            done.wait(timeout=10)

    wide_thread = threading.Thread(target=wide_fit)
    wide_thread.start()
    try:
        wide_entered.wait(timeout=10)
        with solver_blas_threads():
            allow_wide_design(5_000)
            keep_narrow_cap(9)
            seen["overlap"] = _blas_thread_counts()
            with narrow_kernel_blas_threads(9):
                seen["kernel"] = _blas_thread_counts()
            seen["after_kernel"] = _blas_thread_counts()
    finally:
        done.set()
        wide_thread.join(timeout=10)

    assert seen["overlap"] == before  # the wide fit holds the release for the overlap
    assert all(count == 1 for count in seen["kernel"])
    assert seen["after_kernel"] == before
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
    before = _native_blas_counts()
    with solver_blas_threads():
        assert all(count == 1 for count in _blas_thread_counts())
    assert _blas_thread_counts() == before


def test_a_nested_factor_builds_and_inverts_on_one_thread_inside_a_wide_fit(monkeypatch):
    """Perf scout F1: the whole construction and every lazily formed inverse take the cap.

    Inside a wide fit two OpenBLAS pools (numpy's and scipy's) contend when a
    threaded GEMM precedes a LAPACK call; the border's kernels act on its own
    width.  The construction's scatter products and the lazily formed
    ``dpotri`` both run on one thread.  Fails with the cap scoped to
    ``factor_border`` alone, or with a lazy inverse outside it.
    """
    import superglm.solvers._structured.border as border
    import superglm.solvers._structured.nested as nested
    from superglm._blas_threads import allow_wide_design
    from tests.test_nested_schur_factor import CHAIN, _augmented_case

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    _native_blas_counts()
    refs, _, penalized, *_ = _augmented_case("F1")
    seen: dict[str, list[int]] = {"scatter": [], "potri": []}
    scatter, potri = nested._weighted_scatter, border.dpotri

    def recorded_scatter(*args, **kwargs):
        seen["scatter"].extend(_blas_thread_counts())
        return scatter(*args, **kwargs)

    def recorded_potri(*args, **kwargs):
        seen["potri"].extend(_blas_thread_counts())
        return potri(*args, **kwargs)

    monkeypatch.setattr(nested, "_weighted_scatter", recorded_scatter)
    monkeypatch.setattr(border, "dpotri", recorded_potri)
    with solver_blas_threads():
        allow_wide_design(5_000)
        factor = nested.NestedSchurFactor(
            penalized,
            chain_group_names=CHAIN[: refs["fx"]["depth"]],
            chain_group_indices=tuple(range(refs["fx"]["depth"])),
            intercept=True,
        )
        # the border's own lazy inverse, read directly (leverage rows do), and
        # through the factor's lazily formed Q^+
        _ = factor._border.inverse
        factor.solve(np.ones(refs["p"]))
    assert seen["scatter"] and all(count == 1 for count in seen["scatter"])
    assert seen["potri"] and all(count == 1 for count in seen["potri"])


def test_only_the_fs_leaf_route_re_arms_the_cap(monkeypatch):
    """Perf F7 applies to the fs leaf route alone.

    A nested chain's parent blocks are wider than its border, and it needs the
    wide release a wide design grants (pg17_E_discrete ran 1.5x master on
    default threads with the cap re-armed).  Mutation: every structured layout
    re-arms the cap.
    """
    import pandas as pd

    from superglm import Categorical, FactorSmooth, Numeric, SuperGLM
    from superglm.solvers import irls_direct
    from tests.test_nested_structured_fit import _fit

    widths: list[int] = []
    monkeypatch.setattr(irls_direct, "keep_narrow_cap", widths.append)
    nested = _fit("poisson", "structured")
    assert nested._reml_profile.get("structured_chain")
    assert widths == []

    rng = np.random.default_rng(314)
    n = 400
    frame = pd.DataFrame(
        {
            "x": rng.uniform(size=n),
            "x1": rng.normal(size=n),
            "cat": np.array([f"c{c}" for c in rng.integers(0, 3, n)], dtype=object),
            "g": np.array([f"g{c}" for c in rng.integers(0, 6, n)], dtype=object),
        }
    )
    y = np.sin(2 * np.pi * frame["x"].to_numpy()) + rng.normal(0, 0.3, n)
    model = SuperGLM(
        family="gaussian",
        features={"x1": Numeric(), "cat": Categorical()},
        interactions=[FactorSmooth("x", group="g", basis="fs", k=5)],
        selection_penalty=0,
        direct_solve="structured",
    )
    model.fit_reml(frame, y)
    assert widths and all(width == widths[0] for width in widths)


def test_a_structured_route_keeps_the_cap_in_a_wide_fit(monkeypatch):
    """Perf F7 and design §14 T4: a route whose dense kernels act on a narrow border re-arms the cap.

    The wide release is decided on the design width before any route is
    chosen; a structured route's kernels never exceed its border, so its fit
    keeps one BLAS thread (and a result independent of the pool size).  A
    border at or past the break-even, or 'native', leaves the release alone.
    """
    from superglm._blas_threads import allow_wide_design, keep_narrow_cap

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _native_blas_counts()
    with solver_blas_threads():
        allow_wide_design(5_000)
        assert _blas_thread_counts() == before
        keep_narrow_cap(1_500)
        assert _blas_thread_counts() == before
        keep_narrow_cap(9)
        assert all(count == 1 for count in _blas_thread_counts())
    assert _blas_thread_counts() == before
    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "native")
    with solver_blas_threads():
        allow_wide_design(5_000)
        keep_narrow_cap(9)
        assert _blas_thread_counts() == before
