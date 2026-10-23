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


def test_a_pooled_section_runs_blas_on_one_thread_under_every_policy(monkeypatch):
    """A worker pool is the one parallelism level, whatever the BLAS policy says.

    Outside a fit, inside a wide fit that released the automatic cap, and under
    'native' or an explicit integer cap, a pooled section holds every BLAS pool
    at one thread and hands back the state it found.  The explicit cap is the
    case that needs the registration re-made at a different limit.
    """
    from superglm._blas_threads import allow_wide_design, pooled_blas_threads

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _native_blas_counts()
    with pooled_blas_threads():
        assert all(count == 1 for count in _blas_thread_counts())
    assert _blas_thread_counts() == before
    with solver_blas_threads():
        allow_wide_design(5_000)
        assert _blas_thread_counts() == before
        with pooled_blas_threads():
            assert all(count == 1 for count in _blas_thread_counts())
        assert _blas_thread_counts() == before
    assert _blas_thread_counts() == before
    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "native")
    with solver_blas_threads(), pooled_blas_threads():
        assert all(count == 1 for count in _blas_thread_counts())
    assert _blas_thread_counts() == before
    explicit = 2
    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", str(explicit))
    with solver_blas_threads():
        assert all(count == explicit for count in _blas_thread_counts())
        with pooled_blas_threads():
            assert all(count == 1 for count in _blas_thread_counts())
        assert all(count == explicit for count in _blas_thread_counts())
    assert _blas_thread_counts() == before


def test_a_design_the_structured_backend_cannot_take_keeps_the_cap_below_2500(monkeypatch):
    from superglm._blas_threads import allow_wide_design

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _native_blas_counts()
    with solver_blas_threads():
        allow_wide_design(2_499, structured=False)
        assert all(count == 1 for count in _blas_thread_counts())
        allow_wide_design(1_600)
        assert _blas_thread_counts() == before
    with solver_blas_threads():
        # Not yet measured as a complete dense fit: the release is kept there.
        allow_wide_design(2_500, structured=False)
        assert _blas_thread_counts() == before


def test_every_fit_path_reports_whether_the_design_has_a_structured_term(monkeypatch):
    """The release's dispatch, no thread counts: True for a RandomEffect and a
    FactorSmooth design, False for a dense one, on the build and the cached-design path."""
    import pandas as pd

    import superglm._blas_threads as blas
    from superglm import Categorical, FactorSmooth, Numeric, RandomEffect, Spline, SuperGLM

    seen: list[bool] = []
    monkeypatch.setattr(
        blas, "allow_wide_design", lambda p, *, structured=True: seen.append(structured)
    )
    rng = np.random.default_rng(5)
    n = 600
    X = pd.DataFrame(
        {
            "x": rng.uniform(-1.0, 1.0, n),
            "g": rng.choice([f"l{k}" for k in range(8)], n),
        }
    )
    y = 1.0 + 0.3 * X["x"] + rng.normal(scale=0.2, size=n)
    designs = {
        "random effect": SuperGLM(
            family="gaussian", features={"x": Numeric(), "g": RandomEffect()}, selection_penalty=0
        ),
        "factor smooth": SuperGLM(
            family="gaussian",
            features={"z": Numeric()},
            interactions=[FactorSmooth("x", group="g", k=5)],
            selection_penalty=0,
        ),
        "dense": SuperGLM(
            family="gaussian",
            features={"x": Spline(n_knots=5), "g": Categorical()},
            selection_penalty=0,
        ),
    }
    X["z"] = rng.normal(size=n)
    for name, model in designs.items():
        seen.clear()
        model.fit_reml(X, y)
        assert seen == [name != "dense"], name
    # The profile search serves its design from a cache, through its own call;
    # the dense design pins that call's keyword, which the default would hide.
    for name in ("random effect", "dense"):
        model = designs[name]
        model._profile_design_cache = {}
        seen.clear()
        model.fit_reml(X, y)
        assert model._profile_design_cache, name
        model.fit_reml(X, y)
        assert seen == [name != "dense"] * 2, name


def test_a_dense_fit_holds_blas_at_one_thread_throughout(monkeypatch):
    """threadpoolctl probe: a wide design without a RandomEffect or FactorSmooth term, whose
    passes run on the pool (a lookup design: its Grams come from term-pair tables), factors its
    Grams on one BLAS thread, and the process's pools come back after.  A design with a serial
    n·p² pass releases instead (``test_the_chunked_row_route_releases_a_dense_fit``)."""
    import pandas as pd

    import superglm._blas_threads as blas
    import superglm.solvers.rank as rank
    from superglm import Categorical, SuperGLM

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    before = _native_blas_counts()
    # Every design counts as wide (this one stays below the dense threshold),
    # so only the structured test keeps the cap.
    monkeypatch.setattr(blas, "_WIDE_DESIGN_THRESHOLD", 2)
    seen: list[list[int]] = []
    decompose = rank._decompose_gram

    def probe(*args, **kwargs):
        seen.append(_blas_thread_counts())
        return decompose(*args, **kwargs)

    monkeypatch.setattr(rank, "_decompose_gram", probe)
    rng = np.random.default_rng(3)
    n = 2_000
    X = pd.DataFrame({"g": rng.choice(list("pqr"), n), "h": rng.choice(list("uvwxyz"), n)})
    y = rng.poisson(np.exp(0.3 * (X["g"] == "q") - 0.2 * (X["h"] == "u"))).astype(float)
    SuperGLM(
        family="poisson",
        features={"g": Categorical(), "h": Categorical()},
        interactions=[("g", "h")],
    ).fit_reml(X, y)
    assert seen and all(all(count == 1 for count in counts) for counts in seen)
    assert _blas_thread_counts() == before


def test_table_jobs_run_on_one_blas_thread_at_any_worker_count(monkeypatch):
    """Serial or pooled, a lookup-table job runs on one BLAS thread under an explicit policy too."""
    from superglm._group_matrix import _lookup_tables as lookup_tables
    from superglm._parallel import parallel_config

    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "native")
    _native_blas_counts()
    for workers in (1, 3):
        seen: list[list[int]] = []
        with parallel_config(n_jobs=workers):
            lookup_tables._pool_run([lambda: seen.append(_blas_thread_counts())] * 3, 8)
        assert seen and all(all(count == 1 for count in counts) for counts in seen), workers


def test_the_pool_releases_each_result_once_it_is_yielded():
    """A caller that reduces as results arrive holds one at a time: once the next result is
    yielded, the pool no longer references any earlier one."""
    import gc
    import weakref

    from superglm._parallel import parallel_config, pooled_map

    class Part:
        pass

    refs = []
    with parallel_config(n_jobs=3, max_memory=2**30):
        for part in pooled_map(lambda _: Part(), list(range(6)), 1):
            refs.append(weakref.ref(part))
            del part
            gc.collect()
            assert all(ref() is None for ref in refs[:-1]), len(refs)
    assert len(refs) == 6


def test_a_fit_pool_serves_every_section_from_the_same_threads():
    """Inside ``fit_pool`` consecutive sections run on one set of worker threads, in item order,
    capped at the section's own workers, and the threads end with the scope; nested sections and
    a consumer that stops early neither deadlock nor leave a task running."""
    import threading
    import time

    from superglm._parallel import fit_pool, parallel_config, pooled_map

    def on_threads(items=12):
        seen = []

        def task(item):
            seen.append(threading.current_thread())
            time.sleep(0.002)
            return item

        results = list(pooled_map(task, list(range(items)), 1))
        return results, {id(thread): thread for thread in seen}

    with parallel_config(n_jobs=4, max_memory=2**30):
        _, first = on_threads()
        _, second = on_threads()
        assert not set(first) & set(second)  # a section per executor, as before
        with fit_pool():
            results_a, a = on_threads()
            results_b, b = on_threads()
            assert results_a == results_b == list(range(12))
            assert len({*a, *b}) <= 4  # one executor of n_jobs threads serves both
            assert all(name.startswith("superglm-map") for name in (t.name for t in a.values()))
            with parallel_config(n_jobs=2):  # a narrower section keeps to its own workers
                running, peak, lock = [0], [0], threading.Lock()

                def gated(item):
                    with lock:
                        running[0] += 1
                        peak[0] = max(peak[0], running[0])
                    time.sleep(0.01)
                    with lock:
                        running[0] -= 1
                    return item

                assert list(pooled_map(gated, list(range(8)), 1)) == list(range(8))
                assert peak[0] <= 2

            def outer(item):  # a section started from a pool thread runs on its own executor
                return sum(pooled_map(lambda inner: inner, list(range(3)), 1)) + item

            assert list(pooled_map(outer, list(range(6)), 1)) == [3 + item for item in range(6)]

            started, finished = [0], [0]

            def slow(item):
                started[0] += 1
                time.sleep(0.05)
                finished[0] += 1
                return item

            generator = pooled_map(slow, list(range(8)), 1)
            next(generator)
            generator.close()
            assert started[0] == finished[0]  # no task of the section outlives it
        assert not [t for t in threading.enumerate() if t.name.startswith("superglm-map")]


def test_pooled_for_gives_the_serial_list_on_any_crew_and_ends_its_threads():
    """``pooled_for`` is ``[fn(item) ...]`` in item order at any worker count, inside a fit pool or
    on threads it starts itself, with the caller's ``np.errstate`` in every task; the first failure in item order is raised after
    every started task has ended, a nested section does not deadlock, and no crew thread outlives
    its scope."""
    import threading
    import time

    from superglm._parallel import fit_pool, parallel_config, pooled_for

    def named():
        return [t for t in threading.enumerate() if t.name.startswith("superglm-map")]

    items = list(range(23))
    expected = [item * item for item in items]
    for workers in (1, 2, 5, 16):
        with parallel_config(n_jobs=workers, max_memory=2**30):
            assert pooled_for(lambda item: item * item, items, 1) == expected
            with fit_pool():
                assert pooled_for(lambda item: item * item, items, 1) == expected
                assert pooled_for(lambda item: item * item, items[:1], 1) == expected[:1]
                assert pooled_for(lambda item: item, [], 1) == []
            assert not named()

    with parallel_config(n_jobs=4, max_memory=2**30), fit_pool():
        seen: list = []
        with np.errstate(over="raise", invalid="ignore"):

            def probe(item):
                time.sleep(0.001)
                seen.append((np.geterr()["over"], np.geterr()["invalid"]))
                return item

            assert pooled_for(probe, items, 1) == items
        assert len(seen) == len(items)
        assert set(seen) == {("raise", "ignore")}

        started, finished = [0], [0]

        def failing(item):
            started[0] += 1
            time.sleep(0.05 if item == 3 else 0.001)  # item 9 fails while item 3 still runs
            finished[0] += 1
            if item in (3, 9):
                raise ValueError(item)
            return item

        with pytest.raises(ValueError) as caught:
            pooled_for(failing, items, 1)
        assert caught.value.args == (3,)
        assert started[0] == finished[0]  # no task of the section outlives it

        def outer(item):  # a section started from a crew thread runs on threads of its own
            return sum(pooled_for(lambda inner: inner, [1, 2, 3], 1)) + item

        assert pooled_for(outer, items[:8], 1) == [6 + item for item in items[:8]]
    assert not named()


def test_a_fit_crew_starts_only_the_helpers_a_section_admits():
    """The crew grows with the widest section, not to ``n_jobs``: 4 items admit ``(4 + 1) // 2 = 2``
    helpers, so a crew configured for 32 workers holds 2 threads, then 15 for a section of 30."""
    import threading

    from superglm._parallel import fit_pool, parallel_config, pooled_for

    def crew_threads():
        return [t for t in threading.enumerate() if t.name == "superglm-map-crew"]

    with parallel_config(n_jobs=32, max_memory=2**30), fit_pool():
        assert pooled_for(lambda item: item, list(range(4)), 1) == list(range(4))
        assert len(crew_threads()) == 2
        assert pooled_for(lambda item: item, list(range(3)), 1) == list(range(3))
        assert len(crew_threads()) == 2  # a narrower section keeps, not shrinks
        assert pooled_for(lambda item: item, list(range(30)), 1) == list(range(30))
        assert len(crew_threads()) == 15
    assert not crew_threads()


def _bounded(call, seconds=20.0):
    """Run ``call`` on a daemon thread; return ``(outcome, exception)`` or fail if it hangs."""
    import threading

    box: list = []

    def run():
        try:
            box.append((call(), None))
        except BaseException as exc:
            box.append((None, exc))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "the call did not return"
    return box[0]


def _bounded_into(call, errors):
    try:
        call()
    except BaseException as exc:  # an assertion in the section reaches the test through here
        errors.append(exc)


def _named_threads(prefix):
    import threading

    return [t for t in threading.enumerate() if t.name.startswith(prefix)]


def test_a_fit_executor_holds_only_the_workers_a_section_admits():
    """A memory-limited section posts the width ``pool_workers`` admits (2 here), so the fit's
    shared executor holds 2 threads, not ``n_jobs``; a wider section grows it to its own width, in
    item order and with the caller's ``np.errstate`` in every task, and a failure is the first in
    item order with no task left running."""
    import threading

    from superglm._parallel import fit_pool, parallel_config, pool_workers, pooled_map

    def executor_threads():
        return _named_threads("superglm-map_")

    with parallel_config(n_jobs=32, max_memory=128 << 20), fit_pool():
        assert pool_workers(12, 64 << 20) == 2
        pair = threading.Barrier(2, timeout=20)

        def square(item):
            pair.wait()  # each item runs beside another, so the section holds 2 threads at once
            return item * item

        assert list(pooled_map(square, list(range(12)), 64 << 20)) == [
            item * item for item in range(12)
        ]
        assert len(executor_threads()) == 2
        with parallel_config(max_memory=2**30):
            seen = []
            nine = threading.Barrier(9, timeout=20)
            with np.errstate(over="raise", invalid="ignore"):

                def probe(item):
                    nine.wait()  # all nine items run at once: the section needs 9 threads
                    seen.append((np.geterr()["over"], np.geterr()["invalid"]))
                    return item

                assert list(pooled_map(probe, list(range(9)), 1)) == list(range(9))
            assert set(seen) == {("raise", "ignore")} and len(seen) == 9
            assert len(executor_threads()) == 9

            started, finished, later_failed = [], [], threading.Event()

            def failing(item):
                started.append(item)
                if item == 3:
                    assert later_failed.wait(20)  # item 7 has failed while item 3 still runs
                finished.append(item)
                if item == 7:
                    later_failed.set()
                if item in (3, 7):
                    raise ValueError(item)
                return item

            with pytest.raises(ValueError) as caught:
                list(pooled_map(failing, list(range(12)), 1))
            assert caught.value.args == (3,)
            assert len(started) == len(finished)
    assert not executor_threads()


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, MemoryError])
@pytest.mark.parametrize("scoped", [False, True])
def test_an_interruption_while_posting_helpers_propagates_and_waits_only_for_posted_jobs(
    monkeypatch, interrupt, scoped
):
    """``KeyboardInterrupt`` or ``MemoryError`` on the second post of a section of eight items on
    four workers reaches the caller without waiting for the helpers never queued, and the one posted
    helper stops claiming items once released; no crew thread outlives the scope."""
    import contextlib
    import queue
    import threading

    from superglm import _parallel
    from superglm._parallel import fit_pool, parallel_config, pooled_for

    post = _parallel._Crew.post
    posts, raised, gate, started, caught = [0], threading.Event(), threading.Event(), [], []

    def interrupting(self, job, done):
        posts[0] += 1
        if posts[0] == 2:
            raised.set()
            raise interrupt
        post(self, job, done)

    def fn(item):
        started.append(item)
        if item == 0:
            gate.wait(20)  # the posted helper holds item 0 until the test releases it
        return item

    def section():
        with parallel_config(n_jobs=4, max_memory=2**30):
            with fit_pool() if scoped else contextlib.nullcontext():
                try:
                    pooled_for(fn, list(range(8)), 1)
                except BaseException as exc:
                    caught.append(exc)
                if scoped:  # the interrupted section leaves the fit's crew usable
                    assert pooled_for(lambda item: item, list(range(8)), 1) == list(range(8))

    waiting = threading.Event()

    class Finished(queue.SimpleQueue):
        def get(self, *args, **kwargs):
            if threading.current_thread() is thread:
                waiting.set()  # the caller has recorded the interruption and waits for its helper
            return super().get(*args, **kwargs)

    monkeypatch.setattr(_parallel._Crew, "post", interrupting)
    monkeypatch.setattr(_parallel.queue, "SimpleQueue", Finished)
    thread = threading.Thread(target=lambda: _bounded_into(section, caught), daemon=True)
    thread.start()
    assert raised.wait(20)
    assert waiting.wait(20)
    gate.set()
    thread.join(20)
    assert not thread.is_alive(), "the interrupted section did not return"
    assert [type(exc) for exc in caught] == [interrupt]
    assert len(started) <= 1  # the posted helper stopped at the interruption
    assert not _named_threads("superglm-map-crew")


def test_a_crew_that_fails_to_start_a_helper_ends_the_ones_it_started(monkeypatch):
    """A failure to start the second helper re-raises and leaves no crew thread, whether the
    crew is built for one section or by the fit."""
    import threading

    from superglm._parallel import _Crew, fit_pool, parallel_config, pooled_for

    start = threading.Thread.start
    starts = [0]

    def failing_start(self):
        if self.name == "superglm-map-crew":
            starts[0] += 1
            if starts[0] == 2:
                raise RuntimeError("can't start new thread")
        start(self)

    monkeypatch.setattr(threading.Thread, "start", failing_start)
    with pytest.raises(RuntimeError, match="can't start"):
        _Crew(4, started=4)
    assert not _named_threads("superglm-map-crew")

    def section(scoped):
        def run():
            with parallel_config(n_jobs=4, max_memory=2**30):
                if not scoped:
                    return pooled_for(lambda item: item, list(range(8)), 1)
                with fit_pool():
                    return pooled_for(lambda item: item, list(range(8)), 1)

        return run

    starts[0] = 0
    outcome, exc = _bounded(section(False))
    assert isinstance(exc, RuntimeError)
    assert not _named_threads("superglm-map-crew")

    starts[0] = 0
    outcome, exc = _bounded(section(True))
    assert isinstance(exc, RuntimeError)
    assert not _named_threads("superglm-map-crew")  # the fit's crew ended with the fit

    starts[0] = 10  # starts succeed again
    with parallel_config(n_jobs=4, max_memory=2**30), fit_pool():
        assert pooled_for(lambda item: item, list(range(8)), 1) == list(range(8))
    assert not _named_threads("superglm-map-crew")


def test_a_fit_crew_releases_a_finished_sections_buffers(monkeypatch):
    """A crew worker reports a job done only once it holds no reference to it, whose closure holds
    the section's buffers: once the section returns they are collectable, as after a one-worker
    section.  Each worker is held right after it reports, until the buffers have been checked."""
    import gc
    import queue
    import threading
    import weakref

    from superglm import _parallel
    from superglm._parallel import fit_pool, parallel_config, pooled_for

    checked = threading.Event()

    class HeldAfterReport(queue.SimpleQueue):
        def put(self, item, *args, **kwargs):
            super().put(item, *args, **kwargs)
            if threading.current_thread().name == "superglm-map-crew":
                checked.wait(20)

    monkeypatch.setattr(_parallel.queue, "SimpleQueue", HeldAfterReport)
    for workers in (1, 4):
        checked.clear()
        with parallel_config(n_jobs=workers, max_memory=2**30), fit_pool():
            buffer = np.zeros(64)
            ref = weakref.ref(buffer)
            started = threading.Barrier(min(workers, 2), timeout=20)

            def task(item, buffer=buffer):
                if item < 2 and workers > 1:
                    started.wait()  # a helper runs items, so its job held the buffer
                buffer[item] = 1.0
                return item

            assert pooled_for(task, list(range(16)), 1) == list(range(16))
            assert ref() is not None
            del buffer, task
            gc.collect()
            alive = ref() is not None
            checked.set()
            assert not alive, workers


def test_pooled_for_runs_every_task_on_one_blas_thread():
    from superglm._parallel import parallel_config, pooled_for

    _native_blas_counts()
    for workers in (1, 3):
        with parallel_config(n_jobs=workers, max_memory=2**30):
            counts = pooled_for(lambda _: _blas_thread_counts(), list(range(8)), 1)
        assert all(all(count == 1 for count in each) for each in counts), workers


@pytest.mark.parametrize("scoped", [False, True])
def test_numbas_thread_count_caps_a_pooled_section(monkeypatch, scoped):
    """``numba.set_num_threads`` caps a ``pooled_for`` section, as it (and ``NUMBA_NUM_THREADS``)
    capped the ``prange`` loops of SuperLSS's row kernels and batched moments that these sections
    replaced, whatever ``SUPERGLM_N_JOBS`` allows; an ``n_jobs`` the context sets is used as given."""
    import contextlib

    import numba

    from superglm import _parallel
    from superglm._parallel import fit_pool, parallel_config, pooled_for

    grown = []
    grow = _parallel._Crew.grow

    def recording(self, count):
        grown.append(count)
        grow(self, count)

    monkeypatch.setattr(_parallel._Crew, "grow", recording)
    monkeypatch.setenv("SUPERGLM_N_JOBS", "4")
    previous = numba.get_num_threads()
    caps = [1] + ([2] if numba.config.NUMBA_NUM_THREADS >= 2 else [])
    try:
        with fit_pool() if scoped else contextlib.nullcontext():
            for cap in caps:
                numba.set_num_threads(cap)
                grown.clear()
                assert pooled_for(lambda item: item, list(range(8)), 1) == list(range(8))
                assert max(grown, default=0) == cap - 1, cap  # helpers beside the caller
            grown.clear()
            with parallel_config(n_jobs=4):
                assert pooled_for(lambda item: item, list(range(8)), 1) == list(range(8))
            assert max(grown) == 3
    finally:
        numba.set_num_threads(previous)


def test_the_chunked_row_route_releases_a_dense_fit(monkeypatch):
    """A dense design keeps the cap because its row passes run on the pool; every pass that runs
    on the calling thread instead (the chunked solved-row route, the gradient-only pass and the
    serial centred Grams) releases it for the rest of the fit, as before the dense threshold, while
    the pooled support route keeps it."""
    import pandas as pd

    import superglm._blas_threads as blas
    import superglm.reml._leverage_rows as leverage_rows
    import superglm.reml.w_derivatives as w_derivatives
    from superglm import Categorical, Spline, SuperGLM
    from superglm._group_matrix._group_matrix_centered import (
        centered_gram_rhs,
        centered_signed_grams,
    )

    monkeypatch.delenv("SUPERGLM_BLAS_THREADS", raising=False)
    rng = np.random.default_rng(3)
    n = 400
    X = pd.DataFrame({"a": rng.uniform(0.0, 1.0, n), "g": rng.choice(list("pqr"), n)})
    y = rng.poisson(np.exp(0.3 * np.sin(4.0 * X["a"]))).astype(float)
    model = SuperGLM(family="poisson", features={"a": Spline(n_knots=6), "g": Categorical()})
    model.fit(X, y)
    dm = model._dm
    lookup = SuperGLM(family="poisson", features={"g": Categorical(), "h": Categorical()})
    lookup.fit(X.assign(h=rng.choice(list("uvw"), n)), y)
    ldm = lookup._dm
    real_plan = leverage_rows._support_plan
    released = []
    monkeypatch.setattr(blas, "_WIDE_DESIGN_THRESHOLD", 2)
    monkeypatch.setattr(blas, "_release_current_scope", lambda: released.append(True))
    monkeypatch.setattr(leverage_rows, "_support_plan", lambda *args, **kwargs: None)
    inverse, centre = np.eye(dm.p), np.zeros(dm.p)
    leverage_rows.solved_row_blocks(dm, inverse, centre, None, np.ones(dm.n, dtype=bool))
    assert released == [True]
    w_derivatives._leverage_gradient_rhs(dm, inverse, centre, np.ones(dm.n), float(dm.n))
    assert released == [True, True]
    ones = np.ones(dm.n)
    centered_gram_rhs(dm=dm, W=ones, mean_x=centre, z_centered=ones)
    centered_signed_grams(dm=dm, weights=[ones], mean_x=centre)
    assert released == [True] * 4
    # The pooled support route runs on the pool and keeps the cap.
    monkeypatch.setattr(leverage_rows, "_support_plan", real_plan)
    assert real_plan(ldm, np.eye(ldm.p), np.zeros(ldm.p), None) is not None
    leverage_rows.solved_row_blocks(
        ldm, np.eye(ldm.p), np.zeros(ldm.p), None, np.ones(ldm.n, dtype=bool)
    )
    assert released == [True] * 4
    monkeypatch.setattr(leverage_rows, "_support_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(blas, "_WIDE_DESIGN_THRESHOLD", 10**9)  # a narrow design keeps the cap
    leverage_rows.solved_row_blocks(dm, inverse, centre, None, np.ones(dm.n, dtype=bool))
    assert released == [True] * 4
