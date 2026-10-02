"""Structured sufficient-statistic correctness and allocation guards."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import superglm._group_matrix._group_matrix_algebra as group_algebra
import superglm.solvers._structured.moments as structured_moments
from superglm.group_matrix import (
    DenseGroupMatrix,
    DiscretizedSSPGroupMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.solvers.structured import (
    build_block_structured_system,
    build_nested_structured_layout,
    build_nested_structured_system,
    select_structured_group,
    structured_design_matvec,
    structured_design_rmatvec,
)
from superglm.types import GroupSlice, LinearConstraintSet


def _groups(group_matrices: list[GroupMatrix]) -> list[GroupSlice]:
    groups: list[GroupSlice] = []
    start = 0
    for index, gm in enumerate(group_matrices):
        end = start + gm.shape[1]
        groups.append(
            GroupSlice(
                name=f"group_{index}",
                start=start,
                end=end,
                penalized=isinstance(gm, RandomEffectGroupMatrix | SparseSSPGroupMatrix),
            )
        )
        start = end
    return groups


def _factor_smooth_matrix(
    n: int,
    *,
    levels: int,
    k: int = 5,
) -> FactorSmoothGroupMatrix:
    x = np.linspace(-1.0, 1.0, n)
    basis = np.column_stack([x**power for power in range(k)])
    return FactorSmoothGroupMatrix(
        sp.csr_matrix(basis),
        np.arange(n, dtype=np.intp) % levels,
        levels,
        natural_map=np.eye(k),
        levels=tuple(f"level-{index}" for index in range(levels)),
        repeated_penalty_components=(("wiggle", np.eye(k)),),
    )


def test_select_structured_group_chooses_largest_random_effect_and_reports_ineligibility():
    n = 12
    group_matrices: list[GroupMatrix] = [
        RandomEffectGroupMatrix(np.arange(n) % 3, n_levels=3),
        DenseGroupMatrix(np.arange(n, dtype=np.float64)),
        RandomEffectGroupMatrix(np.arange(n) % 7, n_levels=7),
    ]
    groups = _groups(group_matrices)

    selected = select_structured_group(group_matrices, groups, mode="structured")

    assert selected.group_index == 2
    assert selected.group_name == groups[2].name
    assert selected.fallback_reason is None

    no_random_effect = [DenseGroupMatrix(np.arange(n, dtype=np.float64))]
    no_random_groups = _groups(no_random_effect)
    auto = select_structured_group(no_random_effect, no_random_groups, mode="auto")
    assert auto.group_index is None
    assert "RandomEffect" in auto.fallback_reason
    with pytest.raises(ValueError, match="RandomEffect"):
        select_structured_group(no_random_effect, no_random_groups, mode="structured")


def test_structured_selection_rejects_multiple_factor_smooths_before_assembly():
    matrices: list[GroupMatrix] = [
        _factor_smooth_matrix(80, levels=8),
        _factor_smooth_matrix(80, levels=6),
    ]
    groups = _groups(matrices)

    auto = select_structured_group(matrices, groups, mode="auto")
    assert auto.group_index is None
    assert "at most one FactorSmooth" in auto.fallback_reason
    assert groups[0].name in auto.fallback_reason
    assert groups[1].name in auto.fallback_reason

    with pytest.raises(ValueError, match="at most one FactorSmooth"):
        select_structured_group(matrices, groups, mode="structured")


def test_factor_smooth_is_dominant_candidate_even_when_random_effect_is_wider():
    matrices: list[GroupMatrix] = [
        RandomEffectGroupMatrix(np.arange(300) % 50, n_levels=50),
        _factor_smooth_matrix(300, levels=8),
    ]
    groups = _groups(matrices)

    selected = select_structured_group(matrices, groups, mode="structured")

    assert selected.group_index == 1
    assert selected.group_name == groups[1].name


def test_select_structured_group_rejects_constraint_geometry():
    n = 9
    group_matrices: list[GroupMatrix] = [
        DenseGroupMatrix(np.arange(n, dtype=np.float64)),
        RandomEffectGroupMatrix(np.arange(n) % 3, n_levels=3),
    ]
    groups = _groups(group_matrices)
    groups[0].constraints = LinearConstraintSet(A=np.ones((1, 1)), b=np.zeros(1))

    auto = select_structured_group(group_matrices, groups, mode="auto")

    assert auto.group_index is None
    assert "constraint" in auto.fallback_reason.lower()
    with pytest.raises(ValueError, match="constraint"):
        select_structured_group(group_matrices, groups, mode="structured")


class _GuardedRandomEffect(RandomEffectGroupMatrix):
    def gram(self, W):
        raise AssertionError("dominant random-effect gram must not be materialized")

    def toarray(self):
        raise AssertionError("dominant random-effect design must not be materialized")


def test_cached_dense_small_layout_fuses_design_vector_products(monkeypatch):
    rng = np.random.default_rng(918)
    n = 240
    n_levels = 61
    small = [
        DenseGroupMatrix(rng.normal(size=n)),
        DenseGroupMatrix(rng.normal(size=(n, 2))),
    ]
    dominant = RandomEffectGroupMatrix(np.arange(n) % n_levels, n_levels)
    matrices: list[GroupMatrix] = [*small, dominant]
    groups = _groups(matrices)
    layout = build_nested_structured_layout(matrices, groups, chain_group_indices=(2,))
    dense = np.column_stack([matrix.toarray() for matrix in matrices])
    beta = rng.normal(size=dense.shape[1])
    rows = rng.normal(size=n)

    def fail_separate_dense_dispatch(*_args, **_kwargs):
        raise AssertionError("cached dense-small vector product was not used")

    monkeypatch.setattr(DenseGroupMatrix, "matvec", fail_separate_dense_dispatch)
    monkeypatch.setattr(DenseGroupMatrix, "rmatvec", fail_separate_dense_dispatch)

    np.testing.assert_allclose(
        structured_design_matvec(layout, matrices, beta),
        dense @ beta,
    )
    np.testing.assert_allclose(
        structured_design_rmatvec(layout, matrices, rows),
        dense.T @ rows,
    )


def test_discrete_factor_smooth_shared_bin_cross_avoids_spline_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(429)
    n = 90
    n_levels = 6
    block_size = 4
    n_bins = 9
    bin_idx = np.arange(n, dtype=np.intp) % n_bins
    dominant = FactorSmoothGroupMatrix(
        rng.normal(size=(n_bins, 5)),
        (np.arange(n, dtype=np.intp) * 5 + 2) % n_levels,
        n_levels,
        natural_map=rng.normal(size=(5, block_size)),
        levels=tuple(f"level-{level}" for level in range(n_levels)),
        repeated_penalty_components=(("wiggle", np.eye(block_size)),),
        factor_basis="sz",
        bin_idx=bin_idx,
    )
    dense = DenseGroupMatrix(rng.normal(size=(n, 4)))
    spline = DiscretizedSSPGroupMatrix(
        rng.normal(size=(n_bins, 5)),
        rng.normal(size=(5, 3)),
        bin_idx.copy(),
    )
    matrices: list[GroupMatrix] = [dense, spline, dominant]
    groups = _groups(matrices)

    def fail_toarray(_self):
        raise AssertionError("optimized cross must not materialize observation rows")

    monkeypatch.setattr(DiscretizedSSPGroupMatrix, "toarray", fail_toarray)
    monkeypatch.setattr(FactorSmoothGroupMatrix, "toarray", fail_toarray)

    system = build_block_structured_system(
        matrices,
        groups,
        rng.uniform(0.3, 1.8, size=n),
        rng.normal(size=n),
        dominant_group_index=2,
    )

    assert system.operator.C.shape == (n_levels, block_size, 7)


def test_large_dominant_builder_never_requests_full_p_by_p_storage(monkeypatch):
    """A lone random effect's chain of one never forms its level block or any p x p array."""
    n = 600
    n_levels = 5_000
    rng = np.random.default_rng(18)
    small = DenseGroupMatrix(rng.normal(size=(n, 2)))
    dominant = _GuardedRandomEffect(np.arange(n, dtype=np.intp), n_levels)
    group_matrices: list[GroupMatrix] = [small, dominant]
    groups = _groups(group_matrices)
    W = rng.uniform(0.4, 1.6, size=n)
    Wz = rng.normal(size=n)
    p = n_levels + 2
    layout = build_nested_structured_layout(group_matrices, groups, chain_group_indices=(1,))

    def fail_full_block(*args, **kwargs):
        raise AssertionError("full block Gram builder must not be used")

    monkeypatch.setattr(group_algebra, "_block_xtwx", fail_full_block)
    original_zeros = np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        if shape == (p, p):
            raise AssertionError("full p x p allocation requested")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(structured_moments.np, "zeros", guarded_zeros)

    system = build_nested_structured_system(group_matrices, groups, W, Wz, layout=layout)

    assert system.operator.A.shape == (2, 2)
    assert system.operator.leaf.cross.shape == (n_levels, 2)


# T7, memory: an fs term beside a wide border (Opus review P1).  The leaf route
# works on per-level triangles of width p = k + q + 2 (q the border); a stack of
# them, K p^2 doubles, grows with the square of the border, where every array
# the factorization and the published state need is O(K k p) or O(n).  The
# unfixed route kept the triangles and the level Grams (two such stacks) on its
# system, its published factor and its pickle, and formed several more in the
# pass and the signed assembly.  Byte counts, never wall time.
_WIDE = dict(K=400, Q=60, n=2400, k=5)


@pytest.fixture(scope="module", params=["poisson", "tweedie"])
def _wide_border_fs_fit(request):
    """A REML fit (smoothing parameters held) of an fs term beside a 60-level categorical.

    The Tweedie fit's observed REML geometry builds signed leaf systems.
    """
    import warnings

    import pandas as pd

    from superglm import Categorical, FactorSmooth, LambdaPolicy, Numeric, SuperGLM, families

    K, Q, n, k = _WIDE["K"], _WIDE["Q"], _WIDE["n"], _WIDE["k"]
    rng = np.random.default_rng(7)
    g = np.repeat(np.arange(K), n // K)
    x = rng.uniform(size=n)
    c = rng.integers(0, Q, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "g": [f"g{v:03d}" for v in g],
            "z": rng.normal(size=n),
            "cat": [f"c{v:02d}" for v in c],
        }
    )
    mu = np.exp(0.3 + np.sin(3 * x) + rng.normal(0, 0.4, K)[g] + rng.normal(0, 0.3, Q)[c])
    if request.param == "poisson":
        family, y = "poisson", rng.poisson(mu).astype(float)
    else:
        family = families.tweedie(p=1.5)
        y = rng.gamma(2.0, mu / 2.0) * (rng.uniform(size=n) < 0.7)
    policy = {name: LambdaPolicy.fixed(1.0) for name in ("wiggle", "null_0", "null_1")}
    model = SuperGLM(
        family=family,
        features={"z": Numeric(), "cat": Categorical()},
        interactions=[FactorSmooth("x", group="g", k=k, lambda_policy=policy)],
        selection_penalty=0,
        direct_solve="structured",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    system = model._linear_system_state.system
    p = k + len(system.operator.small_indices) + 2
    return model, p


def _largest_reachable_array(root) -> int:
    """The size of the largest NumPy array (owning its data) reachable from ``root``."""
    largest, seen, stack = 0, set(), [root]
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            if item.base is None:
                largest = max(largest, item.size)
            elif isinstance(item.base, np.ndarray):
                stack.append(item.base)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list | tuple | set | frozenset):
            stack.extend(item)
        elif isinstance(getattr(item, "__dict__", None), dict):
            stack.extend(vars(item).values())
    return largest


def test_a_fitted_fs_model_keeps_and_pickles_no_stack_of_level_triangles(_wide_border_fs_fit):
    """Opus review P1: fails on the unfixed route, whose fitted model held the triangles
    and level Grams (and, after a Tweedie fit, the lineage memo's signed system) and
    pickled 2.7 stacks of them.

    Everything the model keeps, the design's caches included, is below the
    triangles' trailing rows alone, ``K p (p - k)``; the pickle is below one
    stack, ``8 K p^2`` bytes (it holds the design's ``O(n)`` arrays and the
    ``O(K k p)`` leaf data and factor).
    """
    import pickle

    model, p = _wide_border_fs_fit
    K, k = _WIDE["K"], _WIDE["k"]
    assert _largest_reachable_array(model) < K * p * (p - k)
    assert len(pickle.dumps(model)) < 8 * K * p * p


@pytest.mark.parametrize("signed", [False, True])
def test_an_fs_leaf_pass_forms_no_stack_beyond_what_the_step_reads(_wide_border_fs_fit, signed):
    """Opus review P1: fails on the unfixed pass, which peaked at 3.0 stacks of level
    triangles on Fisher rows and 7.0 on signed rows.

    Fisher rows keep each level's leading ``k`` rows, the trailing rows' norms
    and their Gram, so the build stays below one stack, ``8 K p^2`` bytes.
    Signed rows keep the pseudo-rows the per-lambda step reads (one stack) and
    nothing else of that size: below two.
    """
    import tracemalloc

    from superglm.solvers._structured.block_leaves import build_factor_smooth_leaf_system
    from superglm.solvers.structured import get_structured_layout

    model, p = _wide_border_fs_fit
    K = _WIDE["K"]
    system = model._linear_system_state.system
    layout = get_structured_layout(
        model._dm, model._groups, dominant_group_index=system.dominant_group_index
    )
    rng = np.random.default_rng(3)
    n = model._dm.n

    def rows():
        W = rng.uniform(0.5, 1.5, n)
        if signed:
            W = np.where(rng.uniform(size=n) < 0.2, -0.3 * W, W)
        return W, rng.normal(size=n)

    # a first build compiles (or loads) the pass's kernels outside the trace
    build_factor_smooth_leaf_system(layout, *rows(), signed=signed)
    W, Wz = rows()
    tracemalloc.start()
    try:
        built = build_factor_smooth_leaf_system(layout, W, Wz, signed=signed)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < (2 if signed else 1) * 8 * K * p * p
    assert built.leaf.triangles is None


def test_the_fs_build_after_a_compile_frees_the_stack_the_compile_pinned(
    _wide_border_fs_fit, monkeypatch
):
    """#432 (cold numba cache): one collection at the next build frees a compile's stack.

    Numba's type inference keeps the overload failures it meets in reference
    cycles whose frames reach every frame on the stack at the compile, so a
    factor built while its kernel compiled kept its construction frame, the
    leaf data and the signed pseudo-rows (one stack, ``8 K p^2`` bytes) until
    a full collection, which a fit rarely reaches.  Numba keeps them only
    while its overload resolution is cold in the process, so the factor's
    caller keeps such a failure itself; an uncached copy of the trial kernel
    makes the factor compile it.  Automatic collection is held off (the
    collector stays enabled), so only the build's own collection can free
    the cycle.  The next build after the compile frees the stack, and a build
    with no compile since collects nothing.  Fails without
    ``collect_after_compile`` (the stack outlives both builds) and without the
    compile listener (no collection).
    """
    import gc
    import weakref

    import numba

    from superglm import _numba_compile
    from superglm.solvers._structured import block_leaves
    from superglm.solvers.structured import get_structured_layout

    model, _ = _wide_border_fs_fit
    K, k = _WIDE["K"], _WIDE["k"]
    system = model._linear_system_state.system
    layout = get_structured_layout(
        model._dm, model._groups, dominant_group_index=system.dominant_group_index
    )
    n = model._dm.n
    rng = np.random.default_rng(5)
    kernel = block_leaves._signed_trials
    options = {
        key: value
        for key, value in kernel.targetoptions.items()
        if key not in ("cache", "nopython")
    }
    monkeypatch.setattr(block_leaves, "_signed_trials", numba.njit(**options)(kernel.py_func))
    q = len(system.operator.small_indices)
    layout_cache = model._dm._structured_layout_cache

    def build():
        return block_leaves.build_factor_smooth_leaf_system(
            layout, rng.uniform(0.5, 1.5, n), rng.normal(size=n), signed=True
        )

    collections = []

    def count(phase, info):
        if phase == "start" and info["generation"] == 2:
            collections.append(phase)

    block_leaves.release_leaf_memo(layout_cache)
    _numba_compile.collect_after_compile()  # whatever compiled before this test
    gc.collect()
    thresholds = gc.get_threshold()
    gc.set_threshold(0)  # no automatic collection; the collector stays enabled
    gc.callbacks.append(count)

    def construct(built):
        penalized = block_leaves.FactorSmoothPenalizedOperator.with_penalties(
            built.operator, np.eye(q), np.broadcast_to(np.eye(k), (K, k, k))
        )
        factor = block_leaves.FactorSmoothLeafFactor(built, penalized)  # compiles the kernel
        try:
            raise RuntimeError("an overload failure numba keeps")
        except RuntimeError as failure:
            kept = failure  # kept -> its traceback -> this frame -> kept
        return factor.rank == factor.shape[0] and kept is not None

    try:
        built = build()
        rows = built.leaf.pseudo_rows
        stack = weakref.ref(rows if rows.base is None else rows.base)
        del rows
        assert construct(built)
        del built
        block_leaves.release_leaf_memo(layout_cache)
        pinned = stack() is not None
        build()
        block_leaves.release_leaf_memo(layout_cache)
        build()
        block_leaves.release_leaf_memo(layout_cache)
    finally:
        gc.callbacks.remove(count)
        gc.set_threshold(*thresholds)
    assert pinned
    assert stack() is None
    assert len(collections) == 1


def test_the_first_build_after_warmup_does_not_collect(_wide_border_fs_fit) -> None:
    """``superglm.warmup()`` compiles outside any fit, so the next build does not collect (#432).

    Warmup compiles an uncached inline helper (``_add_raw_row``) in every
    process, which marked a compile, and a warm-cache process then paid one
    full collection on its first fit: 0.15 s on a 30,000-row ``sz`` sentinel.
    The frames warmup's compiles keep hold no fit's arrays, so it drops its
    mark (``forget_compiles``).  Fails without it: one collection.
    """
    import gc

    import superglm
    from superglm import _numba_compile
    from superglm.solvers._structured import block_leaves
    from superglm.solvers.structured import get_structured_layout

    model, _ = _wide_border_fs_fit
    system = model._linear_system_state.system
    layout = get_structured_layout(
        model._dm, model._groups, dominant_group_index=system.dominant_group_index
    )
    n = model._dm.n
    rng = np.random.default_rng(9)
    layout_cache = model._dm._structured_layout_cache
    collections = []

    def count(phase, info):
        if phase == "start" and info["generation"] == 2:
            collections.append(phase)

    block_leaves.release_leaf_memo(layout_cache)
    _numba_compile._compiled[0] = True  # a compile inside warmup
    superglm.warmup()
    thresholds = gc.get_threshold()
    gc.set_threshold(0)  # no automatic collection; the collector stays enabled
    gc.callbacks.append(count)
    try:
        block_leaves.build_factor_smooth_leaf_system(
            layout, rng.uniform(0.5, 1.5, n), rng.normal(size=n), signed=True
        )
    finally:
        gc.callbacks.remove(count)
        gc.set_threshold(*thresholds)
        block_leaves.release_leaf_memo(layout_cache)
    assert not collections


@pytest.fixture(scope="module")
def _wide_border_sz_fit():
    """A REML fit (smoothing parameters held) of an sz term beside a 60-level categorical."""
    import warnings

    import pandas as pd

    from superglm import Categorical, FactorSmooth, LambdaPolicy, Numeric, Spline, SuperGLM

    K, Q, n, k = _WIDE["K"], _WIDE["Q"], _WIDE["n"], _WIDE["k"]
    rng = np.random.default_rng(11)
    g = np.repeat(np.arange(K), n // K)
    x = rng.uniform(size=n)
    c = rng.integers(0, Q, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "g": [f"g{v:03d}" for v in g],
            "z": rng.normal(size=n),
            "cat": [f"c{v:02d}" for v in c],
        }
    )
    y = np.sin(3 * x) + rng.normal(0, 0.4, K)[g] * x + rng.normal(0, 0.3, Q)[c]
    y = y + rng.normal(0, 0.5, n)
    model = SuperGLM(
        family="gaussian",
        features={
            "z": Numeric(),
            "cat": Categorical(),
            "x": Spline(n_knots=6, lambda_policy=LambdaPolicy.fixed(1.0)),
        },
        interactions=[
            FactorSmooth(
                "x", group="g", basis="sz", k=k, lambda_policy={"wiggle": LambdaPolicy.fixed(1.0)}
            )
        ],
        selection_penalty=0,
        direct_solve="structured",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    system = model._linear_system_state.system
    p = k + len(system.operator.small_indices) + 2
    return model, p


@pytest.mark.parametrize("signed", [False, True])
def test_an_sz_leafs_trailing_factor_forms_no_stack_of_level_triangles(_wide_border_sz_fit, signed):
    """#432 b: the trailing rows' triangular factor an sz leaf reads costs no ``(K, p, p)`` stack.

    The factor (``trailing_root``) was the QR of every level's trailing rows,
    which a second, whole leaf pass formed by keeping every level's triangle
    (and on signed rows its middle factor and error Gram): 3 stacks at its
    peak.  It is now a TSQR merge as the compact pass closes each level
    (``_close_level``): beside a weightless level, in the build's own pass;
    otherwise by one more compact pass on first read.  Peaks as the fs build
    (below one stack, two on signed rows for their pseudo-rows), and the
    factor's Gram is the stacked rows' within both computations' Householder
    backward error, ``(2 gamma~ + gamma~^2) ||t_i|| ||t_j||`` columnwise
    (Higham 2002, Thm 19.4; ``gamma~_k = gamma_(10k)``).
    """
    import tracemalloc

    import superglm.solvers._structured.block_leaves as leaves
    from superglm.solvers.structured import get_structured_layout

    model, p = _wide_border_sz_fit
    K, k = _WIDE["K"], _WIDE["k"]
    system = model._linear_system_state.system
    layout = get_structured_layout(
        model._dm, model._groups, dominant_group_index=system.dominant_group_index
    )
    rng = np.random.default_rng(5)
    n = model._dm.n
    W = rng.uniform(0.5, 1.5, n)
    if signed:
        W = np.where(rng.uniform(size=n) < 0.2, -0.3 * W, W)
    Wz = rng.normal(size=n)
    prior = np.ones(n)
    prior[layout.dominant.codes == 7] = 0.0  # one weightless level: thin
    stack = 8 * K * p * p
    # first builds compile (or load) the kernels outside the traces
    warm = leaves.build_factor_smooth_leaf_system(layout, W * 1.01, Wz, signed=signed)
    warm.leaf.trailing_root()
    leaves.build_factor_smooth_leaf_system(layout, W * 1.02, Wz, prior_weights=prior, signed=signed)

    plain = leaves.build_factor_smooth_leaf_system(layout, W, Wz, signed=signed)
    assert plain.leaf.tail_root is None
    tracemalloc.start()
    try:
        root = plain.leaf.trailing_root()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < (2 if signed else 1) * stack

    tracemalloc.start()
    try:
        thin = leaves.build_factor_smooth_leaf_system(
            layout, W, Wz, prior_weights=prior, signed=signed
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert thin.thin_levels and thin.leaf.tail_root is not None
    assert peak < (2 if signed else 1) * stack

    whole = leaves._leaf_pass(layout, W, Wz, plain.leaf.center, None, signed=signed)[0]
    q1 = p - k - 1
    rows = np.ascontiguousarray(whole[:, k:, k : k + q1]).reshape(-1, q1)
    reference = np.linalg.qr(rows, mode="r")
    norms = np.linalg.norm(rows, axis=0)
    unit = np.finfo(np.float64).eps / 2
    count = 10 * rows.shape[0] * q1
    householder = count * unit / (1 - count * unit)
    bound = 2.0 * (2.0 * householder + householder**2) * np.outer(norms, norms)
    assert np.all(np.abs(root.T @ root - reference.T @ reference) <= bound)


def test_a_thin_level_sz_fit_runs_one_compact_pass_per_leaf_system(monkeypatch):
    """#432 c: beside a thin level every leaf system costs one compact pass, which merges its factor.

    The thin levels' aliases read the trailing rows' factor at every factor
    build, and each leaf system formed it by a second, whole pass: 185 to 215
    passes on the signed fixtures against about 100 systems, 5 against 2 on
    this Fisher one.  The compact pass now merges it (``_close_level``), so
    the counts match and no pass keeps the level triangles.
    """
    import warnings

    import superglm.solvers._structured.block_leaves as leaves
    from tests.test_factor_smooth_sz_thin_and_influence import _model, _signed_aliased_frame

    passes, systems = [], []
    leaf_pass, assemble = leaves._leaf_pass, leaves._assemble_compact

    def counted_pass(*args, **kwargs):
        passes.append((kwargs.get("compact", False), kwargs.get("tail_root", False)))
        return leaf_pass(*args, **kwargs)

    def counted_system(*args, **kwargs):
        systems.append(1)
        return assemble(*args, **kwargs)

    monkeypatch.setattr(leaves, "_leaf_pass", counted_pass)
    monkeypatch.setattr(leaves, "_assemble_compact", counted_system)
    frame, y, weight = _signed_aliased_frame("weightless", response="fisher")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _model("gaussian", "auto", lam=None, numerics=("x1", "x10"))
        model.fit_reml(frame, y, sample_weight=weight)
    assert model._reml_profile["direct_backend"] == "structured"
    assert systems and len(passes) == len(systems)
    assert all(compact and root for compact, root in passes)
