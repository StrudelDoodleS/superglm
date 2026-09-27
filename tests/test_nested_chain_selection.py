"""Rule B's chain choice and the lifetime of the nesting cache.

Rule B (notes/research/2026-09-26-nested-random-effect-elimination.md, section
5) takes, above the largest RandomEffect term, the strictly nested chain that
eliminates the most levels.  The fixtures carry a crossed coarsening of the
leaf (``engine``, a function of ``variant`` crossed with ``model``) beside the
make > model > variant hierarchy, so the finest passing term is not the head
of the heaviest chain.

The nesting cache entries (parent codes, the chain, the chain's tree) read
only RandomEffect codes, so a lambda rebuild of the design carries them and
the O(n) row tests run once per design, not once per REML outer iteration.
"""

from __future__ import annotations

import warnings
from itertools import chain, combinations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import superglm.solvers._structured.layout as layout_module
import superglm.solvers._structured.selection as selection
from superglm import RandomEffect, Spline, SuperGLM
from superglm.dm_builder import rebuild_design_matrix_with_lambdas
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.solvers.structured import (
    find_nested_chain,
    get_structured_layout,
    nested_parent_codes,
)
from superglm.types import GroupSlice

LEAF_LEVELS, OBSERVED_LEAVES = 60, 58


def _codes(engine_levels: int, n: int) -> dict[str, np.ndarray]:
    """Row codes of make (4) > model (12) > variant (60) and its crossed coarsenings.

    Model ``m`` holds variants ``5m .. 5m + 4`` and variant ``v`` has engine
    ``v % engine_levels`` and fuel ``engine % 3``, so engine and fuel are
    functions of variant crossed with make and model.  Model 0 sits under
    make 2 and variants 58 and 59 are unobserved: a pair test that read an
    unobserved leaf as level 0 of every term would refuse model in make.
    """
    rng = np.random.default_rng(engine_levels)
    variant = np.concatenate([np.arange(OBSERVED_LEAVES), rng.integers(0, OBSERVED_LEAVES, n)])
    model = variant // 5
    make_of_model = np.arange(12) % 4
    make_of_model[0] = 2
    engine = variant % engine_levels
    return {
        "make": make_of_model[model],
        "model": model,
        "engine": engine,
        "fuel": engine % 3,
        "crossed": rng.integers(0, 5, len(variant)),
        "variant": variant,
    }


def _design(engine_levels: int, *, fuel: bool, n: int = 400):
    codes = _codes(engine_levels, n)
    names = ["make", "model", "engine", *(["fuel"] if fuel else []), "crossed", "variant"]
    levels = {"make": 4, "model": 12, "engine": engine_levels, "fuel": 3, "crossed": 5}
    levels["variant"] = LEAF_LEVELS
    rows = len(codes["variant"])
    rng = np.random.default_rng(1)
    spline = SparseSSPGroupMatrix(
        sp.random(rows, 4, density=0.5, random_state=2, format="csr"), np.eye(4)
    )
    spline.omega = np.diag([0.0, 1.0, 2.0, 3.0])
    matrices = [
        DenseGroupMatrix(rng.normal(size=(rows, 2))),
        spline,
        *(RandomEffectGroupMatrix(codes[name], levels[name]) for name in names),
    ]
    groups, start = [], 0
    for name, matrix in zip(["x", "spline", *names], matrices, strict=True):
        groups.append(GroupSlice(name=name, start=start, end=start + matrix.shape[1]))
        start += matrix.shape[1]
    return DesignMatrix(matrices, n=rows, p=start), groups


def _names(groups, chain) -> tuple[str, ...]:
    return tuple(groups[index].name for index in chain)


def _row_test(child: np.ndarray, parent: np.ndarray) -> bool:
    """The §3.7 count test: one distinct (child, parent) pair per observed child code."""
    return len(np.unique(np.column_stack((child, parent)), axis=0)) == len(np.unique(child))


def _heaviest_chain_levels(dm: DesignMatrix, groups, leaf: int) -> int:
    """Brute force over every subset of the other RandomEffect terms.

    A chain is totally ordered by nesting, so a valid subset passes the row
    test between consecutive terms once sorted coarsest first.
    """
    matrices = dm.group_matrices
    terms = [
        index
        for index, matrix in enumerate(matrices)
        if isinstance(matrix, RandomEffectGroupMatrix) and index != leaf
    ]
    subsets = chain.from_iterable(combinations(terms, size) for size in range(1, len(terms) + 1))
    best = 0
    for subset in subsets:
        members = [*sorted(subset, key=lambda index: len(np.unique(matrices[index].codes))), leaf]
        if all(
            _row_test(matrices[fine].codes, matrices[coarse].codes)
            for coarse, fine in zip(members[:-1], members[1:], strict=False)
        ):
            best = max(best, sum(matrices[index].n_levels for index in subset))
    return best


@pytest.mark.parametrize(
    ("engine_levels", "fuel", "expected"),
    [
        (14, False, ("make", "model", "variant")),
        (14, True, ("fuel", "engine", "variant")),
        (20, False, ("engine", "variant")),
    ],
    ids=["hierarchy-16-beats-engine-14", "fuel-engine-17-beats-16", "engine-20-beats-16"],
)
def test_the_chain_eliminating_most_levels_is_chosen(engine_levels, fuel, expected) -> None:
    dm, groups = _design(engine_levels, fuel=fuel)
    leaf = len(groups) - 1
    found = find_nested_chain(dm.group_matrices, groups, leaf_index=leaf)
    assert _names(groups, found) == expected
    matrices = dm.group_matrices
    for coarse, fine in zip(found[:-1], found[1:], strict=True):
        assert _row_test(matrices[fine].codes, matrices[coarse].codes)
    eliminated = sum(matrices[index].n_levels for index in found[:-1])
    assert eliminated == _heaviest_chain_levels(dm, groups, leaf)


def test_candidate_pairs_are_tested_on_the_leaf_levels_not_the_rows(monkeypatch) -> None:
    dm, groups = _design(14, fuel=True)
    leaf = len(groups) - 1
    matrices = dm.group_matrices
    calls = []

    def counting(child, parent):
        calls.append((child, parent))
        return nested_parent_codes(child, parent)

    monkeypatch.setattr(selection, "nested_parent_codes", counting)
    cache = selection.shared_nesting_cache(dm._scalar_structured_layout_cache)
    found = find_nested_chain(matrices, groups, leaf_index=leaf, cache=cache)
    # one row test per other RandomEffect term, each against the leaf
    assert len(calls) == leaf - 2
    assert {id(parent) for _, parent in calls} == {id(matrices[i]) for i in range(2, leaf)}
    assert all(child is matrices[leaf] for child, _ in calls)
    pair_entries = {key[1:3]: codes for key, codes in cache.items() if key[0] == "nested_parent"}
    model, make = 3, 2
    assert pair_entries[(model, make)] is not None
    for (child, parent), codes in pair_entries.items():
        reference = nested_parent_codes(matrices[child], matrices[parent])
        assert (codes is None) == (reference is None)
        if reference is not None:
            np.testing.assert_array_equal(codes, reference)
    # the layout finds every consecutive chain pair in the cache
    calls.clear()
    layout = get_structured_layout(dm, groups, dominant_group_index=leaf, chain_group_indices=found)
    assert calls == []
    assert _names(groups, layout.chain_group_indices) == ("fuel", "engine", "variant")


def test_lambda_rebuilds_share_the_nesting_cache_and_rebuild_the_border() -> None:
    dm, groups = _design(14, fuel=False)
    leaf = len(groups) - 1
    decision = selection.resolve_structured_backend(
        list(dm.group_matrices),
        groups,
        direct_solve="structured",
        coefficient_width=dm.p,
        lambda2=1.0,
        nesting_cache=dm._scalar_structured_layout_cache,
    )
    chain = decision.chain_group_indices
    weights = np.ones(dm.n)

    def rebuild(design: DesignMatrix, lam: float) -> DesignMatrix:
        rebuilt = rebuild_design_matrix_with_lambdas(design, groups, {"spline": lam}, weights, 1.0)
        assert rebuilt.group_matrices[1] is not design.group_matrices[1]
        return rebuilt

    def layout(design: DesignMatrix):
        return get_structured_layout(
            design, groups, dominant_group_index=leaf, chain_group_indices=chain
        )

    # the REML driver's designs and the final refit's are both rebuilt from the
    # fitted design; the driver builds the tree, the refit must find it
    driver = rebuild(dm, 3.0)
    driven = layout(driver)
    final = rebuild(dm, 5.0)
    assert set(final._scalar_structured_layout_cache) == {"nesting"}
    refit = layout(final)
    shared = dm._scalar_structured_layout_cache["nesting"]
    assert final._scalar_structured_layout_cache["nesting"] is shared
    assert {key[0] for key in shared} == {"nested_parent", "nested_chain", "nested_tree"}
    assert refit.tree is driven.tree
    # the leaf argsort, the one O(n) entry, is shared rather than recomputed
    assert refit.leaf_order is driven.leaf_order and refit.leaf_starts is driven.leaf_starts
    # the border belongs to each design, and a layout never outlives its design
    assert refit.small_matrices[1] is final.group_matrices[1]
    assert driven.small_matrices[1] is driver.group_matrices[1]
    assert set(rebuild(driver, 7.0)._scalar_structured_layout_cache) == {"nesting"}


def _frame(n: int = 3000) -> tuple[pd.DataFrame, np.ndarray]:
    codes = _codes(14, n)
    rng = np.random.default_rng(3)
    x = rng.uniform(size=len(codes["variant"]))
    eta = (
        -0.5
        + 0.3 * np.sin(6.0 * x)
        + rng.normal(0.0, 0.3, 4)[codes["make"]]
        + rng.normal(0.0, 0.25, 12)[codes["model"]]
        + rng.normal(0.0, 0.25, LEAF_LEVELS)[codes["variant"]]
        + rng.normal(0.0, 0.2, 5)[codes["crossed"]]
    )
    frame = pd.DataFrame({name: [f"{name}{c}" for c in codes[name]] for name in codes})
    frame["x"] = x
    return frame, rng.poisson(np.exp(eta)).astype(float)


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "discrete"])
def test_the_row_nesting_tests_run_once_per_fit(discrete, monkeypatch) -> None:
    frame, y = _frame()
    calls, layouts = [], []

    def counting(child, parent):
        calls.append((child, parent))
        return nested_parent_codes(child, parent)

    build = layout_module.build_nested_structured_layout

    def recording(*args, **kwargs):
        layouts.append(build(*args, **kwargs))
        return layouts[-1]

    monkeypatch.setattr(selection, "nested_parent_codes", counting)
    monkeypatch.setattr(layout_module, "build_nested_structured_layout", recording)
    terms = ("make", "model", "engine", "crossed", "variant")
    model = SuperGLM(
        family="poisson",
        features={"x": Spline(n_knots=8), **{name: RandomEffect() for name in terms}},
        selection_penalty=0,
        direct_solve="structured",
        discrete=discrete,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame[["x", *terms]], y)
    assert model._reml_profile["structured_chain"] == ("make", "model", "variant")
    # every design of the fit (one nested layout each) shares one set of row tests
    assert len(layouts) >= 2
    assert len(calls) == len(terms) - 1
    assert all(layout.tree is layouts[0].tree for layout in layouts)
