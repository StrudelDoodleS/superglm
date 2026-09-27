"""Plumbing of the nested random-effect chain (fix D).

Chain detection, the nested layout and design products, the exact centred
leaf row pass, the nested system and penalty assembly, backend resolution,
dispatch, estimability and the RandomEffect standard-error route.  The factor
algebra itself is tested in ``test_nested_schur_factor.py``.  Section numbers
refer to ``notes/research/2026-09-26-nested-random-effect-elimination.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import superglm.solvers._structured.selection as selection
from superglm import LambdaPolicy, Numeric, RandomEffect, SuperGLM
from superglm.distributions import Gamma, Gaussian, Poisson
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    FactorSmoothGroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.links import LogLink
from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.reml.w_derivatives import reml_w_correction
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.rank import decompose_factor
from superglm.solvers.structured import (
    CenteredBlockOperator,
    NestedStructuredLayout,
    ProfiledNestedSchurFactor,
    ProfiledScalarSchurFactor,
    _random_effect_auto_cost_ratios,
    build_nested_leaf_statistics,
    build_nested_structured_layout,
    build_nested_structured_system,
    build_penalized_nested_operator,
    centered_operator_coefficient_estimable,
    compact_operator_diagonal,
    find_nested_chain,
    get_structured_layout,
    materialize_compact_operator,
    nested_parent_codes,
    resolve_structured_backend,
    structured_design_matvec,
    structured_design_rmatvec,
)
from superglm.types import GroupSlice, PenaltyComponent

EPS = np.finfo(np.float64).eps
NAMES = ("numeric", "make", "spline", "model", "binned", "category", "crossed", "variant")
MAKE, MODEL, CROSSED, VARIANT = 1, 3, 6, 7
CHAIN = (MAKE, MODEL, VARIANT)
# Border columns of the dense block that are constant within every leaf: an
# intercept-like column, a leaf attribute and a root attribute (§3.4).
LEAF_CONSTANT_COLUMNS = (0, 3, 4)


@dataclass
class NestedCase:
    dm: DesignMatrix
    groups: list[GroupSlice]
    weights: np.ndarray
    codes: tuple[np.ndarray, ...]
    sizes: tuple[int, ...]

    @property
    def matrices(self) -> list:
        return list(self.dm.group_matrices)


def _nested_case(seed: int = 20260927, n: int = 600, weight_scale: float = 1.0) -> NestedCase:
    """A make > model > variant chain (3/7/16 observed plus 1/1/2 unobserved levels).

    The border mixes every kernel the leaf row pass must read: dense columns
    (three of them constant within leaves), a sparse spline, a discretized
    spline, a categorical and a crossed random effect whose level count lies
    between two chain levels.  Leaves 3 and 7 carry zero weight.
    """
    rng = np.random.default_rng(seed)
    sizes = (4, 8, 18)
    model_parent = np.concatenate([np.arange(3), rng.integers(0, 3, 4)])
    variant_parent = np.concatenate([np.arange(7), rng.integers(0, 7, 9)])
    variant = rng.integers(0, 16, n)
    model = variant_parent[variant]
    make = model_parent[model]
    leaf_attribute = rng.normal(size=sizes[2])
    root_attribute = rng.normal(size=sizes[0])
    dense = np.column_stack(
        [
            np.ones(n),
            rng.normal(size=n),
            10.0 + 3.0 * rng.normal(size=n),
            leaf_attribute[variant],
            root_attribute[make],
        ]
    )
    spline_basis = sp.random(n, 4, density=0.5, random_state=seed, format="csr")
    spline = SparseSSPGroupMatrix(spline_basis, rng.normal(size=(4, 3)))
    spline.omega = np.diag([0.0, 1.0, 2.0, 3.0])
    binned = DiscretizedSSPGroupMatrix(
        rng.normal(size=(6, 3)), np.eye(3), rng.integers(0, 6, size=n)
    )
    matrices = [
        DenseGroupMatrix(dense),
        RandomEffectGroupMatrix(make, sizes[0]),
        spline,
        RandomEffectGroupMatrix(model, sizes[1]),
        binned,
        CategoricalGroupMatrix(rng.integers(-1, 3, size=n), n_levels=3),
        RandomEffectGroupMatrix(rng.integers(0, 5, size=n), 5),
        RandomEffectGroupMatrix(variant, sizes[2]),
    ]
    groups: list[GroupSlice] = []
    start = 0
    for name, matrix in zip(NAMES, matrices, strict=True):
        groups.append(
            GroupSlice(
                name=name,
                start=start,
                end=start + matrix.shape[1],
                penalized=name not in ("numeric", "category", "binned"),
            )
        )
        start += matrix.shape[1]
    weights = np.exp(rng.normal(size=n)) * weight_scale
    weights[np.isin(variant, (3, 7))] = 0.0
    return NestedCase(
        dm=DesignMatrix(matrices, n=n, p=start),
        groups=groups,
        weights=weights,
        codes=(make, model, variant),
        sizes=sizes,
    )


def _layout(case: NestedCase, chain: tuple[int, ...] = CHAIN) -> NestedStructuredLayout:
    layout = get_structured_layout(
        case.dm, case.groups, dominant_group_index=chain[-1], chain_group_indices=chain
    )
    assert isinstance(layout, NestedStructuredLayout)
    return layout


def _border_rows(layout: NestedStructuredLayout) -> np.ndarray:
    return np.hstack([matrix.toarray() for matrix in layout.small_matrices])


def _identity_components(case: NestedCase) -> list[PenaltyComponent]:
    return [
        PenaltyComponent(
            name=case.groups[index].name,
            group_name=case.groups[index].name,
            group_index=index,
            group_sl=case.groups[index].sl,
            omega_raw=None,
            penalty_kind="identity",
        )
        for index in (MAKE, MODEL, CROSSED, VARIANT)
    ]


def _components(case: NestedCase) -> list[PenaltyComponent]:
    """Identity components for every random effect plus the dense spline penalty.

    The dense oracle ``build_penalty_matrix`` penalizes random effects only
    through components.
    """
    spline = case.matrices[2]
    return [
        *_identity_components(case),
        PenaltyComponent(
            name="spline",
            group_name="spline",
            group_index=2,
            group_sl=case.groups[2].sl,
            omega_raw=spline.omega,
            omega_ssp=spline.R_inv.T @ spline.omega @ spline.R_inv,
        ),
    ]


def _dense_penalty(case: NestedCase, lambda2) -> np.ndarray:
    return build_penalty_matrix(
        case.matrices, case.groups, lambda2, case.dm.p, reml_penalties=_components(case)
    )


def _lambdas(**overrides: float) -> dict[str, float]:
    return {"make": 0.5, "model": 2.0, "crossed": 1.5, "variant": 3.0, "spline": 0.3} | overrides


# ── Chain detection ───────────────────────────────────────────────────────


def test_chain_is_found_coarsest_to_finest_and_skips_the_crossed_term() -> None:
    case = _nested_case()
    chain = find_nested_chain(case.matrices, case.groups, leaf_index=VARIANT)
    assert chain == CHAIN
    assert CROSSED not in chain


def test_parent_codes_agree_with_a_pair_count_reference() -> None:
    case = _nested_case()
    make, model, variant = case.matrices[MAKE], case.matrices[MODEL], case.matrices[VARIANT]
    for child, parent in ((variant, model), (model, make), (variant, make)):
        codes = nested_parent_codes(child, parent)
        pairs = np.unique(np.column_stack((child.codes, parent.codes)), axis=0)
        # The §3.7 count test: one distinct pair per observed child code.
        assert len(pairs) == len(np.unique(child.codes))
        np.testing.assert_array_equal(codes[pairs[:, 0]], pairs[:, 1])
        unobserved = np.setdiff1d(np.arange(child.n_levels), child.codes)
        np.testing.assert_array_equal(codes[unobserved], 0)
    crossed = case.matrices[CROSSED]
    pairs = np.unique(np.column_stack((variant.codes, crossed.codes)), axis=0)
    assert len(pairs) > len(np.unique(variant.codes))
    assert nested_parent_codes(variant, crossed) is None


def test_implicit_nesting_is_not_chained_and_a_forced_chain_names_the_interaction() -> None:
    rng = np.random.default_rng(3)
    n = 400
    model = rng.integers(0, 12, size=n)
    make = model % 3
    # Variant labels 0..3 reused under every model: nested only implicitly.
    implicit_variant = rng.integers(0, 4, size=n)
    matrices = [
        DenseGroupMatrix(rng.normal(size=(n, 2))),
        RandomEffectGroupMatrix(make, 3),
        RandomEffectGroupMatrix(implicit_variant, 4),
        RandomEffectGroupMatrix(model, 12),
    ]
    groups, start = [], 0
    for name, matrix in zip(("x", "make", "variant", "model"), matrices, strict=True):
        groups.append(GroupSlice(name=name, start=start, end=start + matrix.shape[1]))
        start += matrix.shape[1]
    assert find_nested_chain(matrices, groups, leaf_index=3) == (1, 3)
    with pytest.raises(ValueError, match="interaction code"):
        build_nested_structured_layout(matrices, groups, chain_group_indices=(3, 2))


def test_a_factor_smooth_term_keeps_the_single_block_backend() -> None:
    case = _nested_case()
    x = np.linspace(-1.0, 1.0, case.dm.n)
    smooth = FactorSmoothGroupMatrix(
        sp.csr_matrix(np.column_stack((np.ones_like(x), x))),
        case.codes[1],
        case.sizes[1],
        natural_map=np.eye(2),
        levels=tuple(range(case.sizes[1])),
        repeated_penalty_components=(("wiggle", np.eye(2)),),
    )
    matrices = [*case.matrices, smooth]
    groups = [
        *case.groups,
        GroupSlice(name="x:model", start=case.dm.p, end=case.dm.p + smooth.shape[1]),
    ]
    decision = resolve_structured_backend(
        matrices,
        groups,
        direct_solve="structured",
        coefficient_width=case.dm.p + smooth.shape[1],
        lambda2=_lambdas() | {"x:model:wiggle": 1.0},
    )
    assert decision.chain_group_indices == (len(matrices) - 1,)


@pytest.mark.parametrize("route", ["dict_zero", "dict_missing", "override_zero"])
def test_a_zero_penalty_middle_level_leaves_the_chain_and_parents_compose(route) -> None:
    case = _nested_case()
    lambda2 = _lambdas()
    override = None
    if route == "dict_zero":
        lambda2["model"] = 0.0
    elif route == "dict_missing":
        del lambda2["model"]
    else:
        override = _dense_penalty(case, lambda2)
        override[case.groups[MODEL].sl, case.groups[MODEL].sl] = 0.0
    decision = resolve_structured_backend(
        case.matrices,
        case.groups,
        direct_solve="structured",
        coefficient_width=case.dm.p,
        row_weights=case.weights,
        lambda2=lambda2,
        S_override=override,
    )
    assert decision.use_structured
    assert decision.chain_group_indices == (MAKE, VARIANT)
    layout = _layout(case, decision.chain_group_indices)
    make, _model, variant = case.codes
    np.testing.assert_array_equal(layout.tree.parent[1][variant], make)
    assert MODEL in layout.small_group_indices


def test_a_zero_penalty_leaf_keeps_the_single_level_refusal() -> None:
    case = _nested_case()
    with pytest.raises(
        ValueError, match="has zero penalty and is aliased with the fitted intercept"
    ):
        _resolve(case, row_weights=None, lambda2=_lambdas(variant=0.0))


# ── Layout and design products ───────────────────────────────────────────


def test_nested_layout_refuses_malformed_chains() -> None:
    case = _nested_case()
    build = build_nested_structured_layout
    with pytest.raises(ValueError, match="at least two"):
        build(case.matrices, case.groups, chain_group_indices=(VARIANT,))
    with pytest.raises(ValueError, match="not a RandomEffect"):
        build(case.matrices, case.groups, chain_group_indices=(0, VARIANT))
    widened = list(case.groups)
    widened[MODEL] = replace(widened[MODEL], end=widened[MODEL].end - 1)
    with pytest.raises(ValueError, match="slice does not match"):
        build(case.matrices, widened, chain_group_indices=CHAIN)
    constrained = list(case.groups)
    constrained[MAKE] = replace(constrained[MAKE], constraints=object())
    with pytest.raises(ValueError, match="constraints or SCOP"):
        build(case.matrices, constrained, chain_group_indices=CHAIN)
    with pytest.raises(ValueError, match="interaction code"):
        build(case.matrices, case.groups, chain_group_indices=(CROSSED, VARIANT))


def test_nested_layout_border_tree_and_reference_rows() -> None:
    case = _nested_case()
    layout = _layout(case)
    assert layout.chain_group_names == ("make", "model", "variant")
    assert layout.tree.sizes == case.sizes
    assert set(layout.small_group_indices) == set(range(len(NAMES))) - set(CHAIN)
    assert CROSSED in layout.small_group_indices
    for level, index in enumerate(CHAIN):
        np.testing.assert_array_equal(
            layout.level_indices[level], np.arange(case.groups[index].start, case.groups[index].end)
        )
    variant = case.codes[2]
    for leaf in range(case.sizes[2]):
        rows = np.flatnonzero(variant == leaf)
        assert layout.reference_row[leaf] == (rows[0] if len(rows) else -1)
    assert np.all(layout.reference_row[16:] == -1)
    make, model, _ = case.codes
    np.testing.assert_array_equal(layout.tree.parent[2][variant], model)
    np.testing.assert_array_equal(layout.tree.parent[1][model], make)


def test_nested_layout_cache_is_per_chain() -> None:
    case = _nested_case()
    assert _layout(case) is _layout(case)
    assert _layout(case, (MAKE, VARIANT)) is not _layout(case)


def test_nested_design_products_match_the_grouped_design() -> None:
    case = _nested_case()
    layout = _layout(case)
    rng = np.random.default_rng(5)
    beta = rng.normal(size=case.dm.p)
    rows = rng.normal(size=case.dm.n)
    dense = np.hstack([matrix.toarray() for matrix in case.matrices])
    # Each output sums at most p (matvec) or n (rmatvec) rounded products.
    matvec_scale = np.abs(dense) @ np.abs(beta)
    rmatvec_scale = np.abs(dense).T @ np.abs(rows)
    np.testing.assert_array_less(
        np.abs(structured_design_matvec(layout, case.matrices, beta) - case.dm.matvec(beta)),
        2 * (case.dm.p + 2) * EPS * matvec_scale + np.finfo(float).tiny,
    )
    np.testing.assert_array_less(
        np.abs(structured_design_rmatvec(layout, case.matrices, rows) - case.dm.rmatvec(rows)),
        2 * (case.dm.n + 2) * EPS * rmatvec_scale + np.finfo(float).tiny,
    )


# ── Leaf row pass ─────────────────────────────────────────────────────────


def test_leaf_statistics_match_the_dense_formula() -> None:
    case = _nested_case()
    layout = _layout(case)
    stats = build_nested_leaf_statistics(layout, case.matrices, case.weights)
    X = _border_rows(layout)
    w, leaf = case.weights, case.codes[2]
    K = case.sizes[2]
    leaf_weight = np.bincount(leaf, weights=w, minlength=K)
    active = leaf_weight != 0.0
    cross = np.stack([np.bincount(leaf, weights=w * column, minlength=K) for column in X.T], 1)
    mean = np.zeros_like(cross)
    mean[active] = cross[active] / leaf_weight[active, None]
    centered = X - mean[leaf]
    within = centered.T @ (w[:, None] * centered)
    absolute_mass = np.stack(
        [np.bincount(leaf, weights=w * np.abs(column), minlength=K) for column in X.T], 1
    )
    n_leaf = np.bincount(leaf, minlength=K).max()
    # Both means are within (n_leaf + 4) eps of the exact weighted mean, relative
    # to the absolute mass; the scatter is within (n + 4) eps of its absolute
    # value about the exact mean, plus the second-order mean term w_l e e'.
    mean_error = np.zeros_like(mean)
    mean_error[active] = (n_leaf + 4) * EPS * absolute_mass[active] / leaf_weight[active, None]
    absolute_scatter = np.abs(centered).T @ (w[:, None] * np.abs(centered))
    within_bound = 2 * (case.dm.n + 4) * EPS * absolute_scatter + 4 * (
        mean_error.T @ (leaf_weight[:, None] * mean_error)
    )

    np.testing.assert_array_equal(stats.weight, leaf_weight)
    assert np.all(np.abs(stats.cross - cross) <= 2 * (n_leaf + 2) * EPS * absolute_mass)
    assert np.all(np.abs(stats.mean - mean) <= 2 * mean_error)
    np.testing.assert_array_equal(stats.mean[~active], 0.0)
    assert np.all(np.abs(stats.within - within) <= within_bound)
    np.testing.assert_array_equal(stats.within, stats.within.T)
    assert stats.deviation is None


@pytest.mark.parametrize("weight_scale", [1.0, 1e4])
def test_leaf_constant_columns_give_exact_means_and_zero_scatter(weight_scale) -> None:
    case = _nested_case(weight_scale=weight_scale)
    layout = _layout(case)
    stats = build_nested_leaf_statistics(layout, case.matrices, case.weights)
    dense = case.matrices[0].M
    reference = np.zeros((case.sizes[2], dense.shape[1]))
    reference[case.codes[2]] = dense
    active = stats.weight != 0.0
    columns = list(LEAF_CONSTANT_COLUMNS)
    # The shifted mean of a column constant within a leaf is that constant,
    # and its centred scatter row and column are exactly zero (§3.4).
    np.testing.assert_array_equal(
        stats.mean[np.ix_(active, columns)], reference[active][:, columns]
    )
    np.testing.assert_array_equal(stats.within[columns, :], 0.0)
    np.testing.assert_array_equal(stats.within[:, columns], 0.0)
    signed_rows = np.random.default_rng(1).normal(size=case.dm.n) * case.weights
    signed = build_nested_leaf_statistics(layout, case.matrices, signed_rows, mean=stats.mean)
    np.testing.assert_array_equal(signed.deviation[:, columns], 0.0)
    np.testing.assert_array_equal(signed.within[columns, :], 0.0)


def test_leaf_statistics_do_not_depend_on_the_chunking() -> None:
    case = _nested_case()
    layout = _layout(case)
    results = [
        build_nested_leaf_statistics(layout, case.matrices, case.weights, chunk_size=size)
        for size in (1, 7, 8192)
    ]
    X = _border_rows(layout)
    centered = X - results[-1].mean[case.codes[2]]
    absolute_scatter = np.abs(centered).T @ (case.weights[:, None] * np.abs(centered))
    for stats in results[:-1]:
        assert np.all(np.abs(stats.mean - results[-1].mean) <= 16 * EPS * np.abs(stats.mean))
        assert np.all(
            np.abs(stats.within - results[-1].within)
            <= 2 * (case.dm.n + 4) * EPS * absolute_scatter + np.finfo(float).tiny
        )


def test_signed_pass_deviation_matches_exact_rational_sums() -> None:
    case = _nested_case(n=240)
    layout = _layout(case)
    data = build_nested_leaf_statistics(layout, case.matrices, case.weights)
    a = np.random.default_rng(9).normal(size=case.dm.n) * case.weights
    signed = build_nested_leaf_statistics(layout, case.matrices, a, mean=data.mean)
    assert signed.mean is not data.mean
    np.testing.assert_array_equal(signed.mean, data.mean)
    X, leaf = _border_rows(layout), case.codes[2]
    exact = [[Fraction(0)] * X.shape[1] for _ in range(case.sizes[2])]
    for row in range(case.dm.n):
        cell = leaf[row]
        for column in range(X.shape[1]):
            difference = Fraction(float(X[row, column])) - Fraction(float(data.mean[cell, column]))
            exact[cell][column] += Fraction(float(a[row])) * difference
    exact = np.array([[float(value) for value in row] for row in exact])
    centered = np.abs(X - data.mean[leaf]) * np.abs(a)[:, None]
    absolute = np.stack(
        [np.bincount(leaf, weights=column, minlength=case.sizes[2]) for column in centered.T], 1
    )
    n_leaf = np.bincount(leaf).max()
    assert np.all(np.abs(signed.deviation - exact) <= (n_leaf + 4) * EPS * absolute)


# ── System and penalties ─────────────────────────────────────────────────


def _chain_incidence(case: NestedCase) -> np.ndarray:
    return np.hstack([case.matrices[index].toarray() for index in CHAIN])


def test_nested_system_matches_dense_moments() -> None:
    case = _nested_case()
    layout = _layout(case)
    Wz = case.weights * np.random.default_rng(2).normal(size=case.dm.n)
    system = build_nested_structured_system(
        case.matrices, case.groups, case.weights, Wz, layout=layout
    )
    Z, X, w = _chain_incidence(case), _border_rows(layout), case.weights
    bound = (case.dm.n + 2) * EPS
    assert np.all(np.abs(system.xtw_structured - Z.T @ w) <= bound * (Z.T @ np.abs(w)))
    assert np.all(np.abs(system.xtwz_structured - Z.T @ Wz) <= bound * (Z.T @ np.abs(Wz)))
    assert np.all(np.abs(system.xtw_small - X.T @ w) <= bound * (np.abs(X).T @ np.abs(w)))
    assert np.all(
        np.abs(system.operator.A - X.T @ (w[:, None] * X))
        <= 2 * bound * (np.abs(X).T @ (np.abs(w)[:, None] * np.abs(X)))
    )
    assert system.sum_w == pytest.approx(np.sum(w), rel=bound)
    assert system.chain_group_names == ("make", "model", "variant")
    assert system.dominant_group_name == "variant"
    assert system.operator.tree is layout.tree
    assert system.operator.leaf.deviation is None


@pytest.mark.parametrize("source", ["scalar", "dict", "reml_penalties", "override"])
def test_penalized_nested_operator_places_every_penalty_source(source) -> None:
    case = _nested_case()
    layout = _layout(case)
    system = build_nested_structured_system(
        case.matrices, case.groups, case.weights, np.zeros(case.dm.n), layout=layout
    )
    lambda2: float | dict[str, float] = 0.7 if source == "scalar" else _lambdas()
    components = _components(case) if source == "reml_penalties" else None
    dense = _dense_penalty(case, lambda2)
    override = None
    if source == "override":
        override = dense.copy()
        chain_indices = layout.structured_indices
        override[chain_indices, chain_indices] = np.linspace(0.1, 2.0, len(chain_indices))
        dense = override
    penalized = build_penalized_nested_operator(
        system,
        case.matrices,
        case.groups,
        lambda2,
        reml_penalties=components,
        S_override=override,
    )
    node_penalty = np.concatenate(penalized.node_penalty)
    np.testing.assert_array_equal(node_penalty, np.diag(dense)[layout.structured_indices])
    small = layout.small_indices
    np.testing.assert_allclose(
        penalized.border_penalty, dense[np.ix_(small, small)], rtol=4 * EPS, atol=0
    )
    assert penalized.data is system.operator


@pytest.mark.parametrize("coupling", ["cross_level", "chain_to_border"])
def test_penalized_nested_operator_refuses_a_coupled_override(coupling) -> None:
    case = _nested_case()
    layout = _layout(case)
    system = build_nested_structured_system(
        case.matrices, case.groups, case.weights, np.zeros(case.dm.n), layout=layout
    )
    override = _dense_penalty(case, _lambdas())
    first = case.groups[MAKE].start
    other = case.groups[MODEL].start if coupling == "cross_level" else case.groups[0].start
    override[first, other] = override[other, first] = 0.1
    with pytest.raises(ValueError, match="S_override"):
        build_penalized_nested_operator(
            system, case.matrices, case.groups, _lambdas(), S_override=override
        )


# ── Backend resolution ────────────────────────────────────────────────────


def _resolve(case: NestedCase, **kwargs):
    arguments = (
        dict(
            direct_solve="structured",
            coefficient_width=case.dm.p,
            row_weights=case.weights,
            lambda2=_lambdas(),
        )
        | kwargs
    )
    return resolve_structured_backend(case.matrices, case.groups, **arguments)


def test_forced_structured_resolves_the_full_chain() -> None:
    decision = _resolve(_nested_case())
    assert decision.use_structured
    assert decision.chain_group_indices == CHAIN
    assert decision.group_index == VARIANT and decision.group_name == "variant"
    assert decision.nested_fallback_reason is None


def _auto_case(n: int, q: int, sizes: tuple[int, ...]):
    """A strictly nested chain over ``n`` rows beside a ``q``-column border.

    Every level is observed; each coarser code is a function of the next finer
    one.  Selection reads the border only by its shape.
    """
    codes = [np.arange(n) % sizes[-1]]
    for coarse, fine in zip(sizes[-2::-1], sizes[:0:-1], strict=True):
        codes.insert(0, codes[0] * coarse // fine)
    matrices = [DenseGroupMatrix(np.broadcast_to(0.0, (n, q)))]
    groups = [GroupSlice(name="border", start=0, end=q, penalized=False)]
    for level, (size, level_codes) in enumerate(zip(sizes, codes, strict=True)):
        start = groups[-1].end
        matrices.append(RandomEffectGroupMatrix(level_codes, size))
        groups.append(
            GroupSlice(name=f"level{level}", start=start, end=start + size, penalized=True)
        )
    return matrices, groups, groups[-1].end


# The measured winners that anchor the auto rule in selection.py (2026-09-27):
# rows, border width without the intercept, chain sizes coarsest to finest.
@pytest.mark.parametrize(
    ("n", "q", "sizes", "winner"),
    [
        pytest.param(3_885, 104, (51, 407), "single", id="pg17-C-exact-5k-rows"),
        pytest.param(77_014, 104, (87, 942), "single", id="pg17-C"),
        pytest.param(15_794, 35, (91, 1_332), "single", id="dvsa-C-20k-rows"),
        pytest.param(157_593, 36, (223, 3_749), "single", id="dvsa-C-200k-rows"),
        pytest.param(15_794, 35, (91, 1_332, 2_404), "nested", id="dvsa-D-20k-rows"),
        pytest.param(157_593, 36, (223, 3_749, 6_038), "nested", id="dvsa-D-200k-rows"),
    ],
)
def test_auto_takes_the_measured_winner_between_single_level_and_nested(n, q, sizes, winner):
    matrices, groups, p = _auto_case(n, q, sizes)
    decision = resolve_structured_backend(
        matrices, groups, direct_solve="auto", coefficient_width=p
    )
    chain = tuple(range(1, len(sizes) + 1))
    assert decision.use_structured
    assert decision.auto_cost_ratio == _random_effect_auto_cost_ratios(n, p, sizes)[winner]
    if winner == "nested":
        assert decision.chain_group_indices == chain
        assert decision.nested_fallback_reason is None
    else:
        assert decision.chain_group_indices == chain[-1:]
        assert "single-level backend cheaper" in decision.nested_fallback_reason
    forced = resolve_structured_backend(
        matrices, groups, direct_solve="structured", coefficient_width=p
    )
    assert forced.chain_group_indices == chain


@pytest.mark.parametrize(
    ("family", "link", "admitted"),
    [
        (None, None, True),
        (Poisson(), LogLink(), True),
        (Gamma(), LogLink(), True),
        (Gaussian(), LogLink(), False),
    ],
)
def test_the_weight_gate_admits_only_audited_non_negative_curvature(family, link, admitted):
    decision = _resolve(_nested_case(), family=family, link=link)
    assert decision.use_structured
    if admitted:
        assert decision.chain_group_indices == CHAIN
        assert decision.nested_fallback_reason is None
    else:
        assert decision.chain_group_indices == (VARIANT,)
        assert "not audited non-negative" in decision.nested_fallback_reason


def test_a_coupled_override_declines_the_chain_to_the_single_level_backend() -> None:
    case = _nested_case()
    override = _dense_penalty(case, _lambdas())
    make, model = case.groups[MAKE].start, case.groups[MODEL].start
    override[make, model] = override[model, make] = 0.1
    decision = _resolve(case, S_override=override)
    assert decision.use_structured
    assert decision.chain_group_indices == (VARIANT,)
    assert "must be diagonal" in decision.nested_fallback_reason


def test_repeated_resolution_reuses_the_nesting_cache(monkeypatch) -> None:
    case = _nested_case()
    calls = []

    def counting(child, parent):
        calls.append((child, parent))
        return nested_parent_codes(child, parent)

    monkeypatch.setattr(selection, "nested_parent_codes", counting)
    cache = case.dm._scalar_structured_layout_cache
    first = _resolve(case, nesting_cache=cache)
    first_calls = len(calls)
    second = _resolve(case, nesting_cache=cache)
    layout = _layout(case)
    assert first_calls > 0
    assert len(calls) == first_calls
    assert first.chain_group_indices == second.chain_group_indices == layout.chain_group_indices


# ── Estimability ──────────────────────────────────────────────────────────


def test_nested_estimability_matches_the_dense_centred_rank() -> None:
    case = _nested_case(n=300)
    dense = case.matrices[0].M
    # Border columns: a normal, 10 + 3N, the leaf and root attributes (in the
    # tree span) and a duplicate of the normal.  The ones column is left out:
    # its dense centred copy is rounding noise the dense reference misreads.
    columns = np.column_stack((dense[:, 1:5], dense[:, 1]))
    matrices = [DenseGroupMatrix(columns), *case.matrices[1:]]
    dm = DesignMatrix(matrices, n=case.dm.n, p=case.dm.p)
    layout = get_structured_layout(
        dm, case.groups, dominant_group_index=VARIANT, chain_group_indices=CHAIN
    )
    system = build_nested_structured_system(
        matrices, case.groups, case.weights, np.zeros(dm.n), layout=layout
    )
    xtw = np.empty(dm.p)
    xtw[layout.small_indices] = system.xtw_small
    xtw[layout.structured_indices] = system.xtw_structured
    operator = CenteredBlockOperator(
        raw=system.operator, cross=xtw, total=system.sum_w, center=xtw / system.sum_w
    )
    design = np.hstack([matrix.toarray() for matrix in matrices])
    centered = design - (case.weights @ design) / np.sum(case.weights)
    expected = decompose_factor(np.sqrt(case.weights)[:, None] * centered).coefficient_estimable()
    result = centered_operator_coefficient_estimable(operator)
    np.testing.assert_array_equal(result[layout.small_indices], expected[layout.small_indices])
    assert not np.any(result[layout.structured_indices])
    assert not np.any(expected[layout.structured_indices])
    np.testing.assert_array_equal(result[:5], [False, True, False, False, False])


# ── Standard errors of a large RandomEffect ───────────────────────────────


def test_random_effect_feature_se_uses_the_selected_diagonal() -> None:
    rng = np.random.default_rng(12)
    n_levels = 300
    codes = rng.integers(0, n_levels, size=2400)
    z = rng.normal(size=len(codes))
    y = 0.2 * z + rng.normal(scale=0.3, size=n_levels)[codes] + rng.normal(scale=0.1, size=len(z))
    X = pd.DataFrame({"z": z, "group": np.array([f"g{code}" for code in codes], dtype=object)})
    model = SuperGLM(
        family="gaussian",
        features={"z": Numeric(), "group": RandomEffect(lambda_policy=LambdaPolicy.fixed(2.0))},
        selection_penalty=0.0,
        direct_solve="structured",
    )
    model.fit_reml(X, y, max_reml_iter=2, runtime_validation="skip")
    metrics = model.metrics(X, y)
    # More levels than the compact covariance will materialize as one block.
    assert n_levels > 256
    se = metrics.feature_se("group")["se"]
    group = next(group for group in model._groups if group.name == "group")
    _, _, _, covariance, _ = metrics._active_info
    diagonal = covariance.selected_diagonal(np.arange(group.start, group.end) + 1)
    np.testing.assert_array_equal(
        se, np.sqrt(np.maximum(metrics._coefficient_dispersion * diagonal, 0.0))
    )
    assert np.all(np.isfinite(se)) and np.all(se > 0.0)


# ── Dispatch (needs the nested factor) ───────────────────────────────────


def _poisson_response(case: NestedCase) -> np.ndarray:
    rng = np.random.default_rng(4)
    make, model, variant = case.codes
    eta = (
        -0.3
        + 0.2 * case.matrices[0].M[:, 1]
        + rng.normal(scale=0.3, size=case.sizes[0])[make]
        + rng.normal(scale=0.2, size=case.sizes[1])[model]
        + rng.normal(scale=0.1, size=case.sizes[2])[variant]
    )
    return rng.poisson(np.exp(eta)).astype(float)


def _fit(case: NestedCase, groups=None, matrices=None, direct_solve="structured"):
    matrices = case.matrices if matrices is None else matrices
    groups = case.groups if groups is None else groups
    dm = DesignMatrix(matrices, n=case.dm.n, p=groups[-1].end)
    profile: dict = {}
    cache: dict = {}
    components = [
        PenaltyComponent(
            name=group.name,
            group_name=group.name,
            group_index=index,
            group_sl=group.sl,
            omega_raw=None,
            penalty_kind="identity",
        )
        for index, group in enumerate(groups)
        if isinstance(matrices[index], RandomEffectGroupMatrix)
    ]
    result, factor, operator = fit_irls_direct(
        X=dm,
        y=_poisson_response(case),
        weights=np.where(case.weights > 0.0, 1.0, 0.0),
        family=Poisson(),
        link=LogLink(),
        groups=groups,
        lambda2={component.name: 2.0 for component in components},
        max_iter=50,
        tol=1e-10,
        return_xtwx=True,
        direct_solve=direct_solve,
        reml_penalties=components,
        profile=profile,
        cache_out=cache,
        weight_semantics="prior",
    )
    return result, factor, operator, profile, cache, dm


def _dense_border_case(keep: tuple[int, ...]) -> tuple[NestedCase, list, list[GroupSlice]]:
    """The chain beside a dense border without the intercept-aliased ones column."""
    case = _nested_case()
    matrices = [DenseGroupMatrix(case.matrices[0].M[:, 1:3])]
    matrices += [case.matrices[index] for index in keep]
    groups, start = [], 0
    for index, matrix in zip((0, *keep), matrices, strict=True):
        groups.append(replace(case.groups[index], start=start, end=start + matrix.shape[1]))
        start += matrix.shape[1]
    return case, matrices, groups


def test_forced_structured_dispatches_the_nested_factor_on_a_chain() -> None:
    case, matrices, groups = _dense_border_case(CHAIN)
    _result, factor, operator, profile, cache, _dm = _fit(case, groups, matrices)
    assert isinstance(factor, ProfiledNestedSchurFactor)
    assert profile["direct_backend"] == "structured"
    assert profile["structured_chain"] == ("make", "model", "variant")
    assert profile["structured_nested_fallback_reason"] is None
    assert profile["structured_dominant_group"] == "variant"
    assert operator is cache["structured_system"].operator
    assert factor.data_operator is operator


def test_a_single_random_effect_keeps_the_scalar_factor() -> None:
    case, matrices, groups = _dense_border_case((VARIANT,))
    _result, factor, _operator, profile, _cache, _dm = _fit(case, groups, matrices)
    assert isinstance(factor, ProfiledScalarSchurFactor)
    assert profile["structured_chain"] == ("variant",)


def test_signed_operators_are_built_about_the_factors_own_means() -> None:
    case, matrices, groups = _dense_border_case(CHAIN)
    result, factor, _operator, _profile, _cache, dm = _fit(case, groups, matrices)
    correction = reml_w_correction(
        dm,
        LogLink(),
        groups,
        result,
        factor,
        {group.name: 2.0 for group in groups[1:]},
        sample_weight=np.where(case.weights > 0.0, 1.0, 0.0),
        offset_arr=np.zeros(case.dm.n),
        distribution=Poisson(),
        reml_penalties=[
            PenaltyComponent(
                name=group.name,
                group_name=group.name,
                group_index=index,
                group_sl=group.sl,
                omega_raw=None,
                penalty_kind="identity",
            )
            for index, group in enumerate(groups)
            if index
        ],
    )
    assert correction is not None
    _gradient, dH_extra = correction
    for operator in dH_extra.values():
        assert isinstance(operator, CenteredBlockOperator)
        assert operator.raw.tree is factor.data_operator.tree
        np.testing.assert_array_equal(operator.raw.leaf.mean, factor.data_operator.leaf.mean)
        np.testing.assert_array_equal(operator.center, factor.mean_x)
        assert operator.raw.leaf.deviation is not None


def test_nested_centred_data_diagonal_matches_its_materialization() -> None:
    case = _nested_case(n=200)
    layout = _layout(case)
    system = build_nested_structured_system(
        case.matrices, case.groups, case.weights, np.zeros(case.dm.n), layout=layout
    )
    xtw = np.empty(case.dm.p)
    xtw[layout.small_indices] = system.xtw_small
    xtw[layout.structured_indices] = system.xtw_structured
    centered = CenteredBlockOperator(
        raw=system.operator, cross=xtw, total=system.sum_w, center=xtw / system.sum_w
    )
    dense = materialize_compact_operator(centered)
    scale = np.abs(materialize_compact_operator(system.operator)).max()
    np.testing.assert_allclose(
        compact_operator_diagonal(centered), np.diag(dense), rtol=0, atol=64 * EPS * scale
    )
