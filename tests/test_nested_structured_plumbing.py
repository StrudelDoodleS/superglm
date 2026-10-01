"""Plumbing of the nested random-effect chain (fix D).

Chain detection, the nested layout and design products, the exact centred
leaf row pass, the nested system and penalty assembly, backend resolution,
dispatch, estimability and the RandomEffect standard-error route.  The factor
algebra itself is tested in ``test_nested_schur_factor.py``.  Section numbers
refer to ``notes/research/2026-09-26-nested-random-effect-elimination.md``.
"""

from __future__ import annotations

import tracemalloc
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.sparse as sp

import superglm.solvers._structured.selection as selection
from superglm import LambdaPolicy, Numeric, RandomEffect, SuperGLM
from superglm.distributions import Gaussian, Poisson
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    FactorSmoothGroupMatrix,
    RandomEffectGroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm.links import LogLink
from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.reml.w_derivatives import reml_w_correction
from superglm.solvers._structured.moments import _leaf_rows
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.rank import decompose_factor, decompose_gram
from superglm.solvers.structured import (
    CenteredBlockOperator,
    NestedStructuredLayout,
    ProfiledNestedSchurFactor,
    _random_effect_auto_cost_ratio,
    build_augmented_nested_factor,
    build_nested_structured_layout,
    build_nested_structured_system,
    build_penalized_nested_operator,
    centered_operator_coefficient_estimable,
    compact_operator_diagonal,
    find_nested_chain,
    get_structured_layout,
    materialize_compact_operator,
    nested_parent_codes,
    nested_prior_statistics,
    resolve_structured_backend,
    structured_design_matvec,
    structured_design_rmatvec,
)
from superglm.types import GroupSlice, PenaltyComponent

EPS = np.finfo(np.float64).eps


def build_nested_leaf_statistics(layout, matrices, weights, **kwargs):
    """One row pass of a chain, its leaf statistics alone (``moments._nested_pass``)."""
    from superglm.solvers._structured.moments import _nested_pass

    return _nested_pass(layout, matrices, weights, **kwargs)[0]


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


def _border_center(layout: NestedStructuredLayout, prior_weights=None) -> np.ndarray:
    """The border centre the row pass uses for these prior weights (unit weights by default)."""
    return nested_prior_statistics(layout, prior_weights)[0]


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
        _resolve(case, lambda2=_lambdas(variant=0.0))


# ── Layout and design products ───────────────────────────────────────────


def test_nested_layout_refuses_malformed_chains() -> None:
    case = _nested_case()
    build = build_nested_structured_layout
    with pytest.raises(ValueError, match="at least one"):
        build(case.matrices, case.groups, chain_group_indices=())
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


def test_nested_layout_border_tree_and_leaf_row_order() -> None:
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
    np.testing.assert_array_equal(layout.leaf_order, np.argsort(variant, kind="stable"))
    np.testing.assert_array_equal(
        np.diff(layout.leaf_starts), np.bincount(variant, minlength=case.sizes[2])
    )
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


def _leaf_sums(leaf: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    return np.stack([np.bincount(leaf, weights=column, minlength=size) for column in values.T], 1)


def _dense_leaf_reference(case: NestedCase, layout: NestedStructuredLayout, weights: np.ndarray):
    """Centred rows ``x - c``, leaf weights, plain leaf means of ``x - c`` and their allowance.

    The reference rounds ``x - c`` exactly as the row pass does; either mean is
    within ``(n_leaf + 4) eps`` of the exact mean relative to the absolute mass
    ``sum w |x - c|``, so the two differ by at most twice that.
    """
    Xc = _border_rows(layout) - _border_center(layout)
    leaf, K = case.codes[2], case.sizes[2]
    leaf_weight = np.bincount(leaf, weights=weights, minlength=K)
    active = leaf_weight != 0.0
    mean, error = np.zeros((K, Xc.shape[1])), np.zeros((K, Xc.shape[1]))
    n_leaf = np.bincount(leaf, minlength=K).max()
    mean[active] = _leaf_sums(leaf, weights[:, None] * Xc, K)[active] / leaf_weight[active, None]
    mass = _leaf_sums(leaf, np.abs(weights[:, None] * Xc), K)
    error[active] = (n_leaf + 4) * EPS * mass[active] / leaf_weight[active, None]
    return Xc, leaf_weight, mean, 2 * error


def test_leaf_statistics_match_the_dense_formula() -> None:
    case = _nested_case()
    layout = _layout(case)
    stats = build_nested_leaf_statistics(layout, case.matrices, case.weights)
    w, leaf, K = case.weights, case.codes[2], case.sizes[2]
    Xc, leaf_weight, mean, mean_error = _dense_leaf_reference(case, layout, w)
    centered = Xc - mean[leaf]
    within = centered.T @ (w[:, None] * centered)
    # The scatter is within (n + 4) eps of its absolute value about the exact
    # mean, plus the second-order mean term w_l e e'.
    absolute_scatter = np.abs(centered).T @ (w[:, None] * np.abs(centered))
    within_bound = 2 * (case.dm.n + 4) * EPS * absolute_scatter + 4 * (
        mean_error.T @ (leaf_weight[:, None] * mean_error)
    )
    # The absolute mass sums (n + 4)-eps-rounded squares of deviations that each
    # move with their mean, each row charged its own non-negative weight.
    assert np.all(w >= 0.0)
    moved = mean_error[leaf]
    absolute = w @ centered**2
    absolute_bound = w @ (
        (case.dm.n + 4) * EPS * centered**2 + 2 * np.abs(centered) * moved + moved**2
    )
    # The raw cross a_l (m_l + c): the mean's allowance, two roundings, and the
    # reference's own leaf sums.
    X = _border_rows(layout)
    raw_mean = np.abs(mean + _border_center(layout))
    n_leaf = np.bincount(leaf, minlength=K).max()
    cross = _leaf_sums(leaf, w[:, None] * X, K)
    cross_bound = leaf_weight[:, None] * (mean_error + 2 * EPS * raw_mean) + (
        n_leaf + 2
    ) * EPS * _leaf_sums(leaf, np.abs(w[:, None] * X), K)

    np.testing.assert_array_equal(stats.weight, leaf_weight)
    np.testing.assert_array_equal(stats.center, _border_center(layout))
    assert np.all(np.abs(stats.mean - mean) <= mean_error)
    np.testing.assert_array_equal(stats.mean[leaf_weight == 0.0], 0.0)
    assert np.all(np.abs(stats.within - within) <= within_bound)
    np.testing.assert_array_equal(stats.within, stats.within.T)
    assert np.all(np.abs(stats.absolute - absolute) <= absolute_bound)
    assert np.all(np.abs(stats.cross - cross) <= cross_bound)
    assert stats.deviation is None


def _with_dense(case: NestedCase, dense: np.ndarray) -> NestedCase:
    matrices = [DenseGroupMatrix(dense), *case.matrices[1:]]
    return replace(case, dm=DesignMatrix(matrices, n=case.dm.n, p=case.dm.p))


def test_leaf_statistics_carry_no_column_offset() -> None:
    """A column offset by 1e8 gives the same leaf means and scatter up to rounding at its spread.

    The normal column is put on a 2^-20 grid so that ``x + 1e8`` is exact:
    both designs hold the same data up to the offset.  Every column that is
    not one-hot is centred by type on its shifted prior-weighted mean (design
    §3.2), so the offset column's means round at ``|x - c|``; raw means would
    carry ``eps * 1e8`` of rounding.
    """
    dense = _nested_case().matrices[0].M.copy()
    dense[:, 1] = np.round(dense[:, 1] * 2.0**20) / 2.0**20
    case = _with_dense(_nested_case(), dense)
    offset = np.zeros(dense.shape[1])
    offset[1] = 1e8
    shifted_case = _with_dense(case, dense + offset)
    layout, shifted_layout = _layout(case), _layout(shifted_case)
    stats = build_nested_leaf_statistics(layout, case.matrices, case.weights)
    shifted = build_nested_leaf_statistics(shifted_layout, shifted_case.matrices, case.weights)
    assert shifted.center[1] != 0.0 and stats.center[1] != 0.0
    w, leaf = case.weights, case.codes[2]
    Xc, leaf_weight, mean, mean_error = _dense_leaf_reference(case, layout, w)
    # the shifted means about their own centre: x + 1e8 - c' is exact (Sterbenz),
    # so an observed leaf's mean moves by c' - 1e8 at one rounding of its size
    q, active = dense.shape[1], leaf_weight != 0.0
    moved = shifted.mean[:, :q] + (shifted.center[:q] - offset - stats.center[:q])
    error = np.abs(moved - stats.mean[:, :q])[active]
    assert np.all(error <= (2 * mean_error[:, :q] + 2 * EPS * np.abs(mean[:, :q]))[active])
    centered = np.abs(Xc - mean[leaf])
    # each scatter is within its own bound of the exact centred scatter
    within_bound = 4 * (case.dm.n + 4) * EPS * (centered.T @ (w[:, None] * centered)) + 8 * (
        mean_error.T @ (leaf_weight[:, None] * mean_error)
    )
    assert np.all(np.abs(shifted.within - stats.within) <= within_bound)


@pytest.mark.parametrize("chunk_size", [7, 8192])
@pytest.mark.parametrize("weight_scale", [1.0, 1e4])
def test_leaf_constant_columns_give_exact_means_and_zero_scatter(weight_scale, chunk_size) -> None:
    case = _nested_case(weight_scale=weight_scale)
    layout = _layout(case)
    stats = build_nested_leaf_statistics(layout, case.matrices, case.weights, chunk_size=chunk_size)
    dense = case.matrices[0].M
    reference = np.zeros((case.sizes[2], dense.shape[1]))
    reference[case.codes[2]] = dense - _border_center(layout)[: dense.shape[1]]
    active = stats.weight != 0.0
    columns = list(LEAF_CONSTANT_COLUMNS)
    # The shifted mean of a column constant within a leaf is that constant
    # less the centre, and its centred scatter row and column are exactly zero
    # (§3.4).
    np.testing.assert_array_equal(
        stats.mean[np.ix_(active, columns)], reference[active][:, columns]
    )
    np.testing.assert_array_equal(stats.within[columns, :], 0.0)
    np.testing.assert_array_equal(stats.within[:, columns], 0.0)
    signed_rows = np.random.default_rng(1).normal(size=case.dm.n) * case.weights
    signed = build_nested_leaf_statistics(
        layout,
        case.matrices,
        signed_rows,
        mean=stats.mean,
        center=stats.center,
        chunk_size=chunk_size,
    )
    np.testing.assert_array_equal(signed.deviation[:, columns], 0.0)
    np.testing.assert_array_equal(signed.within[columns, :], 0.0)


def test_leaf_means_are_shifted_about_a_weighted_row() -> None:
    """A column constant on a leaf's weighted rows keeps an exact mean and zero scatter.

    Each leaf's first row gets zero weight and a different value in the leaf
    attribute: the mean of the weighted rows is still that attribute exactly,
    which a shift about the zero-weight first row would round.
    """
    case = _nested_case()
    variant = case.codes[2]
    first = np.unique(variant, return_index=True)[1]
    weights = case.weights.copy()
    weights[first] = 0.0
    dense = case.matrices[0].M.copy()
    attribute = dense[:, 3].copy()
    dense[first, 3] += 0.1
    moved = _with_dense(case, dense)
    layout = _layout(moved)
    stats = build_nested_leaf_statistics(layout, moved.matrices, weights)
    active = stats.weight != 0.0
    expected = np.zeros(case.sizes[2])
    expected[variant] = attribute - _border_center(layout)[3]
    np.testing.assert_array_equal(stats.mean[active, 3], expected[active])
    np.testing.assert_array_equal(stats.within[3], 0.0)


@pytest.mark.parametrize("chunk_size", [1, 7, 64])
def test_leaves_cut_by_chunk_edges_give_the_unsplit_statistics(chunk_size) -> None:
    """Pieces of a cut leaf combine to the leaf's statistics (Chan, Golub & LeVeque).

    Compared with the one-chunk pass on non-negative weights and on weights
    with rounding-negative rows (all in zero-weight leaves, so every leaf is
    one-signed and every row is charged ``max |w|``), and for a signed pass
    about the data means.  Bounds: every centre either pass subtracts (a
    leaf row, a piece mean, the heaviest piece's mean) lies in the hull of the
    leaf's rows, so each centred entry is at most ``M_l = 2 D_l + |m_l|`` with
    ``D_l`` the leaf's largest deviation from its mean ``m_l`` (``|m_l|`` for
    the roundings of the centres themselves).  Each pass sums at most ``n``
    row terms, ``P`` piece terms and a few more, each within 4 eps of its
    exact value, and the pairwise update's dropped residual ``sum a (x -
    m_p)`` is the piece mean's error times ``W_p``: each statistic is then
    within ``gamma = (6 n + 2 P + 32) eps`` of the exact one relative to ``M_l``
    (mean), ``A_l M_li M_lj`` (scatter, ``A_l = sum |a|``) or ``E_l M_lj^2``
    (absolute mass, ``E_l = sum e``), and the two passes differ by twice that.
    """
    case = _nested_case()
    layout = _layout(case)
    X = _border_rows(layout) - _border_center(layout)
    leaf, K, n = case.codes[2], case.sizes[2], case.dm.n
    pieces = 2 * -(-n // chunk_size)  # at most two per chunk
    gamma = (6 * n + 2 * pieces + 32) * EPS
    rounding = case.weights.copy()
    rounding[np.flatnonzero(rounding == 0.0)[:3]] = -1e-16 * np.max(rounding)
    data = build_nested_leaf_statistics(layout, case.matrices, case.weights, chunk_size=n)
    signed_rows = np.random.default_rng(9).normal(size=n) * case.weights
    passes = [(case.weights, None), (rounding, None), (signed_rows, data.mean)]
    for weights, mean in passes:
        # a signed pass reads its factor's means and centre together
        center = None if mean is None else data.center
        whole = build_nested_leaf_statistics(
            layout, case.matrices, weights, mean=mean, center=center, chunk_size=n
        )
        cut = build_nested_leaf_statistics(
            layout, case.matrices, weights, mean=mean, center=center, chunk_size=chunk_size
        )
        deviation = np.zeros((K, X.shape[1]))
        np.maximum.at(deviation, leaf, np.abs(X - whole.mean[leaf]))
        M = 2.0 * deviation + np.abs(whole.mean)
        error = np.max(np.abs(weights)) * (weights != 0.0) if np.any(weights < 0.0) else weights
        A = np.bincount(leaf, weights=np.abs(weights), minlength=K)
        E = np.bincount(leaf, weights=error, minlength=K)
        np.testing.assert_array_equal(cut.weight, whole.weight)
        assert np.all(np.abs(cut.mean - whole.mean) <= 2 * gamma * M)
        assert np.all(np.abs(cut.within - whole.within) <= 2 * gamma * (M.T @ (A[:, None] * M)))
        assert np.all(np.abs(cut.absolute - whole.absolute) <= 2 * gamma * (E @ M**2))
        if mean is not None:
            assert np.all(np.abs(cut.deviation - whole.deviation) <= 2 * gamma * A[:, None] * M)


def test_the_row_pass_memory_is_bounded_by_the_chunk_not_the_leaf() -> None:
    """Four leaves of 16 chunks each: the traced peak stays within the chunk budget.

    The pass holds the weights in leaf order (8 bytes a row), after one
    ``n``-row temporary for the leaves' sums of ``|w|``, and at most five
    ``chunk x q`` float64 temporaries at a
    time (the chunk's centred rows, its differences or repeated means and
    weighted rows, and the previous chunk's until they are rebound); the
    budget allows eight for BLAS and einsum workspace, plus the ``(K + q) q``
    statistics.  Materializing one whole leaf needs more than all of that.
    """
    n, q, leaves, chunk = 2**16, 16, 4, 1024
    rng = np.random.default_rng(0)
    codes = np.repeat(np.arange(leaves), n // leaves)
    matrices = [
        DenseGroupMatrix(rng.normal(size=(n, q))),
        RandomEffectGroupMatrix(codes % 2, 2),
        RandomEffectGroupMatrix(codes, leaves),
    ]
    groups, start = [], 0
    for name, matrix in zip(("x", "parent", "leaf"), matrices, strict=True):
        end = start + matrix.shape[1]
        groups.append(GroupSlice(name=name, start=start, end=end, penalized=name != "x"))
        start = end
    dm = DesignMatrix(matrices, n=n, p=start)
    layout = get_structured_layout(dm, groups, dominant_group_index=2, chain_group_indices=(1, 2))
    weights = rng.exponential(size=n)
    build_nested_leaf_statistics(layout, matrices, weights, chunk_size=chunk)  # the cached centre
    tracemalloc.start()
    try:
        build_nested_leaf_statistics(layout, matrices, weights, chunk_size=chunk)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    budget = 9 * n + 8 * (chunk * q * 8) + 8 * (leaves + q) * q * 8
    assert budget < (n // leaves) * q * 8
    assert peak <= budget


def test_signed_pass_deviation_matches_exact_rational_sums() -> None:
    case = _nested_case(n=240)
    layout = _layout(case)
    data = build_nested_leaf_statistics(layout, case.matrices, case.weights)
    a = np.random.default_rng(9).normal(size=case.dm.n) * case.weights
    signed = build_nested_leaf_statistics(
        layout, case.matrices, a, mean=data.mean, center=data.center
    )
    # A signed pass only reads the frozen means it is given, so it shares them:
    # each REML weight-derivative system used to keep its own (K, q) copy,
    # about 30 MiB of big_K25000's peak (T7).  A writable mean is still copied.
    assert signed.mean is data.mean
    writable = np.array(data.mean)
    copied = build_nested_leaf_statistics(
        layout, case.matrices, a, mean=writable, center=data.center
    )
    assert copied.mean is not writable
    np.testing.assert_array_equal(copied.mean, data.mean)
    X, leaf, c = _border_rows(layout), case.codes[2], _border_center(layout)
    exact = [[Fraction(0)] * X.shape[1] for _ in range(case.sizes[2])]
    for row in range(case.dm.n):
        cell = leaf[row]
        for column in range(X.shape[1]):
            mean = Fraction(float(c[column])) + Fraction(float(data.mean[cell, column]))
            difference = Fraction(float(X[row, column])) - mean
            exact[cell][column] += Fraction(float(a[row])) * difference
    exact = np.array([[float(value) for value in row] for row in exact])
    # x - c rounds at |x - c|, the subtraction of the mean at |x - c - m|
    centered = (np.abs(X - c - data.mean[leaf]) + np.abs(X - c)) * np.abs(a)[:, None]
    absolute = _leaf_sums(leaf, centered, case.sizes[2])
    n_leaf = np.bincount(leaf).max()
    assert np.all(np.abs(signed.deviation - exact) <= (n_leaf + 4) * EPS * absolute)
    # without a caller's error scale every row is charged its own |a| (design
    # §3.3: the scale is the rows' builder's, never inferred from their signs)
    squares = np.abs(a) @ (X - c - data.mean[leaf]) ** 2
    assert np.all(np.abs(signed.absolute - squares) <= (case.dm.n + 4) * EPS * squares)


def _indicator_case(n: int = 3000, leaves: int = 400) -> tuple[DesignMatrix, list[GroupSlice]]:
    """A lone random effect of 400 leaves beside indicator blocks.

    The border: two numeric columns, a random effect the leaves nest in (one
    level per leaf, 40 levels), a crossed random effect and a categorical with
    base-level rows whose levels cluster by leaf, a random effect whose level
    0 covers most rows and a categorical whose three levels fill every leaf.
    The random effects go through their cells and the categoricals through
    dense rows, by type, whatever their fill.
    """
    rng = np.random.default_rng(20260929)
    leaf = rng.integers(0, leaves, n)
    dominant = np.where(rng.random(n) < 0.6, 0, 1 + leaf % 19)
    matrices = [
        DenseGroupMatrix(np.column_stack([rng.normal(size=n), 5.0 + rng.normal(size=n)])),
        RandomEffectGroupMatrix(rng.integers(0, 40, leaves)[leaf], 40),
        RandomEffectGroupMatrix((7 * leaf + rng.integers(0, 3, n)) % 200, 200),
        CategoricalGroupMatrix(
            np.where(rng.random(n) < 0.3, -1, (3 * leaf + rng.integers(0, 2, n)) % 30), 30
        ),
        RandomEffectGroupMatrix(dominant, 20),
        CategoricalGroupMatrix(rng.integers(-1, 3, n), 3),
        RandomEffectGroupMatrix(leaf, leaves),
    ]
    groups, start = [], 0
    for index, matrix in enumerate(matrices):
        end = start + matrix.shape[1]
        groups.append(GroupSlice(name=f"g{index}", start=start, end=end, penalized=index > 0))
        start = end
    return DesignMatrix(matrices, n=n, p=start), groups


@pytest.mark.parametrize("chunk_size", [7, 64, 3000])
def test_sparse_indicator_blocks_give_the_dense_pass_statistics(chunk_size) -> None:
    """Random-effect border blocks skip their exact zeros and match the dense pass.

    The sparse route (every random-effect block, by type, one of them with a
    level on most rows) forms each row's centred entries only on its leaf's
    levels; every other entry is an exact zero of the dense pass.  Both passes
    therefore sum the same nonzero terms, in another order, so each is within
    the cut-leaf test's ``gamma`` of the exact statistic and they differ by
    twice that, on non-negative weights, on weights with rounding-negative
    rows and for a signed pass about the data means.  The nested-in random
    effect is constant within leaves: its means are exact and its scatter rows
    exactly zero in both.
    """
    dm, groups = _indicator_case()
    layout = get_structured_layout(dm, groups, dominant_group_index=6, chain_group_indices=(6,))
    assert [cells.block for cells in layout.sparse_indicators] == [1, 2, 4]
    np.testing.assert_array_equal(_border_center(layout)[2:], 0.0)
    dense = replace(layout)
    dense.__dict__["sparse_indicators"] = ()
    n, matrices = dm.n, list(dm.group_matrices)
    leaf, K = matrices[6].codes, matrices[6].n_levels
    X = np.hstack([matrix.toarray() for matrix in layout.small_matrices]) - _border_center(layout)
    rng = np.random.default_rng(3)
    weights = np.exp(rng.normal(size=n))
    weights[rng.random(n) < 0.05] = 0.0
    weights[np.isin(leaf, (3, 7))] = 0.0
    rounding = weights.copy()
    rounding[np.flatnonzero(rounding == 0.0)[:3]] = -1e-16 * np.max(rounding)
    signed_rows = rng.normal(size=n) * weights
    data = build_nested_leaf_statistics(layout, matrices, weights, chunk_size=chunk_size)
    pieces = 2 * -(-n // chunk_size)
    gamma = (6 * n + 2 * pieces + 32) * EPS
    nested = np.arange(2, 42)  # the nested-in random effect's border columns
    passes = [(weights, None), (rounding, None), (signed_rows, data.mean)]
    for rows, mean in passes:
        center = None if mean is None else data.center
        sparse = build_nested_leaf_statistics(
            layout, matrices, rows, mean=mean, center=center, chunk_size=chunk_size
        )
        reference = build_nested_leaf_statistics(
            dense, matrices, rows, mean=mean, center=center, chunk_size=chunk_size
        )
        deviation = np.zeros((K, X.shape[1]))
        np.maximum.at(deviation, leaf, np.abs(X - reference.mean[leaf]))
        M = 2.0 * deviation + np.abs(reference.mean)
        error = np.max(np.abs(rows)) * (rows != 0.0) if np.any(rows < 0.0) else rows
        A = np.bincount(leaf, weights=np.abs(rows), minlength=K)
        E = np.bincount(leaf, weights=error, minlength=K)
        np.testing.assert_array_equal(sparse.weight, reference.weight)
        assert np.all(np.abs(sparse.mean - reference.mean) <= 2 * gamma * M)
        assert np.all(
            np.abs(sparse.within - reference.within) <= 2 * gamma * (M.T @ (A[:, None] * M))
        )
        assert np.all(np.abs(sparse.absolute - reference.absolute) <= 2 * gamma * (E @ M**2))
        np.testing.assert_array_equal(sparse.within, sparse.within.T)
        np.testing.assert_array_equal(sparse.within[nested], 0.0)
        np.testing.assert_array_equal(reference.within[nested], 0.0)
        if mean is None:
            active = sparse.weight != 0.0
            expected = matrices[1].toarray()[np.unique(leaf, return_index=True)[1]]
            np.testing.assert_array_equal(sparse.mean[np.ix_(active, nested)], expected[active])
        else:
            bound = 2 * gamma * A[:, None] * M
            assert np.all(np.abs(sparse.deviation - reference.deviation) <= bound)
            np.testing.assert_array_equal(sparse.deviation[:, nested], 0.0)


def test_the_sparse_indicator_pass_never_materializes_its_blocks() -> None:
    """A 512-level crossed block with two levels per leaf: the pass stays below one dense chunk.

    The dense route materializes ``chunk x q`` border rows.  The sparse route
    holds the ``(K, q)`` means (in its own column order and the layout's), a
    few ``q x q`` products (the scatter, its compensation, the chunk's product
    and indicator block, the compensated addition's temporaries) and its
    row-level arrays with two entries a row.  The budget allows three of the
    first, eight of the second and 32 floats a chunk row; one dense chunk is
    more than all of that.
    """
    n, leaves, levels, chunk = 2**16, 256, 512, 2**15
    rng = np.random.default_rng(4)
    leaf = np.repeat(np.arange(leaves), n // leaves)
    crossed = (4 * leaf + rng.integers(0, 2, n)) % levels
    matrices = [
        DenseGroupMatrix(rng.normal(size=(n, 1))),
        RandomEffectGroupMatrix(crossed, levels),
        RandomEffectGroupMatrix(leaf, leaves),
    ]
    groups = [
        GroupSlice(name="x", start=0, end=1, penalized=False),
        GroupSlice(name="crossed", start=1, end=1 + levels),
        GroupSlice(name="leaf", start=1 + levels, end=1 + levels + leaves),
    ]
    dm = DesignMatrix(matrices, n=n, p=1 + levels + leaves)
    layout = get_structured_layout(dm, groups, dominant_group_index=2, chain_group_indices=(2,))
    assert [cells.block for cells in layout.sparse_indicators] == [1]
    weights = rng.exponential(size=n)
    build_nested_leaf_statistics(layout, matrices, weights, chunk_size=chunk)  # the cached cells
    tracemalloc.start()
    try:
        build_nested_leaf_statistics(layout, matrices, weights, chunk_size=chunk)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    q = 1 + levels
    budget = 8 * (3 * leaves * q + 8 * q * q + 32 * chunk)
    assert budget < 8 * chunk * q
    assert peak <= budget


# ── The compiled row pass ────────────────────────────────────────────────


def _every_border_kind_case(seed: int = 5, n: int = 900):
    """``_nested_case``'s border plus a lossless-support spline and a plain sparse
    block (the pass reads the latter through ``row_subset``), a lone leaf chain."""
    case = _nested_case(seed=seed, n=n)
    rng = np.random.default_rng(seed)
    matrices = case.matrices
    extra = [
        SupportCompressedSSPGroupMatrix(
            rng.normal(size=(9, 4)), rng.normal(size=(4, 3)), rng.integers(0, 9, n)
        ),
        SparseGroupMatrix(sp.random(n, 3, density=0.3, random_state=seed, format="csr")),
    ]
    matrices = [*matrices[:VARIANT], *extra, matrices[VARIANT]]
    groups, start = [], 0
    for index, matrix in enumerate(matrices):
        end = start + matrix.shape[1]
        groups.append(GroupSlice(name=f"g{index}", start=start, end=end, penalized=index > 0))
        start = end
    dm = DesignMatrix(matrices, n=n, p=start)
    leaf = len(matrices) - 1
    layout = get_structured_layout(
        dm, groups, dominant_group_index=leaf, chain_group_indices=(leaf,)
    )
    return layout, matrices, case.weights


def test_the_row_pass_forms_each_border_matrix_own_rows() -> None:
    """The pass's border rows in leaf order are each block's own ``toarray`` rows less ``c``.

    A spline table row is ``(B_unique R_inv)[bin]``, a one-hot row copies a 1
    or a 0, a dense row is gathered and a plain sparse block goes through its
    ``row_subset``: each the value ``toarray`` holds.  A sparse spline forms
    ``B_r R_inv`` from ``B``'s stored entries in order, as scipy's CSR product
    does, so its entries are the same sums; they are held to the ``(k + 1)
    eps |B| |R_inv|`` bound of either evaluation (Higham 2002, eq. 3.4).
    """
    layout, matrices, _ = _every_border_kind_case()
    # the random effects (the leaf's parents and the crossed term) are sparse
    # indicators, not written; every other block is: numeric 5, spline 3,
    # binned 3, category 3, lossless support 3 and plain sparse 3 columns
    dense = ~layout.indicator_columns
    assert dense.sum() == 20
    n, center = matrices[0].shape[0], _border_center(layout)[dense]
    rows = np.empty((n, len(center)))
    for lo in range(0, n, 256):
        hi = min(lo + 256, n)
        _leaf_rows(layout, rows[lo:hi], lo, hi, center)
    expected = (_border_rows(layout)[:, dense] - center)[layout.leaf_order]
    spline, column = np.zeros(len(center), dtype=bool), 0
    for block, matrix in enumerate(layout.small_matrices):
        if block in {cells.block for cells in layout.sparse_indicators}:
            continue
        if type(matrix) is SparseSSPGroupMatrix:
            spline[column : column + matrix.shape[1]] = True
            magnitude = np.abs(matrix.B.toarray()) @ np.abs(matrix.R_inv)
            bound = (matrix.B.shape[1] + 1) * EPS * magnitude[layout.leaf_order]
            error = np.abs(rows - expected)[:, column : column + matrix.shape[1]]
            assert np.all(error <= bound)
        column += matrix.shape[1]
    assert spline.any()
    np.testing.assert_array_equal(rows[:, ~spline], expected[:, ~spline])


def test_the_row_pass_reads_its_own_kernels_never_row_subsets(monkeypatch) -> None:
    """Splines, categorical and random-effect codes and dense blocks are formed from
    their own arrays in leaf order: no chunk copies rows through ``row_subset``."""
    case = _nested_case()
    layout = _layout(case)

    def refused(self, idx):
        raise AssertionError(f"{type(self).__name__}.row_subset in the row pass")

    for kind in (
        DenseGroupMatrix,
        SparseSSPGroupMatrix,
        DiscretizedSSPGroupMatrix,
        CategoricalGroupMatrix,
        RandomEffectGroupMatrix,
    ):
        monkeypatch.setattr(kind, "row_subset", refused)
    stats = build_nested_leaf_statistics(layout, case.matrices, case.weights, chunk_size=97)
    assert np.all(np.isfinite(stats.within))


def test_signed_pass_scatter_matches_exact_rational_sums() -> None:
    """``sum_r a_r d_r d_r'`` about the data means for a vector of both signs, ``d = x - c - m``.

    The pass forms it as ``P'P - N'N`` from ``sqrt|a| d`` (``_root_scatter``):
    within ``(m + 5) eps sum |a| |d_i| |d_j|`` of the scatter of its rounded
    ``d`` (Higham 2002, Lemma 3.1 and eq. 3.4), whose entries are within
    ``delta = 2 eps (|x - c| + |d|)`` of the exact ``d``.  Rows of negative
    weight subtract: charging them with the wrong sign moves an entry by
    ``2 sum_{a < 0} |a| d_i d_j``, far outside the bound.
    """
    case = _nested_case(n=240)
    layout = _layout(case)
    data = build_nested_leaf_statistics(layout, case.matrices, case.weights)
    a = np.random.default_rng(9).normal(size=case.dm.n) * case.weights
    assert np.any(a < 0.0) and np.any(a > 0.0)
    signed = build_nested_leaf_statistics(
        layout, case.matrices, a, mean=data.mean, center=data.center, chunk_size=64
    )
    X, leaf, c = _border_rows(layout), case.codes[2], _border_center(layout)
    q = X.shape[1]
    difference = [
        [
            Fraction(float(X[r, j]))
            - Fraction(float(c[j]))
            - Fraction(float(data.mean[leaf[r], j]))
            for j in range(q)
        ]
        for r in range(case.dm.n)
    ]
    exact = np.array(
        [
            [
                float(
                    sum(
                        Fraction(float(a[r])) * difference[r][i] * difference[r][j]
                        for r in range(case.dm.n)
                    )
                )
                for j in range(q)
            ]
            for i in range(q)
        ]
    )
    d = np.abs(X - c - data.mean[leaf])
    delta = 2 * EPS * (np.abs(X - c) + d)
    weight = np.abs(a)[:, None]
    rounded = (d + delta).T @ (weight * (d + delta))
    moved = d.T @ (weight * delta) + delta.T @ (weight * d) + delta.T @ (weight * delta)
    chunks = -(-case.dm.n // 64)
    bound = (case.dm.n + 5 + 2 * chunks) * EPS * rounded + moved
    assert np.all(np.abs(signed.within - exact) <= bound)
    np.testing.assert_array_equal(signed.within, signed.within.T)


def _rebuilt(matrices: list, spline_index: int, R_inv: np.ndarray) -> list:
    """The border spline at a new ``R_inv`` on the very same bins, every other block
    the very same object: what ``rebuild_design_matrix_with_lambdas`` hands over."""
    spline = matrices[spline_index]
    moved = DiscretizedSSPGroupMatrix(spline.B_unique, R_inv, spline.bin_idx)
    return [*matrices[:spline_index], moved, *matrices[spline_index + 1 :]]


def test_a_lambda_rebuild_reuses_the_leaf_ordered_codes_and_rebuilds_the_tables() -> None:
    """The leaf-ordered codes live in the lineage's nesting cache, keyed by the code
    arrays themselves: a rebuild that keeps them reuses them, a spline table is
    rebuilt for its new ``R_inv``, and a new code array is a new entry.  Each
    layout's statistics are those of the same design with a fresh cache."""
    case = _nested_case()
    layout = _layout(case)
    base = build_nested_leaf_statistics(layout, case.matrices, case.weights, chunk_size=97)
    binned = NAMES.index("binned")
    R_inv = np.random.default_rng(4).normal(size=(3, 3))

    def fresh(matrices):
        dm = DesignMatrix(matrices, n=case.dm.n, p=case.dm.p)
        return get_structured_layout(
            dm, case.groups, dominant_group_index=CHAIN[-1], chain_group_indices=CHAIN
        )

    for matrices, shared in (
        (_rebuilt(case.matrices, binned, R_inv), True),
        (
            [
                *case.matrices[: NAMES.index("category")],
                CategoricalGroupMatrix(np.roll(case.matrices[NAMES.index("category")].codes, 1), 3),
                *case.matrices[NAMES.index("category") + 1 :],
            ],
            False,
        ),
    ):
        dm = DesignMatrix(matrices, n=case.dm.n, p=case.dm.p)
        selection.carry_nesting_cache(case.dm._structured_layout_cache, dm._structured_layout_cache)
        rebuilt = get_structured_layout(
            dm, case.groups, dominant_group_index=CHAIN[-1], chain_group_indices=CHAIN
        )
        assert rebuilt is not layout
        assert (rebuilt.leaf_rows.codes is layout.leaf_rows.codes) is shared
        stats = build_nested_leaf_statistics(rebuilt, matrices, case.weights, chunk_size=97)
        reference = build_nested_leaf_statistics(
            fresh(matrices), matrices, case.weights, chunk_size=97
        )
        for name in ("mean", "within", "absolute"):
            np.testing.assert_array_equal(getattr(stats, name), getattr(reference, name))
        assert not np.array_equal(stats.within, base.within)


def test_a_border_of_random_effects_alone_makes_no_blas_call(monkeypatch) -> None:
    """A chain of one beside random-effect blocks alone has no dense border column:
    the pass forms no dense scatter and hands BLAS no empty operand (``syrk``
    rejects a zero leading dimension, and a reference BLAS stops the process
    there), and its statistics are the dense pass's.  Both sum the same nonzero terms
    in another order, so they differ by twice the cut-leaf test's ``gamma``
    bound (``test_sparse_indicator_blocks_give_the_dense_pass_statistics``).
    """
    case = _nested_case()
    keep = (MAKE, MODEL, CROSSED, VARIANT)
    matrices = [case.matrices[index] for index in keep]
    groups, start = [], 0
    for index, matrix in zip(keep, matrices, strict=True):
        groups.append(GroupSlice(name=NAMES[index], start=start, end=start + matrix.shape[1]))
        start += matrix.shape[1]
    dm = DesignMatrix(matrices, n=case.dm.n, p=start)
    layout = get_structured_layout(dm, groups, dominant_group_index=3, chain_group_indices=(3,))
    assert layout.indicator_columns.all()
    dense = replace(layout)
    dense.__dict__["sparse_indicators"] = ()
    shapes, blas = [], scipy.linalg.get_blas_funcs

    def recorded(names, arrays=(), *args, **kwargs):
        function = blas(names, arrays, *args, **kwargs)

        def call(alpha, a, *rest, **options):
            shapes.append(a.shape)
            return function(alpha, a, *rest, **options)

        return call

    monkeypatch.setattr(scipy.linalg, "get_blas_funcs", recorded)
    sparse = build_nested_leaf_statistics(layout, matrices, case.weights, chunk_size=97)
    assert not any(shape[0] == 0 for shape in shapes)
    monkeypatch.undo()
    reference = build_nested_leaf_statistics(dense, matrices, case.weights, chunk_size=97)
    n, leaf, K = case.dm.n, case.codes[2], layout.tree.sizes[-1]
    X = _border_rows(layout)
    gamma = (6 * n + 2 * 2 * -(-n // 97) + 32) * EPS
    deviation = np.zeros((K, X.shape[1]))
    np.maximum.at(deviation, leaf, np.abs(X - reference.mean[leaf]))
    M = 2.0 * deviation + np.abs(reference.mean)
    A = np.bincount(leaf, weights=case.weights, minlength=K)
    np.testing.assert_array_equal(sparse.weight, reference.weight)
    assert np.all(np.abs(sparse.mean - reference.mean) <= 2 * gamma * M)
    assert np.all(np.abs(sparse.within - reference.within) <= 2 * gamma * (M.T @ (A[:, None] * M)))
    np.testing.assert_array_equal(sparse.within, sparse.within.T)
    # the factor reads the random-effect columns' type from the statistics, and the
    # augmented operator's intercept column is not one of them
    system = build_nested_structured_system(
        matrices, groups, case.weights, case.weights, layout=layout
    )
    assert sparse.indicator.all() and system.operator.leaf.indicator.all()
    np.testing.assert_array_equal(
        system.operator.augmented().leaf.indicator, np.r_[False, np.ones(len(X[0]), dtype=bool)]
    )


def test_a_new_layout_reads_the_current_sparse_spline_values() -> None:
    """A sparse spline's leaf-ordered rows belong to its layout, not to the lineage.

    ``SparseSSPGroupMatrix`` reads its current buffers in every operation.  A
    design built on the same matrix after its values were replaced (a lambda
    rebuild passes the matrix through unchanged) gets a layout whose pass reads
    the new values: its statistics equal those of a fresh lineage bit for bit,
    and differ from the old values' statistics.
    """
    case = _nested_case()
    old = build_nested_leaf_statistics(_layout(case), case.matrices, case.weights, chunk_size=97)
    spline = case.matrices[NAMES.index("spline")]
    spline._data = 2.0 * spline._data

    def layout_of(carry: bool) -> NestedStructuredLayout:
        dm = DesignMatrix(case.matrices, n=case.dm.n, p=case.dm.p)
        if carry:
            selection.carry_nesting_cache(
                case.dm._structured_layout_cache, dm._structured_layout_cache
            )
        return get_structured_layout(
            dm, case.groups, dominant_group_index=CHAIN[-1], chain_group_indices=CHAIN
        )

    rebuilt, fresh = (
        build_nested_leaf_statistics(layout_of(carry), case.matrices, case.weights, chunk_size=97)
        for carry in (True, False)
    )
    for name in ("mean", "within", "absolute"):
        np.testing.assert_array_equal(getattr(rebuilt, name), getattr(fresh, name))
    assert not np.array_equal(rebuilt.within, old.within)


@pytest.mark.parametrize("ratio", [1e-15, 1e-16])
def test_a_mostly_zero_numeric_column_stays_estimable_about_its_prior_weighted_mean(
    ratio,
) -> None:
    """The border centre is the shifted prior-weighted mean (one-engine design §3.2).

    Beside the constant column that aliases the intercept, a column that is 1
    on two rows and 0 elsewhere is carried by genuine prior weights ``ratio
    max|w|``.  The Jacobi-scaled Hessian has one null direction, the alias,
    with no component on that column, so the coefficient is estimable.
    Centred on its prior-weighted mean ``2 ratio max|w| / sum w``, its zeros
    move by that much, far below its weighted spread, and the factor agrees.
    Centred on its unweighted mean they would become ``-2 / n``, a combination
    with the intercept, and the scaled rank decision would take the column
    into the alias's null direction (at 1e-16 also losing a rank): a NaN
    standard error on an estimable coefficient.
    """
    case = _nested_case()
    dense = case.matrices[0].M.copy()
    rows = np.flatnonzero(case.weights)[[10, 200]]
    dense[:, 1] = 0.0
    dense[rows, 1] = 1.0
    matrices = [DenseGroupMatrix(dense), *case.matrices[1:]]
    weights = case.weights.copy()
    weights[rows] = ratio * np.max(weights)
    layout, _, factor = _augmented_factor(
        matrices, case.groups, weights, _lambdas(), _components(case), CHAIN, prior_weights=weights
    )
    # the first weighted row is 0, so the shifted mean is the weighted sum over the total
    center = _border_center(layout, weights)[1]
    expected = 2.0 * weights[rows[0]] / np.sum(weights)
    assert abs(center - expected) <= (len(weights) + 3) * EPS * expected
    p, border = factor.shape[0], factor.small_indices
    H = factor.operator.matvec(np.eye(p))
    scale = 1.0 / np.sqrt(np.diag(H))
    eigenvalues, vectors = np.linalg.eigh(scale[:, None] * (0.5 * (H + H.T)) * scale[None, :])
    null = vectors[:, eigenvalues <= 1e-10 * eigenvalues[-1]]
    assert null.shape[1] == 1 and factor.rank == p - 1
    # the intercept and the constant column carry the alias, the rare column none of
    # it: eigh is backward stable to p eps ||H_s||, so by Davis-Kahan the computed
    # null vector is within p eps lambda_max / lambda_2 of the exact one
    bound = p * EPS * eigenvalues[-1] / eigenvalues[1]
    expected = [2**-0.5, 2**-0.5, 0.0]
    assert np.all(np.abs(np.abs(null[border[:3], 0]) - expected) <= bound)
    assert factor.coefficient_estimable()[border][:3].tolist() == [False, False, True]


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
    assert np.all(np.abs(system.xtwz_small - X.T @ Wz) <= bound * (np.abs(X).T @ np.abs(Wz)))
    # A = within + M' diag(w) M from the rounded raw means M = m + delta: the
    # within bound, sum_l w_l (2 |m| |delta| + 2 delta^2), and both sums' rounding.
    Xc, leaf_weight, mean, mean_error = _dense_leaf_reference(case, layout, w)
    centered = np.abs(Xc - mean[case.codes[2]])
    raw = np.abs(mean + _border_center(layout))
    delta = mean_error + 2 * EPS * raw
    W_l = leaf_weight[:, None]
    A_bound = (
        2 * (case.dm.n + 4) * EPS * (centered.T @ (w[:, None] * centered))
        + raw.T @ (W_l * delta)
        + delta.T @ (W_l * raw)
        + 2 * delta.T @ (W_l * delta)
        + (case.sizes[2] + 2) * EPS * (raw.T @ (W_l * raw))
        + bound * (np.abs(X).T @ (np.abs(w)[:, None] * np.abs(X)))
    )
    assert np.all(np.abs(system.operator.A - X.T @ (w[:, None] * X)) <= A_bound)
    assert system.sum_w == pytest.approx(np.sum(w), rel=bound)
    assert system.chain_group_names == ("make", "model", "variant")
    assert system.dominant_group_name == "variant"
    assert system.operator.tree is layout.tree
    assert system.operator.leaf.deviation is None


@pytest.mark.parametrize("source", ["reml_penalties", "override"])
def test_penalized_nested_operator_places_every_penalty_source(source) -> None:
    case = _nested_case()
    layout = _layout(case)
    system = build_nested_structured_system(
        case.matrices, case.groups, case.weights, np.zeros(case.dm.n), layout=layout
    )
    lambda2: dict[str, float] = _lambdas()
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
    with pytest.raises(ValueError, match="reml_penalties"):
        build_penalized_nested_operator(system, case.matrices, case.groups, lambda2)


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


# The shapes that anchor the auto rule in selection.py: rows, border width
# without the intercept, chain sizes coarsest to finest.  Every leaf is priced
# as its chain, so pg17 C, which the retired scalar factor won on time (2026-09-27),
# takes its chain (2026-09-28).
@pytest.mark.parametrize(
    ("n", "q", "sizes"),
    [
        pytest.param(3_885, 104, (51, 407), id="pg17-C-exact-5k-rows"),
        pytest.param(77_014, 104, (87, 942), id="pg17-C"),
        pytest.param(15_794, 35, (91, 1_332), id="dvsa-C-20k-rows"),
        pytest.param(15_794, 35, (91, 1_332, 2_404), id="dvsa-D-20k-rows"),
        pytest.param(157_593, 36, (223, 3_749, 6_038), id="dvsa-D-200k-rows"),
    ],
)
def test_auto_prices_a_random_effect_leaf_as_its_chain(n, q, sizes):
    matrices, groups, p = _auto_case(n, q, sizes)
    decision = resolve_structured_backend(
        matrices, groups, direct_solve="auto", coefficient_width=p
    )
    chain = tuple(range(1, len(sizes) + 1))
    assert decision.use_structured
    assert decision.chain_group_indices == chain
    assert decision.nested_fallback_reason is None
    assert decision.auto_cost_ratio == _random_effect_auto_cost_ratio(n, p, sizes)
    forced = resolve_structured_backend(
        matrices, groups, direct_solve="structured", coefficient_width=p
    )
    assert forced.chain_group_indices == chain


# Lone levels priced as a chain of one, within 0.75: the #343 stand-in at K=225
# beside 67k rows, which the retired scalar factor also took, and the small-n corner at
# K=271, which the scalar's 0.05 sent to gram, are structured; K=105 beside 67k
# rows pays too many row passes and stays on gram, as it did.
@pytest.mark.parametrize(
    ("n", "q", "size", "structured"),
    [
        pytest.param(67_000, 48, 225, True, id="s343-K225"),
        pytest.param(67_000, 47, 105, False, id="s343-K105"),
        pytest.param(2_000, 99, 271, True, id="small-n-K271"),
    ],
)
def test_auto_prices_a_lone_random_effect_as_a_chain_of_one(n, q, size, structured):
    matrices, groups, p = _auto_case(n, q, (size,))
    decision = resolve_structured_backend(
        matrices, groups, direct_solve="auto", coefficient_width=p
    )
    assert decision.use_structured is structured
    assert decision.chain_group_indices == (1,)
    assert decision.auto_cost_ratio == _random_effect_auto_cost_ratio(n, p, (size,))


def test_the_chain_is_decided_by_the_terms_never_by_the_rows():
    """One-engine design §3.3, §6: signed observed rows keep the chain, so the
    decision reads no family, link, weight or row value; the family table that
    declined unaudited pairs is gone, and so are the row weights the old
    zero-weight-level checks read (``resolve_structured_backend`` takes
    neither)."""
    case = _nested_case()
    decision = _resolve(case)
    assert decision.use_structured
    assert decision.chain_group_indices == CHAIN
    assert decision.nested_fallback_reason is None
    with pytest.raises(TypeError, match="family"):
        _resolve(case, family=Gaussian(), link=LogLink())
    with pytest.raises(TypeError, match="row_weights"):
        _resolve(case, row_weights=case.weights)


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
    cache = case.dm._structured_layout_cache
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


def _augmented_factor(
    matrices, groups, weights, lambdas, components, chain, prior_weights=None, error=None
):
    dm = DesignMatrix(matrices, n=len(weights), p=groups[-1].end)
    layout = get_structured_layout(
        dm, groups, dominant_group_index=chain[-1], chain_group_indices=chain
    )
    system = build_nested_structured_system(
        matrices,
        groups,
        weights,
        0.3 * weights,
        layout=layout,
        prior_weights=prior_weights,
        error=error,
    )
    penalized = build_penalized_nested_operator(
        system, matrices, groups, lambdas, reml_penalties=components
    )
    return layout, system, build_augmented_nested_factor(system, penalized)[0]


@pytest.mark.parametrize("weight_scale", [1.0, 1e6])
def test_the_row_pass_floor_nulls_a_column_carried_by_rounding_weights(weight_scale) -> None:
    """End to end (§3.7): the production row pass into the augmented factor.

    A border column is non-zero on two rows only, whose weights are 0, +1e-16
    or -1e-16 of ``max |w|`` beside a rounding-negative row elsewhere.  Such
    weights come from a cancellation at the scale of ``max |w|``, which the
    rows' builder states as their error scale (design §3.3; the observed
    rows' ``w0 (|u^2/V| + |(y - mu) factor|)``), so the column's pivot is
    inside its bound: an exact null direction beside the constant column in
    all three cases.  A pass charging those rows their own weight keeps the
    +1e-16 column (``1 / Q_jj`` near 1e16) and refuses -1e-16 as materially
    negative.  The logdets
    differ by the two rows' weights and each factor's own rounding, both
    within ``(n + n_nodes + q + 10) eps`` of the scaled ``Q_s`` entrywise, so by
    ``2 q^2 (n + n_nodes + q + 10) eps kappa_s`` (Weyl, ``lambda_max(Q_s) >= 1``).
    """
    case = _nested_case(weight_scale=weight_scale)
    dense = case.matrices[0].M.copy()
    rows = np.flatnonzero(case.weights)[[10, 200]]
    dense[:, 1] = 0.0
    dense[rows, 1] = 1.0
    matrices = [DenseGroupMatrix(dense), *case.matrices[1:]]
    largest = np.max(case.weights)
    weights = case.weights.copy()
    cancelled = np.concatenate(([np.flatnonzero(weights == 0.0)[0]], rows))
    weights[cancelled[0]] = -1e-16 * largest
    factors = []
    for value in (0.0, 1e-16, -1e-16):
        weights[rows] = value * largest
        error = np.abs(weights)
        error[cancelled] = largest
        _, _, factor = _augmented_factor(
            matrices, case.groups, weights, _lambdas(), _components(case), CHAIN, error=error
        )
        factors.append(factor)
    reference = factors[0]
    p, q = reference.shape[0], len(reference.small_indices)
    eigenvalues = reference.scaled_schur_eigenvalues()
    kappa = eigenvalues[-1] / eigenvalues[eigenvalues > 1e-10 * eigenvalues[-1]].min()
    bound = 2 * q**2 * (case.dm.n + p + 10) * EPS * kappa
    assert reference.rank == p - 2
    for factor in factors[1:]:
        assert factor.rank == reference.rank
        np.testing.assert_array_equal(
            factor.coefficient_estimable(), reference.coefficient_estimable()
        )
        assert abs(factor.logdet() - reference.logdet()) <= bound


def _year_age_alias(shift):
    """``year + age = shift`` beside a 4/12/60 chain, which aliases the intercept.

    Returns the layout, the nested system, its augmented factor, the weights
    and the weighted-mean-centred slope data Gram and Hessian ``H_c = gram +
    S``.  Both columns are centred by type at both shifts (one-engine design
    §3.2); the spread rule this replaced left age (mean 3.0, sd 3.2) raw at
    2013, where the centred null vector touched the intercept.
    """
    rng = np.random.default_rng(5)
    n = 3000
    variant = rng.integers(0, 60, n)
    model = variant // 5
    make = model % 4
    year = rng.integers(2005, 2016, n).astype(float)
    matrices = [
        DenseGroupMatrix(np.column_stack([year, shift - year])),
        RandomEffectGroupMatrix(make, 4),
        RandomEffectGroupMatrix(model, 12),
        RandomEffectGroupMatrix(variant, 60),
    ]
    groups, start = [], 0
    for name, matrix in zip(("numeric", "make", "model", "variant"), matrices, strict=True):
        end = start + matrix.shape[1]
        groups.append(GroupSlice(name=name, start=start, end=end, penalized=name != "numeric"))
        start = end
    lambdas = {"make": 3.0, "model": 5.0, "variant": 8.0}
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
        if index
    ]
    weights = np.exp(rng.normal(size=n))
    layout, system, factor = _augmented_factor(
        matrices, groups, weights, lambdas, components, (1, 2, 3)
    )
    design = np.hstack([matrix.toarray() for matrix in matrices])
    centered = design - (weights @ design) / np.sum(weights)
    ridge = np.concatenate(
        [np.zeros(2), *(np.full(groups[i].size, lambdas[groups[i].name]) for i in (1, 2, 3))]
    )
    gram = centered.T @ (weights[:, None] * centered)
    gram = 0.5 * (gram + gram.T)
    return layout, system, factor, weights, gram, gram + np.diag(ridge)


@pytest.mark.parametrize("shift", [2020.0, 2013.0])
def test_a_truncated_logdet_is_the_dense_backends_centred_pseudo_determinant(shift) -> None:
    """The logdet on both aliases of ``year + age = shift`` equals the gram backend's.

    The dense backend reports ``log sum_w + log pdet(H_c)``, a
    pseudo-determinant that no choice of centre moves.  Both are backward
    stable in their Jacobi-scaled metrics to ``(n + p) eps`` entrywise, so by
    Weyl they differ by at most ``2 p^2 (n + p) eps / lambda_min`` over the
    retained scaled spectrum of ``H_c``.
    """
    layout, _, factor, weights, _, H = _year_age_alias(shift)
    assert np.all(_border_center(layout)[:2] != 0.0)
    n = len(weights)
    dense = decompose_gram(H)
    scale = 1.0 / np.sqrt(np.diag(H))
    retained = np.linalg.eigvalsh(scale[:, None] * H * scale[None, :])[1:]
    p = factor.shape[0]
    assert factor.rank == p - 1 and dense.rank == p - 2
    expected = np.log(np.sum(weights)) + dense.log_pdet
    assert abs(factor.logdet() - expected) <= 2 * p**2 * (n + p) * EPS / retained.min()


@pytest.mark.parametrize("shift", [2020.0, 2013.0])
def test_truncated_profiled_edf_is_the_dense_backends(shift) -> None:
    """EDF and EDF1 of the profiled identity routes equal the gram backend's on both aliases.

    The slope block of the augmented ``H^+`` is a generalized inverse of
    ``H_c``.  The super-root keeps every null ``H``-orthogonal to the intercept,
    so ``H^+ H e_0 = e_0`` at both shifts (the witness, design §3.6); under the
    spread rule it failed at 2013, where the raw slope diagonal exceeded
    ``diag(H_c^+ H_c)`` by ``mean_x[j] (H^+ H)_(j+1, 0)`` (0.599 of EDF).  Diagonal entries on the null support {year, age} depend
    on the generalized inverse (the gram backend pivots one out); their sum,
    every other entry and the traces do not (Rao & Mitra 1971, Lemma 2.2.4 and
    Theorem 2.4.1).  With both backends backward stable to ``(n + p) eps``
    entrywise (above), ``||E|| <= p (n + p) eps``: EDF is the rank less the
    retained eigenvalues in [0, 1] of the scaled ``H^+ S``, each moved at most
    ``||E|| / lambda_min`` (Weyl), and an entry of ``H^+`` at most ``||E|| /
    lambda_min^2`` (first order), doubled for the two-entry support sum; both
    for each backend, and EDF1 at most twice as sensitive.
    """
    _, system, factor, weights, gram, H = _year_age_alias(shift)
    witness = factor.inverse_operator_diagonal(factor.operator.data)[0]
    data = system.operator
    xtw = np.empty(data.shape[0])
    xtw[data.small_indices], xtw[data.structured_indices] = system.xtw_small, system.xtw_structured
    profiled = ProfiledNestedSchurFactor(
        augmented_factor=factor, sum_w=system.sum_w, xtw=xtw, data_operator=data
    )
    dense = decompose_gram(H)
    influence = np.column_stack([dense.solve(column) for column in gram.T])
    scale = 1.0 / np.sqrt(np.diag(H))
    retained = np.linalg.eigvalsh(scale[:, None] * H * scale[None, :])[1:]
    p, n = H.shape[0], len(weights)
    trace = 2 * p**2 * (n + p) * EPS / retained.min()
    entry = 4 * p * (n + p) * EPS / retained.min() ** 2
    assert abs(witness - 1.0) <= entry

    def grouped(diagonal):
        return np.r_[diagonal[:2].sum(), diagonal[2:]]

    edf = profiled.inverse_operator_diagonal(data)
    edf1 = profiled.inverse_operator_square_diagonal(data)
    squared = influence @ influence
    assert abs(profiled.trace_inverse_operator(data) - np.trace(influence)) <= trace
    assert abs(np.sum(edf1) - np.trace(squared)) <= 2 * trace
    assert np.all(np.abs(grouped(edf) - grouped(np.diag(influence))) <= entry)
    assert np.all(np.abs(grouped(edf1) - grouped(np.diag(squared))) <= 2 * entry)


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


def _fit(case: NestedCase, groups=None, matrices=None, direct_solve="structured", family=None):
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
        family=Poisson() if family is None else family,
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


def test_a_single_random_effect_is_a_chain_of_one() -> None:
    case, matrices, groups = _dense_border_case((VARIANT,))
    _result, factor, _operator, profile, _cache, _dm = _fit(case, groups, matrices)
    assert isinstance(factor, ProfiledNestedSchurFactor)
    assert factor.chain_group_names == ("variant",)
    assert profile["structured_chain"] == ("variant",)
    assert profile["structured_nested_fallback_reason"] is None


def test_gaussian_log_keeps_its_chain_of_one() -> None:
    """Gaussian/log observed rows can be negative; the chain of one factors
    them (one-engine design §3.3), so the fit keeps it with no decline."""
    case, matrices, groups = _dense_border_case((VARIANT,))
    _result, factor, _operator, profile, _cache, _dm = _fit(
        case, groups, matrices, family=Gaussian()
    )
    assert isinstance(factor, ProfiledNestedSchurFactor)
    assert profile["structured_chain"] == ("variant",)
    assert profile["structured_nested_fallback_reason"] is None


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
