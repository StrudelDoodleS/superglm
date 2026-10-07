"""Representation gates behind the packed centered build.

``packed_centered_gram_rhs`` is all-or-nothing: one group it does not admit
sends the whole design to the chunked dense fallback, which materializes rows
in blocks.  Every group in an ordered-categorical/categorical pricing design
is one-hot or support-compressed, so the packed path must accept all of them.
These tests pin the representation each spec emits, because the cost of losing
one is paid by the entire design rather than by that group.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM
from superglm._group_matrix._group_matrix_centered import (
    anchor_support_centered_gram_rhs,
    centered_gram_rhs,
    packed_centered_gram_rhs,
)
from superglm._group_matrix._group_matrix_discretized import (
    SupportCompressedSSPGroupMatrix,
)
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DesignMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSSPGroupMatrix,
)
from superglm.model.base import model_build_design_matrix

N = 600
LEVELS_A = 7
LEVELS_B = 5


def _frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ord": rng.choice([f"o{i}" for i in range(6)], size=N),
            "cat_a": rng.choice([f"a{i}" for i in range(LEVELS_A)], size=N),
            "cat_b": rng.choice([f"b{i}" for i in range(LEVELS_B)], size=N),
        }
    )


def _build(df: pd.DataFrame, *, specials: list[str] | None = None, interaction: bool = True):
    features = {
        "ord": OrderedCategorical(
            values={f"o{i}": float(i) for i in range(6)},
            basis=Spline(kind="cr", k=4),
            specials=specials,
        ),
        "cat_a": Categorical(base="most_exposed"),
        "cat_b": Categorical(base="most_exposed"),
    }
    model = SuperGLM(
        family="gaussian",
        link="identity",
        selection_penalty=0.0,
        features=features,
        interactions=[("cat_a", "cat_b")] if interaction else None,
    )
    rng = np.random.default_rng(1)
    y = rng.normal(size=N)
    w = np.ones(N)
    model_build_design_matrix(model, df, y, w, None)
    return model


def _named_group(model, name: str):
    for group, gm in zip(model._groups, model._dm.group_matrices):
        if group.name == name or group.feature_name == name:
            yield group, gm


def test_categorical_interaction_builds_as_a_categorical_group() -> None:
    """A two-categorical interaction is itself categorical: one cell per row."""
    model = _build(_frame())
    matches = [
        gm
        for group, gm in zip(model._groups, model._dm.group_matrices)
        if "cat_a" in group.name and "cat_b" in group.name
    ]
    assert matches, "no interaction group was emitted"
    assert isinstance(matches[0], CategoricalGroupMatrix)
    assert matches[0].shape[1] == (LEVELS_A - 1) * (LEVELS_B - 1)


def test_interaction_columns_match_an_explicit_one_hot() -> None:
    """The codes representation must reproduce the pair-indicator columns exactly."""
    df = _frame()
    model = _build(df)
    spec = next(iter(model._interaction_specs.values()))
    built = None
    for group, gm in zip(model._groups, model._dm.group_matrices):
        if "cat_a" in group.name and "cat_b" in group.name:
            built = gm.toarray()
    assert built is not None

    expected = np.column_stack(
        [
            ((df["cat_a"].to_numpy() == lev1) & (df["cat_b"].to_numpy() == lev2)).astype(float)
            for lev1, lev2 in spec._pairs
        ]
    )
    np.testing.assert_array_equal(built, expected)


def test_specials_block_builds_as_a_categorical_group() -> None:
    """A row carries at most one special, so that block is one-hot too."""
    model = _build(_frame(), specials=["o0"])
    special = [
        gm
        for group, gm in zip(model._groups, model._dm.group_matrices)
        if group.subgroup_type == "special"
    ]
    assert special, "no special block was emitted"
    assert isinstance(special[0], CategoricalGroupMatrix)


def _support_compressed_group(n: int, width: int, n_support: int, seed: int):
    """A lossless support-compressed spline block, built directly.

    Built at the group-matrix level rather than through a spec: whether the
    builder chooses compression is a cost-model decision with its own
    thresholds, and this test is about what the packed gate does once it is
    handed such a group -- not about when the builder produces one.
    """
    rng = np.random.default_rng(seed)
    b_unique = rng.normal(size=(n_support, width))
    row_index = rng.integers(0, n_support, size=n).astype(np.intp)
    r_inv = np.eye(width, dtype=np.float64)
    return SupportCompressedSSPGroupMatrix(b_unique, r_inv, row_index)


def _mixed_design(n: int = N):
    rng = np.random.default_rng(5)
    groups = [
        _support_compressed_group(n, 3, 6, seed=11),
        _support_compressed_group(n, 4, 9, seed=12),
        CategoricalGroupMatrix(rng.integers(-1, 6, size=n).astype(np.intp), 6),
        CategoricalGroupMatrix(rng.integers(-1, 24, size=n).astype(np.intp), 24),
    ]
    return DesignMatrix(groups, n, sum(g.shape[1] for g in groups))


def test_packed_path_accepts_a_lossless_support_compressed_group() -> None:
    """The gate must not reject a subclass that adds no state.

    ``SupportCompressedSSPGroupMatrix`` is a ``DiscretizedSSPGroupMatrix`` with
    ``__slots__ = ()``; an exact-type test rejected it, and because the packed
    build is all-or-nothing, one such group sent an entire design to the
    chunked dense fallback.
    """
    dm = _mixed_design()
    rng = np.random.default_rng(3)
    W = rng.uniform(0.5, 2.0, dm.n)
    z = rng.normal(size=dm.n)
    z_centered = z - float(np.dot(W, z) / W.sum())
    assert packed_centered_gram_rhs(dm=dm, W=W, z_centered=z_centered) is not None


def test_packed_and_chunked_builds_agree() -> None:
    """The two routes must differ only in cost."""
    dm = _mixed_design()
    rng = np.random.default_rng(4)
    W = rng.uniform(0.5, 2.0, dm.n)
    z = rng.normal(size=dm.n)
    z_centered = z - float(np.dot(W, z) / W.sum())

    packed = packed_centered_gram_rhs(dm=dm, W=W, z_centered=z_centered)
    assert packed is not None
    mean_x, gram_packed, rhs_packed = packed
    gram_chunked, rhs_chunked = centered_gram_rhs(dm=dm, W=W, mean_x=mean_x, z_centered=z_centered)
    np.testing.assert_allclose(gram_packed, gram_chunked, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(rhs_packed, rhs_chunked, rtol=1e-9, atol=1e-10)


def test_a_wide_categorical_support_is_rejected_before_it_is_materialised(monkeypatch):
    """One wide block must fall back without building its dense support first.

    A categorical block's anchor support is a dense ``(K + 1, K)`` identity and
    its Gram costs O(K^3).  Once a crossed interaction builds as a categorical
    block, ``K`` is ``(L1 - 1) * (L2 - 1)`` -- multiplicative in the parents'
    cardinalities rather than additive.  The pairwise cell check cannot see
    this case at all: with a single categorical block there is no pair to
    compare, so nothing rejects the plan and the identity is both materialized
    and cubed.
    """
    from superglm._group_matrix import _group_matrix_centered as centered

    rng = np.random.default_rng(3)
    n, levels = 400, 24
    group = CategoricalGroupMatrix(rng.integers(-1, levels, size=n).astype(np.intp), levels)
    dm = DesignMatrix([group], n, group.shape[1])
    support_rows = levels + 1

    W = rng.uniform(0.5, 2.0, n)
    z = rng.normal(size=n)
    z_centered = z - float(np.dot(W, z) / W.sum())

    # A cap this block's own support exceeds, but that no PAIR could trip:
    # there is only one support, so ``supports[i + 1:]`` is always empty.
    monkeypatch.setattr(centered, "_MAX_PACKED_HIST_CELLS", support_rows * support_rows - 1)
    assert centered.packed_centered_gram_rhs(dm=dm, W=W, z_centered=z_centered) is None, (
        "an oversized support must fall back, not be materialized and cubed"
    )

    # Directly under the cap it still builds, so the guard is not blanket-off.
    monkeypatch.setattr(centered, "_MAX_PACKED_HIST_CELLS", support_rows * support_rows)
    assert centered.packed_centered_gram_rhs(dm=dm, W=W, z_centered=z_centered) is not None


@pytest.mark.parametrize("route", ["anchor_gram", "centred_rhs"])
def test_an_oversized_compact_support_is_refused_before_it_is_allocated(monkeypatch, route):
    """Every caller that materialises supports sizes them from metadata first.

    A categorical's compact support is a dense ``(K + 1, K)`` identity: for a
    3,000-level block that is 72 MB allocated only to be rejected by the cell
    cap. The anchor route's Gram needs ``(K + 1)^2`` cells, the centred
    right-hand side (``_compact_centered_rmatvec``, which the SCOP mode score
    reaches when a frequent indicator trips the centring guard) ``(K + 1) K``.
    Mutation check: building the supports first and checking their shapes
    afterwards calls ``_compact_support`` on the oversized block, which this
    test forbids; on af53c8d4 the right-hand side did exactly that.
    """
    from superglm._group_matrix import _group_matrix_centered as centered

    rng = np.random.default_rng(3)
    n, levels = 400, 24
    group = CategoricalGroupMatrix(rng.integers(-1, levels, size=n).astype(np.intp), levels)
    dm = DesignMatrix([group], n, group.shape[1])
    support_rows = levels + 1
    W = rng.uniform(0.5, 2.0, n)
    z = rng.normal(size=n)
    z_centered = z - float(np.dot(W, z) / W.sum())
    if route == "anchor_gram":
        cells = support_rows * support_rows

        def build():
            return centered.anchor_support_centered_gram_rhs(dm=dm, W=W, z_centered=z_centered)

    else:
        cells = support_rows * levels
        mean_x = dm.toarray().T @ W / W.sum()

        def build():
            return centered._compact_centered_rmatvec(
                dm=dm, rows=W * z_centered, mean_x=mean_x, mean_lo=None
            )

    built = []
    real_compact_support = centered._compact_support

    def counting_compact_support(gm):
        built.append(gm)
        return real_compact_support(gm)

    monkeypatch.setattr(centered, "_compact_support", counting_compact_support)

    monkeypatch.setattr(centered, "_MAX_PACKED_HIST_CELLS", cells - 1)
    assert build() is None
    assert built == [], "an oversized support must be refused before it is materialised"

    monkeypatch.setattr(centered, "_MAX_PACKED_HIST_CELLS", cells)
    assert build() is not None
    assert built == [group]


def test_a_rejected_tensor_raw_rung_is_paid_once_per_fit(monkeypatch):
    """The tensor rung's raw moments are not recomputed after their certificate rejects.

    The categorical level carries about 3/4 of the weight, so its mean exceeds
    its centred RMS and the pattern rung's certificate rejects; the
    anchor-centred supports serve the design either way.  Repeating the
    rejected rung every iteration doubled the centring cost of a 678k-row fit.
    """
    from superglm._group_matrix import _group_matrix_centered as centered
    from superglm.group_matrix import DiscretizedTensorGroupMatrix
    from superglm.solvers.centered_system import TabmatCenteringState

    rng = np.random.default_rng(20261004)
    n = 200
    B1, B2 = rng.uniform(size=(5, 4)), rng.uniform(size=(4, 3))
    idx1 = rng.integers(0, len(B1), size=n, dtype=np.intp)
    idx2 = rng.integers(0, len(B2), size=n, dtype=np.intp)
    cells, pair_idx = np.unique(idx1 * len(B2) + idx2, return_inverse=True)
    B_joint = (B1[cells // len(B2), :, None] * B2[cells % len(B2), None, :]).reshape(len(cells), -1)
    tensor = DiscretizedTensorGroupMatrix(
        B1, B2, idx1, idx2, B_joint, rng.normal(size=(12, 6)), pair_idx.astype(np.intp), 1
    )
    frequent = CategoricalGroupMatrix(np.where(rng.uniform(size=n) < 0.75, 0, -1), 1)
    dm = DesignMatrix([tensor, frequent], n, 7)
    W = rng.uniform(0.5, 2.0, n)
    z = rng.normal(size=n)
    z_centered = z - float(np.dot(W, z) / W.sum())

    attempts = []
    pattern = centered._try_pattern_tensor_centering

    def counted(**kwargs):
        attempts.append(1)
        return pattern(**kwargs)

    monkeypatch.setattr(centered, "_try_pattern_tensor_centering", counted)
    state = TabmatCenteringState()
    first = centered.packed_centered_gram_rhs(dm=dm, W=W, z_centered=z_centered, state=state)
    second = centered.packed_centered_gram_rhs(dm=dm, W=W, z_centered=z_centered, state=state)

    assert first is not None and second is not None
    assert state.tensor_raw_eligible is False, "the frequent level must trip the certificate"
    assert len(attempts) == 1
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_a_rejected_tensor_raw_rung_stays_live_when_the_anchor_route_declines(monkeypatch):
    """A rejection latches only when the anchor-centred supports served the build.

    The tensor has more cells than the anchor route accepts, so after the
    certificate rejects, the build falls to the chunked dense pass.  Latching
    there skipped the tensor rungs for the rest of the fit, so every later
    build paid that pass even when a later iterate's weights would have
    passed (ten tensor pairs on the 678k-row book: 5 chunked builds against
    2).  The rung must be retried at every build instead.
    """
    from superglm._group_matrix import _group_matrix_centered as centered
    from superglm.group_matrix import DiscretizedTensorGroupMatrix
    from superglm.solvers.centered_system import TabmatCenteringState

    rng = np.random.default_rng(20261005)
    n = 20_000
    B1, B2 = rng.uniform(size=(60, 4)), rng.uniform(size=(60, 3))
    idx1 = rng.integers(0, len(B1), size=n, dtype=np.intp)
    idx2 = rng.integers(0, len(B2), size=n, dtype=np.intp)
    cells, pair_idx = np.unique(idx1 * len(B2) + idx2, return_inverse=True)
    assert len(cells) ** 2 > centered._MAX_PACKED_HIST_CELLS, "the anchor route must decline"
    B_joint = (B1[cells // len(B2), :, None] * B2[cells % len(B2), None, :]).reshape(len(cells), -1)
    tensor = DiscretizedTensorGroupMatrix(
        B1, B2, idx1, idx2, B_joint, rng.normal(size=(12, 6)), pair_idx.astype(np.intp), 1
    )
    frequent = CategoricalGroupMatrix(np.where(rng.uniform(size=n) < 0.75, 0, -1), 1)
    dm = DesignMatrix([tensor, frequent], n, 7)
    W = rng.uniform(0.5, 2.0, n)
    z = rng.normal(size=n)
    z_centered = z - float(np.dot(W, z) / W.sum())

    attempts = []
    pattern = centered._try_pattern_tensor_centering

    def counted(**kwargs):
        attempted, result = pattern(**kwargs)
        attempts.append((attempted, result is not None))
        return attempted, result

    monkeypatch.setattr(centered, "_try_pattern_tensor_centering", counted)
    state = TabmatCenteringState()
    first = centered.packed_centered_gram_rhs(dm=dm, W=W, z_centered=z_centered, state=state)
    second = centered.packed_centered_gram_rhs(dm=dm, W=W, z_centered=z_centered, state=state)

    assert first is None and second is None
    assert attempts == [(True, False), (True, False)], "the rung must be retried, not latched"
    assert state.tensor_raw_eligible is not False


def test_spline_by_category_design_is_centred_on_compact_supports(monkeypatch):
    """A factor-by smooth beside a frequent indicator: no chunked Gram, exact to its bound.

    The indicator's level carries about 3/4 of the weight, so the raw-moment
    rung's certificate rejects, and ``packed_centered_gram_rhs`` does not admit
    spline-by-category levels: every Gram fell to row chunks (2.2 s a build at
    678k rows).  The compact fallback centres each support row about its
    anchor before any product.  Each entry is a sum over support rows of a
    (joint) bin mass times two centred, transformed support entries; with
    ``|c_b T|_j <= alpha_j = 4 sum_r max_b |v_br| |T_rj|`` and ``k = n + 2
    n_bins + K + 6`` roundings per factor, ``|C - C*|_jk <= gamma_{3k + n_bins}
    S alpha_j alpha_k`` (Higham 2002, sec. 3.1), against the exact Gram of
    the exact rows ``v_b T``.
    """
    from fractions import Fraction

    from superglm._group_matrix import _group_matrix_centered as centered
    from superglm.group_matrix import DiscretizedSplineCategoricalGroupMatrix as SplineCategorical
    from superglm.group_matrix import DiscretizedSSPGroupMatrix
    from superglm.solvers.centered_system import build_centered_system

    rng = np.random.default_rng(20261004)
    n, n_bins = 60, 5
    B, R = rng.uniform(size=(n_bins, 4)), rng.normal(size=(4, 3))
    bins = rng.integers(0, n_bins, n)
    level = rng.integers(0, 3, n)  # level 0 is the base of the by-factor
    groups = [
        DiscretizedSSPGroupMatrix(B, R, bins),
        SplineCategorical(B, R, bins, np.flatnonzero(level == 1)),
        SplineCategorical(B, R, bins, np.flatnonzero(level == 2)),
        CategoricalGroupMatrix(np.where(rng.uniform(size=n) < 0.75, 0, -1), 1),
    ]
    dm = DesignMatrix(groups, n, 10)
    W = rng.uniform(0.5, 2.0, n)

    def no_rows(self, idx):
        raise AssertionError("a compact design must not be materialised")

    monkeypatch.setattr(DesignMatrix, "row_subset", no_rows)
    monkeypatch.setattr(centered, "_MIN_MIXED_RAW_MOMENT_CELLS", 0)  # let the raw rung run
    profile: dict = {}
    system = build_centered_system(
        dm=dm, W=W, z_off=rng.normal(size=n), penalty=np.zeros((10, 10)), profile=profile
    )
    assert "centered_raw_moment_hits" not in profile, "the frequent level must trip the certificate"
    assert profile["centered_anchor_support_hits"] == 1

    # Exact rows v_b T, column majorants alpha, exact centred Gram.
    supports = [(B, bins, R)] * 3 + [(np.array([[1.0], [0.0]]), (groups[3].codes == 1), None)]
    exact_rows = [[Fraction(0)] * 10 for _ in range(n)]
    alpha = []
    column = 0
    for g, (values, codes, transform) in enumerate(supports):
        T = np.eye(values.shape[1]) if transform is None else transform
        peak = np.max(np.abs(values), axis=0)
        alpha.extend(4 * Fraction(float(v)) for v in peak @ np.abs(T))
        for i in range(n):
            inside = g == 0 or g == 3 or level[i] == g
            if not inside:
                continue
            row = values[int(codes[i])]
            for j in range(T.shape[1]):
                exact_rows[i][column + j] = sum(
                    Fraction(row[r]) * Fraction(T[r, j]) for r in range(values.shape[1])
                )
        column += T.shape[1]
    weights = [Fraction(w) for w in W]
    total = sum(weights)
    means = [sum(w * x[j] for w, x in zip(weights, exact_rows)) / total for j in range(10)]
    u = Fraction(2) ** -53
    k = 3 * (n + 2 * n_bins + B.shape[1] + 6) + n_bins
    gamma = k * u / (1 - k * u)
    for j in range(10):
        for m in range(10):
            exact = sum(
                w * (x[j] - means[j]) * (x[m] - means[m]) for w, x in zip(weights, exact_rows)
            )
            bound = gamma * total * alpha[j] * alpha[m]
            assert abs(Fraction(system.data_gram[j, m]) - exact) <= bound, (j, m)


def test_compact_centred_rhs_centres_support_rows_and_forms_no_design_row(monkeypatch):
    """``centered_rhs`` over compact supports, against the exact sum over the same rows.

    Every term is a centred support row ``s_b - c`` (one rounding, as the
    chunked pass forms it) times its bin's sum of ``w z`` (one rounding each,
    then at most ``n - 1`` in the bin sum and ``n_bins`` in the product over
    bins), so ``|r_j - r*_j| <= gamma_k sum_i |x_ij - c_j| |w_i z_i|`` with
    ``k = n + n_bins + 2`` (Higham 2002, sec. 3.1).  The SCOP support sits at
    1e6: aggregating raw and subtracting ``c`` afterwards would round at ``u
    1e6 sum |w z|``, far above that bound.
    """
    from fractions import Fraction

    from superglm._group_matrix._group_matrix_centered import centered_rhs
    from superglm.group_matrix import (
        DenseGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSSPGroupMatrix,
    )

    rng = np.random.default_rng(20261004)
    n, n_bins = 96, 7
    groups = [
        DiscretizedSSPGroupMatrix(
            rng.uniform(size=(n_bins, 4)), rng.normal(size=(4, 3)), rng.integers(0, n_bins, n)
        ),
        DiscretizedSCOPGroupMatrix(1.0e6 + rng.uniform(size=(5, 2)), rng.integers(0, 5, n)),
        CategoricalGroupMatrix(np.where(rng.uniform(size=n) < 0.8, 0, -1), 1),
        DenseGroupMatrix((1.0e10 + rng.normal(size=n))[:, None]),
    ]
    dm = DesignMatrix(groups, n, 7)
    W = rng.uniform(0.5, 2.0, n)
    z = rng.normal(size=n)
    rows = dm.toarray()
    centre = rows.T @ W / W.sum()

    def no_rows(self, idx):
        raise AssertionError("a compact design must not be materialised")

    monkeypatch.setattr(DesignMatrix, "row_subset", no_rows)
    computed = centered_rhs(dm=dm, W=W, mean_x=centre, z_centered=z)

    u = Fraction(2) ** -53
    k = Fraction(n + n_bins + 2)
    gamma = k * u / (1 - k * u)
    weighted = [Fraction(W[i]) * Fraction(z[i]) for i in range(n)]
    for j in range(dm.p):
        offsets = [Fraction(rows[i, j]) - Fraction(centre[j]) for i in range(n)]
        exact = sum(d * r for d, r in zip(offsets, weighted, strict=True))
        bound = gamma * sum(abs(d * r) for d, r in zip(offsets, weighted, strict=True))
        assert abs(Fraction(computed[j]) - exact) <= bound, j


@pytest.mark.parametrize("tensor_position", ["last", "middle"])
def test_tensor_margin_cross_grams_come_from_the_tensor_tables(monkeypatch, tensor_position):
    """A tensor's own margins take their cross-Grams from its tables: fewer passes, bounded error.

    Two groups whose codes are the tensor's marginal bins, one other spline and
    a frequent indicator.  Per pair the Gram costs ten row passes; deriving the
    margins' joint tables from the tensor's (``_add_family_cross``) needs one
    per other group plus the one pair outside the family: three.  The derived
    sums associate the same terms differently, so the check is against the
    exact Gram of the computed centred supports ``v``:
    ``|G_jk - sum_r W_r v(r)_j v(r)_k| <= gamma_K sum_r W_r |v(r)_j| |v(r)_k|``,
    ``K = n + 2 max(bins, cells) + 2`` (``_anchor_support_gram_rhs``).
    """
    from fractions import Fraction

    from superglm._group_matrix import _group_matrix_centered as centered
    from superglm.group_matrix import DiscretizedSSPGroupMatrix, DiscretizedTensorGroupMatrix

    rng = np.random.default_rng(20261004)
    n = 150
    B1, B2, B3 = rng.uniform(size=(6, 4)), rng.uniform(size=(5, 3)), rng.uniform(size=(4, 3))
    idx1, idx2 = rng.integers(0, 6, n), rng.integers(0, 5, n)
    cells, cell_idx = np.unique(idx1 * 5 + idx2, return_inverse=True)
    joint = (B1[cells // 5, :, None] * B2[cells % 5, None, :]).reshape(len(cells), -1)
    tensor = DiscretizedTensorGroupMatrix(
        B1, B2, idx1, idx2, joint, rng.normal(size=(12, 7)), cell_idx.astype(np.intp), 1
    )
    margin1 = DiscretizedSSPGroupMatrix(B1, rng.normal(size=(4, 3)), idx1)
    margin2 = DiscretizedSSPGroupMatrix(B2, rng.normal(size=(3, 2)), idx2)
    other = DiscretizedSSPGroupMatrix(B3, rng.normal(size=(3, 2)), rng.integers(0, 4, n))
    frequent = CategoricalGroupMatrix(np.where(rng.uniform(size=n) < 0.75, 0, -1), 1)
    groups = (
        [margin1, margin2, other, frequent, tensor]
        if tensor_position == "last"
        else [margin1, tensor, other, margin2, frequent]
    )
    dm = DesignMatrix(groups, n, 15)
    W = rng.uniform(0.5, 2.0, n)
    z = rng.normal(size=n)
    z_centered = z - float(np.dot(W, z) / W.sum())

    passes = []
    histogram = centered._disc_disc_2d_hist

    def counted(*args):
        passes.append(1)
        return histogram(*args)

    monkeypatch.setattr(centered, "_disc_disc_2d_hist", counted)
    result = centered.anchor_support_centered_gram_rhs(dm=dm, W=W, z_centered=z_centered)
    assert result is not None
    _mean, gram, _rhs = result
    assert len(passes) == 3, f"{len(passes)} row passes for one Gram"

    # The computed centred supports both routes share, and their exact Gram.
    sum_w = float(np.sum(W))
    rows = [[] for _ in range(n)]
    for matrix in groups:
        values, codes, transform = centered._compact_support(matrix)
        support = centered._anchor_center_support(
            values=values, codes=codes, W=W, Wz=W * z_centered, sum_w=sum_w, transform=transform
        )
        for r in range(n):
            rows[r].extend(Fraction(float(v)) for v in support.values[codes[r]])
    weights = [Fraction(float(w)) for w in W]
    u = Fraction(2) ** -53
    k = n + 2 * len(cells) + 2
    gamma = k * u / (1 - k * u)
    for j in range(15):
        for m in range(15):
            exact = sum(w * v[j] * v[m] for w, v in zip(weights, rows, strict=True))
            scale = sum(w * abs(v[j] * v[m]) for w, v in zip(weights, rows, strict=True))
            assert abs(Fraction(gram[j, m]) - exact) <= gamma * scale, (j, m)


def _two_bins(n: int) -> np.ndarray:
    bins = np.zeros(n, dtype=np.intp)
    bins[9 * n // 10 :] = 1
    return bins


def test_the_anchor_route_projects_the_anchor_and_the_shift_apart():
    """An SSP column's centre is its centred support's through a projection that cancels the anchor.

    ``B_unique`` rows ``(1e16 +- 2, 1e16)`` and ``R_inv = (1, -1)'`` give the
    column ``4 1[bin 0] - 2`` exactly, bin 0 holding 90% of the unit weight,
    so the anchor is row 0 and the shift ``(-0.4, 0)``.  Every difference and
    product of the support is exact (Sterbenz), so the centre
    ``fl(v_h T) + fl(shift T)`` lies within ``gamma_2 (2 + 0.4)`` of 8/5: only
    the shift's division and the final sum round.  ``fl(v_h + shift)`` rounds
    the shift away against 1e16 and gives 2.
    """
    n = 1000
    ssp = DiscretizedSSPGroupMatrix(
        np.array([[1e16 + 2, 1e16], [1e16 - 2, 1e16]]), np.array([[1.0], [-1.0]]), _two_bins(n)
    )
    result = anchor_support_centered_gram_rhs(
        dm=DesignMatrix([ssp], n, 1), W=np.ones(n), z_centered=np.zeros(n)
    )
    assert result is not None
    u = Fraction(2) ** -53
    gamma_2 = 2 * u / (1 - 2 * u)
    assert abs(Fraction(float(result[0][0])) - Fraction(8, 5)) <= gamma_2 * Fraction(12, 5)


def test_the_anchor_route_reads_an_integer_support_as_float64():
    """An int64 SCOP support gives the products of the float64 column every reader sees.

    Anchored at ``-2**63`` (90% of the rows), ``1 - (-2**63)`` wrapped in
    int64 and reversed the centred column.  Read as float64 first, the integer
    and float64 supports go through identical arithmetic.
    """
    n = 1000
    rng = np.random.default_rng(5)
    W, z = rng.uniform(0.5, 2.0, n), rng.normal(size=n)
    support = np.array([[-(2**63)], [1]], dtype=np.int64)
    integer, real = (
        anchor_support_centered_gram_rhs(
            dm=DesignMatrix([DiscretizedSCOPGroupMatrix(values, _two_bins(n))], n, 1),
            W=W,
            z_centered=z,
        )
        for values in (support, support.astype(np.float64))
    )
    assert integer is not None and real is not None
    for got, expected in zip(integer, real, strict=True):
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("part", ["complex_support", "complex_transform", "object_support"])
def test_the_anchor_route_declines_a_support_it_cannot_read_as_float64(part):
    """A complex or object support or transform declines the anchor route, which reads them as float64.

    The design then takes the chunked pass.  Real dtypes of eight bytes or
    fewer are read (``test_the_anchor_route_reads_an_integer_support_as_float64``).
    Mutation: without the guard the route converts them and returns products.
    """
    n = 1000
    support = np.array([[2.0, 1.0], [1.0, 3.0]])
    transform = np.array([[1.0], [-1.0]])
    if part == "complex_support":
        support = support.astype(np.complex128)
    elif part == "complex_transform":
        transform = transform.astype(np.complex128)
    else:
        support = support.astype(object)
    ssp = DiscretizedSSPGroupMatrix(support, transform, _two_bins(n))
    result = anchor_support_centered_gram_rhs(
        dm=DesignMatrix([ssp], n, 1), W=np.ones(n), z_centered=np.zeros(n)
    )
    assert result is None
