"""Representation gates behind the packed centered build.

``packed_centered_gram_rhs`` is all-or-nothing: one group it does not admit
sends the whole design to the chunked dense fallback, which materializes rows
in blocks.  Every group in an ordered-categorical/categorical pricing design
is one-hot or support-compressed, so the packed path must accept all of them.
These tests pin the representation each spec emits, because the cost of losing
one is paid by the entire design rather than by that group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from superglm import Categorical, OrderedCategorical, Spline, SuperGLM
from superglm._group_matrix._group_matrix_centered import (
    centered_gram_rhs,
    packed_centered_gram_rhs,
)
from superglm._group_matrix._group_matrix_discretized import (
    SupportCompressedSSPGroupMatrix,
)
from superglm.group_matrix import CategoricalGroupMatrix, DesignMatrix
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
