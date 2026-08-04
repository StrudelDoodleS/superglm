"""The constrained solve must report the geometry it actually ran on.

``solve_constrained_qp`` decomposes ``H`` and every pure-``H`` solve runs on
that decomposition, but ``QPResult`` used to carry none of it, and the
constrained branch of ``fit_irls_direct`` hardcoded ``_cond_est = 0.0`` and
``_used_svd = False``.  ``0.0`` reads as "perfectly conditioned" to the three
consumers that inspect it, so the branch with the least visible linear algebra
claimed the most certainty.

Without these fields the equivalence evidence issue #203 asks for cannot be
collected at all: there is nothing to compare.
"""

from __future__ import annotations

import numpy as np

from superglm.solvers.constrained_qp import solve_constrained_qp
from superglm.solvers.rank import decompose_gram


def _monotone_rows(width: int) -> np.ndarray:
    """First-difference rows: ``beta[i + 1] >= beta[i]``."""
    return np.diff(np.eye(width), axis=0)


def test_unconstrained_return_reports_the_decomposition_geometry() -> None:
    rng = np.random.default_rng(20260805)
    X = rng.normal(size=(80, 5))
    H = X.T @ X
    g = X.T @ rng.normal(size=80)

    result = solve_constrained_qp(H, g, np.zeros((0, 5)), np.zeros(0))
    reference = decompose_gram(0.5 * (H + H.T))

    assert result.rank == reference.rank == 5
    assert result.width == reference.width == 5
    assert result.method == reference.method
    assert result.condition == reference.pre_truncation_condition
    assert result.condition > 0.0


def test_rank_deficient_h_reports_the_truncated_rank_not_the_width() -> None:
    """A duplicated column must show up as ``rank < width``, not as silence."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, 4))
    X[:, 3] = X[:, 0]
    H = X.T @ X
    g = X.T @ rng.normal(size=60)

    result = solve_constrained_qp(H, g, np.zeros((0, 4)), np.zeros(0))

    assert result.width == 4
    assert result.rank == 3
    assert result.method != ""


def test_the_constrained_path_reports_geometry_too() -> None:
    """The active-set return carries it, not only the early unconstrained ones.

    The constraint is chosen to bind, so this exercises the loop's return
    rather than the ``_is_feasible(beta_unc)`` shortcut.
    """
    rng = np.random.default_rng(3)
    width = 5
    X = rng.normal(size=(90, width))
    beta_true = np.array([3.0, 1.5, 0.5, -1.0, -2.5])
    y = X @ beta_true + 0.05 * rng.normal(size=90)
    H = X.T @ X
    g = X.T @ y
    A = _monotone_rows(width)

    result = solve_constrained_qp(H, g, A, np.zeros(A.shape[0]))

    assert result.active_set, "expected the monotone constraint to bind"
    assert result.rank == width
    assert result.width == width
    assert result.condition > 0.0
    assert result.method != ""


def test_condition_is_the_same_quantity_the_unconstrained_branch_publishes() -> None:
    """Comparability is the point: both must be ``pre_truncation_condition``.

    If these two ever diverge, the constrained and unconstrained branches of a
    single fit are reporting different quantities under one name.
    """
    rng = np.random.default_rng(101)
    X = rng.normal(size=(50, 3))
    X[:, 2] = X[:, 0] + 1e-7 * X[:, 1]
    H = X.T @ X

    result = solve_constrained_qp(H, X.T @ rng.normal(size=50), np.zeros((0, 3)), np.zeros(0))

    assert result.condition == decompose_gram(0.5 * (H + H.T)).pre_truncation_condition
