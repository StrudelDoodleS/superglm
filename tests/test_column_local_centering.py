"""Column-local centring: a column the raw-moment certificate rejects is recentred alone.

The certificate is per column, ``kappa^2 = 1 + mean^2 / RMS^2 <= 2`` (Chan,
Golub & LeVeque 1983, eq. 3.3).  A 0/1 indicator whose level carries more than
half the working weight fails it, and beside two or more tensor terms, whose
supports the anchor-centred route declines, every Gram build of the whole
design fell to the chunked ``O(n p^2)`` pass: on the freMTPL2 book with ten
tensor pairs, VehGas (Diesel at 51% of the weight) cost 21 such builds.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from superglm._group_matrix import _group_matrix_centered as centered
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DesignMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
)
from superglm.solvers import centered_system
from superglm.solvers.centered_system import TabmatCenteringState, build_centered_system


def _tensor(rng, n, bins, k, p, tensor_id, offset=0.0):
    """A factored tensor on a full grid; ``offset`` moves its columns' means far from zero."""
    B1 = offset + rng.normal(size=(bins[0], k[0]))
    B2 = offset + rng.normal(size=(bins[1], k[1]))
    idx1 = rng.integers(0, bins[0], size=n, dtype=np.intp)
    idx2 = rng.integers(0, bins[1], size=n, dtype=np.intp)
    joint = np.einsum("ia,jb->ijab", B1, B2).reshape(bins[0] * bins[1], k[0] * k[1])
    R = rng.normal(size=(k[0] * k[1], p)) / np.sqrt(k[0] * k[1])
    return DiscretizedTensorGroupMatrix(
        B1, B2, idx1, idx2, joint, R, idx1 * bins[1] + idx2, tensor_id
    )


def _design(rng, n, bins, share, scop=False):
    """Two tensors and a 0/1 indicator whose level holds ``share`` of the rows.

    ``scop`` adds a discretized SCOP group, which the packed tensor rungs do
    not admit, so the general raw-moment rung forms the moments instead.
    """
    first, second = (_tensor(rng, n, bins, (3, 3), 6, tensor_id) for tensor_id in (1, 2))
    heavy = CategoricalGroupMatrix(np.where(rng.uniform(size=n) < share, 0, -1), 1)
    groups = [first, heavy, second]
    if scop:
        groups.append(
            DiscretizedSCOPGroupMatrix(rng.normal(size=(12, 3)), rng.integers(0, 12, size=n))
        )
    return DesignMatrix(groups, n, sum(group.shape[1] for group in groups))


def _counted(store, function):
    def counted(*args, **kwargs):
        store.append(1)
        return function(*args, **kwargs)

    return counted


@pytest.mark.parametrize("scop", [False, True])
@pytest.mark.parametrize("repaired", [True, False])
def test_a_heavy_indicator_beside_two_tensors_takes_no_chunked_build(monkeypatch, repaired, scop):
    """The indicator alone is recentred; every other column keeps its raw moments.

    Its level holds 55% of the weight, so ``kappa^2 = 1 / 0.45 > 2``, and the
    50 x 50 tensor grids exceed the anchor-centred route's support cap.  Three
    builds of one fit: none takes the chunked pass, and each forms the raw
    moments once.  Without ``scop`` the packed factored rung forms them, and
    the raw-moment rung no longer recomputes them (the parent formed them
    twice a build); with it the raw-moment rung forms them, and its refusal
    no longer latches while the repair serves (latched, builds two and three
    formed none and took the chunked pass).  ``repaired=False`` is the
    mutation check: without the repair every build takes the chunked pass.
    """
    rng = np.random.default_rng(20261006)
    n = 20_000
    dm = _design(rng, n, (50, 50), 0.55, scop=scop)
    assert dm.group_matrices[0].B_unique.shape[0] ** 2 > centered._MAX_PACKED_HIST_CELLS, (
        "the anchor-centred route must decline"
    )
    chunked, moments = [], []
    monkeypatch.setattr(
        centered_system, "centered_gram_rhs", _counted(chunked, centered_system.centered_gram_rhs)
    )
    plan = type(dm.execution_plan)
    monkeypatch.setattr(
        plan, "_moments_prevalidated", _counted(moments, plan._moments_prevalidated)
    )
    if not repaired:
        monkeypatch.setattr(centered_system, "column_local_centering", lambda **_: None)
    state, profile, builds = TabmatCenteringState(), {}, 3
    for _ in range(builds):
        build_centered_system(
            dm=dm,
            W=rng.uniform(0.5, 2.0, n),
            z_off=rng.normal(size=n),
            penalty=np.zeros((dm.p, dm.p)),
            tabmat_state=state,
            profile=profile,
        )
    if not repaired:
        assert len(chunked) == builds
        return
    assert len(moments) == builds, "one raw-moment pass a build"
    assert chunked == []
    assert profile["centered_column_local_hits"] == builds
    assert profile["centered_column_local_columns"] == builds, "only the indicator is recentred"


@pytest.mark.parametrize("route", ["certified", "anchor_support"])
def test_the_repair_is_reached_only_where_the_chunked_pass_was(monkeypatch, route):
    """Where the certificate passes, or the anchor-centred route serves, nothing changes.

    ``certified``: no level above half the weight, so the factored rung's
    raw moments serve.  ``anchor_support``: the indicator fails, but 6 x 5
    tensor grids fit the anchor-centred route, which serves as before.  The
    repair is never called.  Mutation check: calling it ahead of the
    anchor-centred route (inside ``packed_centered_gram_rhs``) fails the
    second case.
    """
    rng = np.random.default_rng(20261007)
    n = 4_000
    dm = _design(rng, n, (6, 5), 0.3 if route == "certified" else 0.55)
    anchored = []
    monkeypatch.setattr(
        centered,
        "_anchor_support_gram_rhs",
        _counted(anchored, centered._anchor_support_gram_rhs),
    )
    monkeypatch.setattr(
        centered_system,
        "column_local_centering",
        lambda **_: pytest.fail("the repair must not replace a route that serves"),
    )
    build_centered_system(
        dm=dm,
        W=rng.uniform(0.5, 2.0, n),
        z_off=rng.normal(size=n),
        penalty=np.zeros((dm.p, dm.p)),
        tabmat_state=TabmatCenteringState(),
    )
    assert len(anchored) == int(route == "anchor_support")


def _gamma(k: int) -> float:
    u = 2.0**-53
    return k * u / (1 - k * u)


def test_recentred_columns_match_the_two_pass_gram_within_their_bound(monkeypatch):
    """Recentred rows against the exact two-pass Gram; admitted entries bitwise the raw rung's.

    Two groups hold failing columns: a 2-level categorical whose first level
    holds all but 1e-9 of the weight (``kappa^2 = 1e9``; its light level is
    admitted), and a tensor whose margins sit at 10.  An admitted tensor and
    spline keep their raw-moment entries bit for bit, and so does the light
    level.  Each recentred entry ``(j, k)`` is a sum over rows of ``W c_j
    y_k``, ``c_j`` the anchor-centred row, ``y_k`` the other column's centred
    row or its raw value less its mean; with ``A_j(r) = (|v(r) - v*| +
    |shift|) |T|`` (recentred) or ``|v(r)| |T|`` (admitted) majorising the
    factors each kernel multiplies, and ``K = n + 2 n_s + 4 q + 10``
    roundings a term (``n_s`` support rows, ``q`` raw width: support
    centring and projection, weighting, row sums, the partner's projection,
    the mean correction), ``|C - C*|_jk <= 2 gamma_K sum_r W_r A_j(r)
    (A_k(r) + |m_k|)`` (Higham 2002, sec. 3.1); the factor 2 covers the
    centre's own rounding, which enters only at second order.  The raw
    subtraction rounds the indicator's Gram at ``u sum W``, about 1e9 times
    that bound on its row.
    """
    from superglm._group_matrix import _column_local_centering as local
    from superglm._group_matrix._group_matrix_centered import RawMomentRejection

    rng = np.random.default_rng(20261008)
    n = 240
    levels = rng.choice([0, 1, -1], size=n, p=[0.8, 0.1, 0.1])
    bins = rng.integers(0, 7, size=n)
    groups = [
        _tensor(rng, n, (6, 5), (3, 2), 4, 1),
        CategoricalGroupMatrix(levels, 2),
        _tensor(rng, n, (5, 4), (2, 3), 4, 2, offset=10.0),
        DiscretizedSSPGroupMatrix(rng.normal(size=(7, 4)), rng.normal(size=(4, 3)), bins),
    ]
    dm = DesignMatrix(groups, n, 13)
    W = np.where(levels == 0, rng.uniform(0.5, 2.0, n), 1e-9 * rng.uniform(0.5, 2.0, n))
    z = rng.normal(size=n)
    S = float(np.sum(W))
    weighted_z = W * (z - float(np.dot(W, z)) / S)
    moments = dm.execution_plan._moments_prevalidated(W, rhs=(weighted_z,), include_xtw=True)
    rejection = RawMomentRejection()
    rejection.record(
        "factored",
        raw_gram=moments.gram,
        xtw=moments.xtw,
        raw_rhs=moments.xt_rhs[0],
        weighted_z=weighted_z,
    )
    mean, gram, rhs, columns = local.column_local_centering(dm=dm, W=W, rejected=rejection, sum_w=S)
    raw_mean = moments.xtw / S
    raw = moments.gram - np.outer(moments.xtw, raw_mean)
    raw = 0.5 * (raw + raw.T)
    certified = centered._raw_centering_admitted(raw_mean, np.sqrt(np.diag(raw) / S))
    owner = np.repeat(np.arange(4), [4, 2, 4, 3])
    assert set(owner[~certified]) == {1, 2}, "the indicator and the offset tensor must fail"
    assert certified[5], "the light level must be admitted"
    recentred = ~certified
    assert columns == int(np.count_nonzero(recentred))

    # Admitted entries: bitwise the raw rung's subtraction.
    kept = certified
    np.testing.assert_array_equal(gram[np.ix_(kept, kept)], raw[np.ix_(kept, kept)])
    np.testing.assert_array_equal(mean[kept], raw_mean[kept])
    np.testing.assert_array_equal(
        rhs[kept], (moments.xt_rhs[0] - raw_mean * float(np.sum(weighted_z)))[kept]
    )

    # Exact rows, their exact two-pass Gram and right-hand side.
    supports = [local._compact_support(group) for group in groups]
    exact = np.empty((n, 13), dtype=object)
    majorant = np.empty((n, 13), dtype=np.float64)
    centre_majorant = np.zeros(13)
    column = 0
    for values, codes, transform in supports:
        T = np.eye(values.shape[1]) if transform is None else transform
        width = T.shape[1]
        rows = np.abs(values)
        mass = np.bincount(codes, weights=W, minlength=len(values))
        anchor = values[int(np.argmax(mass))]
        shift = mass @ (values - anchor) / S
        anchored = np.abs(values - anchor) + np.abs(shift)
        for j in range(column, column + width):
            source = anchored if recentred[j] else rows
            majorant[:, j] = (source @ np.abs(T[:, j - column]))[codes]
            centre_majorant[j] = (np.abs(anchor) + np.abs(shift)) @ np.abs(T[:, j - column])
        for b in np.unique(codes):
            value = [
                sum(Fraction(values[b, q]) * Fraction(T[q, j]) for q in range(T.shape[0]))
                for j in range(width)
            ]
            exact[codes == b, column : column + width] = value
        column += width
    weights = [Fraction(w) for w in W]
    total = sum(weights)
    centre = [
        sum(w * x for w, x in zip(weights, exact[:, j], strict=True)) / total for j in range(13)
    ]
    K = n + 2 * max(len(v) for v, _, _ in supports) + 4 * max(v.shape[1] for v, _, _ in supports)
    K += 10
    for j in np.flatnonzero(recentred):
        c_j = [x - centre[j] for x in exact[:, j]]
        for k in range(13):
            exact_jk = sum(
                w * c * (x - centre[k]) for w, c, x in zip(weights, c_j, exact[:, k], strict=True)
            )
            bound = (
                2
                * _gamma(K)
                * float(np.sum(W * majorant[:, j] * (majorant[:, k] + abs(raw_mean[k]))))
            )
            assert abs(Fraction(gram[j, k]) - exact_jk) <= Fraction(bound), (j, k)
        exact_rhs = sum(Fraction(wz) * c for wz, c in zip(weighted_z, c_j, strict=True))
        bound = 2 * _gamma(K) * float(np.sum(np.abs(weighted_z) * majorant[:, j]))
        assert abs(Fraction(rhs[j]) - exact_rhs) <= Fraction(bound), j
        bound = 2 * _gamma(K) * (centre_majorant[j] + float(np.sum(W * majorant[:, j])) / S)
        assert abs(Fraction(mean[j]) - centre[j]) <= Fraction(bound), j

    # A failing column whose group has no compact support declines to the chunked pass.
    monkeypatch.setattr(local, "_compact_support_rows", lambda group: None)
    assert local.column_local_centering(dm=dm, W=W, rejected=rejection, sum_w=S) is None
