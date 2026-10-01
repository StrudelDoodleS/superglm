"""Exact-reference tests of the nested chain Schur factor (fix D; spec §8-§9).

Every protocol method of ``NestedSchurFactor`` and ``ProfiledNestedSchurFactor``
is checked against an exact rational assembly of ``H`` from the same float64
rows, with its exact inverse and determinant.  Bounds are derived per
quantity from the dimensions, ``eps`` and the Jacobi-scaled condition number
``kappa_s(Q)`` of the exact Schur complement in the factor's centred
coordinates: ``gamma_tree = n_tree eps`` for the recursions that sum
non-negative terms, ``gamma_Q = (n + k + q + 10) eps`` for the PSD-sum ``Q``
(``_bounds``) and ``gamma_border = q kappa_s(Q) gamma_Q`` for one pass
through ``Q^-1``, with small integer multiples counting the passes.  The
bounds carry no column offset: the leaf statistics are formed about the
global centre ``c`` (``_center``), so their rounding scales with ``|x - c|``,
which F9 checks with a column offset by 1e7.
Positive quantities are measured relative to their value, signed quantities
against a Cauchy-Schwarz scale, never against ``sum |parts|``.  Solutions of
``H x = r`` are certified by their residual on every fixture and compared
forward only on the well-conditioned ones; the stress fixtures test the
stable observables (§8).
"""

from __future__ import annotations

import math
import tracemalloc
from dataclasses import replace
from fractions import Fraction
from functools import cache

import numpy as np
import pytest

from superglm.solvers._structured.nested import (
    NestedDataOperator,
    NestedLeafStatistics,
    NestedPenalizedOperator,
    NestedSchurFactor,
    NestedTree,
    ProfiledNestedSchurFactor,
    _weighted_scatter,
)
from superglm.solvers._structured.operators import (
    BlockSymmetricOperator,
    CenteredBlockOperator,
    LowRankSymmetricOperator,
    SumBlockOperator,
    SymmetricBlockOperator,
)
from superglm.solvers.hessian_factor import DerivativeCrossTraceFactor, HessianFactor
from superglm.types import PenaltyComponent

EPS = np.finfo(np.float64).eps
CHAIN = ("make", "model", "variant", "trim")


# ---------------------------------------------------------------- fixtures
def _make_fixture(
    seed,
    sizes,
    n,
    lam,
    *,
    wscale=1.0,
    zero_leaves=(3, 7),
    extra=(1, 1, 2),
    crossed=0,
    duplicate=None,
    width=5,
    normal_scale=1.0,
    offset=0.0,
):
    """Chain coarse to fine with ``sizes`` observed and ``extra`` unobserved nodes per level.

    The border is the intercept, a normal (times ``normal_scale``), a column
    with mean ``10 + offset``, a leaf attribute and a root attribute (``width=5``);
    ``crossed`` adds that many indicator columns of a crossed factor with its
    own identity penalty; ``duplicate`` repeats that border column
    unpenalised; ``width=0`` is the chain-only design.  Ported from the
    review's ``crit.make_fixture`` with the same seeds.
    """
    rng = np.random.default_rng(seed)
    depth = len(sizes)
    parents = [None]
    for j in range(1, depth):
        kp, kc = sizes[j - 1], sizes[j]
        par = np.concatenate([np.arange(kp), rng.integers(0, kp, kc - kp)])
        par[kc - 1] = 0
        rng.shuffle(par)
        counts = np.bincount(par, minlength=kp)
        if counts[0] > 1 and kp > 1:
            par[np.flatnonzero(par == 0)[1:]] = 1
        parents.append(np.concatenate([par, np.zeros(extra[j], dtype=int)]).astype(np.intp))
    K = [sizes[j] + extra[j] for j in range(depth)]
    parents[0] = np.full(K[0], -1, dtype=np.intp)
    leaf = rng.integers(0, sizes[-1], n)
    w = np.exp(rng.normal(size=n)) * wscale
    w[np.isin(leaf, zero_leaves)] = 0.0
    root_of_leaf = np.arange(K[-1])
    for j in range(depth - 1, 0, -1):
        root_of_leaf = parents[j][root_of_leaf]
    leaf_attr = rng.normal(size=K[-1])
    root_attr = rng.normal(size=K[0])
    columns = [
        np.ones(n),
        normal_scale * rng.normal(size=n),
        offset + 10.0 + 3.0 * rng.normal(size=n),
        leaf_attr[leaf],
        root_attr[root_of_leaf[leaf]],
    ][:width]
    S_b = np.zeros((width, width))
    Omega_b = np.zeros((width, width))
    if width:
        S_b[1:3, 1:3] = np.array([[2.0, 0.5], [0.5, 1.0]])
        Omega_b[1:3, 1:3] = np.array([[1.0, 0.25], [0.25, 0.5]])
    components = []
    if crossed:
        codes = rng.integers(0, crossed, n)
        columns.append((codes[:, None] == np.arange(crossed)[None, :]).astype(np.float64))
        start = S_b.shape[0]
        S_b = np.pad(S_b, (0, crossed))
        Omega_b = np.pad(Omega_b, (0, crossed))
        S_b[range(start, start + crossed), range(start, start + crossed)] = 0.4
        components.append(("crossed", slice(start, start + crossed), 0.4))
    if duplicate is not None:
        # unpenalised on both copies: a penalty on either breaks the aliasing
        columns.append(columns[duplicate])
        S_b = np.pad(S_b, (0, 1))
        Omega_b = np.pad(Omega_b, (0, 1))
        S_b[duplicate, :] = S_b[:, duplicate] = 0.0
    X = np.column_stack(columns) if columns else np.empty((n, 0))
    a = rng.normal(size=n) * w
    a2 = np.random.default_rng(99).normal(size=n) * w
    return dict(
        K=K,
        depth=depth,
        parents=parents,
        leaf=leaf,
        w=w,
        X=X,
        S_b=S_b,
        Omega_b=Omega_b,
        lam=np.asarray(lam, dtype=np.float64),
        a=a,
        a2=a2,
        border_components=components,
        duplicate=duplicate,
    )


_DUPLICATED = dict(seed=8, sizes=[3, 7, 16], n=600)
FIXTURES = {
    "F1": dict(seed=1, sizes=[3, 7, 16], n=600, lam=[0.7, 0.05, 3.0]),
    "F2": dict(seed=2, sizes=[3, 7, 16], n=600, lam=[1e-7, 1e-7, 1e-7], wscale=1e4),
    "F3": dict(seed=3, sizes=[3, 7, 16], n=600, lam=[1e8, 1e8, 1e8]),
    "F4": dict(seed=4, sizes=[3, 7, 16], n=600, lam=[1e8, 1e-7, 1.0], wscale=1e4),
    "F5": dict(seed=5, sizes=[3, 7, 16], n=600, lam=[1e-7, 1e8, 1e-3], wscale=1e-3),
    "F6": dict(seed=6, sizes=[2, 4, 8, 14], n=500, lam=[1e-4, 2.0, 1e-6, 5.0], extra=(1, 0, 1, 1)),
    "F7": dict(seed=1, sizes=[3, 7, 16], n=600, lam=[0.7, 0.05, 3.0], crossed=3),
    # F4's penalties with the mean-10 column offset by 1e7: kappa_s of the raw Q
    # is ~1e12 larger than the centred one the factor works on
    "F9": dict(seed=4, sizes=[3, 7, 16], n=600, lam=[1e8, 1e-7, 1.0], wscale=1e4, offset=1e7),
    # a duplicated leaf attribute (zero within-leaf scatter: every entry of its
    # Schur rows is on the lambda scale) and a duplicated normal column, plain
    # and at scale 1e3 (within-leaf scatter on the raw weight scale, 1e13
    # above the intercept pivot at lambda = 1e-7)
    "F8a": dict(**_DUPLICATED, lam=[1e-4] * 3, wscale=1e2, duplicate=3),
    "F8b": dict(**_DUPLICATED, lam=[1e-5] * 3, wscale=1e2, duplicate=3),
    "F8c": dict(**_DUPLICATED, lam=[1e-7] * 3, wscale=1e4, duplicate=3),
    "F8d": dict(**_DUPLICATED, lam=[1e-7] * 3, wscale=1e4, duplicate=1),
    "F8e": dict(**_DUPLICATED, lam=[1e-7] * 3, wscale=1e4, duplicate=1, normal_scale=1e3),
    "F8f": dict(**_DUPLICATED, lam=[1e-9] * 3, wscale=1e6, duplicate=1),
}
MAIN = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F9"]
TRUNCATED = ["F8a", "F8b", "F8c", "F8d", "F8e", "F8f"]
PROFILED = ["F1", "F2", "F4", "F6", "F7", "F9"]
# Fixtures whose H is well conditioned enough for forward solve comparisons.
WELL_CONDITIONED = {"F1", "F7"}


def _offsets(K):
    return np.concatenate([[0], np.cumsum(K)]).astype(np.intp)


def _incidence(fx):
    """Leaf-to-node incidence ``M`` (K_leaf x k) as a dense 0/1 array."""
    K, depth, parents = fx["K"], fx["depth"], fx["parents"]
    off = _offsets(K)
    M = np.zeros((K[-1], off[-1]))
    node = np.arange(K[-1])
    for j in range(depth - 1, -1, -1):
        M[np.arange(K[-1]), off[j] + node] = 1.0
        if j > 0:
            node = parents[j][node]
    return M


# ------------------------------------------------- production objects
def _bincols(index, values, size):
    columns = [
        np.bincount(index, weights=values[:, i], minlength=size) for i in range(values.shape[1])
    ]
    return np.stack(columns, axis=1) if columns else np.zeros((size, 0))


def _center(fx):
    """The builder's global centre, the rule of ``border_center``: the mean of each
    column whose offset exceeds its spread, 0 elsewhere, on the crossed one-hot
    columns and on the intercept column 0."""
    mean, spread = fx["X"].mean(axis=0), fx["X"].std(axis=0)
    center = np.where(np.abs(mean) > spread, mean, 0.0)
    center[:1] = 0.0
    for name, columns, _ in fx["border_components"]:
        if name == "crossed":
            center[columns] = 0.0
    return center


def _leaf_statistics(fx, weights, mean=None, *, shifted=True):
    """Leaf statistics of ``weights`` from the dense rows about ``mean`` (the data pass when None).

    The rows are centred on ``_center`` as the production row pass centres them.
    """
    center = _center(fx)
    X, leaf, K = fx["X"] - center, fx["leaf"], fx["K"][-1]
    total = np.bincount(leaf, weights=weights, minlength=K)
    if mean is None:
        order = np.argsort(leaf, kind="stable")
        first = order[np.r_[True, leaf[order][1:] != leaf[order][:-1]]]
        reference = np.zeros((K, X.shape[1]))
        if shifted:
            reference[leaf[first]] = X[first]
        shift = _bincols(leaf, weights[:, None] * (X - reference[leaf]), K)
        scaled = np.divide(
            shift, total[:, None], out=np.zeros_like(shift), where=total[:, None] != 0.0
        )
        mean = np.where(total[:, None] != 0.0, reference + scaled, 0.0)
        deviation = None
    else:
        deviation = _bincols(leaf, weights[:, None] * (X - mean[leaf]), K)
    residual = X - mean[leaf]
    within = (residual * weights[:, None]).T @ residual
    # the row pass's weight error scale: each row's own weight when none is
    # negative, max |w| on every weighted row otherwise
    signed = np.any(weights < 0.0)
    error = np.max(np.abs(weights)) * (weights != 0.0) if signed else weights
    absolute = error @ residual**2
    return NestedLeafStatistics(
        weight=total,
        mean=mean,
        within=0.5 * (within + within.T),
        absolute=absolute,
        center=center,
        deviation=deviation,
    )


def _tree(fx):
    return NestedTree(tuple(fx["K"]), tuple(fx["parents"]))


def _operator(fx, tree, weights, mean=None, *, columns=None):
    """Leaf-form operator in the exact ordering (nodes first, then the border)."""
    stats = _leaf_statistics(fx, weights, mean)
    if columns is not None:
        stats = NestedLeafStatistics(
            weight=stats.weight,
            mean=stats.mean[:, columns],
            within=stats.within[np.ix_(columns, columns)],
            absolute=stats.absolute[columns],
            center=stats.center[columns],
            deviation=None if stats.deviation is None else stats.deviation[:, columns],
        )
    k, q = tree.n_nodes, stats.width
    return NestedDataOperator(
        tree=tree, leaf=stats, small_indices=np.arange(k, k + q), structured_indices=np.arange(k)
    )


def _penalized(fx, data, columns=None):
    S_b = fx["S_b"] if columns is None else fx["S_b"][np.ix_(columns, columns)]
    return NestedPenalizedOperator(
        data=data,
        node_penalty=tuple(np.full(K, lam) for K, lam in zip(fx["K"], fx["lam"], strict=True)),
        border_penalty=S_b,
    )


def _factor(fx, *, intercept=True, **kwargs):
    tree = _tree(fx)
    data = _operator(fx, tree, fx["w"])
    return NestedSchurFactor(
        _penalized(fx, data),
        chain_group_names=CHAIN[: fx["depth"]],
        chain_group_indices=tuple(range(fx["depth"])),
        intercept=intercept,
        **kwargs,
    )


def _component(name, start, stop, omega=None, index=0):
    return PenaltyComponent(
        name=name,
        group_name=name,
        group_index=index,
        group_sl=slice(start, stop),
        omega_raw=None,
        omega_ssp=omega,
        penalty_kind="identity" if omega is None else "dense",
    )


def _components(fx, k, border_shift=0):
    """Level components, the dense border penalty and any crossed identity block.

    Nodes come first in both layouts; ``border_shift=-1`` gives the slope
    coordinates, whose border columns drop the intercept.
    """
    off = _offsets(fx["K"])
    levels = [_component(f"level{j}", off[j], off[j + 1], index=j) for j in range(fx["depth"])]
    border = _component(
        "border",
        k + 1 + border_shift,
        k + 3 + border_shift,
        fx["Omega_b"][1:3, 1:3],
        index=fx["depth"],
    )
    crossed = [
        _component(
            name, k + sl.start + border_shift, k + sl.stop + border_shift, index=fx["depth"] + 1 + i
        )
        for i, (name, sl, _) in enumerate(fx["border_components"])
    ]
    return levels, border, crossed


# ------------------------------------------------------- exact references
# Exact matrices are pairs ``(N, d)``: an object array of Python integers over
# one integer denominator.  Every float is a dyadic rational, so products and
# sums stay integer arithmetic and only the final ``int / int`` rounds, and it
# rounds correctly.  The inverse is the fraction-free Gauss-Jordan
# elimination, whose final left block is ``det I`` and right block ``adj``.


def _frac(x):
    return Fraction(float(x))


def _fraction_gram(fx, weights, M):
    """Exact ``[M[leaf] | X]' diag(weights) [M[leaf] | X]`` as Fractions, nodes first."""
    K, X, leaf = fx["K"], fx["X"], fx["leaf"]
    k, q = M.shape[1], X.shape[1]
    total = [Fraction(0)] * K[-1]
    cross = [[Fraction(0)] * q for _ in range(K[-1])]
    A = [[Fraction(0)] * q for _ in range(q)]
    for r in range(len(weights)):
        wr = _frac(weights[r])
        if wr == 0:
            continue
        xr = [_frac(v) for v in X[r]]
        total[leaf[r]] += wr
        for i in range(q):
            cross[leaf[r]][i] += wr * xr[i]
            for j in range(i, q):
                A[i][j] += wr * xr[i] * xr[j]
    for i in range(q):
        for j in range(i):
            A[i][j] = A[j][i]
    p = k + q
    gram = [[Fraction(0)] * p for _ in range(p)]
    for u in range(k):
        leaves_u = np.flatnonzero(M[:, u])
        for v in range(k):
            common = leaves_u[M[leaves_u, v] > 0]
            gram[u][v] = sum((total[ell] for ell in common), Fraction(0))
        for i in range(q):
            gram[u][k + i] = gram[k + i][u] = sum((cross[ell][i] for ell in leaves_u), Fraction(0))
    for i in range(q):
        for j in range(q):
            gram[k + i][k + j] = A[i][j]
    return gram


def _fraction_penalty(fx, k, q):
    p = k + q
    S = [[Fraction(0)] * p for _ in range(p)]
    lamnode = np.concatenate([np.full(K, lam) for K, lam in zip(fx["K"], fx["lam"], strict=True)])
    for u in range(k):
        S[u][u] = _frac(lamnode[u])
    for i in range(q):
        for j in range(q):
            S[k + i][k + j] = _frac(fx["S_b"][i, j])
    return S


def _fraction_sum(A, B):
    return [[a + b for a, b in zip(ra, rb, strict=True)] for ra, rb in zip(A, B, strict=True)]


def _pair(rows):
    """``(N, d)`` of a matrix of Fractions or floats over the lcm of their denominators."""
    fractions = [[x if isinstance(x, Fraction) else _frac(x) for x in row] for row in rows]
    den = 1
    for row in fractions:
        for x in row:
            den = math.lcm(den, x.denominator)
    scaled = [[x * den for x in row] for row in fractions]
    assert all(x.denominator == 1 for row in scaled for x in row)
    return np.array([[int(x) for x in row] for row in scaled], dtype=object), den


def _inverse(N):
    """Return ``(adj, det)`` of an integer matrix by fraction-free Gauss-Jordan, ``None`` if singular."""
    n = len(N)
    M = [[int(v) for v in row] + [int(i == j) for j in range(n)] for i, row in enumerate(N)]
    previous = 1
    for k in range(n):
        pivot = M[k][k]
        if pivot == 0:
            return None
        for i in range(n):
            if i != k:
                factor = M[i][k]
                M[i] = [
                    (a * pivot - factor * b) // previous for a, b in zip(M[i], M[k], strict=True)
                ]
        previous = pivot
    return np.array([row[n:] for row in M], dtype=object), previous


def _inverse_pair(pair):
    """Exact inverse of ``N / d`` as a pair, or ``None`` when singular."""
    inverse = _inverse(pair[0])
    if inverse is None:
        return None
    adj, det = inverse
    return adj * pair[1], det


def _log_det(pair):
    """``log det(N / d)`` for a positive definite pair."""
    _, det = _inverse(pair[0])
    return math.log(det) - len(pair[0]) * math.log(pair[1])


def _product(left, right):
    return left[0] @ right[0], left[1] * right[1]


def _plus(left, right):
    return left[0] * right[1] + right[0] * left[1], left[1] * right[1]


def _block(pair, rows, cols):
    return pair[0][np.ix_(list(rows), list(cols))], pair[1]


def _floats(pair):
    return np.array([[int(v) / pair[1] for v in row] for row in pair[0]], dtype=np.float64)


def _diagonal(pair):
    return np.array([int(pair[0][i, i]) / pair[1] for i in range(len(pair[0]))])


def _trace_product(left, right):
    """``tr(L R) = sum L_ij R_ji`` exactly, then one rounding."""
    return int((left[0] * right[0].T).sum()) / (left[1] * right[1])


def _row_products(left, right):
    """``sum_j L_ij R_ji`` per row exactly, then one rounding per entry."""
    sums = (left[0] * right[0].T).sum(axis=1)
    return np.array([int(v) / (left[1] * right[1]) for v in sums])


def _frobenius(pair):
    return int((pair[0] * pair[0]).sum()) / pair[1] ** 2


def _selector_pair(p, indices, scale=1.0):
    E = np.zeros((p, p))
    E[list(indices), list(indices)] = scale
    return _pair(E)


def _embed_pair(p, block, indices):
    E = np.zeros((p, p))
    E[np.ix_(list(indices), list(indices))] = block
    return _pair(E)


def _scaled_condition(Q):
    d = 1.0 / np.sqrt(np.diag(Q))
    return float(np.linalg.cond(d[:, None] * Q * d[None, :]))


def _bounds(fx, q, kappa_s):
    """``gamma_tree`` and ``gamma_border`` of the module docstring.

    The PSD-sum ``Q`` adds the within-leaf scatter, one product over all ``n``
    rows in this file's builder (``n eps`` componentwise against ``sqrt(W_ii
    W_jj)``), the between-child terms over the ``k`` nodes and the shifted
    means, so ``gamma_Q = (n + k + q + 10) eps``; ``n_tree`` covers only the
    per-node recursions.  What the constants assume: each mean is known to
    ``eps |x - c|`` with ``c`` the global centre, so ``gamma_Q`` holds when a
    column's ``max |x - c|`` is of the order of its spread (the offset of a
    raw column cancels in ``x - c``; its range does not), and ``kappa_s`` is
    that of the centred ``Q`` the factor decides rank on.
    """
    rows_per_leaf = np.bincount(fx["leaf"], minlength=fx["K"][-1]).max()
    fan = max(np.bincount(fx["parents"][j]).max() for j in range(1, fx["depth"]))
    n_tree = int(rows_per_leaf + fx["depth"] * (fan + 4))
    gamma_tree = n_tree * EPS
    gamma_Q = (len(fx["leaf"]) + sum(fx["K"]) + q + 10) * EPS
    return gamma_tree, q * kappa_s * gamma_Q


def _exact_pivots(H_frac, fx, k):
    """Exact LDL pivots of the tree block in the leaves-first order."""
    K, depth = fx["K"], fx["depth"]
    off = _offsets(K)
    order = [u for j in range(depth - 1, -1, -1) for u in range(off[j], off[j + 1])]
    T = [[H_frac[u][v] for v in order] for u in order]
    pivots = {}
    for c in range(len(order)):
        pivots[order[c]] = T[c][c]
        for r in range(c + 1, len(order)):
            if T[r][c] != 0:
                f = T[r][c] / T[c][c]
                for s in range(c + 1, len(order)):
                    if T[c][s] != 0:
                        T[r][s] -= f * T[c][s]
    return np.array([float(pivots[u]) for u in range(k)])


@cache
def _case(name):
    """Fixture and every exact reference of one fixture (both layouts)."""
    fx = _make_fixture(**FIXTURES[name])
    K, depth = fx["K"], fx["depth"]
    off = _offsets(K)
    M = _incidence(fx)
    k, q = M.shape[1], fx["X"].shape[1]
    p = k + q
    H_frac = _fraction_sum(_fraction_gram(fx, fx["w"], M), _fraction_penalty(fx, k, q))
    H = _pair(H_frac)
    S = _pair(_fraction_penalty(fx, k, q))
    O1, O2 = _pair(_fraction_gram(fx, fx["a"], M)), _pair(_fraction_gram(fx, fx["a2"], M))
    refs = dict(fx=fx, k=k, q=q, p=p, H=H, H_frac=H_frac, H_f=_floats(H), S=S, O1=O1)
    Hinv = _inverse_pair(H)
    refs["Hinv"] = Hinv
    if Hinv is None:
        return refs
    Hinv_f = _floats(Hinv)
    refs["Hinv_f"] = Hinv_f
    refs["logdet"] = _log_det(H)
    border = range(k, p)
    # the Schur complement in the factor's centred coordinates: R' Q R with
    # R = I - e_0 c' on the border (the intercept is border column 0)
    R = _pair(np.eye(q) - np.outer(np.eye(q)[0], _center(fx)))
    Q = _product(_product((R[0].T, R[1]), _inverse_pair(_block(Hinv, border, border))), R)
    refs["Q"] = _floats(Q)
    refs["kappa_s"] = _scaled_condition(refs["Q"])
    # the intercept-profiled Q the factor decides rank on (the super-root
    # eliminates the intercept first): the inverse of Q^-1's rest block
    rest = range(1, q)
    refs["Q_profiled"] = _floats(_inverse_pair(_block(_inverse_pair(Q), rest, rest)))
    refs["kappa_profiled"] = _scaled_condition(refs["Q_profiled"])
    refs["gamma_tree"], refs["gamma_border"] = _bounds(fx, q, refs["kappa_s"])
    refs["pivots"] = _exact_pivots(H_frac, fx, k)
    everything = range(p)
    levels = [range(off[j], off[j + 1]) for j in range(depth)]
    refs["frob"] = [_frobenius(_block(Hinv, lvl, lvl)) for lvl in levels]
    refs["pairs"] = {
        (i, j): _frobenius(_block(Hinv, levels[i], levels[j]))
        for i in range(depth)
        for j in range(i, depth)
    }
    HOmega = _product(Hinv, _embed_pair(p, fx["Omega_b"], border))
    refs["level_border"] = [
        _trace_product(_block(HOmega, lvl, everything), _block(Hinv, everything, lvl))
        for lvl in levels
    ]
    refs["border_border"] = _trace_product(HOmega, HOmega)
    crossed = {
        name: _selector_pair(p, range(k + sl.start, k + sl.stop), sc)
        for name, sl, sc in fx["border_components"]
    }
    HO1, HO2 = _product(Hinv, O1), _product(Hinv, O2)
    refs["tr11"], refs["tr22"] = _trace_product(HO1, HO1), _trace_product(HO2, HO2)
    refs["tr12"] = _trace_product(HO1, HO2)
    refs["trace_O1"] = float(np.sum(_diagonal(HO1)))
    refs["diag_O1"] = _diagonal(HO1)
    refs["scale_diag_O1"] = np.sqrt(np.diag(Hinv_f) * _row_products(O1, HO1))
    refs["level_O1"] = [
        _trace_product(_block(HO1, lvl, everything), _block(Hinv, everything, lvl))
        for lvl in levels
    ]
    refs["border_O1"] = _trace_product(HOmega, HO1)
    refs["crossed_O1"] = {
        name: _trace_product(_product(Hinv, E), HO1) for name, E in crossed.items()
    }
    HS = _product(Hinv, S)
    refs["edf"] = 1.0 - _diagonal(HS)
    refs["edf1"] = 1.0 - 2.0 * _diagonal(HS) + _row_products(HS, HS)
    S_f = _floats(S)
    refs["edf1_scale"] = (
        1.0
        + 2.0 * np.abs(_diagonal(HS))
        + np.sum(np.abs(Hinv_f @ S_f @ Hinv_f) * np.abs(S_f).T, axis=1)
    )
    refs["square_O1"] = _row_products(HO1, HO1)
    HO1_f = _floats(HO1)
    refs["HO1_f"], refs["O1_f"] = HO1_f, _floats(O1)
    refs["square_O1_scale"] = np.sum(np.abs(HO1_f) * np.abs(HO1_f).T, axis=1)
    directions = [
        _plus(
            _product(Hinv, _selector_pair(p, levels[i], fx["lam"][i])),
            HO1 if i % 2 == 0 else HO2,
        )
        for i in range(depth)
    ]
    directions.append(HOmega)
    directions.extend(_product(Hinv, E) for E in crossed.values())
    refs["batched"] = np.array([[_trace_product(a, b) for b in directions] for a in directions])

    # Profiled references: slope coordinates are the nodes then border columns 1..q-1.
    slope = list(range(k)) + list(range(k + 1, p))
    pc = p - 1
    Hinv_c = _block(Hinv, slope, slope)
    refs["Hinv_c_f"] = _floats(Hinv_c)
    # H_c = H_ss - x x' / H_00, the Schur complement of the intercept
    H_ss, intercept_row = _block(H, slope, slope), H[0][k, slope]
    H_c = (H_ss[0] * H[0][k, k] - np.outer(intercept_row, intercept_row), H_ss[1] * H[0][k, k])
    refs["H_c_f"] = _floats(H_c)
    leaf_weight = np.bincount(fx["leaf"], weights=fx["w"], minlength=K[-1])
    xtw = np.concatenate([M.T @ leaf_weight, fx["X"][:, 1:].T @ fx["w"]])
    sum_w = float(np.sum(fx["w"]))
    refs["xtw"], refs["sum_w"] = xtw, sum_w
    refs["mean_x"] = xtw / sum_w
    # The profiled identities are exact for the augmented system's own centre,
    # the exact ratio of its intercept row; the factor's float mean_x only has
    # to be bitwise the operators' centre.  A reference centred at the float
    # mean instead would differ by |H_c^-1 delta| |y|, a 1/lambda-amplified
    # eps that is a property of the centring, not of the factor.
    exact_sum_w = H_frac[k][k]
    refs["logdet_c"] = refs["logdet"] - (
        math.log(exact_sum_w.numerator) - math.log(exact_sum_w.denominator)
    )
    mean, mean_den = _pair([[H_frac[k][j] / exact_sum_w for j in slope]])
    mean = mean[0]

    def centred(pair):
        """``P' O P`` for the float centre: ``O_c = O_ss - m x' - x m' + m m' O_00``."""
        N, den = pair
        sub = N[np.ix_(slope, slope)]
        row = N[k, slope]
        matrix = (
            sub * (mean_den * mean_den)
            - np.outer(mean, row) * mean_den
            - np.outer(row, mean) * mean_den
            + N[k, k] * np.outer(mean, mean)
        )
        return matrix, den * mean_den * mean_den

    O1c, O2c = centred(O1), centred(O2)
    Sc = _block(S, slope, slope)
    all_c = range(pc)
    HO1c, HO2c = _product(Hinv_c, O1c), _product(Hinv_c, O2c)
    refs["trace_O1_c"] = float(np.sum(_diagonal(HO1c)))
    # The factor's diagonal is (H_aug^-1 O_aug P)_jj with P built on the float
    # mean_x (the operators' centre): the exact-centre value less
    # (mean_x - mean)_j (H_aug^-1 y)_j, y = O_aug e_0.
    Hy = _product(Hinv, (O1[0][:, [k]], O1[1]))
    refs["diag_O1_c"] = _diagonal(HO1c) - np.array(
        [
            float(
                (_frac(refs["mean_x"][j]) - Fraction(int(mean[j]), mean_den))
                * Fraction(int(Hy[0][slope[j], 0]), Hy[1])
            )
            for j in range(pc)
        ]
    )
    refs["scale_diag_O1_c"] = np.sqrt(np.diag(refs["Hinv_c_f"]) * _row_products(O1c, HO1c))
    refs["tr11_c"], refs["tr22_c"] = _trace_product(HO1c, HO1c), _trace_product(HO2c, HO2c)
    refs["tr12_c"] = _trace_product(HO1c, HO2c)
    refs["level_O1_c"] = [
        _trace_product(_block(HO1c, lvl, all_c), _block(Hinv_c, all_c, lvl)) for lvl in levels
    ]
    border_c = range(k, pc)
    HOmega_c = _product(Hinv_c, _embed_pair(pc, fx["Omega_b"][1:, 1:], border_c))
    refs["border_O1_c"] = _trace_product(HOmega_c, HO1c)
    refs["border_border_c"] = _trace_product(HOmega_c, HOmega_c)
    refs["frob_c"] = [_frobenius(_block(Hinv_c, lvl, lvl)) for lvl in levels]
    HSc = _product(Hinv_c, Sc)
    refs["edf_c"] = 1.0 - _diagonal(HSc)
    refs["edf1_c"] = 1.0 - 2.0 * _diagonal(HSc) + _row_products(HSc, HSc)
    Sc_f = _floats(Sc)
    refs["edf1_scale_c"] = (
        1.0
        + 2.0 * np.abs(_diagonal(HSc))
        + np.sum(np.abs(refs["Hinv_c_f"] @ Sc_f @ refs["Hinv_c_f"]) * np.abs(Sc_f).T, axis=1)
    )
    basis = np.random.default_rng(17).normal(size=(pc, 2)) / np.sqrt(sum_w)
    refs["low_rank_basis"] = basis
    B = _pair(basis)
    R = _pair(np.array([[0.0, -sum_w], [-sum_w, 0.0]]))
    low = _product(_product(B, R), (B[0].T, B[1]))
    HLc = _product(Hinv_c, low)
    refs["trace_low_c"] = float(np.sum(_diagonal(HLc)))
    directions_c = [
        _plus(
            _product(Hinv_c, _selector_pair(pc, levels[i], fx["lam"][i])),
            HO1c if i % 2 == 0 else HO2c,
        )
        for i in range(depth)
    ]
    directions_c.append(HOmega_c)
    for name, sl, sc in fx["border_components"]:
        selector = _selector_pair(pc, range(k + sl.start - 1, k + sl.stop - 1), sc)
        directions_c.append(_product(Hinv_c, selector))
    refs["batched_c"] = np.array(
        [[_trace_product(a, b) for b in directions_c] for a in directions_c]
    )
    directions_c[1] = _plus(directions_c[1], HLc)
    refs["batched_c_low"] = np.array(
        [[_trace_product(a, b) for b in directions_c] for a in directions_c]
    )
    return refs


def _augmented_case(name):
    """The augmented factor on the exact ordering (border column 0 is the intercept)."""
    refs = _case(name)
    fx = refs["fx"]
    tree = _tree(fx)
    data = _operator(fx, tree, fx["w"])
    penalized = _penalized(fx, data)
    factor = NestedSchurFactor(
        penalized,
        chain_group_names=CHAIN[: fx["depth"]],
        chain_group_indices=tuple(range(fx["depth"])),
        intercept=True,
    )
    mean = data.leaf.mean
    O1 = _operator(fx, tree, fx["a"], mean)
    O2 = _operator(fx, tree, fx["a2"], mean)
    levels, border, crossed = _components(fx, refs["k"])
    return refs, factor, penalized, O1, O2, levels, border, crossed


def _profiled_case(name):
    """The profiled adapter built from the slope operator through ``augmented``."""
    refs = _case(name)
    fx = refs["fx"]
    tree = _tree(fx)
    columns = np.arange(1, refs["q"])
    data = _operator(fx, tree, fx["w"], columns=columns)
    penalized = _penalized(fx, data, columns)
    k = refs["k"]
    augmented = NestedSchurFactor(
        penalized.augmented(),
        chain_group_names=CHAIN[: fx["depth"]],
        chain_group_indices=tuple(range(fx["depth"])),
        intercept=True,
    )
    profiled = ProfiledNestedSchurFactor(
        augmented_factor=augmented, sum_w=refs["sum_w"], xtw=refs["xtw"], data_operator=data
    )
    # the data leaf means over every column; the slope operator carries their slope columns
    mean = _leaf_statistics(fx, fx["w"]).mean
    assert np.array_equal(data.leaf.mean, mean[:, columns])
    Xs = np.column_stack([_incidence(fx)[fx["leaf"]], fx["X"][:, 1:]])

    def centred(weights):
        raw = _operator(fx, tree, weights, mean, columns=columns)
        return CenteredBlockOperator(
            raw=raw, cross=Xs.T @ weights, total=float(np.sum(weights)), center=refs["mean_x"]
        )

    levels, border, crossed = _components(fx, k, border_shift=-1)
    return refs, profiled, data, centred(fx["a"]), centred(fx["a2"]), levels, border, crossed


def _certified_solve(operator, solution, rhs, H_f, Hinv_f, gamma):
    """Assert the residual is within its backward-error bound; return the forward allowance.

    ``|x - x_exact| <= |H^-1| (|H x - r| + rounding of the residual)``, with the
    residual's own rounding at ``gamma (|H| |x| + |r|)``.
    """
    residual = operator.matvec(solution) - rhs
    rounding = gamma * (np.abs(H_f) @ np.abs(solution) + np.abs(rhs))
    assert np.all(np.abs(residual) <= rounding)
    return np.abs(Hinv_f) @ (np.abs(residual) + rounding)


def _unit_columns(p, indices):
    unit = np.zeros((p, len(indices)))
    unit[indices, np.arange(len(indices))] = 1.0
    return unit


# ------------------------------------------------------------ the checks
@pytest.mark.parametrize("name", MAIN)
def test_pivots_logdet_and_inverse_diagonal(name):
    refs, factor, penalized, *_ = _augmented_case(name)
    gamma_tree, gamma_border = refs["gamma_tree"], refs["gamma_border"]
    k, q, p = refs["k"], refs["q"], refs["p"]
    assert isinstance(factor, HessianFactor) and isinstance(factor, DerivativeCrossTraceFactor)
    assert factor.rank == p and not factor.rank_truncated
    assert factor.shape == (p, p) and factor.backend == "structured"
    assert factor.dominant_group_name == factor.chain_group_names[-1]
    exact_min = refs["pivots"].min()
    assert abs(factor.minimum_local_diagonal - exact_min) <= gamma_tree * exact_min
    assert abs(factor.logdet() - refs["logdet"]) <= k * gamma_tree + q * gamma_border
    diagonal = factor.selected_inverse_diagonal(np.arange(p))
    exact = np.diag(refs["Hinv_f"])
    assert np.all(np.abs(diagonal - exact) <= (gamma_tree + gamma_border) * exact)
    reordered = factor.selected_inverse_diagonal(np.array([p - 1, 0, k]))
    assert np.array_equal(reordered, diagonal[[p - 1, 0, k]])
    # a mixed block by solves: two nodes per level and two border columns,
    # certified by its residual and compared forward where H allows it
    off = _offsets(refs["fx"]["K"])
    indices = np.concatenate([off[:-1], off[:-1] + 1, [k, k + 2]])
    unit = _unit_columns(p, indices)
    gamma = 4 * (gamma_tree + gamma_border)
    allowance = _certified_solve(
        penalized, factor.solve(unit), unit, refs["H_f"], refs["Hinv_f"], gamma
    )
    block = factor.selected_inverse_block(indices)
    exact_block = refs["Hinv_f"][np.ix_(indices, indices)]
    assert np.array_equal(block, block.T)
    allowance = allowance[indices]
    assert np.all(np.abs(block - exact_block) <= 0.5 * (allowance + allowance.T))
    if name in WELL_CONDITIONED:
        scale = np.sqrt(np.outer(np.diag(exact_block), np.diag(exact_block)))
        assert np.all(np.abs(block - exact_block) <= gamma * scale)
    assert np.all(factor.coefficient_estimable())
    # Every rank decision reads Q_s = D_s Q D_s (§3.7) of the intercept-profiled
    # Q, the super-root having eliminated the intercept (one-engine design
    # §3.1).  The PSD sum knows Q to gamma_Q sqrt(Q_ii Q_jj) per entry (§3.4,
    # §8), and the float D_s adds gamma_Q, so Q_s is within 2 gamma_Q plus three
    # roundings of the exact one.  Plain (unshifted) parent means leave eps |m|
    # noise in d = m_c - m_p on a column constant within the parent and break
    # this before any observable.
    exact_Q = refs["Q_profiled"]
    scale = 1.0 / np.sqrt(np.diag(exact_Q))
    gamma_Q = gamma_border / (q * refs["kappa_s"])
    Q_scaled_error = np.abs(factor._scaled_matrix - scale[:, None] * exact_Q * scale[None, :])
    assert np.all(Q_scaled_error <= 2 * gamma_Q + 3 * EPS)
    eigenvalues = factor.scaled_schur_eigenvalues()
    assert eigenvalues.shape == (q - 1,) and eigenvalues[0] > 0
    assert np.all(np.diff(eigenvalues) >= 0)
    # Q_s is known to gamma_Q per entry and lambda_min(Q_s) >= 1 / kappa_s, so
    # the eigenvalue ratio is within a few gamma_border of the exact kappa_s.
    kappa = refs["kappa_profiled"]
    ratio = eigenvalues[-1] / eigenvalues[0]
    assert abs(ratio - kappa) <= 4 * (q - 1) * kappa * gamma_Q * kappa
    # one path: no fallback, a Rump-verified full retained rank (§3.6)
    certificate = factor.border_certificate
    assert certificate.rank == q - 1 and certificate.decrements == 0
    assert np.isnan(certificate.trailing_bound) and not certificate.directions


@pytest.mark.parametrize("name", MAIN)
def test_solve_backward_and_forward_error(name):
    refs, factor, penalized, *_ = _augmented_case(name)
    gamma = 4 * (refs["gamma_tree"] + refs["gamma_border"])
    p = refs["p"]
    rhs = np.random.default_rng(5).normal(size=(p, 3))
    solution = factor.solve(rhs)
    assert solution.shape == rhs.shape
    allowance = _certified_solve(penalized, solution, rhs, refs["H_f"], refs["Hinv_f"], gamma)
    exact = refs["Hinv_f"] @ rhs
    assert np.all(np.abs(solution - exact) <= allowance)
    if name in WELL_CONDITIONED:
        assert np.all(np.abs(solution - exact) <= gamma * (np.abs(refs["Hinv_f"]) @ np.abs(rhs)))
    vector = factor.solve(rhs[:, 0])
    assert vector.shape == (p,)
    allowance = _certified_solve(penalized, vector, rhs[:, 0], refs["H_f"], refs["Hinv_f"], gamma)
    assert np.all(np.abs(vector - exact[:, 0]) <= allowance)


@pytest.mark.parametrize("name", MAIN)
def test_penalty_traces_and_pairs(name):
    refs, factor, _, _, _, levels, border, crossed = _augmented_case(name)
    gamma_tree, gamma_border = refs["gamma_tree"], refs["gamma_border"]
    depth, Hinv, k = refs["fx"]["depth"], refs["Hinv_f"], refs["k"]
    off = _offsets(refs["fx"]["K"])
    for lev_i, component in enumerate(levels):
        exact = float(np.sum(np.diag(Hinv)[off[lev_i] : off[lev_i + 1]]))
        value = factor.trace_inverse_penalty(component)
        assert abs(value - exact) <= (gamma_tree + gamma_border) * exact
    exact_border = float(np.sum(Hinv[k:, k:] * refs["fx"]["Omega_b"]))
    value = factor.trace_inverse_penalty(border)
    assert abs(value - exact_border) <= 2.0 * (gamma_tree + gamma_border) * abs(exact_border)
    frob = refs["frob"]
    for lev_i in range(depth):
        for lev_j in range(lev_i, depth):
            value = factor.penalty_cross_trace(levels[lev_i], levels[lev_j], 1.5, 0.5)
            exact = 0.75 * refs["pairs"][(lev_i, lev_j)]
            scale = 0.75 * math.sqrt(frob[lev_i] * frob[lev_j])
            bound = (4 * gamma_tree + 4 * gamma_border) * scale
            assert abs(value - exact) <= bound, (lev_i, lev_j, abs(value - exact) / bound)
            swapped = factor.penalty_cross_trace(levels[lev_j], levels[lev_i], 0.5, 1.5)
            assert swapped == pytest.approx(value, rel=1e-13)
        value = factor.penalty_cross_trace(levels[lev_i], border, 1.0, 1.0)
        scale = math.sqrt(frob[lev_i] * refs["border_border"])
        assert (
            abs(value - refs["level_border"][lev_i]) <= (2 * gamma_tree + 3 * gamma_border) * scale
        )
    value = factor.penalty_cross_trace(border, border, 1.0, 1.0)
    assert abs(value - refs["border_border"]) <= 4 * gamma_border * abs(refs["border_border"])
    for component in crossed:
        exact = refs["batched"][-1, -1]
        value = factor.penalty_cross_trace(component, component, 0.4, 0.4)
        assert abs(value - exact) <= 4 * gamma_border * exact


@pytest.mark.parametrize("name", MAIN)
def test_operator_traces_and_diagonals(name):
    refs, factor, penalized, O1, O2, levels, border, crossed = _augmented_case(name)
    gamma_tree, gamma_border = refs["gamma_tree"], refs["gamma_border"]
    p, depth = refs["p"], refs["fx"]["depth"]
    value = factor.trace_inverse_operator(O1)
    scale = math.sqrt(p * refs["tr11"])
    assert abs(value - refs["trace_O1"]) <= (4 * gamma_tree + 2 * gamma_border) * scale
    # the identity route on the factor's own data operator and its sum
    edf = factor.inverse_operator_diagonal(penalized.data)
    assert np.all(np.abs(edf - refs["edf"]) <= 2 * gamma_tree + 2 * gamma_border)
    total = factor.trace_inverse_operator(penalized.data)
    assert abs(total - float(np.sum(refs["edf"]))) <= p * (2 * gamma_tree + 2 * gamma_border)
    # the row-pass generic route on a signed operator, against sqrt(H^-1_uu (O H^-1 O)_uu)
    diagonal = factor.inverse_operator_diagonal(O1)
    scale = np.maximum(refs["scale_diag_O1"], EPS * refs["scale_diag_O1"].max())
    assert np.all(np.abs(diagonal - refs["diag_O1"]) <= (4 * gamma_tree + 2 * gamma_border) * scale)
    value = factor.operator_cross_trace(O1, O2)
    scale = math.sqrt(refs["tr11"] * refs["tr22"])
    assert abs(value - refs["tr12"]) <= (8 * gamma_tree + 4 * gamma_border) * scale
    assert factor.operator_cross_trace(O2, O1) == pytest.approx(value, rel=1e-13)
    for lev_i in range(depth):
        value = factor.penalty_operator_cross_trace(levels[lev_i], 1.0, O1)
        scale = math.sqrt(refs["frob"][lev_i] * refs["tr11"])
        assert abs(value - refs["level_O1"][lev_i]) <= (8 * gamma_tree + 4 * gamma_border) * scale
    value = factor.penalty_operator_cross_trace(border, 1.0, O1)
    scale = math.sqrt(refs["border_border"] * refs["tr11"])
    assert abs(value - refs["border_O1"]) <= (8 * gamma_tree + 4 * gamma_border) * scale
    for component in crossed:
        exact = refs["crossed_O1"][component.name]
        value = factor.penalty_operator_cross_trace(component, 0.4, O1)
        scale = math.sqrt(refs["batched"][-1, -1] * refs["tr11"])
        assert abs(value - exact) <= (8 * gamma_tree + 4 * gamma_border) * scale
    edf1 = factor.inverse_operator_square_diagonal(penalized.data)
    bound = (4 * gamma_tree + 4 * gamma_border) * refs["edf1_scale"]
    assert np.all(np.abs(edf1 - refs["edf1"]) <= bound)
    if name == "F1":
        # the explicit-column route: sum_v (O H^-1)_vi (H^-1 O)_vi with each column a
        # solve, so its error is the Skeel forward bound gamma |H^-1||H||x| of the two
        # solves times the other factor, plus the products' own rounding
        square = factor.inverse_operator_square_diagonal(O1)
        gamma = 4 * (gamma_tree + gamma_border)
        amplify = np.abs(refs["Hinv_f"]) @ np.abs(refs["H_f"])
        left = np.abs(refs["HO1_f"])
        right = np.abs(refs["HO1_f"]).T
        solve_error = gamma * (amplify @ left) * right
        apply_error = left * (gamma * np.abs(refs["O1_f"]) @ amplify @ np.abs(refs["Hinv_f"]))
        bound = np.sum(solve_error + apply_error, axis=0) + gamma * refs["square_O1_scale"]
        assert np.all(np.abs(square - refs["square_O1"]) <= bound)


@pytest.mark.parametrize("name", MAIN)
def test_derivative_cross_traces_batched_and_pairwise(name):
    refs, factor, _, O1, O2, levels, border, crossed = _augmented_case(name)
    gamma_tree, gamma_border = refs["gamma_tree"], refs["gamma_border"]
    lam = refs["fx"]["lam"]
    directions = [(levels[i], lam[i], O1 if i % 2 == 0 else O2) for i in range(len(levels))]
    directions.append((border, 1.0, None))
    directions.extend((component, 0.4, None) for component in crossed)
    batched = factor.derivative_cross_traces(directions)
    exact = refs["batched"]
    scale = np.sqrt(np.outer(np.diag(exact), np.diag(exact)))
    assert batched.shape == exact.shape and np.array_equal(batched, batched.T)
    assert np.all(np.abs(batched - exact) <= (8 * gamma_tree + 4 * gamma_border) * scale)
    for i, (ci, si, oi) in enumerate(directions):
        for j, (cj, sj, oj) in enumerate(directions):
            pairwise = factor.penalty_cross_trace(ci, cj, si, sj)
            if oj is not None:
                pairwise += factor.penalty_operator_cross_trace(ci, si, oj)
            if oi is not None:
                pairwise += factor.penalty_operator_cross_trace(cj, sj, oi)
            if oi is not None and oj is not None:
                pairwise += factor.operator_cross_trace(oi, oj)
            assert (
                abs(batched[i, j] - pairwise) <= (8 * gamma_tree + 4 * gamma_border) * scale[i, j]
            )


@pytest.mark.parametrize("name", PROFILED)
def test_profiled_factor_against_the_exact_centred_reference(name):
    refs, profiled, data, C1, C2, levels, border, crossed = _profiled_case(name)
    gamma_tree, gamma_border = refs["gamma_tree"], refs["gamma_border"]
    gamma = 4 * (gamma_tree + gamma_border)
    k, p, depth = refs["k"], refs["p"], refs["fx"]["depth"]
    pc = p - 1
    Hinv_c = refs["Hinv_c_f"]
    assert profiled.shape == (pc, pc) and profiled.rank == pc and not profiled.rank_truncated
    assert np.array_equal(profiled.mean_x, refs["mean_x"]) and profiled.sum_w == refs["sum_w"]
    assert np.array_equal(profiled.structured_indices, np.arange(k))
    assert np.array_equal(profiled.small_indices, np.arange(k, pc))
    assert profiled.chain_group_indices == tuple(range(depth))
    assert abs(profiled.logdet() - refs["logdet_c"]) <= k * gamma_tree + refs["q"] * gamma_border
    diagonal = profiled.selected_inverse_diagonal(np.arange(pc))
    exact = np.diag(Hinv_c)
    assert np.all(np.abs(diagonal - exact) <= (gamma_tree + gamma_border) * exact)
    # solves and blocks through the augmented factor, certified by their residual
    # on the augmented system (permutation: intercept first, nodes, slope border)
    perm = np.concatenate([[k], np.arange(k), np.arange(k + 1, p)])
    H_aug, Hinv_aug = refs["H_f"][np.ix_(perm, perm)], refs["Hinv_f"][np.ix_(perm, perm)]
    augmented = profiled.augmented_factor
    rhs = np.random.default_rng(9).normal(size=(pc, 2))
    padded = np.vstack([np.zeros((1, 2)), rhs])
    allowance = _certified_solve(
        augmented.operator, augmented.solve(padded), padded, H_aug, Hinv_aug, gamma
    )
    solution = profiled.solve(rhs)
    assert np.all(np.abs(solution - Hinv_c @ rhs) <= allowance[1:])
    indices = np.array([0, 1, k, k + 1, k - 1])
    unit = _unit_columns(p, indices + 1)
    allowance = _certified_solve(
        augmented.operator, augmented.solve(unit), unit, H_aug, Hinv_aug, gamma
    )
    block = profiled.selected_inverse_block(indices)
    exact_block = Hinv_c[np.ix_(indices, indices)]
    allowance = allowance[indices + 1]
    assert np.all(np.abs(block - exact_block) <= 0.5 * (allowance + allowance.T))
    if name in WELL_CONDITIONED:
        scale = np.sqrt(np.outer(np.diag(exact_block), np.diag(exact_block)))
        assert np.all(np.abs(block - exact_block) <= gamma * scale)
        assert np.all(np.abs(solution - Hinv_c @ rhs) <= gamma * (np.abs(Hinv_c) @ np.abs(rhs)))
    # penalties through the index shift
    off = _offsets(refs["fx"]["K"])
    for lev_i in range(depth):
        exact = float(np.sum(np.diag(Hinv_c)[off[lev_i] : off[lev_i + 1]]))
        value = profiled.trace_inverse_penalty(levels[lev_i])
        assert abs(value - exact) <= (gamma_tree + gamma_border) * exact
        for lev_j in range(lev_i, depth):
            value = profiled.penalty_cross_trace(levels[lev_i], levels[lev_j], 1.0, 1.0)
            exact = refs["pairs"][(lev_i, lev_j)]
            scale = math.sqrt(refs["frob_c"][lev_i] * refs["frob_c"][lev_j])
            assert abs(value - exact) <= (4 * gamma_tree + 4 * gamma_border) * scale
    value = profiled.penalty_cross_trace(border, border, 1.0, 1.0)
    assert abs(value - refs["border_border_c"]) <= 4 * gamma_border * refs["border_border_c"]
    # centred operators by the e0 identities
    value = profiled.trace_inverse_operator(C1)
    scale = math.sqrt(pc * refs["tr11_c"])
    assert abs(value - refs["trace_O1_c"]) <= (4 * gamma_tree + 2 * gamma_border) * scale
    diagonal = profiled.inverse_operator_diagonal(C1)
    scale = np.maximum(refs["scale_diag_O1_c"], EPS * refs["scale_diag_O1_c"].max())
    assert np.all(
        np.abs(diagonal - refs["diag_O1_c"]) <= (4 * gamma_tree + 2 * gamma_border) * scale
    )
    value = profiled.operator_cross_trace(C1, C2)
    scale = math.sqrt(refs["tr11_c"] * refs["tr22_c"])
    assert abs(value - refs["tr12_c"]) <= (8 * gamma_tree + 4 * gamma_border) * scale
    for lev_i in range(depth):
        value = profiled.penalty_operator_cross_trace(levels[lev_i], 1.0, C1)
        scale = math.sqrt(refs["frob_c"][lev_i] * refs["tr11_c"])
        assert abs(value - refs["level_O1_c"][lev_i]) <= (8 * gamma_tree + 4 * gamma_border) * scale
    value = profiled.penalty_operator_cross_trace(border, 1.0, C1)
    scale = math.sqrt(refs["border_border_c"] * refs["tr11_c"])
    assert abs(value - refs["border_O1_c"]) <= (8 * gamma_tree + 4 * gamma_border) * scale
    # the identity routes for the centred data operator, raw and wrapped
    centred_data = CenteredBlockOperator(
        raw=data, cross=refs["xtw"], total=refs["sum_w"], center=refs["mean_x"]
    )
    for operator in (data, centred_data):
        edf = profiled.inverse_operator_diagonal(operator)
        assert np.all(np.abs(edf - refs["edf_c"]) <= 2 * gamma_tree + 2 * gamma_border)
        edf1 = profiled.inverse_operator_square_diagonal(operator)
        bound = (4 * gamma_tree + 4 * gamma_border) * refs["edf1_scale_c"]
        assert np.all(np.abs(edf1 - refs["edf1_c"]) <= bound)
        total = profiled.trace_inverse_operator(operator)
        assert abs(total - float(np.sum(refs["edf_c"]))) <= pc * (2 * gamma_tree + 2 * gamma_border)
    # a second-order low-rank piece through profiled solves, alone and in a sum
    core = np.array([[0.0, -refs["sum_w"]], [-refs["sum_w"], 0.0]])
    low = LowRankSymmetricOperator(basis=refs["low_rank_basis"], core=core)
    # Low-rank pieces go through profiled solves: their error is the Skeel forward
    # bound of the solve, gamma |H_c^-1| |H_c| |H_c^-1 U|, times the other factor,
    # plus the products' own rounding.
    amplify = np.abs(Hinv_c) @ np.abs(refs["H_c_f"])
    columns = np.abs(Hinv_c @ low.basis)
    weighted = np.abs(low.basis @ low.core)
    bound = gamma * np.sum(weighted * (amplify @ columns)) + gamma * np.sum(weighted * columns)
    assert abs(profiled.trace_inverse_operator(low) - refs["trace_low_c"]) <= bound
    value = profiled.trace_inverse_operator(SumBlockOperator((C1, low)))
    exact = refs["trace_O1_c"] + refs["trace_low_c"]
    bound += (4 * gamma_tree + 2 * gamma_border) * math.sqrt(pc * refs["tr11_c"])
    assert abs(value - exact) <= bound
    assert np.all(profiled.coefficient_estimable())
    assert np.array_equal(profiled.scaled_schur_eigenvalues(), augmented.scaled_schur_eigenvalues())


def test_a_profiled_direction_does_not_keep_its_augmented_leaf_copies():
    """A slope direction's ``[1 | X]`` operator only forms its record's piece.

    ``derivative_cross_traces`` holds every direction's record through the
    whole cross matrix; records that kept their augmented operators held two
    ``(K, q + 1)`` leaf copies per weight-derivative direction, about 54 MiB of
    big_K25000's peak (T7).  With the collector off, the augmented operator
    must go when the call returns, and the traces must not change.
    """
    import gc
    import weakref

    refs, profiled, data, C1, C2, levels, border, crossed = _profiled_case(PROFILED[0])
    directions = [(levels[0], refs["fx"]["lam"][0], C1), (None, 0.0, C2)]
    expected = profiled.derivative_cross_traces(directions)
    held = []
    augment = profiled._augment

    def spy(operator):
        augmented = augment(operator)
        held.append(weakref.ref(augmented))
        return augmented

    profiled._augment = spy
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        records = [profiled._direction(*direction) for direction in directions]
        assert len(held) == 2
        assert all(reference() is None for reference in held)
        np.testing.assert_array_equal(profiled._cross_matrix(records), expected)
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("name", PROFILED)
def test_profiled_derivative_cross_traces(name):
    refs, profiled, data, C1, C2, levels, border, crossed = _profiled_case(name)
    gamma_tree, gamma_border = refs["gamma_tree"], refs["gamma_border"]
    lam = refs["fx"]["lam"]
    # Low-rank pieces go through solves (the frozen interface), which are
    # forward-error limited on the stress fixtures, so the second-order piece
    # joins direction 1 on the well-conditioned fixtures only.
    with_low = name in WELL_CONDITIONED
    core = np.array([[0.0, -refs["sum_w"]], [-refs["sum_w"], 0.0]])
    low = LowRankSymmetricOperator(basis=refs["low_rank_basis"], core=core)
    directions = []
    for i in range(len(levels)):
        operator = C1 if i % 2 == 0 else C2
        if i == 1 and with_low:
            operator = SumBlockOperator((operator, low))
        directions.append((levels[i], lam[i], operator))
    directions.append((border, 1.0, None))
    directions.extend((component, 0.4, None) for component in crossed)
    batched = profiled.derivative_cross_traces(directions)
    exact = refs["batched_c_low"] if with_low else refs["batched_c"]
    scale = np.sqrt(np.outer(np.diag(exact), np.diag(exact)))
    assert np.array_equal(batched, batched.T)
    assert np.all(np.abs(batched - exact) <= (8 * gamma_tree + 4 * gamma_border) * scale)
    for i, (ci, si, oi) in enumerate(directions):
        for j, (cj, sj, oj) in enumerate(directions):
            pairwise = profiled.penalty_cross_trace(ci, cj, si, sj)
            if oj is not None:
                pairwise += profiled.penalty_operator_cross_trace(ci, si, oj)
            if oi is not None:
                pairwise += profiled.penalty_operator_cross_trace(cj, sj, oi)
            if oi is not None and oj is not None:
                pairwise += profiled.operator_cross_trace(oi, oj)
            assert (
                abs(batched[i, j] - pairwise) <= (8 * gamma_tree + 4 * gamma_border) * scale[i, j]
            )


def test_the_raw_coordinate_factor_is_retired():
    """No raw-coordinate coefficient factor exists (one-engine design §3.6, §3.10).

    The slope covariance is ``M_ss``, the slope block of the augmented
    inverse; a factor without the intercept to absorb the border centre is a
    ``ValueError``, not a second route.
    """
    fx = _make_fixture(**FIXTURES["F1"])
    with pytest.raises(ValueError, match="raw-coordinate nested factor") as raised:
        _factor(fx, intercept=False)
    # a user-facing message does not cite the design document
    assert "design" not in str(raised.value)


def test_chain_only_factor_with_an_intercept_border():
    """A chain beside the intercept alone: the super-root is the whole border (§3.1).

    The profiled border is empty, so the factor is the tree recursion plus the
    super-root pivot ``D_0 = sum_r s_r``; every quantity is checked against the
    exact rational ``H_aug``.
    """
    fx = _make_fixture(seed=1, sizes=[3, 7, 16], n=600, lam=[0.7, 0.05, 3.0], width=0)
    fx["X"] = np.ones((len(fx["w"]), 1))
    fx["S_b"], fx["Omega_b"] = np.zeros((1, 1)), np.zeros((1, 1))
    tree = _tree(fx)
    data = _operator(fx, tree, fx["w"])
    penalized = _penalized(fx, data)
    factor = NestedSchurFactor(
        penalized, chain_group_names=CHAIN[:3], chain_group_indices=(0, 1, 2), intercept=True
    )
    M = _incidence(fx)
    k = tree.n_nodes
    H = _pair(_fraction_sum(_fraction_gram(fx, fx["w"], M), _fraction_penalty(fx, k, 1)))
    Hinv = _inverse_pair(H)
    Hinv_f = _floats(Hinv)
    gamma_tree, _ = _bounds(fx, 1, 1.0)
    k = k + 1  # the intercept sits beside the nodes in every check below
    assert factor.shape == (k, k) and factor.rank == k and factor.schur_condition_estimate == 1.0
    assert abs(factor.logdet() - _log_det(H)) <= k * gamma_tree
    diagonal = factor.selected_inverse_diagonal(np.arange(k))
    assert np.all(np.abs(diagonal - np.diag(Hinv_f)) <= 2 * gamma_tree * np.diag(Hinv_f))
    rhs = np.random.default_rng(4).normal(size=(k, 2))
    solution = factor.solve(rhs)
    allowance = _certified_solve(penalized, solution, rhs, _floats(H), Hinv_f, 4 * gamma_tree)
    assert np.all(np.abs(solution - Hinv_f @ rhs) <= allowance)
    levels, _, _ = _components(fx, k)
    off = _offsets(fx["K"])
    for lev_i in range(3):
        for lev_j in range(lev_i, 3):
            rows, cols = range(off[lev_i], off[lev_i + 1]), range(off[lev_j], off[lev_j + 1])
            exact = _frobenius(_block(Hinv, rows, cols))
            scale = math.sqrt(
                _frobenius(_block(Hinv, rows, rows)) * _frobenius(_block(Hinv, cols, cols))
            )
            value = factor.penalty_cross_trace(levels[lev_i], levels[lev_j], 1.0, 1.0)
            assert abs(value - exact) <= 4 * gamma_tree * scale
    edf = factor.inverse_operator_diagonal(data)
    S = _floats(_pair(_fraction_penalty(fx, k - 1, 1)))
    assert np.all(np.abs(edf - (1.0 - np.diag(Hinv_f @ S))) <= 2 * gamma_tree)
    assert np.all(factor.coefficient_estimable())
    assert factor.scaled_schur_eigenvalues().shape == (0,)


@pytest.mark.parametrize("name", TRUNCATED)
def test_rank_deficient_border_is_truncated_by_exactly_one_direction(name):
    refs = _case(name)
    fx = refs["fx"]
    assert refs["Hinv"] is None, "the duplicated column makes H exactly singular"
    factor = _factor(fx)
    k, q, p = refs["k"], refs["q"], refs["p"]
    pair = [k + fx["duplicate"], p - 1]
    assert factor.rank_truncated
    assert factor.rank == p - 1 and factor.schur_condition_estimate == float("inf")
    # §3.6 step 5: the one truncated direction is disclosed on the duplicated
    # pair (rest coordinates sit one after the intercept), unpenalized, so an
    # alias and not a weakly identified direction
    certificate = factor.border_certificate
    assert len(certificate.directions) == 1 and not certificate.weak.any()
    assert sorted(certificate.directions[0] + 1) == sorted([fx["duplicate"], q - 1])
    estimable = np.ones(p, dtype=bool)
    estimable[pair] = False
    assert np.array_equal(factor.coefficient_estimable(), estimable)
    # The null vector of H is n = [0; e_i - e_j] (identical columns, so F z = 0
    # and the null is uncoupled).  Rank decisions and the retained inverse are
    # taken in the Jacobi-scaled border coordinates (§3.7), where the null
    # vector is resolved to eps: Q^+ = D_s Q_s^+ D_s, the dense gram_eigh
    # convention.  A consistent solve therefore satisfies H x = r with
    # D_s^-1 x_b orthogonal to the scaled null vector; on a duplicated pair that
    # is also the unscaled Moore-Penrose solution in exact arithmetic, while the
    # computed one carries D_s[i] ||D_s^-1 x_b|| eps along e_i - e_j, which no
    # unscaled identity bounds.  H^+ = (H + n n'/2)^-1 - n n'/2 is the exact
    # Moore-Penrose inverse, log pdet(H) = log det(H + n n'/2), and the
    # identity routes read diag(H^+ H) = 1 - ||Z_s[j]||^2 on the border.
    null = np.zeros(p)
    null[pair] = [1.0, -1.0]
    penalized = factor.operator
    target = np.random.default_rng(6).normal(size=p)
    rhs = penalized.matvec(target)
    solution = factor.solve(rhs)
    eigenvalues = factor.scaled_schur_eigenvalues()
    kappa_s = float(eigenvalues[-1] / eigenvalues[1])
    gamma_tree, gamma_border = _bounds(fx, q, kappa_s)
    gamma = 4 * (gamma_tree + gamma_border)
    residual = penalized.matvec(solution) - rhs
    assert np.all(
        np.abs(residual) <= gamma * (np.abs(refs["H_f"]) @ np.abs(solution) + np.abs(rhs))
    )
    H_plus = [row[:] for row in refs["H_frac"]]
    half = Fraction(1, 2)
    for i in pair:
        for j in pair:
            H_plus[i][j] += half if i == j else -half
    plus = _pair(H_plus)
    assert abs(factor.logdet() - _log_det(plus)) <= k * gamma_tree + q * gamma_border
    null_projector = np.zeros((p, p))
    null_projector[np.ix_(pair, pair)] = [[-0.5, 0.5], [0.5, -0.5]]
    pseudo = _plus(_inverse_pair(plus), _pair(null_projector))
    exact = _diagonal(pseudo)
    diagonal = factor.selected_inverse_diagonal(np.arange(p))
    assert np.all(np.abs(diagonal - exact) <= (gamma_tree + gamma_border) * exact)
    HS = _product(pseudo, refs["S"])
    projector = _diagonal(_product(pseudo, refs["H"]))
    edf = factor.inverse_operator_diagonal(penalized.data)
    assert np.all(np.abs(edf - (projector - _diagonal(HS))) <= 2 * gamma_tree + 2 * gamma_border)
    edf1 = factor.inverse_operator_square_diagonal(penalized.data)
    edf1_exact = projector - 2.0 * _diagonal(HS) + _row_products(HS, HS)
    HS_f = _floats(HS)
    scale = 1.0 + 2.0 * np.abs(_diagonal(HS)) + np.sum(np.abs(HS_f) * np.abs(HS_f).T, axis=1)
    assert np.all(np.abs(edf1 - edf1_exact) <= (4 * gamma_tree + 4 * gamma_border) * scale)
    scaled_border = solution[k:] * factor._border_root
    assert abs(scaled_border @ null[k:]) <= gamma * np.linalg.norm(scaled_border) * np.sqrt(2.0)


def _with_border_column(fx, column):
    fx = dict(fx)
    fx["X"] = np.column_stack([fx["X"], column])
    fx["S_b"], fx["Omega_b"] = np.pad(fx["S_b"], (0, 1)), np.pad(fx["Omega_b"], (0, 1))
    return fx


def test_a_truncated_factor_maps_its_centred_null_to_raw_coordinates():
    """A null direction that touches the intercept in raw coordinates is mapped by ``R`` (§3.7).

    A column that is 5 on the weighted rows and 0 on the zero-weight ones has
    the raw null ``e_j - 5 e_0``; ``diag(H^+ H)`` on the border must be what the
    factor's own solves give.  A column that is 0 on the weighted rows and 7 on
    the (majority of) zero-weight rows has the raw null ``e_j``: the intercept
    stays estimable.  The fixture's centre is the old spread rule's, a
    different ``c`` from the prior-weighted one; the factor is exact algebra
    for any fixed centre.
    """
    fx = _make_fixture(**FIXTURES["F1"])
    fx = _with_border_column(fx, np.where(fx["w"] != 0.0, 5.0, 0.0))
    factor = _factor(fx)
    border, p = factor.small_indices, factor.shape[0]
    assert factor.rank == p - 1 and factor._center[-1] != 0.0
    eigenvalues = factor.scaled_schur_eigenvalues()
    gamma_tree, gamma_border = _bounds(fx, len(border), eigenvalues[-1] / eigenvalues[1])
    gamma = 4 * (gamma_tree + gamma_border)
    unit = _unit_columns(p, border)
    projector = np.diag(factor.solve(factor.operator.matvec(unit))[border])
    penalty = np.zeros((p, p))
    penalty[np.ix_(border, border)] = factor.operator.border_penalty
    HS = np.diag(factor.solve(penalty @ unit)[border])
    edf = factor.inverse_operator_diagonal(factor.operator.data)[border]
    assert np.all(np.abs(edf - (projector - HS)) <= gamma * (1.0 + np.abs(projector) + np.abs(HS)))

    fx = _make_fixture(**(FIXTURES["F1"] | dict(zero_leaves=tuple(range(4, 16)))))
    fx = _with_border_column(fx, np.where(fx["w"] != 0.0, 0.0, 7.0))
    factor = _factor(fx)
    assert factor.rank == factor.shape[0] - 1
    estimable = factor.coefficient_estimable()[factor.small_indices]
    assert np.array_equal(estimable, [True] * 5 + [False])


def test_rank_decisions_are_taken_on_the_scaled_schur_complement():
    """F8c has cond(Q) near 1e15 unscaled: the scaled rule keeps five directions, nullity one."""
    refs = _case("F8c")
    factor = _factor(refs["fx"])
    eigenvalues = factor.scaled_schur_eigenvalues()
    assert eigenvalues[0] <= 1e-10 * eigenvalues[-1]
    assert np.all(eigenvalues[1:] > 1e-3 * eigenvalues[-1])
    assert factor.rank == refs["p"] - 1
    border = refs["H_f"][refs["k"] :, refs["k"] :]
    assert np.linalg.cond(border) > 1e12


# ------------------------------------------------------------- refusals
def _factor_with_leaf(fx, data, leaf):
    """The intercept factor of ``fx`` with the data leaf statistics replaced by ``leaf``."""
    operator = NestedDataOperator(
        tree=data.tree,
        leaf=leaf,
        small_indices=data.small_indices,
        structured_indices=data.structured_indices,
    )
    return NestedSchurFactor(
        _penalized(fx, operator),
        chain_group_names=CHAIN[: fx["depth"]],
        chain_group_indices=tuple(range(fx["depth"])),
        intercept=True,
    )


def _with_within(fx, data, changes):
    """``fx``'s factor with ``changes`` (``(i, j, value)``) added to the within-leaf scatter."""
    within = np.array(data.leaf.within)
    for i, j, value in changes:
        within[i, j] += value
        if i != j:
            within[j, i] += value
    return _factor_with_leaf(fx, data, replace(data.leaf, within=within))


def test_material_negative_curvature_is_refused_by_the_scaled_eigenvalue_test():
    """Only the eigenvalue test on Q_s sees an indefinite block with a positive diagonal (§3.7)."""
    fx = _make_fixture(**FIXTURES["F1"])
    data = _operator(fx, _tree(fx), fx["w"])
    within = data.leaf.within
    coupling = 3.0 * np.sqrt(within[1, 1] * within[2, 2])
    with pytest.raises(np.linalg.LinAlgError, match="negative Schur curvature"):
        _with_within(fx, data, [(1, 2, coupling)])
    # a materially negative diagonal (the leaf attribute has no within-leaf
    # scatter and no border penalty, so its pivot is the between-child mass)
    with pytest.raises(np.linalg.LinAlgError, match="negative Schur curvature"):
        _with_within(fx, data, [(3, 3, -1e3)])


def test_a_border_column_carried_by_rounding_weights_is_an_exact_null_direction():
    """On weights with a rounding-negative row, 0, +1e-16 and -1e-16 under a column
    give one factor (§3.7).

    Such a vector was formed by a cancellation, so each weighted row is charged
    ``max|w|``.  The column is non-zero on two rows only; with those rows'
    weights at rounding level its pivot ``Q_jj`` is ``+-1e-16`` against a floor
    of ``gamma_Q max|w| sum (x - m)^2``, so it is an exact null direction in
    all three cases.  A sign test on the pivot, or a floor charged only the
    column's own weights, keeps ``+1e-16`` with ``1/Q_jj = 1e16`` in the
    inverse and refuses ``-1e-16`` as materially negative.
    """
    fx = dict(_make_fixture(**FIXTURES["F1"]))
    n, q = len(fx["w"]), fx["X"].shape[1]
    rows = np.flatnonzero(fx["w"])[[10, 200]]
    # a zero-weight row elsewhere rounds negative in every case
    fx["w"] = np.array(fx["w"])
    fx["w"][np.flatnonzero(fx["w"] == 0.0)[0]] = -1e-16
    rare = np.zeros(n)
    rare[rows] = 1.0
    fx["X"] = np.column_stack([fx["X"], rare])
    fx["S_b"], fx["Omega_b"] = np.pad(fx["S_b"], (0, 1)), np.pad(fx["Omega_b"], (0, 1))
    p = _tree(fx).n_nodes + q + 1
    factors = []
    for value in (0.0, 1e-16, -1e-16):
        fx["w"] = np.array(fx["w"])
        fx["w"][rows] = value
        factors.append(_factor(fx))
    reference = factors[0]
    exact = reference.selected_inverse_diagonal(np.arange(p))
    assert reference.rank_truncated and reference.rank == p - 1
    assert not reference.coefficient_estimable()[p - 1]
    for factor in factors[1:]:
        assert factor.rank == p - 1
        assert np.array_equal(factor.coefficient_estimable(), reference.coefficient_estimable())
        assert abs(factor.logdet() - reference.logdet()) <= 8 * EPS * abs(reference.logdet())
        # the two rows moved every other statistic by at most 1e-16 of their weight
        diagonal = factor.selected_inverse_diagonal(np.arange(p))
        assert np.all(np.abs(diagonal - exact) <= 64 * EPS * (np.abs(exact) + exact.max()))


@pytest.mark.parametrize("ratio", [1e-14, 1e-15])
def test_a_border_column_carried_by_tiny_positive_weights_keeps_its_pivot(ratio):
    """Non-negative weights are charged componentwise, so a column carried only by
    two rows of weight ``ratio max|w|`` is not null (§3.7).

    Fisher weights are each accurate to a few ulp, so these are genuine weights:
    below ``gamma_Q max|w|`` but far above their own rounding.  The factor keeps
    the rank of the Jacobi-scaled augmented ``H`` at ``1e-10 lambda_max``, the
    dense convention; a floor charged ``max|w|`` per row nulls the column.
    """
    fx = dict(_make_fixture(**FIXTURES["F1"]))
    rows = np.flatnonzero(fx["w"])[[10, 200]]
    rare = np.zeros(len(fx["w"]))
    rare[rows] = 1.0
    fx = _with_border_column(fx, rare)
    fx["w"] = np.array(fx["w"])
    fx["w"][rows] = ratio * np.max(fx["w"])
    factor = _factor(fx)
    p = factor.shape[0]
    H = factor.operator.matvec(np.eye(p))
    scale = 1.0 / np.sqrt(np.diag(H))
    eigenvalues = np.linalg.eigvalsh(scale[:, None] * (0.5 * (H + H.T)) * scale[None, :])
    assert np.sum(eigenvalues > 1e-10 * eigenvalues[-1]) == p
    assert factor.rank == p and not factor.rank_truncated
    assert factor.coefficient_estimable()[p - 1]


def test_refusals_by_the_iterate_are_linalg_errors():
    fx = _make_fixture(**FIXTURES["F1"])
    tree = _tree(fx)
    data = _operator(fx, tree, fx["w"])
    stats = data.leaf
    bad_weight = np.where(np.arange(len(stats.weight)) == 2, np.nan, stats.weight)
    with pytest.raises(np.linalg.LinAlgError, match="finite"):
        replace(stats, weight=bad_weight)
    with pytest.raises(np.linalg.LinAlgError, match="finite"):
        replace(stats, absolute=np.full(stats.width, np.inf))
    # a zero per-node penalty at an unobserved leaf: lambda_u = 0 and omega_u = 0
    penalty = [np.full(K, lam) for K, lam in zip(fx["K"], fx["lam"], strict=True)]
    penalty[-1][-1] = 0.0
    with pytest.raises(np.linalg.LinAlgError, match="zero penalty"):
        NestedSchurFactor(
            NestedPenalizedOperator(
                data=data, node_penalty=tuple(penalty), border_penalty=fx["S_b"]
            ),
            chain_group_names=CHAIN[:3],
            chain_group_indices=(0, 1, 2),
            intercept=True,
        )

    def with_weight(weight):
        return NestedDataOperator(
            tree=tree,
            leaf=replace(stats, weight=weight),
            small_indices=data.small_indices,
            structured_indices=data.structured_indices,
        )

    # signed leaf weights driving a pivot to or within its certified
    # uncertainty E_u + 4 eps (|w_u| + lambda_u) (signed-rows note §4.4, with
    # E_u = gamma_Q |w_u| for statistics that carry no error mass): exactly
    # zero, and then positive but inside it (w_0 = -lambda (1 - 20 eps) leaves
    # the pivot 20 eps lambda against more than gamma_Q lambda), which a zero
    # floor would accept at the leaf and refuse only at its parent
    signed = np.array(stats.weight)
    for weight in (-fx["lam"][-1], -fx["lam"][-1] * (1.0 - 20.0 * EPS)):
        signed[0] = weight
        with pytest.raises(np.linalg.LinAlgError, match="'variant'.*certified uncertainty"):
            NestedSchurFactor(
                _penalized(fx, with_weight(signed)),
                chain_group_names=CHAIN[:3],
                chain_group_indices=(0, 1, 2),
                intercept=True,
            )
    # rows negative only by rounding do not refuse
    rounding = np.array(stats.weight)
    rounding[3] = -1e-15
    accepted = NestedSchurFactor(
        _penalized(fx, with_weight(rounding)),
        chain_group_names=CHAIN[:3],
        chain_group_indices=(0, 1, 2),
        intercept=True,
    )
    reference = _factor(fx)
    assert abs(accepted.logdet() - reference.logdet()) <= 1e-12 * abs(reference.logdet())


@pytest.mark.parametrize("lam", [(1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0)])
def test_a_whole_level_without_penalty_is_intercept_aliasing(lam):
    fx = _make_fixture(seed=1, sizes=[3, 7, 16], n=600, lam=lam, zero_leaves=(), extra=(0, 0, 0))
    # the super-root pivot sum_r s_r is exactly zero: the intercept is aliased
    with pytest.raises(np.linalg.LinAlgError, match="aliased with the fitted intercept"):
        _factor(fx, intercept=True)


def test_malformed_calls_are_value_errors_and_foreign_operators_type_errors():
    refs, factor, penalized, O1, _, levels, border, _ = _augmented_case("F1")
    fx, k, p, q = refs["fx"], refs["k"], refs["p"], refs["q"]
    tree = _tree(fx)
    plain = _leaf_statistics(fx, fx["a"], None, shifted=False)
    other_means = NestedDataOperator(
        tree=tree,
        leaf=plain,
        small_indices=O1.small_indices,
        structured_indices=O1.structured_indices,
    )
    with pytest.raises(ValueError, match="leaf means"):
        factor.trace_inverse_operator(other_means)
    other_centre = replace(O1, leaf=replace(O1.leaf, center=np.zeros(q)))
    with pytest.raises(ValueError, match="leaf means"):
        factor.inverse_operator_diagonal(other_centre)
    moved = list(fx["parents"])
    moved[2] = np.array(moved[2])
    moved[2][5] = 1 - moved[2][5]
    other_tree = NestedTree(tuple(fx["K"]), tuple(moved))
    foreign = NestedDataOperator(
        tree=other_tree,
        leaf=O1.leaf,
        small_indices=O1.small_indices,
        structured_indices=O1.structured_indices,
    )
    with pytest.raises(ValueError, match="forest"):
        factor.inverse_operator_diagonal(foreign)
    swapped = NestedDataOperator(
        tree=tree,
        leaf=O1.leaf,
        small_indices=O1.small_indices[::-1],
        structured_indices=O1.structured_indices,
    )
    with pytest.raises(ValueError, match="partitions"):
        factor.operator_cross_trace(swapped, O1)
    with pytest.raises(ValueError, match="straddles"):
        factor.trace_inverse_penalty(_component("straddle", k - 2, k + 2))
    with pytest.raises(ValueError, match="exactly one nested chain level"):
        factor.penalty_cross_trace(_component("partial", 1, 3), levels[0], 1.0, 1.0)
    with pytest.raises(ValueError, match="exactly one nested chain level"):
        factor.trace_inverse_penalty(_component("dense", 0, fx["K"][0], np.eye(fx["K"][0])))
    capped = _factor(fx, max_structured_inverse_block=4)
    with pytest.raises(ValueError, match="request its diagonal instead"):
        capped.selected_inverse_block(np.arange(5))
    assert capped.selected_inverse_block(np.array([0, 1, 2, 3, k, k + 1])).shape == (6, 6)
    with pytest.raises(ValueError):
        factor.selected_inverse_diagonal(np.array([0, 0]))
    with pytest.raises(IndexError):
        factor.selected_inverse_diagonal(np.array([p]))
    with pytest.raises(ValueError):
        factor.solve(np.zeros(p + 1))
    single = SymmetricBlockOperator(
        A=np.eye(q),
        C=np.zeros((k, q)),
        d=np.ones(k),
        small_indices=O1.small_indices,
        structured_indices=O1.structured_indices,
    )
    with pytest.raises(TypeError):
        factor.trace_inverse_operator(single)
    block = BlockSymmetricOperator(
        A=np.eye(q),
        C=np.zeros((k, 1, q)),
        D=np.ones((k, 1, 1)),
        small_indices=O1.small_indices,
        structured_indices=O1.structured_indices.reshape(k, 1),
    )
    with pytest.raises(TypeError):
        factor.inverse_operator_square_diagonal(block)
    centred = CenteredBlockOperator(raw=O1, cross=np.zeros(p), total=1.0, center=np.zeros(p))
    with pytest.raises(TypeError):
        factor.operator_cross_trace(centred, O1)
    with pytest.raises(ValueError):
        NestedSchurFactor(
            penalized, chain_group_names=("a",), chain_group_indices=(0,), intercept=True
        )


def test_profiled_factor_refuses_other_centres_and_raw_foreign_operators():
    refs, profiled, data, C1, *_ = _profiled_case("F1")
    k, q = refs["k"], refs["q"]
    other_centre = CenteredBlockOperator(
        raw=C1.raw, cross=C1.cross, total=C1.total, center=np.zeros_like(C1.center)
    )
    with pytest.raises(ValueError, match="mean_x"):
        profiled.trace_inverse_operator(other_centre)
    with pytest.raises(ValueError, match="passed raw"):
        profiled.inverse_operator_diagonal(C1.raw)
    # the right centre but a raw operator built about other leaf means: its
    # row-pass quantities would be silently wrong, so the augmented factor's
    # bitwise mean check refuses it on every route
    fx = refs["fx"]
    columns = np.arange(1, q)
    mean = np.nextafter(_leaf_statistics(fx, fx["w"]).mean, np.inf)
    foreign = CenteredBlockOperator(
        raw=_operator(fx, _tree(fx), fx["a"], mean, columns=columns),
        cross=C1.cross,
        total=C1.total,
        center=C1.center,
    )
    for route in (profiled.trace_inverse_operator, profiled.inverse_operator_diagonal):
        with pytest.raises(ValueError, match="other leaf means"):
            route(foreign)
    with pytest.raises(ValueError, match="other leaf means"):
        profiled.derivative_cross_traces([(None, 0.0, foreign)])
    single = SymmetricBlockOperator(
        A=np.eye(q - 1),
        C=np.zeros((k, q - 1)),
        d=np.ones(k),
        small_indices=data.small_indices,
        structured_indices=data.structured_indices,
    )
    with pytest.raises(TypeError):
        profiled.trace_inverse_operator(single)
    with pytest.raises(ValueError, match="positive and finite"):
        ProfiledNestedSchurFactor(
            augmented_factor=profiled.augmented_factor,
            sum_w=0.0,
            xtw=refs["xtw"],
            data_operator=data,
        )
    with pytest.raises(ValueError, match="widths"):
        ProfiledNestedSchurFactor(
            augmented_factor=profiled.augmented_factor,
            sum_w=refs["sum_w"],
            xtw=refs["xtw"][:-1],
            data_operator=data,
        )


def test_augmentation_matches_the_operator_applied_to_the_intercept_column():
    refs, _, _, O1, *_ = _augmented_case("F1")
    fx, k, p = refs["fx"], refs["k"], refs["p"]
    tree = _tree(fx)
    columns = np.arange(1, refs["q"])
    slope = _operator(fx, tree, fx["a"], O1.leaf.mean, columns=columns)
    Xs = np.column_stack([_incidence(fx)[fx["leaf"]], fx["X"][:, 1:]])
    cross = Xs.T @ fx["a"]
    augmented = slope.augmented()
    e0 = np.zeros(p)
    e0[0] = 1.0
    column = augmented.matvec(e0)
    # the intercept column sums the leaf statistics: sum_l a_l, and C_leaf' 1
    # from a_l (m_l + c) + dev_l, rounding at the rows' absolute mass
    rows_per_leaf = np.bincount(fx["leaf"]).max()
    mass = np.abs(fx["a"]) @ np.abs(Xs)
    assert abs(column[0] - np.sum(fx["a"])) <= (len(fx["a"]) + 2) * EPS * np.sum(np.abs(fx["a"]))
    assert np.array_equal(column[1 : k + 1], np.concatenate(tree.subtree_sum(slope.leaf.weight)))
    assert np.all(np.abs(column[k + 1 :] - cross[k:]) <= 8 * rows_per_leaf * EPS * mass[k:])
    assert np.array_equal(augmented.leaf.mean[:, 0], np.ones(fx["K"][-1]))
    assert augmented.leaf.center[0] == 0.0 and augmented.leaf.absolute[0] == 0.0
    assert np.all(augmented.leaf.within[0] == 0.0) and np.all(augmented.leaf.within[:, 0] == 0.0)
    assert augmented.leaf.deviation is not None and np.all(augmented.leaf.deviation[:, 0] == 0.0)
    assert augmented.tree is tree
    # the augmented operator equals the exact-ordering operator up to the permutation
    perm = np.concatenate([[k], np.arange(k), np.arange(k + 1, p)])
    dense_exact = _floats(refs["O1"])
    dense_augmented = augmented.matvec(np.eye(p))
    rows_per_leaf = np.bincount(fx["leaf"]).max()
    atol = 8 * rows_per_leaf * EPS * np.abs(dense_exact).max()
    assert np.allclose(dense_augmented, dense_exact[np.ix_(perm, perm)], rtol=0.0, atol=atol)
    assert np.allclose(augmented.diagonal(), np.diag(dense_exact)[perm], rtol=0.0, atol=atol)
    penalized = _penalized(fx, slope, columns).augmented()
    assert np.all(penalized.border_penalty[0] == 0.0)
    assert penalized.border_penalty.shape == (refs["q"], refs["q"])


def test_no_dense_node_or_coefficient_matrix_is_formed():
    """Peak allocation over the build and every production method stays far below one dense square."""
    rng = np.random.default_rng(11)
    sizes = (40, 400, 6000)
    fx = _make_fixture(
        seed=12, sizes=list(sizes), n=20000, lam=[0.5, 0.2, 1.0], extra=(0, 0, 0), zero_leaves=()
    )
    for j in (1, 2):
        random_parents = rng.integers(0, sizes[j - 1], sizes[j] - sizes[j - 1])
        fx["parents"][j] = np.concatenate([np.arange(sizes[j - 1]), random_parents]).astype(np.intp)
    tree = _tree(fx)
    data = _operator(fx, tree, fx["w"])
    penalized = _penalized(fx, data)
    signed = _operator(fx, tree, fx["a"], data.leaf.mean)
    k = tree.n_nodes
    p = k + data.leaf.width
    levels, border, _ = _components(fx, k)
    tracemalloc.start()
    tracemalloc.reset_peak()
    factor = NestedSchurFactor(
        penalized, chain_group_names=CHAIN[:3], chain_group_indices=(0, 1, 2), intercept=True
    )
    factor.logdet()
    factor.solve(rng.normal(size=(p, 2)))
    factor.selected_inverse_diagonal(np.arange(p))
    factor.selected_inverse_block(np.array([0, 5, k, k + 1]))
    factor.trace_inverse_penalty(levels[1])
    factor.penalty_cross_trace(levels[0], levels[2], 1.0, 1.0)
    factor.penalty_cross_trace(levels[1], border, 1.0, 1.0)
    factor.trace_inverse_operator(signed)
    factor.inverse_operator_diagonal(signed)
    factor.inverse_operator_diagonal(penalized.data)
    factor.inverse_operator_square_diagonal(penalized.data)
    factor.operator_cross_trace(signed, signed)
    factor.penalty_operator_cross_trace(levels[2], 1.0, signed)
    factor.derivative_cross_traces(
        [
            (levels[0], 0.5, signed),
            (levels[1], 0.2, None),
            (levels[2], 1.0, signed),
            (border, 1.0, None),
        ]
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # the smallest dense square a wrong route could form is (K_leaf, K_leaf)
    dense_bytes = 8 * sizes[-1] ** 2
    assert peak < 0.05 * dense_bytes, (peak, dense_bytes, p)


def test_border_penalty_traces_read_only_their_own_rows_and_columns():
    """A two-column border penalty beside a 300-column crossed block forms no ``q x q`` array.

    Embedded in border coordinates the penalty and its products with ``Q^-1``
    are ``(q, q)``; on its own rows and columns every trace reads ``(k, q)``
    blocks at most.  Their values against the exact references are
    ``test_penalty_traces_and_pairs``.
    """
    fx = _make_fixture(
        seed=13, sizes=[2000], n=4000, lam=[0.5], extra=(0,), zero_leaves=(), crossed=300
    )
    factor = _factor(fx)
    levels, border, crossed = _components(fx, factor.structured_indices.size)
    q = factor.small_indices.size
    factor.penalty_cross_trace(levels[0], levels[0], 1.0, 1.0)  # caches G_I and the level pair
    tracemalloc.start()
    tracemalloc.reset_peak()
    factor.trace_inverse_penalty(border)
    factor.penalty_cross_trace(border, border, 1.0, 1.0)
    factor.penalty_cross_trace(levels[0], border, 1.0, 1.0)
    factor.penalty_cross_trace(border, crossed[0], 1.0, 1.0)
    factor.derivative_cross_traces([(levels[0], 0.5, None), (border, 1.0, None)])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 8 * q * q, (peak, q)


def test_the_between_node_scatter_skips_the_exact_zeros_of_indicator_columns():
    """``values' diag(s) values`` with its random-effect columns sparse equals the dense
    product without its copy.

    Both routes sum the same nonzero products, each within ``gamma_K sum_l
    |s_l d_li d_lj|`` of the exact entry (one more rounding for the weight), so
    they differ by twice that; a column with no nonzero keeps exact zeros.  The
    route is the columns' type (``indicator``), never their fill: without
    indicator columns it is the dense product itself, and a fully dense
    column marked as one gives the same sums.  The sparse route never forms
    the ``(K, q)`` weighted copy the dense product needs.
    """
    rng = np.random.default_rng(14)
    K, q = 20000, 300
    values = np.zeros((K, q))
    values[:, :10] = rng.normal(size=(K, 10))
    values[rng.integers(0, K, 3 * K), rng.integers(10, q - 1, 3 * K)] = rng.normal(size=3 * K)
    weight = rng.exponential(size=K)
    weight[::11] = 0.0
    indicator = np.arange(q) >= 10
    dense = (weight[:, None] * values).T @ values
    scatter, mass = _weighted_scatter(values, weight, indicator)
    bound = 2 * (K + 2) * EPS * ((weight[:, None] * np.abs(values)).T @ np.abs(values))
    assert np.all(np.abs(scatter - dense) <= bound)
    # the absolute mass is the diagonal of the same sum with |weight| (here weight >= 0)
    assert np.all(np.abs(mass - np.diag(dense)) <= np.diag(bound))
    np.testing.assert_array_equal(scatter[-1], 0.0)
    np.testing.assert_array_equal(scatter[:, -1], 0.0)
    for plain in (None, np.zeros(q, dtype=bool)):
        np.testing.assert_array_equal(_weighted_scatter(values, weight, plain)[0], dense)
    marked = np.arange(q) < 12  # ten dense columns and two sparse ones marked
    assert np.all(np.abs(_weighted_scatter(values, weight, marked)[0] - dense) <= bound)
    # a centre on the dense columns centres their slice only: the product of
    # the centred values, the one-hot columns keeping their exact zeros
    center = np.where(indicator, 0.0, rng.normal(size=q))
    centred = values - center
    expected = (weight[:, None] * centred).T @ centred
    centred_bound = 2 * (K + 3) * EPS * ((weight[:, None] * np.abs(centred)).T @ np.abs(centred))
    shifted, shifted_mass = _weighted_scatter(values, weight, indicator, center=center)
    assert np.all(np.abs(shifted - expected) <= centred_bound)
    assert np.all(np.abs(shifted_mass - np.diag(expected)) <= np.diag(centred_bound))
    tracemalloc.start()
    tracemalloc.reset_peak()
    _weighted_scatter(values, weight, indicator, center=center)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 0.5 * 8 * K * q, (peak, 8 * K * q)
