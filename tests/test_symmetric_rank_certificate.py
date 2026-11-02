"""The rank of the exact matrix a formed symmetric one approximates: ``rank.decompose_symmetric_deflated``.

A penalized curvature ``H = sum_r X_r' C_r X_r`` is formed in floating point,
and forming it moves it by up to ``gamma_n |X|'|C||X|`` (Higham 2002, eq.
3.13), which in Jacobi-scaled coordinates is above the rank cut.  So a
direction within that of the cut is decided by rounding: an exact alias
the formation lifts over the cut is kept, and a near-null estimable
direction it pushes under is dropped.  The fixtures FORCE that event: the
formed matrix is the correctly rounded exact matrix plus a controlled
perturbation ``Delta G`` along the deciding direction, inside a realistic
formation bound, so no BLAS rounding has to land anywhere.  ``project``
evaluates the grey block from the rows in float64 with Higham's bounds, and
every published number is checked against an exact rational reference built
here from the same float rows.
"""

import dataclasses
import math
import pickle
from fractions import Fraction

import numpy as np
import pytest

import superglm.solvers.rank as rank

_U = np.finfo(np.float64).eps / 2.0
ROWS = 600


def _gamma(count):
    return count * _U / (1.0 - count * _U)


def _alias_rows(weights, seed=0):
    """Location columns ``1[~b], 1[b], 1[b]`` (an exact alias, columns 1 and 2) and ``z``.

    ``positive``: one scalar channel in ``[0.1, 2]``.  ``signed``: one scalar
    channel ``1.5 + N(0, 1)``, negative on about a third of the rows.
    ``pair``: two predictors (location, and scale ``[1, z]``) with ``2 x 2``
    channels whose cross term exceeds ``sqrt(c11 c22)`` on some rows, so
    ``det C_r < 0`` there, as observed GaussianLS curvature has it.
    """
    rng = np.random.default_rng(seed)
    b = rng.random(ROWS) < 0.5
    z = rng.normal(size=ROWS)
    location = np.column_stack(((~b).astype(float), b.astype(float), b.astype(float)))
    if weights == "pair":
        c11, c22 = rng.uniform(0.5, 2.0, ROWS), rng.uniform(1.0, 3.0, ROWS)
        c12 = rng.choice([-1.0, 1.0], ROWS) * rng.uniform(0.5, 1.5, ROWS) * np.sqrt(c11 * c22)
        assert np.any(c11 * c22 - c12 * c12 < 0.0)
        return [location, np.column_stack((np.ones(ROWS), z))], {
            (0, 0): c11,
            (0, 1): c12,
            (1, 1): c22,
        }
    c = rng.uniform(0.1, 2.0, ROWS) if weights == "positive" else 1.5 + rng.normal(size=ROWS)
    assert weights == "positive" or np.any(c < 0.0)
    return [np.column_stack((location, z))], {(0, 0): c}


def _near_pair_rows(seed=3, separation=1e-7):
    """``1, x, x + separation * e``: one estimable direction near ``separation^2 / 2`` relative."""
    rng = np.random.default_rng(seed)
    x, e = rng.normal(size=ROWS), rng.normal(size=ROWS)
    design = np.column_stack((np.ones(ROWS), x, x + separation * e))
    return [design], {(0, 0): rng.uniform(0.1, 2.0, ROWS)}


def _ordered_pairs(channels):
    for (left, right), c in channels.items():
        yield left, right, c
        if left != right:
            yield right, left, c


def _exact_matrix(blocks, channels):
    """``sum_r X_r' C_r X_r`` of the float rows, exactly, as Fractions."""
    offsets = np.cumsum([0] + [block.shape[1] for block in blocks])
    width = int(offsets[-1])
    exact = [[Fraction(0)] * width for _ in range(width)]
    for a, b, c in _ordered_pairs(channels):
        weights = [Fraction(float(value)) for value in c]
        left = [[Fraction(float(value)) for value in row] for row in blocks[a]]
        right = [[Fraction(float(value)) for value in row] for row in blocks[b]]
        for i in range(blocks[a].shape[1]):
            for j in range(blocks[b].shape[1]):
                exact[offsets[a] + i][offsets[b] + j] += sum(
                    left[r][i] * weights[r] * right[r][j] for r in range(ROWS) if left[r][i]
                )
    return exact


def _exact_inverse(exact, keep):
    """The exact inverse of the principal block on ``keep`` (Gauss-Jordan in Fractions)."""
    size = len(keep)
    rows = [
        [exact[i][j] for j in keep] + [Fraction(int(a == b)) for b in range(size)]
        for a, i in enumerate(keep)
    ]
    for column in range(size):
        pivot = next(row for row in range(column, size) if rows[row][column] != 0)
        rows[column], rows[pivot] = rows[pivot], rows[column]
        rows[column] = [value / rows[column][column] for value in rows[column]]
        for row in range(size):
            if row != column and rows[row][column] != 0:
                factor = rows[row][column]
                rows[row] = [a - factor * b for a, b in zip(rows[row], rows[column])]
    return [row[size:] for row in rows]


def _projection(blocks, channels):
    """``project`` from the rows: ``T = X B`` first, then ``T' C T`` and ``X' (C T)``.

    Bounds per Higham (2002) eq. 3.13: ``|fl(T) - X B| <= gamma_(q+1) |X||B|``,
    each channel's sum over the rows ``gamma_(n+8)`` (the extra terms cover
    adding the channels), doubled to cover the bounds' own rounding.
    """
    offsets = np.cumsum([0] + [block.shape[1] for block in blocks])
    width = int(offsets[-1])

    def project(basis):
        parts = [basis[offsets[k] : offsets[k + 1]] for k in range(len(blocks))]
        T = [X @ B for X, B in zip(blocks, parts)]
        dT = [_gamma(X.shape[1] + 1) * (np.abs(X) @ np.abs(B)) for X, B in zip(blocks, parts)]
        g = _gamma(ROWS + 8)
        k = basis.shape[1]
        block, block_bound = np.zeros((k, k)), np.zeros((k, k))
        cross, cross_bound = np.zeros((width, k)), np.zeros((width, k))
        for a, b, c in _ordered_pairs(channels):
            weight, magnitude = c[:, None], np.abs(c)[:, None]
            block += T[a].T @ (weight * T[b])
            block_bound += g * (np.abs(T[a]).T @ (magnitude * np.abs(T[b]))) + (1.0 + g) * (
                dT[a].T @ (magnitude * np.abs(T[b]))
                + np.abs(T[a]).T @ (magnitude * dT[b])
                + dT[a].T @ (magnitude * dT[b])
            )
            rows = slice(offsets[a], offsets[a + 1])
            cross[rows] += blocks[a].T @ (weight * T[b])
            cross_bound[rows] += g * (np.abs(blocks[a]).T @ (magnitude * np.abs(T[b]))) + (
                1.0 + g
            ) * (np.abs(blocks[a]).T @ (magnitude * dT[b]))
        return rank.SymmetricGreyProjection(block, cross, 2.0 * block_bound, 2.0 * cross_bound)

    return project


def _formation_bound(blocks, channels, formed):
    """The bound a formation would carry: ``gamma_(n+p+4) ||D^-1 (sum |X|'|C||X|) D^-1||_2``, doubled."""
    offsets = np.cumsum([0] + [block.shape[1] for block in blocks])
    width = int(offsets[-1])
    majorant = np.zeros((width, width))
    for a, b, c in _ordered_pairs(channels):
        majorant[offsets[a] : offsets[a + 1], offsets[b] : offsets[b + 1]] += np.abs(
            blocks[a]
        ).T @ (np.abs(c)[:, None] * np.abs(blocks[b]))
    scale = np.sqrt(np.diag(formed))
    return 2.0 * _gamma(ROWS + width + 4) * np.linalg.norm(majorant / np.outer(scale, scale), 2)


def _counted(project):
    calls = []

    def counted(basis):
        calls.append(basis.shape)
        return project(basis)

    return counted, calls


def _forced(exact, direction, size):
    """The correctly rounded exact matrix plus ``size`` along ``direction`` (Jacobi-scaled, unit)."""
    rounded = np.array([[float(value) for value in row] for row in exact])
    scale = np.sqrt(np.diag(rounded))
    lifted = scale * direction / np.linalg.norm(direction)
    return rounded + size * np.outer(lifted, lifted)


def _cut(matrix):
    scale = np.sqrt(np.diag(matrix))
    largest = np.linalg.eigvalsh(matrix / np.outer(scale, scale))[-1]
    return rank._eigensolver_relative_bar(matrix.shape[0]) * largest


@pytest.mark.parametrize("weights", ["positive", "signed", "pair"])
@pytest.mark.parametrize("route", ["gram_eigh", "cholesky_shortcut"])
def test_an_exact_alias_lifted_over_the_cut_is_certified_dropped(route, weights):
    """The formed matrix keeps the alias, through either route; the exact one has it as a null.

    ``gram_eigh`` lifts the alias to five times the cut, ``cholesky_shortcut``
    to ``1e-8``, where the Cholesky shortcut certifies full rank without an
    eigendecomposition (as on the constant-weight reproducer, at ``2e-13``)
    and its probe's residual stays far inside its tolerance.  Mutation
    check, in the test: ``decompose_gram`` (the unfixed rule) reports full
    rank on both.  The rows give the grey block at the rows' own resolution,
    the alias is dropped with a certificate, and the published inverse is the
    representative block's exact inverse.
    """
    blocks, channels = _alias_rows(weights)
    exact = _exact_matrix(blocks, channels)
    width = len(exact)
    alias = np.zeros(width)
    alias[1], alias[2] = 1.0, -1.0
    rounded = np.array([[float(value) for value in row] for row in exact])
    size = 5.0 * _cut(rounded) if route == "gram_eigh" else 1e-8
    formed = _forced(exact, alias, size)
    bound = _formation_bound(blocks, channels, formed) + size

    unfixed = rank.decompose_gram(formed)
    assert unfixed.rank == width
    assert (unfixed.retained_values is None) == (route == "cholesky_shortcut")

    project, calls = _counted(_projection(blocks, channels))
    certified = rank.decompose_symmetric_deflated(formed, project, formation_bound=bound)
    assert calls == [(width, 1)]
    assert certified.rank == width - 1
    assert certified.grey_subspace["certified"] == 1.0 and not certified.resolution_limited
    assert certified.method == "pivoted_cholesky" and 2 not in certified.active_columns
    null = certified.null_basis()[:, 0]
    assert abs(null @ alias) / (np.linalg.norm(null) * np.linalg.norm(alias)) > 1.0 - 1e-12
    np.testing.assert_array_equal(
        certified.coefficient_estimable(), [index not in (1, 2) for index in range(width)]
    )
    keep = [index for index in range(width) if index != 2]
    reference = np.zeros((width, width))
    inverse = _exact_inverse(exact, keep)
    reference[np.ix_(keep, keep)] = [[float(value) for value in row] for row in inverse]
    np.testing.assert_allclose(
        certified.pseudo_inverse(), reference, rtol=1e-9, atol=1e-12 * np.abs(reference).max()
    )


@pytest.mark.parametrize("push", [-1.5, 0.2], ids=["pushed_under_the_cut", "pushed_up"])
def test_a_near_null_estimable_direction_is_kept_with_an_accurate_variance(push):
    """An estimable direction near ``5e-15`` of the largest eigenvalue, the formed one moved by ``push`` of itself.

    Pushed under the cut (to minus half itself, inside the negative
    tolerance, so clipped), the unfixed rule drops it (rank 2); pushed up by
    a fifth, it keeps it with variances off by about that fifth.  Both are the
    mutation checks, in the test.  The grey block from the rows retains it
    with a certificate, and every published variance is within ``1e-8`` of
    the exact rational inverse.
    """
    blocks, channels = _near_pair_rows()
    exact = _exact_matrix(blocks, channels)
    rounded = np.array([[float(value) for value in row] for row in exact])
    scale = np.sqrt(np.diag(rounded))
    values, vectors = np.linalg.eigh(rounded / np.outer(scale, scale))
    formed = _forced(exact, vectors[:, 0], push * values[0])
    inverse = _exact_inverse(exact, [0, 1, 2])
    variances = np.array([float(inverse[i][i]) for i in range(3)])
    # The exact equilibrated matrix's smallest eigenvalue lies in [1/t, 3/t],
    # t the trace of its inverse: well above the cut, and inside the band.
    trace = float(
        sum(Fraction(float(np.sqrt(formed[i, i]))) ** 2 * inverse[i][i] for i in range(3))
    )
    cut = _cut(formed)
    assert 2.0 * cut < 1.0 / trace and 3.0 / trace < 1e-13

    unfixed = rank.decompose_gram(formed)
    if push < 0.0:
        assert unfixed.rank == 2
    else:
        assert unfixed.rank == 3
        assert np.max(np.abs(np.diag(unfixed.pseudo_inverse()) / variances - 1.0)) > 1e-2

    project, calls = _counted(_projection(blocks, channels))
    bound = _formation_bound(blocks, channels, formed)
    certified = rank.decompose_symmetric_deflated(formed, project, formation_bound=bound)
    assert len(calls) == 1
    assert certified.rank == 3 and certified.grey_subspace["certified"] == 1.0
    assert certified.grey_subspace["retained"] == 1.0
    np.testing.assert_allclose(np.diag(certified.pseudo_inverse()), variances, rtol=1e-8)
    assert certified.coefficient_estimable().all()


def test_a_grey_direction_inside_its_bound_is_decided_by_the_rows_and_flagged():
    """A row block whose own bound straddles the cut: decided by the row estimate, never refused.

    The near pair pushed under the cut, with ``project``'s block bound
    inflated to ten times the block, so the certificate cannot place the
    grey eigenvalue on either side.  The rank is the rows' estimate (3, where
    the formed matrix says 2), flagged ``resolution_limited`` with
    ``certified = 0``.  Mutation check: a certificate that ignores
    ``block_bound`` certifies this and fails the flag.
    """
    blocks, channels = _near_pair_rows()
    exact = _exact_matrix(blocks, channels)
    rounded = np.array([[float(value) for value in row] for row in exact])
    scale = np.sqrt(np.diag(rounded))
    values, vectors = np.linalg.eigh(rounded / np.outer(scale, scale))
    formed = _forced(exact, vectors[:, 0], -1.5 * values[0])
    project = _projection(blocks, channels)

    def loose(basis):
        rows = project(basis)
        return rows._replace(block_bound=10.0 * np.abs(rows.block) + rows.block_bound)

    decided = rank.decompose_symmetric_deflated(
        formed, loose, formation_bound=_formation_bound(blocks, channels, formed)
    )
    assert rank.decompose_gram(formed).rank == 2
    assert decided.rank == 3
    assert decided.grey_subspace["certified"] == 0.0
    assert decided.grey_subspace["unresolved"] == 1.0
    assert decided.resolution_limited


def test_past_the_cap_the_formed_decision_stands_flagged():
    """More grey directions than ``max_grey``: no row is read, the formed decision is kept and flagged."""
    blocks, channels = _alias_rows("positive")
    exact = _exact_matrix(blocks, channels)
    alias = np.array([0.0, 1.0, -1.0, 0.0])
    rounded = np.array([[float(value) for value in row] for row in exact])
    formed = _forced(exact, alias, 5.0 * _cut(rounded))
    project, calls = _counted(_projection(blocks, channels))
    kept = rank.decompose_symmetric_deflated(
        formed, project, formation_bound=_formation_bound(blocks, channels, formed), max_grey=0
    )
    assert calls == []
    assert kept.rank == rank.decompose_gram(formed).rank == 4
    assert kept.grey_subspace["capped"] == 1.0 and kept.grey_subspace["certified"] == 0.0
    assert kept.resolution_limited


def _assert_same_bits(left, right, *, skip=()):
    for field in dataclasses.fields(left):
        if field.name in skip:
            continue
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), field.name
        else:
            assert pickle.dumps(a) == pickle.dumps(b), field.name


def _correlated(rho):
    """``D C D``: ``C`` a correlation matrix with ``lambda_min = 1 - rho``, ``D`` spanning five decades."""
    correlation = np.eye(3)
    correlation[0, 1] = correlation[1, 0] = rho
    scale = np.diag([1.0, 1e3, 1e-2])
    return scale @ correlation @ scale


@pytest.mark.parametrize("route", ["cholesky_shortcut", "gram_eigh"])
def test_a_matrix_without_a_grey_set_reads_no_row(route):
    """Every eigenvalue past the cut by more than the error: full rank, certified, ``project`` never called.

    ``cholesky_shortcut``: a well-conditioned matrix, certified by the
    shortcut's shifted Cholesky, and otherwise bit for bit the unbounded
    decomposition.  ``gram_eigh``: ``lambda_min = 1e-14`` of a matrix taken
    as exact, under the shortcut's own threshold (so it goes to the
    eigendecomposition) and past ``tau + e``.
    """
    formed = _correlated(0.5 if route == "cholesky_shortcut" else 1.0 - 1e-14)
    bound = 1e-12 if route == "cholesky_shortcut" else 0.0
    unbounded = rank.decompose_gram(formed)
    assert (unbounded.retained_values is None) == (route == "cholesky_shortcut")
    project, calls = _counted(lambda basis: pytest.fail("no grey direction to project"))
    certified = rank.decompose_symmetric_deflated(formed, project, formation_bound=bound)
    assert calls == []
    assert certified.rank == 3
    assert certified.grey_subspace["certified"] == 1.0 and certified.grey_subspace["k"] == 0.0
    _assert_same_bits(certified, unbounded, skip=("grey_subspace",))


def test_the_bounded_shortcut_certifies_only_what_it_proves():
    """``decompose_gram``'s shortcut with and without a formation bound.

    Without one, every field is what the explicit ``None`` gives, which is
    the decomposition from before the keyword existed.  With a bound that the
    matrix clears, the decomposition is the
    shortcut's plus a certificate.  With a bound the alias lifted to
    ``1e-8`` cannot clear, the shortcut is refused and the
    eigendecomposition decides, uncertified.  Mutation check, in the test:
    the unbounded call takes the shortcut on that alias.
    """
    clear = _correlated(0.5)
    _assert_same_bits(rank.decompose_gram(clear), rank.decompose_gram(clear, formation_bound=None))
    bounded = rank.decompose_gram(clear, formation_bound=1e-12)
    assert bounded.grey_subspace["certified"] == 1.0
    _assert_same_bits(bounded, rank.decompose_gram(clear), skip=("grey_subspace",))

    blocks, channels = _alias_rows("positive")
    formed = _forced(_exact_matrix(blocks, channels), np.array([0.0, 1.0, -1.0, 0.0]), 1e-8)
    bound = _formation_bound(blocks, channels, formed) + 1e-8
    unbounded = rank.decompose_gram(formed)
    assert unbounded.method == "cholesky" and unbounded.retained_values is None
    refused = rank.decompose_gram(formed, formation_bound=bound)
    assert refused.retained_values is not None and refused.grey_subspace is None


def test_the_shifted_cholesky_proves_only_true_floors():
    """Rump's test on matrices whose smallest eigenvalue is exact: it never proves a floor at or above it."""
    for smallest in (1.0, 1e-6, 1e-10):
        matrix = np.diag([2.0, 1.0, smallest])
        assert rank._shifted_cholesky_clears(matrix, smallest * (1.0 - 1e-3))
        assert not rank._shifted_cholesky_clears(matrix, smallest)
        assert not rank._shifted_cholesky_clears(matrix, smallest * (1.0 + 1e-12))


def test_a_dropped_coupled_grey_direction_keeps_the_published_pseudo_determinant():
    """``log_pdet`` is that of ``E diag(r) E'`` when a grey column coupled to the top block is dropped.

    The rows couple the near-null direction to a top eigenvector (Jacobi-scaled
    coupling 0.05, forced here rather than left to rounding), so the Schur
    complement drops it and the factor ``E`` is orthogonal to the null basis
    but not orthonormal: its volume is part of the determinant.  The
    reference is the exact rational determinant of ``E' E`` for the published
    float ``E``.  Mutation check: the complementary-minor shortcut on ``E``
    itself misses ``log det(E' E)``, 1.8e-3 here.
    """
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    scale = np.array([1.0, 2.0**10, 2.0**-6])
    formed = (scale[:, None] * q) * np.array([1e-10, 1.0, 2.0]) @ (q.T * scale[None, :])
    formed = 0.5 * (formed + formed.T)
    unit = np.sqrt(np.diag(formed))
    _, vectors = np.linalg.eigh(formed / np.outer(unit, unit))
    grey, top = vectors[:, 0], vectors[:, 1]
    exact = formed + np.outer(unit, unit) * 0.05 * (np.outer(top, grey) + np.outer(grey, top))

    def project(basis):
        cross = exact @ basis
        error = 16.0 * _U * (np.abs(exact) @ np.abs(basis))
        return rank.SymmetricGreyProjection(basis.T @ cross, cross, np.abs(basis).T @ error, error)

    decided = rank.decompose_symmetric_deflated(formed, project, formation_bound=1e-6)
    assert decided.rank == 2 and decided.grey_subspace["retained"] == 0.0
    basis = [[Fraction(float(v)) for v in row] for row in decided.estimable_functional_basis]
    gram = [[sum(basis[r][i] * basis[r][j] for r in range(3)) for j in range(2)] for i in range(2)]
    determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
    reference = math.log(determinant) + float(np.sum(np.log(decided.retained_values)))
    assert abs(math.log(determinant)) > 1e-3
    assert abs(decided.log_pdet - reference) <= 1e-12 * max(1.0, abs(reference))
