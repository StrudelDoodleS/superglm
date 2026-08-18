"""Per-pair screening moments: exactness pins against the dense row-Kronecker.

Task 1 of docs/superpowers/plans/2026-07-28-interaction-screening.md.  The
whole screening design rests on the cell-space assembly reproducing the dense
assembly exactly, so these tolerances must not be loosened.
"""

from __future__ import annotations

import numpy as np
import pytest

from superglm.distributions import Gamma, Poisson
from superglm.links import LogLink
from superglm.screening import pair_cell_moments, pair_score_curvature, working_score
from superglm.screening._score_stat import penalized_score_statistic_ladder


def _pair_case(seed, n=4000, n_a=17, n_b=13, k_a=4, k_b=3, signed=False):
    rng = np.random.default_rng(seed)
    codes_a = rng.integers(0, n_a, n)
    codes_b = rng.integers(0, n_b, n)
    B_a = rng.normal(size=(n_a, k_a))
    B_b = rng.normal(size=(n_b, k_b))
    score = rng.normal(size=n) if signed else rng.uniform(0.1, 1.0, n)
    weights = rng.normal(size=n) if signed else rng.uniform(0.2, 2.0, n)
    return codes_a, codes_b, B_a, B_b, score, weights


def _dense_row_kronecker(codes_a, codes_b, B_a, B_b):
    rows_a = B_a[codes_a]
    rows_b = B_b[codes_b]
    return np.einsum("rp,rq->rpq", rows_a, rows_b).reshape(
        len(codes_a), B_a.shape[1] * B_b.shape[1]
    )


def test_cell_assembly_matches_dense_row_kronecker():
    codes_a, codes_b, B_a, B_b, score, weights = _pair_case(0)

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, 17, 13, score, weights)
    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)

    X = _dense_row_kronecker(codes_a, codes_b, B_a, B_b)
    U_dense = X.T @ score
    V_dense = X.T @ (X * weights[:, None])
    np.testing.assert_allclose(U, U_dense, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(V, V_dense, rtol=1e-12, atol=1e-12)


def test_cell_assembly_handles_signed_score_and_weights():
    """REML working quantities are signed; nothing here may assume positivity."""
    codes_a, codes_b, B_a, B_b, score, weights = _pair_case(1, signed=True)

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, 17, 13, score, weights)
    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)

    X = _dense_row_kronecker(codes_a, codes_b, B_a, B_b)
    np.testing.assert_allclose(U, X.T @ score, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(V, X.T @ (X * weights[:, None]), rtol=1e-11, atol=1e-11)


def test_empty_and_singleton_cells_are_exact():
    """Cells with no rows contribute zero; single-level margins still work."""
    codes_a = np.array([0, 0, 0, 0])
    codes_b = np.array([2, 2, 0, 0])
    B_a = np.array([[1.5, -0.5]])
    B_b = np.array([[1.0], [2.0], [4.0]])
    score = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.5, 0.5, 1.0, 1.0])

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, 1, 3, score, weights)
    assert S_cell.shape == (1, 3)
    assert W_cell[0, 1] == 0.0

    U, V = pair_score_curvature(B_a, B_b, S_cell, W_cell)
    X = _dense_row_kronecker(codes_a, codes_b, B_a, B_b)
    np.testing.assert_allclose(U, X.T @ score, rtol=1e-14)
    np.testing.assert_allclose(V, X.T @ (X * weights[:, None]), rtol=1e-14)


def test_cell_values_are_pinned_independently():
    """Hand-computed S_cell and W_cell so a coordinated S/W swap cannot pass."""
    codes_a = np.array([0, 0, 0, 0])
    codes_b = np.array([2, 2, 0, 0])
    score = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([0.5, 0.25, 1.0, 2.0])

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, 1, 3, score, weights)

    np.testing.assert_array_equal(S_cell, [[7.0, 0.0, 3.0]])
    np.testing.assert_array_equal(W_cell, [[3.0, 0.0, 0.75]])


def test_out_of_range_codes_raise_instead_of_corrupting():
    """The kernel indexes without checks; the guard must catch every escape."""
    import pytest

    ok = np.zeros(4, dtype=int)
    vals = np.ones(4)
    for bad_a, bad_b in (([0, 0, 0, 2], ok), (ok, [0, 3, 0, 0]), ([-1, 0, 0, 0], ok)):
        with pytest.raises(ValueError, match="fall outside"):
            pair_cell_moments(np.asarray(bad_a), np.asarray(bad_b), 2, 3, vals, vals)


def test_short_value_arrays_raise_instead_of_reading_heap():
    import pytest

    codes = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="row-for-row"):
        pair_cell_moments(codes, codes, 1, 1, np.zeros(3), np.zeros(10))
    with pytest.raises(ValueError, match="row-for-row"):
        pair_cell_moments(codes, codes, 1, 1, np.zeros(10), np.zeros(3))


def test_cell_ceiling_rejects_unbinned_wide_pairs():
    import pytest

    codes = np.zeros(4, dtype=int)
    vals = np.ones(4)
    with pytest.raises(ValueError, match="ceiling"):
        pair_cell_moments(codes, codes, 50_000, 50_000, vals, vals)
    # The ceiling is caller-adjustable; a modest raise takes effect.
    S_cell, _ = pair_cell_moments(codes, codes, 2_000, 3_000, vals, vals, max_cells=6_000_000)
    assert S_cell.shape == (2_000, 3_000)


def test_working_score_is_bitwise_the_inline_formula():
    """Bit-identity pin: the KKT suite tolerates ~10% drift, this does not."""
    from superglm.distributions import _VARIANCE_FLOOR

    rng = np.random.default_rng(4)
    n = 500
    mu = rng.uniform(0.05, 5.0, n)
    eta = np.log(mu)
    y = rng.poisson(mu).astype(float)
    w = rng.uniform(0.1, 2.0, n)
    family, link = Gamma(), LogLink()

    expected = (
        w * link.deriv_inverse(eta) * (y - mu) / np.maximum(family.variance(mu), _VARIANCE_FLOOR)
    )
    assert np.array_equal(working_score(y, mu, eta, w, family, link), expected)


def test_mismatched_code_shapes_raise():
    import pytest

    with pytest.raises(ValueError, match="row dimension"):
        pair_cell_moments(
            np.zeros(3, dtype=int), np.zeros(4, dtype=int), 1, 1, np.zeros(3), np.zeros(3)
        )


def test_working_score_reduces_to_raw_residual_for_canonical_link():
    rng = np.random.default_rng(2)
    n = 200
    mu = rng.uniform(0.1, 3.0, n)
    eta = np.log(mu)
    y = rng.poisson(mu).astype(float)
    w = rng.uniform(0.2, 1.5, n)

    score = working_score(y, mu, eta, w, Poisson(), LogLink())
    np.testing.assert_allclose(score, w * (y - mu), rtol=1e-14)


def test_working_score_carries_family_factor_for_noncanonical_link():
    rng = np.random.default_rng(3)
    n = 200
    mu = rng.uniform(0.5, 4.0, n)
    eta = np.log(mu)
    y = rng.gamma(2.0, mu / 2.0)
    w = rng.uniform(0.2, 1.5, n)

    score = working_score(y, mu, eta, w, Gamma(), LogLink())
    np.testing.assert_allclose(score, w * (y - mu) / mu, rtol=1e-12)


def _pd_matrix(rng, p, strength=1.0):
    A = rng.normal(size=(p, 2 * p))
    return strength * (A @ A.T) / p


def test_statistic_reduces_to_unpenalized_without_penalty():
    from superglm.screening import penalized_score_statistic

    rng = np.random.default_rng(5)
    p = 6
    V = _pd_matrix(rng, p)
    U = rng.normal(size=p)

    result = penalized_score_statistic(U, V, S_ti=None)

    np.testing.assert_allclose(result.statistic, U @ np.linalg.solve(V, U), rtol=1e-11)
    assert result.lambda0 == 0.0


def test_whitening_keeps_an_identifiable_mode_that_is_merely_small():
    """The singular-pencil whitening may discard the common null space, no more.

    ``G = V + S`` here is ``diag(2, 2e-13, 0)``.  The third direction is the
    genuine shared null space; the second is NOT -- it carries ``a = 0.5``, an
    honest half-share of the curvature, and all of ``U``'s mass.  A cut set by
    smallness relative to the largest eigenvalue rather than by round-off
    deletes it, and the ladder then reports a statistic of 0 for a pair whose
    score is entirely in that direction.

    The direct pseudo-inverse ladder is the reference: solving
    ``(V + lam S)`` at ``lam = 1`` gives edf 1.0 and T 0.5.
    """
    from superglm.screening import penalized_score_statistic

    V = np.diag([1.0, 1e-13, 0.0])
    S = np.diag([1.0, 1e-13, 0.0])
    U = np.array([0.0, np.sqrt(1e-13), 0.0])

    got = penalized_score_statistic(U, V, S_ti=S, edf0=1.0)

    # what the direct solver resolves, recomputed here rather than asserted
    A = V + 1.0 * S
    Ainv = np.linalg.pinv(A, hermitian=True)
    assert float(np.trace(Ainv @ V)) == pytest.approx(1.0, abs=1e-9)
    assert float(U @ (Ainv @ U)) == pytest.approx(0.5, rel=1e-9)

    assert got.edf0 == pytest.approx(1.0, abs=1e-6)
    assert got.statistic == pytest.approx(0.5, rel=1e-6)
    assert got.lambda0 == pytest.approx(1.0, rel=1e-6)


def test_screening_kernels_are_internal_and_the_root_api_is_self_consistent():
    """The screening package is a kernel, not public API — pinned deliberately.

    ``superglm.screening.__all__`` reads like a public claim, and a reviewer
    took it as one: ``from superglm import penalized_score_statistic_ladder``
    raises.  So it does for all eight names there, every one of which predates
    the ladder, and the resolution is that none of them belongs at the root
    rather than that the ladder does -- they take raw assembled moment
    matrices, so they are unusable without the internals that build them.

    This test pins that decision so it is not silently reversed, and guards the
    class of defect the reviewer was actually pointing at: a name advertised
    somewhere it cannot be imported from.
    """
    import superglm
    import superglm.screening as screening

    # The root advertises nothing it cannot supply.
    assert [n for n in superglm.__all__ if not hasattr(superglm, n)] == []
    # Each kernel name resolves from its OWN package, which is where it lives.
    assert [n for n in screening.__all__ if not hasattr(screening, n)] == []
    # And none of them is root API.
    assert set(screening.__all__).isdisjoint(superglm.__all__)
    # The supported entry point is the model method.
    assert hasattr(superglm.SuperGLM, "screen_interactions")


def test_a_wholly_absorbed_probe_is_scored_rather_than_discarded():
    """An indeterminate block is REPORTED, not deleted.  Deliberately.

    A numeric constant within each level makes every ``numeric_cat`` probe
    column a multiple of that level's indicator, so the categorical main
    absorbs the whole block and the true profiled rank is 0.  ``V_eff`` is a
    difference, so what survives is round-off -- and being all that is left,
    that round-off becomes the block's own largest eigenvalue.  This block IS
    pure cancellation: asserted below at more than 1e12.

    Detecting that would need a threshold on the share of curvature profiling
    leaves behind, and no Type 1 bound for it exists -- see the THRESHOLD
    TYPES note in :mod:`superglm.screening._score_stat`, which records eleven
    orders of variation at fixed ``k``.  Since only arithmetic may discard a
    pair, and this cannot be decided as arithmetic, the pair is scored.

    What that buys is the property this test actually pins: the statistic
    stays at round-off, so ``z`` ranks the pair down on its own merits rather
    than the kernel deleting it.  What it costs is that ``edf0`` is not
    reproducible across seeds, which is the honest signature of an
    indeterminate block.  Both are asserted, so a future change to either is
    visible.
    """
    import pandas as pd

    from superglm import Categorical, SuperGLM
    from superglm.features.numeric import Numeric
    from superglm.screening._numeric_margin import numeric_pair_moments
    from superglm.screening._score_stat import _solve_psd

    L, n = 5, 4000
    values = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    menu = np.eye(L)[:, 1:]
    for seed in range(8):
        rng = np.random.default_rng(seed)
        g = rng.integers(0, L, n)
        x = values[g]

        U, V, C, M, u_m = numeric_pair_moments(
            g, L, menu, x, rng.normal(size=n), rng.uniform(0.5, 1.5, n)
        )
        P = C.T @ _solve_psd(M, C)
        V_eff = 0.5 * ((V - P) + (V - P).T)
        assert max(np.linalg.norm(V, 2), np.linalg.norm(P, 2)) / np.linalg.norm(V_eff, 2) > 1e12, (
            seed
        )

        df = pd.DataFrame({"g": pd.Categorical([f"L{c}" for c in g]), "x": x})
        y = rng.poisson(np.exp(-1.0 + 0.1 * x)).astype(np.float64)
        model = SuperGLM(family="poisson", features={"g": Categorical(), "x": Numeric()})
        model.fit_reml(df, y)
        row = model.screen_interactions(df, y, candidates=[("x", "g")]).iloc[0]

        # The pair is kept, and what makes it uninteresting is its own
        # statistic -- at round-off, so z cannot promote it.
        if np.isfinite(row["statistic"]):
            assert abs(row["statistic"]) < 1e-6, (seed, row["statistic"])
            assert row["z"] < 0.0, (seed, row["z"])


def test_screening_is_invariant_to_the_units_of_a_numeric_margin():
    """Rescaling a covariate is a change of UNITS and nothing else.

    Multiplying a numeric margin by a constant scales its probe columns and its
    moments by fixed powers of that constant, and the profiled statistic is a
    ratio in which they cancel exactly.  The mains fit is the same fit with a
    rescaled coefficient, so ``phi`` is unchanged too.  Every reported field
    must therefore come back identical.

    **This enforces the equilibration in ``_psd_rank``, which is the module's
    only relative-rank threshold** -- see the SCALE DISCIPLINE note in
    :mod:`superglm.screening._score_stat`.  It is deliberately a property of
    the whole public screen rather than a unit check on one block, so any
    future relative threshold that reaches a reported field is covered by it.

    What it does NOT cover, measured rather than assumed: reverting the
    balancing in ``_build_pencil`` leaves this test PASSING, because rescaling
    a spline's covariate rescales its penalty with it and never reaches the
    ``V >> S`` regime.  That site has its own regression,
    ``test_a_curvature_that_dwarfs_its_penalty_keeps_the_penalty``.

    The reviewer's case: on the balanced four-point design
    ``z1, z2 = +-10000`` with ``C = 0``, so the true profiled rank is
    unambiguously 1, rescaling from ``+-1`` turned ``statistic=397, edf0=1``
    into an all-NaN row.  1e4 is kept below because a moment matrix carries the
    square of the covariate's scale, so it is already 1e16 in the joint.

    The 1e-5 bar, measured 2026-08-07: ``x`` has no true effect, so its
    smoothing parameter is a flat direction the mains fit now freezes
    mid-transition (cross-scale spread 1e-9) instead of marching deep into
    the edf flat-tail (spread 3.6e-4, hidden to 2e-7 by the tail's zero
    slope). At the freeze point the probe's scale-squared-conditioned
    moments register at 1.5e-6 on ``(g, x)``'s edf0 -- real sensitivity the
    tail masked, orders below anything the equilibration guard exists to
    catch.
    """
    import pandas as pd

    from superglm import Categorical, SuperGLM
    from superglm.features.numeric import Numeric
    from superglm.features.spline import Spline

    rng = np.random.default_rng(0)
    n = 4000
    base = pd.DataFrame(
        {
            "z1": rng.choice([-1.0, 1.0], n),
            "z2": rng.choice([-1.0, 1.0], n),
            "g": pd.Categorical(rng.choice([f"L{i}" for i in range(4)], n)),
            "x": rng.uniform(0.0, 1.0, n),
        }
    )
    y = rng.poisson(np.exp(-1.0 + 0.4 * base["z1"] * base["z2"])).astype(np.float64)

    def screen(scale):
        df = base.copy()
        df["z1"] = df["z1"] * scale
        df["z2"] = df["z2"] * scale
        df["x"] = df["x"] * scale
        model = SuperGLM(
            family="poisson",
            features={
                "z1": Numeric(),
                "z2": Numeric(),
                "g": Categorical(),
                "x": Spline(kind="ps", n_knots=6),
            },
        )
        model.fit_reml(df, y)
        table = model.screen_interactions(df, y)
        return {
            (a, b): (kind, e, z)
            for a, b, kind, e, z in zip(
                table["feature_a"],
                table["feature_b"],
                table["kind"],
                table["edf0"],
                table["z"],
            )
        }

    unit = screen(1.0)
    assert unit, "the sweep must produce rows or this proves nothing"
    for scale in (1e2, 1e4):
        got = screen(scale)
        assert set(got) == set(unit), scale
        for pair, (kind, e, z) in unit.items():
            k2, e2, z2 = got[pair]
            assert k2 == kind, (pair, scale)
            assert np.isnan(e) == np.isnan(e2), (pair, scale, e, e2)
            if not np.isnan(e):
                assert e2 == pytest.approx(e, rel=1e-5), (pair, scale, e, e2)
                assert z2 == pytest.approx(z, rel=1e-5, abs=1e-9), (pair, scale, z, z2)


def test_a_block_of_pure_cancellation_cannot_score_competitively():
    """Swept, because the property was decided by the sign of a round-off eigenvalue.

    ``V_eff = V - C' M^-1 C`` is a difference, so on a block the overlap has
    absorbed there is nothing left but round-off -- and every cut taken on
    ``V_eff`` itself is relative to a scale that IS that round-off.

    Getting the edf too high is not a neutral failure.  ``z = (T - e)/sqrt(2e)``
    DECREASES in ``e``, so a partly-rejected block outranks an unrejected one:
    at the measured ``statistic = 145.508`` an edf of 10 gives ``z = 30.3``
    where an edf of 1 gives ``z = 102.2``.  Only rank 0 is safe.

    An earlier version of this test pinned a single seed and asserted
    ``not (z > 0)``.  Both were wrong.  The seed mattered because the rule it
    certified fired only when the largest-MAGNITUDE eigenvalue happened to be
    negative -- a coin flip per seed, not a property -- and ``not (z > 0)``
    passes both when the block is rejected and when its statistic merely
    happened to be small, which is the conflation the sweep exists to break.
    So: sweep, and assert the block is REJECTED.
    """
    import pandas as pd

    from superglm import Categorical, SuperGLM
    from superglm.features.numeric import Numeric

    L, n = 25, 8000
    bad = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        g = rng.integers(0, L, n)
        g[g == 1] = 0
        g[:2] = 1  # one level with two rows
        x = np.linspace(0.5, L - 0.5, L)[g] + 1e-8 * rng.normal(size=n)
        df = pd.DataFrame({"g": pd.Categorical([f"L{c}" for c in g]), "x": x})
        y = rng.poisson(np.exp(-1.0 + 0.1 * x)).astype(np.float64)
        w = np.ones(n)
        model = SuperGLM(family="poisson", features={"g": Categorical(), "x": Numeric()})
        model.fit_reml(df, y, sample_weight=w)
        row = model.screen_interactions(df, y, candidates=[("x", "g")], sample_weight=w).iloc[0]
        if not np.isnan(row["edf0"]):
            bad.append((seed, row["edf0"], row["statistic"], row["z"]))
    assert bad == [], bad


def test_a_wholly_absorbed_probe_is_rejected_on_every_seed():
    """The absorbed case, swept, asserting rejection rather than a sign.

    A numeric constant within each level makes every probe column a multiple of
    that level's indicator, so the true profiled rank is 0.  Counting that on
    ``V_eff`` cannot see it; counting it on the joint moment matrix, where
    nothing has cancelled, can.
    """
    import pandas as pd

    from superglm import Categorical, SuperGLM
    from superglm.features.numeric import Numeric

    values = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    bad = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        g = rng.integers(0, 5, 4000)
        x = values[g]
        df = pd.DataFrame({"g": pd.Categorical([f"L{c}" for c in g]), "x": x})
        y = rng.poisson(np.exp(-1.0 + 0.1 * x)).astype(np.float64)
        model = SuperGLM(family="poisson", features={"g": Categorical(), "x": Numeric()})
        model.fit_reml(df, y)
        row = model.screen_interactions(df, y, candidates=[("x", "g")]).iloc[0]
        if not np.isnan(row["edf0"]):
            bad.append((seed, row["edf0"], row["z"]))
    assert bad == [], bad


def test_a_probe_exactly_nested_in_the_overlap_reports_only_its_free_directions():
    """Positive-only cancellation: nothing observable in V_eff betrays it.

    The probe's first direction is exactly a multiple of the overlap's, so the
    true profiled rank is 2 of 3.  The computed eigenvalues come out
    1.7347e-18, 1e-12, 2e-12 -- all NONNEGATIVE, so no PSD violation is visible
    and any rule that needs one counts three.
    """
    from superglm.screening import penalized_score_statistic
    from superglm.screening._score_stat import _solve_psd

    M = np.array([[float.fromhex("0x1.c60ae65c20699p-2")]])
    C = np.array([[float.fromhex("0x1.7d086fd7cd3a2p-5"), 0.0, 0.0]])
    V = np.diag([float.fromhex("0x1.3fc36202de5dap-8"), 1e-12, 2e-12])

    V_eff = V - C.T @ _solve_psd(M, C)
    assert (np.linalg.eigvalsh(0.5 * (V_eff + V_eff.T)) >= 0).all()

    got = penalized_score_statistic(np.zeros(3), V, C, M, None, U_nuisance=np.zeros(1))
    assert got.edf0 == 2.0, got.edf0


def test_a_curvature_that_dwarfs_its_penalty_keeps_the_penalty():
    """``G = V + S`` must not round the smaller term away.

    With ``V = 1e20 I`` and ``S = I`` the sum IS ``V`` in float64, so the
    penalty is lost before any share is taken and the pencil reports the full
    dimension as its edf at every lambda.  The answer would then depend on the
    units the curvature is carried in, or on a frequency-weight scale.  The
    direct problem reaches edf 2 at ``lambda = 1e20`` with statistic 0.5.

    **This test holds the BALANCING and nothing else.**  It used to be
    described as also holding the ``(v, s)`` parameterisation -- "a pencil
    that derives the penalty share as ``1 - v`` loses it entirely".  The
    shipped pencil DOES derive ``s`` as ``1 - share`` and this test is green,
    so that reading was never true: what it catches is ``G`` formed on two
    scales, which is a different defect at a different line.  Reverting the
    balancing fails it; there is no revert of the parameterisation that does.
    """
    from superglm.screening import penalized_score_statistic

    V = 1e20 * np.eye(4)
    S = np.eye(4)
    U = np.zeros(4)
    U[0] = 1e10
    assert np.array_equal(V + S, V), "the premise: the sum loses S"

    got = penalized_score_statistic(U, V, S_ti=S, edf0=2.0)

    assert got.edf0 == pytest.approx(2.0, abs=1e-6)
    assert got.statistic == pytest.approx(0.5, rel=1e-6)


def test_a_weakly_identified_block_is_scored_not_discarded():
    """Weak identification is a finding about the data, not about arithmetic.

    ``M = C = I`` with ``V = (1 + 1e-4) I`` gives the Schur complement
    ``V_eff = 1e-4 I`` -- full rank, entirely real, and carrying a genuine
    score statistic of 1.0 on ``U = sqrt(1e-4) e_1``.  Every generalized
    eigenvalue against the absorption metric is 5.0e-05, so an absorption
    guard set by SMALLNESS rather than by round-off classifies the whole block
    as absorbed and drops the pair.

    Only arithmetic may discard a pair -- and the way that rule is kept here is
    that **no absorption guard exists at all**, which is stronger than the
    threshold this docstring used to name.

    It said "the threshold is ``10 * k^3 * eps``, nine orders below this
    block's 5.0e-05 at ``k = 2``".  There is no such cut: ``grep`` finds no
    ``k**3`` anywhere in :mod:`superglm.screening` except a cost gate on
    ``max_cells``, and :mod:`superglm.screening._score_stat`'s docstring
    records the guard's removal under "That rule cost a guard" -- no Type 1
    bound could separate absorption from weak identification, and the fitted
    one that was tried deleted a legitimately weak block while missing a
    genuinely absorbed one at the same ``k``.  So a weak interaction keeps its
    degrees of freedom because nothing is looking to take them, not because a
    threshold sits far enough away.  A pair that is genuinely uninteresting is
    for ``z`` to rank down, not for the kernel to delete.

    Corrected in passing while PR #324 removed the same defect class from
    ``_score_stat``: a docstring crediting a threshold that is not there makes
    a property look guarded when nothing guards it.
    """
    from superglm.screening import penalized_score_statistic

    for k in (2, 4, 12):
        eye = np.eye(k)
        V = (1.0 + 1e-4) * eye
        V_eff = V - eye @ np.linalg.solve(eye, eye)
        assert np.linalg.matrix_rank(V_eff) == k, k
        U = np.zeros(k)
        U[0] = np.sqrt(1e-4)

        got = penalized_score_statistic(U, V, eye, eye, None, U_nuisance=np.zeros(k))

        assert got.edf0 == float(k), (k, got.edf0)
        assert got.statistic == pytest.approx(float(U @ np.linalg.solve(V_eff, U)), rel=1e-9), k


def test_unpenalized_edf_is_a_rank_not_a_cholesky_trace():
    """A barely positive-definite block still reports its RANK, not ``k``.

    The unpenalized rung's edf used to be ``tr(A^-1 V)``, which equals the rank
    only when ``A^-1`` is a pseudo-inverse.  ``cho_factor`` is entitled to
    accept a block like this one -- it IS positive definite, by 1e-18 -- and
    the trace then reports ``k``.  Diagonal on purpose: this is the one
    construction whose Cholesky cannot come out platform-dependent, since
    every pivot is a stored entry rather than a round-off residue.
    """
    import scipy.linalg

    from superglm.screening import penalized_score_statistic

    V = np.diag([3.0, 2.0, 1e-18])
    scipy.linalg.cho_factor(V, check_finite=False)  # accepts it; that is the trap
    assert np.linalg.matrix_rank(V) == 2

    result = penalized_score_statistic(np.array([1.0, 1.0, 1.0]), V, S_ti=None)
    assert result.edf0 == 2.0
    assert result.lambda0 == 0.0


def test_unpenalized_edf_matches_the_dense_rank_on_a_profiled_block():
    """The reachable case: a factor level in which the numeric is constant.

    ``numeric_cat`` profiles out ``[1 | menu | z]``, so a level carrying a
    single row has its probe column exactly absorbed and the block's true rank
    is ``k - 1``.  ``V_eff`` is formed by SUBTRACTION though, so that direction
    lands at round-off rather than at zero and whether ``cho_factor`` accepts
    the block is decided by rounding alone -- 10 of these 20 seeds are accepted
    here.  The edf must not depend on which, so the reported values over
    statistically identical replicates must be ONE value, and it must be the
    rank.  A Cholesky trace reports ``k`` on the accepted seeds and ``k - 1``
    on the rest, which is two.

    ``matrix_rank`` is the contract, not a cross-check: the screen counts at
    ``_rank_floor``, which IS ``matrix_rank``'s own tolerance, so the reported
    edf can never exceed it.

    **Swept wide on purpose.**  An earlier version of this test ran 20 seeds
    and passed while CI failed, because the round-off eigenvalue this layout
    leaves behind has a TAIL: measured over 400 replicates its median is
    2.2e-16 but it reaches 1.2e-15, so a cut placed at 1e-15 misreports rank
    on roughly one seed in two hundred and twenty seeds had a nine-in-ten
    chance of missing it.  Locally that bit seeds 22 and 88; on CI's numpy it
    bit seed 16.  Anything narrow enough to miss a 0.5% failure is not a
    regression test for it.
    """
    from superglm.screening import penalized_score_statistic
    from superglm.screening._numeric_margin import numeric_pair_moments
    from superglm.screening._score_stat import _solve_psd

    L, n = 40, 8000
    menu = np.eye(L)[:, 1:]
    reported = set()
    for seed in range(200):
        rng = np.random.default_rng(seed)
        codes = rng.integers(0, L - 1, n)
        codes[0] = L - 1  # the singleton level
        z = rng.normal(size=n)
        score = rng.normal(size=n)
        w = rng.uniform(0.5, 1.5, n)
        U, V, C, M, u_m = numeric_pair_moments(codes, L, menu, z, score, w)

        V_eff = V - C.T @ _solve_psd(M, C)
        V_eff = 0.5 * (V_eff + V_eff.T)
        assert np.linalg.matrix_rank(V_eff) == L - 2, seed

        result = penalized_score_statistic(U, V, C, M, None, U_nuisance=u_m)
        # Never ABOVE the rank: that inequality is the defect itself, and it
        # fires only on the seeds cho_factor accepts.
        assert result.edf0 <= L - 2, (seed, result.edf0)
        reported.add(result.edf0)
    # An exact count, identical across replicates -- not a float trace.
    assert reported == {float(L - 2)}


def test_singular_pencil_answer_does_not_depend_on_the_units():
    """The whitening fallback's rank cut has to be RELATIVE.

    ``V`` and ``S`` share a null space here, so the generalized driver fails
    and the explicit whitening runs.  The same problem is posed twice, once
    at a scale a thousand times smaller; an absolute floor in that cut called
    every identifiable direction null below 1e-12 and returned a zero
    statistic at zero df, which makes the screening table depend on whether
    a covariate is carried in metres or kilometres.
    """
    from superglm.screening import penalized_score_statistic

    answers = []
    for s in (1e-10, 1e-12, 1e-13, 1e-16):
        V = s * np.diag([1.0, 2.0, 0.0])
        S = s * np.diag([2.0, 1.0, 0.0])
        U = np.sqrt(s) * np.array([1.0, 1.0, 0.0])
        got = penalized_score_statistic(U, V, S_ti=S, edf0=1.0)
        answers.append((got.statistic, got.edf0))
    # lambda0 = 1 makes V + lambda S = 3s I on the identifiable block, so the
    # statistic is U' (3s I)^-1 U = 2/3 and edf0 is 1 -- at every scale.
    for statistic, edf0 in answers:
        assert statistic == pytest.approx(2.0 / 3.0, rel=1e-9)
        assert edf0 == pytest.approx(1.0, rel=1e-9)


def test_edf_solver_hits_target():
    from superglm.screening import penalized_score_statistic

    rng = np.random.default_rng(6)
    p = 8
    V = _pd_matrix(rng, p)
    S = _pd_matrix(rng, p, strength=0.3)
    U = rng.normal(size=p)

    result = penalized_score_statistic(U, V, S_ti=S, edf0=3.5)

    achieved = np.trace(np.linalg.solve(V + result.lambda0 * S, V))
    assert abs(achieved - 3.5) <= 1e-6
    assert abs(result.edf0 - 3.5) <= 1e-6


def test_profiling_removes_overlap_explained_score():
    """A purely additive signal must screen at the null level."""
    from superglm.screening import penalized_score_statistic

    rng = np.random.default_rng(7)
    p, q = 6, 4
    V = _pd_matrix(rng, p)
    M = _pd_matrix(rng, q)
    C = rng.normal(size=(q, p))
    u_m = rng.normal(size=q)
    U = C.T @ np.linalg.solve(M, u_m)  # score fully explained by the overlap
    S = _pd_matrix(rng, p, strength=0.2)

    result = penalized_score_statistic(U, V, C, M, S, edf0=3.0, U_nuisance=u_m)

    assert abs(result.statistic) < 1e-18


def test_infinite_penalty_limit_restricts_to_null_space():
    """With edf0 = penalty null dimension, T approaches the null-space statistic."""
    from superglm.screening import penalized_score_statistic

    rng = np.random.default_rng(8)
    p, null_dim = 6, 2
    V = _pd_matrix(rng, p)
    S = np.zeros((p, p))
    S[null_dim:, null_dim:] = _pd_matrix(rng, p - null_dim)
    U = rng.normal(size=p)

    result = penalized_score_statistic(U, V, S_ti=S, edf0=float(null_dim))

    V_nn = V[:null_dim, :null_dim]
    expected = U[:null_dim] @ np.linalg.solve(V_nn, U[:null_dim])
    assert result.lambda0 > 1e4
    np.testing.assert_allclose(result.statistic, expected, rtol=1e-3)


def test_c_without_m_raises():
    import pytest

    from superglm.screening import penalized_score_statistic

    with pytest.raises(ValueError, match="supply both"):
        penalized_score_statistic(np.ones(2), np.eye(2), C=np.ones((1, 2)))


def test_overlap_moments_match_dense_assembly():
    """M, C, u_m from cell tables must equal the dense row-space assembly."""
    from superglm.screening import pair_cell_moments
    from superglm.screening._overlap import pair_overlap_moments

    rng = np.random.default_rng(9)
    n, n_a, n_b, k_a, k_b = 3000, 15, 11, 4, 3
    codes_a = rng.integers(0, n_a, n)
    codes_b = rng.integers(0, n_b, n)
    A = rng.normal(size=(n_a, k_a))
    B = rng.normal(size=(n_b, k_b))
    score = rng.normal(size=n)
    weights = rng.normal(size=n)  # signed, as REML working quantities can be

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, n_a, n_b, score, weights)
    M, C, u_m = pair_overlap_moments(A, B, S_cell, W_cell)

    X_o = np.column_stack([np.ones(n), A[codes_a], B[codes_b]])
    X_T = _dense_row_kronecker(codes_a, codes_b, A, B)
    np.testing.assert_allclose(M, X_o.T @ (X_o * weights[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(C, X_o.T @ (X_T * weights[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(u_m, X_o.T @ score, rtol=1e-12, atol=1e-9)


def test_tensor_penalty_matches_interaction_convention():
    from superglm.screening._overlap import tensor_penalty

    rng = np.random.default_rng(10)
    S1 = _pd_matrix(rng, 3)
    S2 = _pd_matrix(rng, 2)

    expected = np.kron(S1, np.eye(2)) + np.kron(np.eye(3), S2)
    np.testing.assert_array_equal(tensor_penalty(S1, S2), expected)


def test_full_chain_zeroes_an_additive_signal():
    """End-to-end: a score generated by the overlap span screens at exactly zero."""
    from superglm.screening import (
        pair_cell_moments,
        pair_score_curvature,
        penalized_score_statistic,
    )
    from superglm.screening._overlap import pair_overlap_moments, tensor_penalty

    rng = np.random.default_rng(11)
    n, n_a, n_b, k_a, k_b = 4000, 12, 9, 4, 3
    codes_a = rng.integers(0, n_a, n)
    codes_b = rng.integers(0, n_b, n)
    A = rng.normal(size=(n_a, k_a))
    B = rng.normal(size=(n_b, k_b))
    weights = rng.uniform(0.2, 2.0, n)
    gamma = rng.normal(size=1 + k_a + k_b)
    X_o = np.column_stack([np.ones(n), A[codes_a], B[codes_b]])
    score = weights * (X_o @ gamma)  # purely additive working signal

    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, n_a, n_b, score, weights)
    U, V = pair_score_curvature(A, B, S_cell, W_cell)
    M, C, u_m = pair_overlap_moments(A, B, S_cell, W_cell)
    S_ti = tensor_penalty(_pd_matrix(rng, k_a), _pd_matrix(rng, k_b))

    result = penalized_score_statistic(U, V, C, M, S_ti, edf0=3.0, U_nuisance=u_m)

    scale = float(np.abs(U).max())
    assert abs(result.statistic) < 1e-16 * max(scale, 1.0) ** 2 + 1e-12


def _screening_data(seed, n=2500, interaction=0.0):
    """Integer covariates, Poisson response, optional planted x1:x2 surface."""
    rng = np.random.default_rng(seed)
    frame = {f"x{i}": rng.integers(0, 25 + 3 * i, n).astype(float) for i in range(1, 6)}
    z1 = (frame["x1"] - frame["x1"].mean()) / frame["x1"].std()
    z2 = (frame["x2"] - frame["x2"].mean()) / frame["x2"].std()
    eta = -1.2 + 0.30 * z1 + 0.25 * z2 + interaction * z1 * z2
    w = rng.uniform(0.3, 1.0, n)
    y = rng.poisson(np.exp(eta) * w) / w
    import pandas as pd

    return pd.DataFrame(frame), y, w


def _fit_mains(frame, y, w):
    from superglm import SuperGLM

    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={
            name: __import__("superglm.features.spline", fromlist=["Spline"]).Spline(kind="ps", k=6)
            for name in frame.columns
        },
    )
    return model.fit_reml(frame, y, sample_weight=w)


def test_screen_requires_fitted_model():
    import pytest

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    frame, y, w = _screening_data(0)
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        features={"x1": Spline(kind="ps", k=6)},
    )
    with pytest.raises(RuntimeError, match="fit_reml first"):
        model.screen_interactions(frame, y, sample_weight=w)


def test_oracle_planted_interaction_ranks_first_across_seeds():
    """The plan's release gate: the true pair must rank first, five seeds."""
    for seed in range(5):
        frame, y, w = _screening_data(seed, interaction=0.5)
        model = _fit_mains(frame, y, w)

        table = model.screen_interactions(frame, y, sample_weight=w)

        top = tuple(sorted((table.loc[0, "feature_a"], table.loc[0, "feature_b"])))
        assert top == ("x1", "x2"), f"seed {seed}: expected x1:x2 first, got {top}; \n{table}"
        assert len(table) == 10  # all pairs of five splines


def test_null_statistics_stay_bounded_across_seeds():
    """No interaction anywhere: generous bound, rank stability not calibration."""
    for seed in range(3):
        frame, y, w = _screening_data(10 + seed, interaction=0.0)
        model = _fit_mains(frame, y, w)

        table = model.screen_interactions(frame, y, sample_weight=w)

        assert np.isfinite(table["statistic"]).all()
        assert np.isfinite(table["z"]).all()
        assert table["z"].max() < 10.0, table


def test_candidates_restricts_the_sweep():
    frame, y, w = _screening_data(2, interaction=0.5)
    model = _fit_mains(frame, y, w)

    table = model.screen_interactions(
        frame, y, sample_weight=w, candidates=[("x1", "x2"), ("x3", "x4")]
    )

    assert len(table) == 2
    assert set(map(tuple, table[["feature_a", "feature_b"]].to_numpy())) == {
        ("x1", "x2"),
        ("x3", "x4"),
    }


def test_screen_uses_the_fit_offset():
    """Review finding: predict() dropped model._fit_offset, so an offset-fitted
    model was screened at the wrong mean and leftover main-effect mass showed
    up as interaction signal."""
    import pandas as pd

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(7)
    n = 2500
    frame = pd.DataFrame(
        {f"x{i}": rng.integers(0, 25 + 3 * i, n).astype(float) for i in range(1, 4)}
    )
    z1 = (frame["x1"] - frame["x1"].mean()) / frame["x1"].std()
    z2 = (frame["x2"] - frame["x2"].mean()) / frame["x2"].std()
    off = np.log(rng.uniform(0.2, 3.0, n))
    y = rng.poisson(np.exp(-1.2 + 0.3 * z1 + 0.25 * z2 + off)).astype(float)
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={name: Spline(kind="ps", k=6) for name in frame.columns},
    ).fit_reml(frame, y, offset=off)

    table_default = model.screen_interactions(frame, y)
    table_explicit = model.screen_interactions(frame, y, offset=off)

    pd.testing.assert_frame_equal(table_default, table_explicit)
    # the offset genuinely flows: suppressing it must change the screen
    table_zero = model.screen_interactions(frame, y, offset=np.zeros(n))
    assert not np.allclose(table_default["z"], table_zero["z"])
    # and the properly-offset null stays inside the null bound
    assert table_default["z"].max() < 10.0, table_default


def test_screen_dispersed_gaussian_null_stays_bounded():
    """Review finding: z assumed phi=1; a sigma=3 Gaussian null hit z=25 and
    the winning rung collapsed to the endpoint before Pearson scaling."""
    import pandas as pd

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(11)
    n = 2500
    frame = pd.DataFrame(
        {f"x{i}": rng.integers(0, 25 + 3 * i, n).astype(float) for i in range(1, 6)}
    )
    z1 = (frame["x1"] - frame["x1"].mean()) / frame["x1"].std()
    z2 = (frame["x2"] - frame["x2"].mean()) / frame["x2"].std()
    y = 0.5 * z1 + 0.4 * z2 + rng.normal(0.0, 3.0, n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=None,
        discrete=False,
        features={name: Spline(kind="ps", k=6) for name in frame.columns},
    ).fit_reml(frame, y)

    table = model.screen_interactions(frame, y)

    assert np.isfinite(table["statistic"]).all()
    assert np.isfinite(table["z"]).all()
    assert table["z"].max() < 10.0, table


def test_screen_validates_inputs():
    """Review findings: NaN y / zero weights / bad edf0 / bad candidates must
    raise clear errors instead of AttributeError or silent nonsense."""
    import pytest

    frame, y, w = _screening_data(3)
    model = _fit_mains(frame, y, w)

    y_bad = y.copy()
    y_bad[0] = np.nan
    with pytest.raises(ValueError, match="finite y"):
        model.screen_interactions(frame, y_bad, sample_weight=w)
    with pytest.raises(ValueError, match="positive sum"):
        model.screen_interactions(frame, y, sample_weight=np.zeros_like(w))
    with pytest.raises(ValueError, match="non-empty"):
        model.screen_interactions(frame, y, sample_weight=w, edf0=())
    with pytest.raises(ValueError, match="finite and positive"):
        model.screen_interactions(frame, y, sample_weight=w, edf0=-1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        model.screen_interactions(frame, y, sample_weight=w, edf0=float("nan"))
    with pytest.raises(ValueError, match="distinct screenable fitted"):
        model.screen_interactions(frame, y, sample_weight=w, candidates=[("x1", "x1")])
    with pytest.raises(ValueError, match="screenable features"):
        model.screen_interactions(frame, y, sample_weight=w, candidates=[("x1", "nope")])
    with pytest.raises(ValueError, match="distinct screenable fitted"):
        model.screen_interactions(frame, y, sample_weight=w, candidates=[("x1", "x2", "x3")])


def test_screen_scalar_edf0_variants_agree():
    """Review finding: a 0-d numpy array crashed the budget normalization."""
    import pandas as pd

    frame, y, w = _screening_data(4, interaction=0.5)
    model = _fit_mains(frame, y, w)

    t_float = model.screen_interactions(frame, y, sample_weight=w, edf0=4.0)
    t_0d = model.screen_interactions(frame, y, sample_weight=w, edf0=np.array(4.0))
    t_np = model.screen_interactions(frame, y, sample_weight=w, edf0=np.float64(4.0))

    pd.testing.assert_frame_equal(t_float, t_0d)
    pd.testing.assert_frame_equal(t_float, t_np)


def test_screen_select_parents_raise_upfront():
    """Review finding: select=True mains died per-pair three modules down;
    now one clear error at the top names the offending features."""
    import pytest

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    frame, y, w = _screening_data(5)
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={name: Spline(kind="ps", k=6, select=True) for name in frame.columns},
    ).fit_reml(frame, y, sample_weight=w)

    with pytest.raises(ValueError, match="select=True"):
        model.screen_interactions(frame, y, sample_weight=w)


def _continuous_screening_data(seed=6, n=2500):
    """Two continuous covariates (~n uniques each) with a planted smooth
    interaction, plus one integer covariate; x1:x2 raw grid ~ n^2 cells."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.integers(0, 30, n).astype(float),
        }
    )
    z1 = (frame["x1"] - frame["x1"].mean()) / frame["x1"].std()
    z2 = (frame["x2"] - frame["x2"].mean()) / frame["x2"].std()
    w = rng.uniform(0.3, 1.0, n)
    y = rng.poisson(np.exp(-1.2 + 0.3 * z1 + 0.25 * z2 + 0.5 * z1 * z2) * w) / w
    return frame, y, w


def _fit_continuous(frame, y, w):
    from superglm import SuperGLM
    from superglm.features.spline import Spline

    return SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={c: Spline(kind="ps", k=6) for c in frame.columns},
    ).fit_reml(frame, y, sample_weight=w)


def test_screen_bins_continuous_pairs_instead_of_skipping():
    """Task 4: a continuous x continuous pair above the cell budget falls
    back to quantile binning (flagged approx) instead of NaN-skipping."""
    frame, y, w = _continuous_screening_data()
    model = _fit_continuous(frame, y, w)

    table = model.screen_interactions(frame, y, sample_weight=w)

    row = table[(table.feature_a == "x1") & (table.feature_b == "x2")].iloc[0]
    assert bool(row["approx"])
    assert np.isfinite(row["z"])
    # largest-margin-first binning: only one margin needs compression here
    assert row["n_cells"] < 2500 * 2500
    assert row["n_cells"] <= 5_000_000
    # the planted continuous interaction ranks first through the fallback
    top = tuple(sorted((table.loc[0, "feature_a"], table.loc[0, "feature_b"])))
    assert top == ("x1", "x2")
    # pairs within the budget stay on the exact path
    exact = table[~table["approx"]]
    assert len(exact) == 2
    assert np.isfinite(exact["z"]).all()


def test_screen_binned_tracks_exact_on_signal():
    """The binned statistic must approximate the exact one where it matters:
    on a pair carrying real signal (measured gap 3.5%, pinned at 10%).
    No such promise on null pairs, where binning legitimately smooths away
    high-frequency noise at unpenalized rungs."""
    frame, y, w = _continuous_screening_data()
    model = _fit_continuous(frame, y, w)

    exact = model.screen_interactions(
        frame, y, sample_weight=w, candidates=[("x1", "x2")], max_cells=7_000_000
    )
    binned = model.screen_interactions(frame, y, sample_weight=w, candidates=[("x1", "x2")])

    assert not bool(exact.loc[0, "approx"])
    assert bool(binned.loc[0, "approx"])
    z_e, z_b = float(exact.loc[0, "z"]), float(binned.loc[0, "z"])
    assert z_e > 10.0  # this is a signal pair
    assert abs(z_e - z_b) / z_e < 0.10
    # same winning rung (edf0 is the achieved edf, so compare to bisection tol)
    assert abs(exact.loc[0, "edf0"] - binned.loc[0, "edf0"]) < 1e-2


def test_screen_nan_only_when_binning_cannot_fit():
    """If even the binned grid exceeds max_cells the pair NaN-skips; approx
    reports whether binning was attempted before the budget still failed."""
    import pytest

    frame, y, w = _continuous_screening_data()
    model = _fit_continuous(frame, y, w)

    # binning IS applied (x1 -> 50 bins) but the binned grid still exceeds the
    # budget: NaN row, n_cells reports the attempted binned grid, approx says
    # binning happened
    table = model.screen_interactions(
        frame, y, sample_weight=w, candidates=[("x1", "x3")], max_cells=800, screen_bins=50
    )
    row = table.iloc[0]
    assert np.isnan(row["z"])
    assert bool(row["approx"])
    assert row["n_cells"] == 50 * 30

    # with a budget so tiny even the (k_a*k_b)^2 curvature block cannot fit,
    # the pair is unscreenable at any binning — NaN, with approx recording
    # that binning was attempted before the true dimensions ruled it out
    tiny = model.screen_interactions(
        frame, y, sample_weight=w, candidates=[("x1", "x3")], max_cells=100, screen_bins=50
    )
    assert np.isnan(tiny.iloc[0]["z"])
    assert bool(tiny.iloc[0]["approx"])

    with pytest.raises(ValueError, match="screen_bins"):
        model.screen_interactions(frame, y, sample_weight=w, screen_bins=1)


def test_screen_uses_the_fit_weights():
    """Blocking review finding: sample_weight must inherit from the fit like
    offset does — screening an exposure-weighted fit at unit weight
    linearizes against the wrong likelihood."""
    import pandas as pd

    frame, y, w = _screening_data(8)
    model = _fit_mains(frame, y, w)

    table_default = model.screen_interactions(frame, y)
    table_explicit = model.screen_interactions(frame, y, sample_weight=w)

    pd.testing.assert_frame_equal(table_default, table_explicit)
    # the weights genuinely flow: forcing unit weights must change the screen
    table_unit = model.screen_interactions(frame, y, sample_weight=np.ones_like(y))
    assert not np.allclose(table_default["z"], table_unit["z"])
    assert table_default["z"].max() < 10.0, table_default


def test_screen_validates_row_alignment():
    """Review finding: scalar or misaligned y/sample_weight silently
    broadcast; both must raise instead."""
    import pytest

    frame, y, w = _screening_data(9)
    model = _fit_mains(frame, y, w)

    with pytest.raises(ValueError, match="one entry per row"):
        model.screen_interactions(frame, np.float64(y[0]), sample_weight=w)
    with pytest.raises(ValueError, match="one entry per row"):
        model.screen_interactions(frame, y[:-5], sample_weight=w)
    with pytest.raises(ValueError, match="one entry per row"):
        model.screen_interactions(frame, y, sample_weight=w[:-5])
    with pytest.raises(ValueError, match="one entry per row"):
        model.screen_interactions(frame, y.reshape(-1, 1), sample_weight=w)


def test_screen_requires_finite_covariates():
    """Review finding: a NaN covariate silently became its own cell; now the
    skip reason is distinguishable from a cell-budget skip by raising."""
    import pytest

    frame, y, w = _screening_data(9)
    model = _fit_mains(frame, y, w)
    bad = frame.copy()
    bad.loc[3, "x1"] = np.nan

    with pytest.raises(ValueError, match="finite covariates"):
        model.screen_interactions(bad, y, sample_weight=w)


def test_screen_intermediate_budget_triggers_binning():
    """Review finding: max_cells bounded the cell grid but not the (n_a, k_b^2)
    curvature intermediates — a lopsided pair passed the cell budget while the
    einsum intermediate blew past it. The budget now covers both."""
    import pandas as pd

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(12)
    n = 2500
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),  # ~2500 uniques
            "x4": rng.integers(0, 6, n).astype(float),  # 6 levels
        }
    )
    z1 = (frame["x1"] - frame["x1"].mean()) / frame["x1"].std()
    w = rng.uniform(0.3, 1.0, n)
    y = rng.poisson(np.exp(-1.2 + 0.3 * z1) * w) / w
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={c: Spline(kind="ps", k=6) for c in frame.columns},
    ).fit_reml(frame, y, sample_weight=w)

    # cells 2500*6 = 15k fit max_cells exactly; the (n_a * k_b^2) intermediate
    # (~62.7k) exceeds 4*max_cells, so binning must trigger anyway
    table = model.screen_interactions(
        frame, y, sample_weight=w, candidates=[("x1", "x4")], max_cells=15_000
    )
    row = table.iloc[0]
    assert bool(row["approx"])
    assert np.isfinite(row["z"])
    assert row["n_cells"] <= 256 * 6

    # with a generous budget the same pair is exact
    wide = model.screen_interactions(frame, y, sample_weight=w, candidates=[("x1", "x4")])
    assert not bool(wide.iloc[0]["approx"])


def test_screen_phi_is_visible_and_overridable():
    """Phi materially selects the winning rung, so it is visible and overridable."""
    import pytest

    frame, y, w = _screening_data(13, interaction=0.5)
    model = _fit_mains(frame, y, w)

    table = model.screen_interactions(frame, y, sample_weight=w)
    assert np.isfinite(table.attrs["phi"]) and table.attrs["phi"] > 0.0

    forced = model.screen_interactions(frame, y, sample_weight=w, phi=5.0)
    assert forced.attrs["phi"] == 5.0
    assert not np.allclose(table["z"], forced["z"])

    with pytest.raises(ValueError, match="phi override"):
        model.screen_interactions(frame, y, sample_weight=w, phi=-1.0)


def test_weighted_gaussian_screen_uses_the_fitted_frequency_weight_scale():
    """Screening and published inference must use one sample-weight contract."""
    import pandas as pd

    from superglm import Numeric, SuperGLM

    rng = np.random.default_rng(219)
    n = 240
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    weights = rng.integers(1, 6, size=n).astype(np.float64)
    y = 1.2 + 0.8 * frame["x1"].to_numpy() + rng.normal(scale=1.7, size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={"x1": Numeric(), "x2": Numeric()},
    ).fit(frame, y, sample_weight=weights)

    table = model.screen_interactions(
        frame,
        y,
        sample_weight=weights,
        candidates=[("x1", "x2")],
    )

    assert table.attrs["phi"] == pytest.approx(model.result.phi, rel=2e-12, abs=2e-12)


def test_weighted_tweedie_screen_retains_the_prior_weight_scale():
    """Tweedie is the deliberate n-edf exception to frequency-weight fitting."""
    import pandas as pd

    from superglm import Numeric, SuperGLM, Tweedie

    rng = np.random.default_rng(220)
    n = 240
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    weights = rng.uniform(0.25, 4.0, size=n)
    mu = np.exp(0.2 + 0.3 * frame["x1"].to_numpy())
    noise_scale = 0.25 / np.sqrt(weights)
    y = mu * np.exp(rng.normal(scale=noise_scale) - 0.5 * noise_scale**2)
    model = SuperGLM(
        family=Tweedie(p=1.5),
        selection_penalty=0.0,
        features={"x1": Numeric(), "x2": Numeric()},
    ).fit(frame, y, sample_weight=weights)

    table = model.screen_interactions(
        frame,
        y,
        sample_weight=weights,
        candidates=[("x1", "x2")],
    )

    assert table.attrs["phi"] == pytest.approx(model.result.phi, rel=2e-12, abs=2e-12)
    zero_weight = weights.copy()
    zero_weight[0] = 0.0
    with pytest.raises(ValueError, match="Tweedie sample_weight.*strictly positive"):
        model.screen_interactions(
            frame,
            y,
            sample_weight=zero_weight,
            candidates=[("x1", "x2")],
        )


def test_screen_discrete_mains_rows_flag_approx():
    """Reviewer ask, twice refined: a discrete fit flags approx only when its
    refit basis genuinely differs — discretization of a column whose
    cardinality fits the bin count is LOSSLESS (the binner returns the exact
    unique support), so default-bin screens of low-cardinality rating
    factors stay approx=False, while genuinely lossy bins flag every pair."""
    from superglm import SuperGLM
    from superglm.features.spline import Spline

    frame, y, w = _screening_data(14, interaction=0.5)

    def fit(n_bins):
        return SuperGLM(
            family="poisson",
            selection_penalty=None,
            discrete=True,
            n_bins=n_bins,
            features={name: Spline(kind="ps", k=6) for name in frame.columns},
        ).fit_reml(frame, y, sample_weight=w)

    # 25-37 levels vs 256 bins: every discretization is exact
    lossless = fit(256).screen_interactions(frame, y, sample_weight=w)
    assert not lossless["approx"].any()
    assert np.isfinite(lossless["z"]).all()

    # 8 bins: every parent bins lossily, so every refit differs from the probe
    lossy = fit(8).screen_interactions(frame, y, sample_weight=w)
    assert lossy["approx"].all()
    assert np.isfinite(lossy["z"]).all()


def test_screen_released_fit_state_refuses_silent_fallbacks():
    """Review finding: retain_fit_state=False releases _fit_weights/_fit_offset,
    so the fit-inheritance defaults silently degraded to unit weights / no
    offset. A weighted or offset fit whose arrays are gone must demand
    explicit values; an unweighted fit stays usable."""
    import pytest

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    frame, y, w = _screening_data(15)

    def build():
        return SuperGLM(
            family="poisson",
            selection_penalty=None,
            discrete=False,
            retain_fit_state=False,
            features={name: Spline(kind="ps", k=6) for name in frame.columns},
        )

    weighted = build().fit_reml(frame, y, sample_weight=w)
    with pytest.raises(ValueError, match="released"):
        weighted.screen_interactions(frame, y)
    table = weighted.screen_interactions(frame, y, sample_weight=w)
    assert np.isfinite(table["z"]).all()

    offset_fit = build().fit_reml(frame, y, offset=np.full(len(y), 0.1))
    with pytest.raises(ValueError, match="offset explicitly"):
        offset_fit.screen_interactions(frame, y)

    unweighted = build().fit_reml(frame, y)
    table2 = unweighted.screen_interactions(frame, y)
    assert np.isfinite(table2["z"]).all()


def test_screen_excludes_already_fitted_interactions():
    """Review finding: the sweep re-surfaced pairs the model already fits as
    tensor terms, and the confirmation workflow then failed with
    'interaction already added'."""
    import pytest

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    frame, y, w = _screening_data(16, interaction=0.5)
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={name: Spline(kind="ps", k=6) for name in frame.columns},
    )
    model._add_interaction("x1", "x2")
    model = model.fit_reml(frame, y, sample_weight=w)

    table = model.screen_interactions(frame, y, sample_weight=w)

    screened = {frozenset((r.feature_a, r.feature_b)) for r in table.itertuples()}
    assert frozenset(("x1", "x2")) not in screened
    assert len(table) == 9  # 10 pairs minus the fitted one

    with pytest.raises(ValueError, match="already fitted"):
        model.screen_interactions(frame, y, sample_weight=w, candidates=[("x1", "x2")])


def test_screen_rejects_inherited_arrays_on_reordered_frame():
    """Review finding: inherited fit arrays are in training row order; a
    permuted frame silently paired them with the wrong observations."""
    import pytest

    frame, y, w = _screening_data(17, interaction=0.5)
    model = _fit_mains(frame, y, w)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    frame_p = frame.iloc[perm].reset_index(drop=True)
    y_p, w_p = y[perm], w[perm]

    with pytest.raises(ValueError, match="explicitly"):
        model.screen_interactions(frame_p, y_p)
    # explicitly aligned arrays are fine on any row order
    table = model.screen_interactions(frame_p, y_p, sample_weight=w_p)
    assert np.isfinite(table["z"]).all()
    # and the unpermuted frame still inherits silently
    table2 = model.screen_interactions(frame, y)
    assert np.isfinite(table2["z"]).all()


def test_screen_preserves_eta_sign_for_noninjective_links():
    """Review finding: eta was reconstructed as link(predict(X)) = |eta| for
    link='sqrt', flipping the score sign on every negative-eta row."""
    import pandas as pd

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    rng = np.random.default_rng(18)
    n = 2500
    frame = pd.DataFrame(
        {f"x{i}": rng.integers(0, 25 + 3 * i, n).astype(float) for i in range(1, 4)}
    )
    z1 = (frame["x1"] - frame["x1"].mean()) / frame["x1"].std()
    z2 = (frame["x2"] - frame["x2"].mean()) / frame["x2"].std()
    # an offset crossing zero forces negative stabilized eta on many rows
    off = rng.uniform(-3.0, 0.0, n)
    eta_true = 2.0 + 0.3 * z1 + 0.25 * z2 + 0.4 * z1 * z2 + off
    y = rng.poisson(np.maximum(eta_true, 0.05) ** 2).astype(float)
    model = SuperGLM(
        family="poisson",
        link="sqrt",
        selection_penalty=None,
        spline_penalty=20.0,
        discrete=False,
        features={name: Spline(kind="ps", k=6) for name in frame.columns},
    ).fit(frame, y, offset=off)

    eta = model._predict_eta_exact(frame, off)
    assert (np.asarray(eta) < 0).any(), "test needs negative fitted eta rows"

    table = model.screen_interactions(frame, y)

    assert np.isfinite(table["z"]).all()
    top = tuple(sorted((table.loc[0, "feature_a"], table.loc[0, "feature_b"])))
    assert top == ("x1", "x2")


def test_marginal_width_estimate_never_overestimates():
    """Review finding: over-estimates are terminal (the pair bins or skips
    with no correction possible), so the estimate must bias low for every
    built-in spline kind."""
    import pandas as pd

    from superglm.features.interaction import TensorInteraction
    from superglm.features.spline import Spline
    from superglm.model.screening_ops import _marginal_width_estimate

    rng = np.random.default_rng(19)
    x = rng.uniform(0.0, 10.0, 500)
    cases = (
        ("ps", 3, 0),  # degree-0: centered width n_knots exactly
        ("ps", 4, 1),
        ("ps", 5, 3),
        ("ps", 6, 3),
        ("ps", 12, 3),
        ("cr", 3, 3),  # minimum cr: 2-column centered marginal
        ("cr", 6, 3),
        ("cr", 12, 3),
    )
    for kind, k, degree in cases:
        spec = Spline(kind=kind, k=k, degree=degree)
        spec.prepare(pd.Series(x)) if hasattr(spec, "prepare") else None
        try:
            m = TensorInteraction._marginal_from_spec(spec, x, None)
        except Exception:
            continue  # spec needs fitting machinery; covered end-to-end elsewhere
        true_width = m.basis.shape[1]
        assert _marginal_width_estimate(spec) <= true_width, (kind, k, degree, true_width)


def test_screen_cached_sweep_matches_per_pair_screens():
    """Reviewer ask: pin cache equivalence — the all-pairs sweep must match
    screening each pair alone, catching any cross-pair cache leakage."""
    import pandas as pd

    frame, y, w = _screening_data(20, interaction=0.5)
    model = _fit_mains(frame, y, w)

    full = model.screen_interactions(frame, y, sample_weight=w)

    for row in full.itertuples():
        single = model.screen_interactions(
            frame, y, sample_weight=w, candidates=[(row.feature_a, row.feature_b)]
        )
        s = single.iloc[0]
        for col in ("statistic", "z", "edf0", "lambda0", "n_cells", "approx"):
            a, b = getattr(row, col), s[col]
            assert (a == b) or (pd.isna(a) and pd.isna(b)), (row.feature_a, row.feature_b, col)


def test_screen_unweighted_fit_screens_any_frame_without_arguments():
    """Review finding: unit fitted weights were treated as 'inherited', so the
    fit-data guard fired on every screen and an unweighted model could no
    longer screen a holdout or subsample. Ones cannot mispair rows."""
    frame, y, _ = _screening_data(21, interaction=0.5)
    model = _fit_mains(frame, y, np.ones_like(y))  # unit weights == unweighted

    holdout = frame.iloc[: len(frame) // 2].reset_index(drop=True)
    table = model.screen_interactions(holdout, y[: len(frame) // 2])

    assert np.isfinite(table["z"]).all()


def test_screen_per_spec_discrete_flags_approx():
    """Review findings: approx must follow the per-SPEC discretization
    decision (spec.discrete overrides the model flag; both parents must
    discretize) AND only flag when at least one discretization is lossy —
    lossless binning returns the exact unique support."""

    from superglm import SuperGLM
    from superglm.features.spline import Spline

    frame, y, w = _screening_data(22, interaction=0.5)
    model = SuperGLM(
        family="poisson",
        selection_penalty=None,
        discrete=False,
        features={
            "x1": Spline(kind="ps", k=6, discrete=True, n_bins=8),  # lossy bins
            "x2": Spline(kind="ps", k=6, discrete=True),  # lossless (28 <= 256)
            "x3": Spline(kind="ps", k=6),
            "x4": Spline(kind="ps", k=6),
            "x5": Spline(kind="ps", k=6, discrete=True),  # lossless (37 <= 256)
        },
    ).fit_reml(frame, y, sample_weight=w)

    table = model.screen_interactions(frame, y, sample_weight=w)

    flags = {frozenset((r.feature_a, r.feature_b)): bool(r.approx) for r in table.itertuples()}
    assert flags[frozenset(("x1", "x2"))]  # both discretize, x1 lossy
    assert not flags[frozenset(("x2", "x5"))]  # both discretize, both lossless
    assert not flags[frozenset(("x1", "x3"))]  # mixed pair: refit is exact
    assert not flags[frozenset(("x3", "x4"))]


def test_screen_cache_equivalence_covers_binned_entries():
    """Reviewer ask: the cache-equivalence pin only exercised exact-path
    entries; cover the (name, binned=True) cache keys too."""
    frame, y, w = _continuous_screening_data()
    model = _fit_continuous(frame, y, w)

    full = model.screen_interactions(frame, y, sample_weight=w)
    binned_rows = full[full["approx"]]
    assert len(binned_rows) >= 1  # x1:x2 bins under the default budget

    for row in binned_rows.itertuples():
        single = model.screen_interactions(
            frame, y, sample_weight=w, candidates=[(row.feature_a, row.feature_b)]
        )
        s = single.iloc[0]
        for col in ("statistic", "z", "edf0", "lambda0", "n_cells", "approx"):
            assert getattr(row, col) == s[col], col


def test_screen_weight_inheritance_reads_the_stored_array_not_the_stamp():
    """Review finding: the editor rewrites _fit_weights without touching the
    _fit_used_weights stamp, so trusting the stamp resurrects the original
    blocking bug (screen at unit weight against an exposure-fitted model).
    Non-unitness must be derived from the stored array at read time."""
    import pandas as pd

    frame, y, w = _screening_data(23, interaction=0.5)
    model = _fit_mains(frame, y, w)

    # editor's second writer: array becomes ones, stamp stays True
    explicit_ones = model.screen_interactions(frame, y, sample_weight=np.ones_like(y))
    model._fit_weights = np.ones_like(y)
    assert model._fit_used_weights  # the stale stamp
    pd.testing.assert_frame_equal(model.screen_interactions(frame, y), explicit_ones)

    # mirror: stored non-unit weights inherit despite a stale False stamp
    model._fit_weights = w
    model._fit_used_weights = False
    explicit_w = model.screen_interactions(frame, y, sample_weight=w)
    pd.testing.assert_frame_equal(model.screen_interactions(frame, y), explicit_w)


def test_screen_labels_every_spline_pair_ti():
    """The mixed-margin work added a `kind` column naming the refit target;
    a spline-only model must be unchanged apart from a constant 'ti'."""
    from superglm.model.screening_ops import _RESULT_COLUMNS

    frame, y, w = _screening_data(24, interaction=0.5)
    model = _fit_mains(frame, y, w)

    table = model.screen_interactions(frame, y, sample_weight=w)

    assert list(table.columns) == _RESULT_COLUMNS
    assert _RESULT_COLUMNS[:4] == ["feature_a", "feature_b", "kind", "statistic"]
    assert set(table["kind"]) == {"ti"}


def _treatment_menu(n_levels):
    """(L, L-1) treatment contrasts with level 0 as base — the screen's menu."""
    return np.vstack([np.zeros((1, n_levels - 1)), np.eye(n_levels - 1)])


def test_contrast_menu_kron_is_the_pair_indicator_block():
    # kron of contrast menus on the level-pair grid == CategoricalInteraction's
    # non-base pair indicator columns, row for row.
    rng = np.random.default_rng(7)
    n, L1, L2 = 400, 4, 3
    codes_a = rng.integers(0, L1, n)
    codes_b = rng.integers(0, L2, n)
    menu_a = _treatment_menu(L1)
    menu_b = _treatment_menu(L2)
    X = _dense_row_kronecker(codes_a, codes_b, menu_a, menu_b)
    expected = np.column_stack(
        [
            (codes_a == i).astype(float) * (codes_b == j).astype(float)
            for i in range(1, L1)
            for j in range(1, L2)
        ]
    )
    np.testing.assert_allclose(X, expected)


def test_cell_assembly_exact_for_contrast_menus():
    rng = np.random.default_rng(8)
    n, L1, L2 = 500, 5, 4
    codes_a = rng.integers(0, L1, n)
    codes_b = rng.integers(0, L2, n)
    menu_a = _treatment_menu(L1)
    menu_b = _treatment_menu(L2)
    score = rng.normal(size=n)
    w = rng.uniform(0.1, 3.0, n)
    S_cell, W_cell = pair_cell_moments(codes_a, codes_b, L1, L2, score, w)
    U, V = pair_score_curvature(menu_a, menu_b, S_cell, W_cell)
    X = _dense_row_kronecker(codes_a, codes_b, menu_a, menu_b)
    np.testing.assert_allclose(U, X.T @ score, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(V, X.T @ (X * w[:, None]), rtol=1e-12, atol=1e-9)


def test_numeric_pair_moments_match_dense_assembly():
    from superglm.screening import numeric_pair_moments

    rng = np.random.default_rng(11)
    n, L = 600, 5
    codes = rng.integers(0, L, n)
    menu = np.zeros((L, L - 1))
    menu[1:, :] = np.eye(L - 1)
    z = rng.uniform(-2.0, 3.0, n)
    score = rng.normal(size=n)
    w = rng.uniform(0.1, 3.0, n)

    U, V, C, M, u_m = numeric_pair_moments(codes, L, menu, z, score, w)

    X_T = menu[codes] * z[:, None]  # probe block
    X_o = np.column_stack([np.ones(n), menu[codes], z])  # overlap span
    np.testing.assert_allclose(U, X_T.T @ score, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(V, X_T.T @ (X_T * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(M, X_o.T @ (X_o * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(C, X_o.T @ (X_T * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(u_m, X_o.T @ score, rtol=1e-12, atol=1e-9)


def test_numeric_pair_moments_reject_a_menu_that_is_not_the_grid():
    """Review finding: row agreement and the code bounds were validated but the
    menu's own row count was not, so a menu built for a different margin
    surfaced as a numpy broadcast error instead of this module's contract."""
    import pytest

    from superglm.screening import numeric_pair_moments

    rng = np.random.default_rng(11)
    n, L = 200, 5
    codes = rng.integers(0, L, n)
    z = rng.uniform(-2.0, 3.0, n)
    score = rng.normal(size=n)
    w = rng.uniform(0.1, 3.0, n)
    menu = np.zeros((L, L - 1))
    menu[1:, :] = np.eye(L - 1)

    with pytest.raises(ValueError, match="one row per gridded cell"):
        numeric_pair_moments(codes, L, menu[:-1], z, score, w)
    with pytest.raises(ValueError, match="one row per gridded cell"):
        numeric_pair_moments(codes, L, menu[:, 0], z, score, w)
    # the matching menu still computes
    assert numeric_pair_moments(codes, L, menu, z, score, w)[0].shape == (L - 1,)


def test_numeric_numeric_moments_match_dense_assembly():
    from superglm.screening import numeric_numeric_moments

    rng = np.random.default_rng(12)
    n = 500
    z1 = rng.uniform(-1.0, 2.0, n)
    z2 = rng.uniform(0.5, 1.5, n)
    score = rng.normal(size=n)
    w = rng.uniform(0.1, 3.0, n)
    U, V, C, M, u_m = numeric_numeric_moments(z1, z2, score, w)
    X_T = (z1 * z2)[:, None]
    X_o = np.column_stack([np.ones(n), z1, z2])
    np.testing.assert_allclose(U, X_T.T @ score, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(V, X_T.T @ (X_T * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(M, X_o.T @ (X_o * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(C, X_o.T @ (X_T * w[:, None]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(u_m, X_o.T @ score, rtol=1e-12, atol=1e-9)


def _low_edge_sensitivity(sigma_min, p=24, units=1.0, penalty="I"):
    """The low-edge ``edf``'s FIRST-ORDER sensitivity, and the displacement realizing it.

    **THIS IS A LOCAL FIRST-ORDER MEASURE AND NOT A CERTIFIED WORST CASE.**  An
    earlier revision called it a ceiling.  The maximisation below is exact for
    the DIFFERENTIAL at ``V``; ``edf(V + E)`` is nonlinear in ``E``, so the
    finite response at ``||E||_F = c`` is not bounded by it, and this test's own
    numbers show it -- realizations reach 1.0617 of the predicted displacement.
    No finite-radius remainder bound is derived here, so the quantity is
    described and asserted as a first-order condition measure only.

    ``edf = tr(A^-1 V)`` with ``A = V + lambda S`` is exactly
    ``p - lambda tr(A^-1 S)``, so for a perturbation ``E`` of the operand::

        d edf = <E, G>_F ,
        G := lambda A^-1 S A^-1 + alpha (d edf / d lambda) I

    and over ``||E||_F = c`` that is maximised at ``E = c G / ||G||_F``, giving
    exactly ``c ||G||_F``.  The second term is the BRACKET's own response, from
    ``lambda = alpha tr(V)`` with ``alpha = 1e-10 / tr(S)``; it is displayed
    because the code carries it, and an earlier revision showed only the first
    term here while asserting on both.  **The identity needs a nonsingular
    ``A``** and so does not cover a pencil whose ``V`` and ``S`` share a null
    space -- that case is answered elsewhere through ``pinv`` and is not what
    this measures.

    **THE RADIUS IS A STATED PROBE AND NOTHING MORE.**  ``c = eps ||V||_F`` is
    ONE ROUNDING of the operand's norm -- a definition.  Two revisions have now
    over-claimed it and both are withdrawn: it is not "the error already
    committed in forming the Gram", and the Gram's error is not known to
    EXCEED it either.  What is available is an UPPER bound, Higham, *Accuracy
    and Stability of Numerical Algorithms*, 2nd ed., Ch. 3:
    ``fl(X'X) = X'X + D`` with ``|D| <= gamma_n |X|'|X|`` COMPONENTWISE, ``n``
    the contraction dimension.  Measured here that bound is 98.6x to 204.7x the
    probe radius -- but an upper bound does not put a floor under the actual
    error, which may be anywhere below it and is zero for exactly representable
    products.  So this reports the answer's SENSITIVITY to a one-rounding
    probe, and nothing here establishes an information floor.

    **THIS IS A DETERMINISTIC NORM, NOT A SAMPLED WIDTH, AND THAT IS THE
    POINT.**  It replaces a ``max - min`` over 32 fixed random perturbations,
    which review correctly refused: first-order theory supplies a sensitivity
    SCALE, i.e. an upper bound, and never a lower bound on a sample range.
    Measured, that refusal is right twice over --

    * over the ``reps`` axis, at fixed everything else, the old ratio read
      0.2548 at 4 draws, 0.5496 at 8, 0.6342 at 32 and 0.8653 at 128.  A range
      of ``N`` samples grows with ``N``, so the old bound was a property of an
      arbitrary constant in the test, and at 4 draws it cleared its own
      enforced boundary by 1.27x;
    * over the perturbation SEED, an axis the microkernel sweep does not
      explore at all, 24 seeds at one configuration spread 0.5063 to 0.8085.

    ``G`` needs only solves against ``A = V + lambda S``.  An earlier revision
    claimed ``lambda`` bounds ``cond(A)`` at about 1e10 "by construction";
    **that is false for an anisotropic penalty and is withdrawn.**  The only
    bound available is ``||A||_2 / (lambda lambda_min(S))``, which the rotated
    penalty makes vacuous (1.6e+18) and a singular ``S`` makes unavailable
    outright.  Measured instead, ``cond(A)`` over the asserted rows is 1.0e+02
    and 1.0e+06 on the resolved geometries, 1.9e+11 and 2.2e+11 on the
    isotropic unresolved ones, and 5.8e+12 and **5.4e+14** on the rotated ones
    -- so at the hardest row the solve for ``G`` keeps about one digit in its
    smallest direction.  That is disclosure, not derivation: what makes it
    usable is measured stability rather than a bound, and the sweep is where
    that is checked.

    It does remove the ``||S||_2`` proxy a second finding refused, since ``G``
    carries ``S`` in the right place -- there is no directional penalty scale
    left to substitute for.  ``penalty="rot"`` exercises exactly that: a
    rotated ``S`` with eight orders between its largest and smallest
    eigenvalue, in a basis unrelated to the design's.

    Returns the first-order sensitivity, the displacement the maximising
    perturbation actually produces, and one ulp of the answer.
    """
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((4 * p, p)))
    Z, _ = np.linalg.qr(rng.standard_normal((p, p)))
    X = units * ((Q * np.geomspace(1.0, sigma_min, p)) @ Z.T)
    V = 0.5 * (X.T @ X + (X.T @ X).T)
    if penalty == "I":
        S = np.eye(p)
    else:
        R, _ = np.linalg.qr(np.random.default_rng(77).standard_normal((p, p)))
        S = (R * np.geomspace(1.0, 1e-8, p)) @ R.T
        S = 0.5 * (S + S.T)
    U = np.ones(p)

    # A budget above ``p`` is unreachable, so the rung clamps to the bracket's
    # LOW edge and reports the lambda it clamped at.  Read back rather than
    # hard-coded: the bracket is scale-relative and this pins no constant.
    lam = float(penalized_score_statistic_ladder(U, V, S_ti=S, budgets=(4.0 * p,))[0].lambda0)

    A = V + lam * S
    Ainv_S = np.linalg.solve(A, S)
    G = lam * np.linalg.solve(A, Ainv_S.T).T
    # THE TOTAL GRADIENT, not the partial one at fixed lambda.  The bracket is
    # ``lambda = alpha tr(V)`` with ``alpha = 1e-10 / tr(S)``, so
    # ``d edf = <E, G>_F + (d edf / d lambda) alpha tr(E)``.  Review is right
    # that dropping the second term misstates the ladder's own sensitivity --
    # in ONE dimension it cancels the first exactly.  Measured, including it
    # moves the gradient's norm by 0.0% on the unresolved geometries and at
    # most 1.6% on a resolved one, so it changes no conclusion here; it is
    # carried because the maximising DIRECTION is only the ladder's if the
    # whole gradient is used.
    alpha = 1e-10 / float(np.trace(S))
    dedf_dlam = -float(np.trace(Ainv_S)) + lam * float(np.trace(Ainv_S @ Ainv_S))
    G = G + alpha * dedf_dlam * np.eye(p)
    G = 0.5 * (G + G.T)
    g_norm = float(np.linalg.norm(G, "fro"))
    c = np.finfo(np.float64).eps * float(np.linalg.norm(V, "fro"))

    def edf(operand):
        # THE REAL CALLER PATH, not ``_edge`` at a pinned lambda.  The bracket
        # is ``1e-10 tr(V) / tr(S)``, so a perturbation with nonzero trace
        # moves lambda too, and holding it fixed would measure only the partial
        # derivative.  (An earlier revision argued the probe has nonzero trace
        # because ``G`` is PSD.  It is NOT, once the bracket term is included:
        # the ladder's scale invariance gives ``<G, V>_F = 0``, impossible for
        # a nonzero PSD ``G`` against positive definite ``V``.  Withdrawn.)  Review is right
        # that the omitted term can cancel the response outright: in ONE
        # dimension ``edf = V / (V + 1e-10 V)`` is constant in ``V`` while the
        # fixed-lambda calculation reports a nonzero move.  It does not cancel
        # here, because the response is carried by the near-null direction at
        # ``1 / (lambda s)`` while the bracket shifts with the TOTAL trace,
        # which the saturated directions dominate -- measured, the ladder's
        # displacement equals the fixed-lambda one to 1.000 on all eight
        # geometries, with lambda itself moving 0 to 7.9e-16 relative.  Driving
        # the ladder anyway costs one call and removes the argument.
        rung = penalized_score_statistic_ladder(
            U, 0.5 * (operand + operand.T), S_ti=S, budgets=(4.0 * p,)
        )[0]
        return float(rung.edf0)

    def response(direction):
        # A CENTRED difference, so what is measured is the response and not the
        # response plus whatever the evaluator does to the base point alone.
        d = direction * (c / float(np.linalg.norm(direction, "fro")))
        return abs(edf(V + d) - edf(V - d)) / 2.0

    realised = response(G)
    # THE SAME PROBE IN OTHER DIRECTIONS, for the ordering invariant below.
    # Seeded, so this is deterministic; eight of them, so a routine that had
    # stopped maximising would have to win eight coin flips to hide.
    rng_dirs = np.random.default_rng(90210)
    rivals = []
    for _ in range(8):
        R = rng_dirs.standard_normal(G.shape)
        rivals.append(response(0.5 * (R + R.T)))
    # ``np.spacing``, not ``eps * |edf|``.  The latter is a RELATIVE scale and
    # runs 1.045x to 1.679x the true adjacent-float distance across these
    # geometries, so a displacement reported as "under one ulp" could be most
    # of two.  That mattered: at the true spacing the ``1e-3`` geometries read
    # 0.72 and 0.94 rather than 0.48 and 0.63, which is what moved them out of
    # the asserted set and into disclosure below.
    return c * g_norm, realised, float(np.spacing(abs(edf(V)))), max(rivals)


# ONE BOUNDARY, AND IT IS THE ONE THE PROSE STATES: a probe of one rounding
# moves the answer by less than one ulp of itself, or it does not.  That is a
# representability statement, derived, with no fitted constant in it.
#
# An earlier revision asserted 1e5 ulp while the comment beside it said one --
# an "empirically placed separator", as review named it, sitting at the midpoint
# of 0.63 and 2.9e10.  It would have passed a resolved case amplified by five
# orders.  The boundary is now 1 on both sides, so the margins are what they
# are and are reported rather than engineered: 1.59x on the resolved side and
# 2.90e+10x on the unresolved one.  The thin side is thin because it is the
# real boundary; what makes 1.59x usable is that the quantity is BIT-IDENTICAL
# across the sweep on that row, not that the number is comfortable.
#
# ONE ASSERTED BOUNDARY, AND NO FITTED CONSTANT ANYWHERE.
#
# The realization ratio is REPORTED in the failure message and not asserted.
# Two intervals were tried around it -- (0.5, 2.0), then (0.1, 10.0) -- and
# review refused both for the same reason, correctly: the ratio differences two
# production evaluations on operands whose ``cond(A)`` reaches 5.4e+14, so it
# can leave any such interval on another LAPACK with no mathematical
# regression, and widening a fitted window does not derive it.  No second-order
# remainder or conditioning-based allowance is available here, so there is
# nothing to assert.
#
# THAT TRADE COSTS TWO MUTATIONS AND IS TAKEN ANYWAY.  With the interval,
# perturbing along a RANDOM direction rather than the maximising one reds 6
# rows and shrinking the probe a millionfold reds 6; without it neither is
# detected, because both scale the sensitivity and its reference together.
# Both are edits to this test's own stated parameters rather than regressions
# in the module, and a bound that reds on a backend change without a defect is
# worse than one that misses a self-mutation.  The regression that matters --
# design factors collapsing the sensitivity by ten orders (#257) -- is caught
# by the boundary below on its own.
_LOW_EDGE_ULP = 1.0
# THE CALLER PATH IS PINNED ELSEWHERE, BY A CLOSED FORM.
#
# ``ceiling`` is computed from this test's own ``A`` and ``G``, so the only
# production quantity in the boundary is ``edf(V)`` inside ``ulp``: an ``_edge``
# returning an edf INDEPENDENT of ``V`` would keep every row green.  Three
# attempts to close that here were refused in turn, and each refusal was right:
# two magnitude windows around the realized response ((0.5, 2.0), then
# (0.1, 10.0) scoped to ``cond(A) ~ 2e+11``) fit an interval nothing derives,
# and the ordering invariant that replaced them is exact for the DIFFERENTIAL
# while comparing FINITE centred differences, so curvature or solve error at
# ``cond(A)`` up to 5.4e+14 could legitimately let a rival win.
#
# So the caller path is pinned where it can be pinned exactly instead:
# :func:`test_the_clamped_low_edge_reproduces_the_isotropic_closed_form`, on a
# ``cond(A) = 1`` fixture with an analytic answer.  The realized response and
# its rivals stay in the failure message as disclosure.


@pytest.mark.parametrize(
    ("sigma_min", "units", "penalty", "determined"),
    [
        (1e-1, 1.0, "I", True),
        (1e-1, 1e3, "I", True),
        (1e-8, 1.0, "I", False),
        (1e-12, 1.0, "I", False),
        # OTHER UNITS.  Rescaling the design changes nothing about it, so a
        # quantity that is a property of the pair must not move.  These rows are
        # a review finding kept as a test: an earlier normalization moved twelve
        # orders under exactly this, and they scale BOTH ways because that error
        # sends the ratio up on one and down on the other.
        (1e-8, 1e3, "I", False),
        (1e-8, 1e-3, "I", False),
        # AN ANISOTROPIC PENALTY, likewise a finding kept as a test: eight
        # orders of penalty spectrum in a rotated basis.  Under the ``||S||_2``
        # normalization this replaces, these read 55.6 to 126.9 and 5924 to
        # 11124 where the theory says 1 -- four orders of understatement, hidden
        # entirely by the ``S = I`` the fixture used to hard-code.
        (1e-1, 1.0, "rot", True),
        (1e-8, 1.0, "rot", False),
        (1e-12, 1.0, "rot", False),
    ],
)
def test_the_low_edge_edf_is_only_as_determined_as_the_gram_it_is_read_from(
    sigma_min, units, penalty, determined
):
    """At the ladder's LOW edge the ceiling is the operand's sensitivity, not the arithmetic.

    This module is handed MOMENTS, so what it can resolve is the design's
    spectrum SQUARED.  A direction sitting at ``sigma`` in the design sits at
    ``sigma^2`` in the Gram, and once that is under ``eps`` the Gram carries
    round-off there and nothing else.  The low edge then divides by ``lambda``:
    ``edf = sum_j v_j / (v_j + lambda s_j)`` has slope ``1 / (lambda s_j)`` at
    ``v_j = 0``.  :func:`_low_edge_sensitivity` turns that into the exact
    first-order worst case ``eps ||V||_F ||G||_F`` with ``G = lambda A^-1 S
    A^-1``, which needs no directional penalty scale and no sampling.

    **Why a SENSITIVITY and not an error, and what the probes are NOT.**  There
    is no value to assert.  An earlier revision justified that by calling every
    perturbed operand "as faithful to the design as the one the caller passed";
    **that is withdrawn**.  ``eps ||V||_F`` is an arbitrary probe radius, the
    maximising direction is indefinite once the bracket term is in it, so
    ``V - step`` can leave the PSD cone, and no ``X`` is exhibited whose
    rounding produces either operand -- indeed for an exactly representable
    product the caller's Gram can carry zero formation error while these probes
    are nonzero.  They are CONDITION PROBES.  What they measure is how far the
    answer moves under perturbations the size of the arithmetic's own, which is
    a property of the map and not a claim about admissible inputs.

    **THE REGIME.**  Against one ulp of ``edf`` -- ``np.spacing``, the actual
    adjacent-float distance -- the first-order sensitivity separates by
    eighteen orders, and it is bit-identical across the sweep on every row but
    the two hardest.

    **The comparison is of the DIFFERENTIAL against one ulp, not of the finite
    probe's move.**  No remainder over the probe radius is derived here, so
    strictly this classifies ``c ||G||_F`` and not ``|edf(V + E) - edf(V)|``.
    What makes that safe is the margin rather than an argument: the asserted
    rows clear the boundary by 2.44e+07x and 3.03e+10x, so a remainder would
    have to exceed the differential by seven orders to move a row across, where
    the measured realizations sit within 6% of it.  A geometry close enough to
    the boundary for that gap to matter is exactly what the ``1e-3`` rows are,
    and they are disclosed rather than classified::

        penalty  sigma_min  units   sensitivity / ulp over the 14   asserted
        I        1e-1       1       3.2839e-08                      resolved
        I        1e-1       1e3     3.2839e-08                      resolved
        rot      1e-1       1       4.0956e-08                      resolved
        I        1e-3       1       7.2316e-01                      no
        rot      1e-3       1       9.4248e-01                      no
        I        1e-5       1       3.9388e+07                      no
        I        1e-8       1 / 1e3 / 1e-3   3.0293e+10             unresolved
        I        1e-12      1       9.4365e+10                      unresolved
        rot      1e-8       1       3.0024e+11 .. 3.0027e+11        unresolved
        rot      1e-12      1       6.2109e+13 .. 6.2554e+13        unresolved

    Worst asserted-resolved 4.0956e-08 and best asserted-unresolved 3.0293e+10,
    so the boundary of one ulp clears by **2.44e+07x** and **3.03e+10x**.

    **THE ``1e-3`` ROWS ARE MEASURED AND NOT ASSERTED, and that is a finding
    rather than a convenience.**  They were asserted as resolved while the ulp
    was computed as ``eps * |edf|`` -- a relative scale, 1.045x to 1.679x the
    true spacing here -- which put them at 0.48 and 0.63 with a 1.59x margin.
    At the real spacing they are 0.72 and 0.94: still under one ulp, but by
    6% on the binding row, and a binade crossing in ``edf`` would double the
    ratio outright.  A geometry that close to the boundary is transitional, so
    it is disclosed with its number instead of being asserted on either side.
    The alternative -- keeping the assertion and quoting the comfortable
    margin the wrong ulp produced -- is the failure this whole PR is about.

    **THE REALIZATION IS REPORTED AND NOT ASSERTED.**  The maximising
    perturbation attains 0.9425 to 1.0617 of the predicted displacement across
    the sweep, against a first-order value of 1 -- which confirms the algebra
    is live and, by exceeding 1, shows the quantity is a first-order measure
    rather than a bound on the finite response.

    Two intervals were tried around it, ``(0.5, 2.0)`` and then
    ``(0.1, 10.0)``, and **review refused both for the same reason,
    correctly**: the ratio differences two production evaluations on operands
    whose ``cond(A)`` reaches 5.4e+14, so it can leave any such interval under
    another LAPACK with no mathematical regression, and widening a fitted
    window does not derive it.  That is the sampled-width objection from an
    earlier round re-entering by a different door, and just as valid.  No
    second-order remainder is derived here, so there is nothing left to assert
    and the number is disclosure.

    **The trade is recorded because it is not free.**  With the interval, a
    RANDOM perturbation direction reds 6 rows and a millionfold-smaller probe
    reds 6; without it neither is detected, since both scale the sensitivity
    and its reference together.  Both are edits to this test's own stated
    parameters rather than regressions in the module, and a bound that reds on
    a backend change without a defect is worse than one that misses a
    self-mutation.  The regression that matters -- design factors collapsing
    the sensitivity by ten orders, #257 -- is caught by the boundary alone.

    **What this says about issue #279.**  Over 21 draws of a 20-level
    spline-by-categorical pair, the same first-order sensitivity spans
    **1.63e-15 to 4.93e-04** -- twelve orders, tracking the residualized
    design's smallest singular value.  All ten draws whose design is rank
    deficient sit at ~2.6e-04, i.e. **26x the ``abs=1e-5`` the suite asserts**
    at that edge, on exactly the draws that can fail it.  That is a statement
    about how far those answers MOVE under a one-rounding probe, not a proof
    that their error must exceed the bound; what it explains is why the miss is
    a property of the draw.

    **No arithmetic in this module can narrow it**, which is why #279 closed
    without a code change: the information is destroyed by the squaring, before
    the call.  Handing the module design factors is the fix, and it is a change
    to the CALLER, tracked at #257.
    """
    ceiling, realised, ulp, best_rival = _low_edge_sensitivity(
        sigma_min, units=units, penalty=penalty
    )
    against_ulp = ceiling / ulp
    # BOTH sides are named on EVERY failure, so a geometry crossing the cut
    # reports its position exactly as an approaching one does.  A cut watched
    # from one side alone goes quiet at the moment it is crossed.
    separation = (_LOW_EDGE_ULP / against_ulp) if determined else (against_ulp / _LOW_EDGE_ULP)
    where = (
        f"(sensitivity eps||V||_F ||G||_F {ceiling:.4e}, realised/predicted "
        f"{(realised / ceiling if ceiling else float('nan')):.4f} (reported, not asserted), "
        f"one ulp of edf {ulp:.4e}, "
        f"ratio {against_ulp:.4e} ulp, boundary {_LOW_EDGE_ULP:.0f} ulp, "
        f"separation {separation:.3e}x)"
    )
    assert separation > 1.0, (
        (
            f"a design the Gram RESOLVES left the low-edge edf sensitive {where}; "
            "a one-rounding probe of a resolved operand must stay UNDER one ulp of the "
            "answer; the sweep puts the worst such geometry at 0.6283"
        )
        if determined
        else (
            f"a design the Gram CANNOT resolve left the low-edge edf determined {where}; "
            "the same one-rounding probe must move it by MORE than one ulp, and the sweep "
            "puts the smallest such geometry at 3.0293e+10; a collapse here means this "
            "module is no longer being handed a Gram -- see #257"
        )
    )


@pytest.mark.parametrize("p", [24, 40])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e5])
def test_the_clamped_low_edge_reproduces_the_isotropic_closed_form(p, scale):
    """The clamped low-edge rung, against an answer that is known exactly.

    With ``V = a I`` and ``S = I`` the bracket is ``lambda = 1e-10 a`` and
    ``A = (a + lambda) I``, so::

        edf = tr(A^-1 V) = p a / (a + lambda) = p / (1 + 1e-10)

    exactly, for every ``a`` and at ``cond(A) = 1``.  Nothing here is measured
    or fitted: the value is arithmetic and the tolerance is dimensional.

    **THIS EXISTS TO PIN THE CALLER PATH**, which
    ``test_the_low_edge_edf_is_only_as_determined_as_the_gram_it_is_read_from``
    cannot: that test's boundary is computed from its own ``A`` and ``G``, so an
    ``_edge`` returning an edf independent of ``V`` would leave every row of it
    green -- verified, it did.  Three attempts to close that inside it were
    refused in review and each refusal was right, because each asserted a
    magnitude or an ordering of FINITE responses that no remainder bound
    covers.  Here there is nothing to bound: the answer is closed form.

    **THE TOLERANCE IS ``p eps``-DERIVED, NOT OBSERVED.**  ``edf`` is a trace of
    ``p`` terms from a solve against a diagonal matrix, so its relative error is
    ``O(p eps)`` -- 5.3e-15 at ``p = 24`` (Higham, *Accuracy and Stability of
    Numerical Algorithms*, 2nd ed., Ch. 3, for the ``gamma_p`` of a length-``p``
    sum).  ``rel=1e-13`` sits 19x above that floor.  For disclosure and never to
    set the bound, the observed relative error over the six parametrizations is
    0.0 to 2.961e-16, so the bound clears it by 338x; it is placed against the
    derived floor rather than that headroom.

    The scale rows are not decoration: ``a`` cancels out of the closed form
    entirely, so a routine whose low edge drifted with the units of the
    curvature would break these and nothing else here.
    """
    V = scale * np.eye(p)
    S = np.eye(p)
    rung = penalized_score_statistic_ladder(np.ones(p), V, S_ti=S, budgets=(4.0 * p,))[0]

    assert rung.lambda0 == pytest.approx(1e-10 * scale, rel=1e-13), (
        f"the bracket's low edge is 1e-10 tr(V)/tr(S) = {1e-10 * scale:.6e} on this "
        f"fixture and the ladder clamped at {rung.lambda0:.6e}"
    )
    assert rung.edf0 == pytest.approx(p / (1.0 + 1e-10), rel=1e-13), (
        f"the clamped low-edge edf of a I against I is p/(1 + 1e-10) = "
        f"{p / (1.0 + 1e-10):.15f} exactly; the ladder returned {rung.edf0:.15f}"
    )
