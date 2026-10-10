"""Profiled-intercept geometry for cached discrete REML trials."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from superglm.distributions import Gaussian
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import IdentityLink
from superglm.reml.discrete import _cached_centred_intercept, _solve_cached_profiled_system
from superglm.reml.objective import reml_laml_objective
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.mode_score import centred_matvec, prior_weighted_centre
from superglm.solvers.pirls import PIRLSResult
from superglm.types import GroupSlice


def _translated_gaussian_fixture() -> tuple[DesignMatrix, np.ndarray, np.ndarray, GroupSlice]:
    rng = np.random.default_rng(20260718)
    n = 80
    centered = rng.normal(size=(n, 3))
    centered -= np.mean(centered, axis=0)
    X = centered + np.array([1.0e10, -3.0e10, 7.0e9])
    beta = np.array([0.7, -0.25, 0.4])
    y = 1.3 + centered @ beta + rng.normal(scale=0.05, size=n)
    weights = rng.uniform(0.4, 1.7, size=n)
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=X.shape[1])
    return dm, y, weights, GroupSlice(name="x", start=0, end=X.shape[1])


def test_direct_working_cache_retains_stable_profiled_system() -> None:
    dm, y, weights, group = _translated_gaussian_fixture()
    cache: dict[str, object] = {}
    penalty = np.diag([0.2, 0.4, 0.8])

    fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=[group],
        lambda2=1.0,
        max_iter=2,
        tol=1.0e-12,
        cache_out=cache,
        S_override=penalty,
        compute_rank_info=False,
        _return_working_system=True,
        _compute_fit_statistics=False,
        weight_semantics="frequency",
    )

    assert set(cache) >= {
        "centered_XtWX",
        "centered_rhs",
        "mean_x",
        "mean_z",
        "sum_W",
    }
    assert np.all(np.isfinite(cache["centered_XtWX"]))
    assert np.all(np.isfinite(cache["centered_rhs"]))


@pytest.mark.parametrize("lambda_value", [1.0e-4, 0.7, 1.0e4])
def test_cached_trial_matches_full_profiled_objective_after_large_translation(
    lambda_value: float,
) -> None:
    dm, y, weights, group = _translated_gaussian_fixture()
    family = Gaussian()
    link = IdentityLink()
    base_penalty = np.diag([0.3, 0.9, 1.7])
    trial_penalty = lambda_value * base_penalty
    cache: dict[str, object] = {}

    fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=family,
        link=link,
        groups=[group],
        lambda2=1.0,
        max_iter=2,
        tol=1.0e-12,
        cache_out=cache,
        S_override=base_penalty,
        compute_rank_info=False,
        _return_working_system=True,
        _compute_fit_statistics=False,
        weight_semantics="frequency",
    )

    beta_cached, intercept_cached, log_det_cached, hessian_rank_cached = (
        _solve_cached_profiled_system(
            cache["centered_XtWX"],
            trial_penalty,
            cache["centered_rhs"],
            cache["mean_x"],
            cache["sum_W"],
            cache["mean_z"],
        )
    )
    full_result, _, full_xtwx = fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=family,
        link=link,
        groups=[group],
        lambda2=lambda_value,
        max_iter=3,
        tol=1.0e-12,
        return_xtwx=True,
        S_override=trial_penalty,
        compute_rank_info=False,
        _compute_fit_statistics=False,
        weight_semantics="frequency",
    )

    np.testing.assert_allclose(beta_cached, full_result.beta, rtol=2.0e-11, atol=2.0e-11)
    assert intercept_cached == pytest.approx(full_result.intercept, rel=2.0e-11, abs=2.0e-4)
    assert log_det_cached == pytest.approx(full_result.log_det_H, rel=2.0e-12, abs=2.0e-12)
    assert hessian_rank_cached == full_result.reml_hessian_rank

    # the linear predictor about the fixed prior-weighted centre, as the cached
    # trials (``reml.discrete``) and every PIRLS state form it: at a 1e10
    # translation X beta against the raw intercept errs by ~1e-6 per row
    centre = prior_weighted_centre(dm, weights)
    alpha_cached = _cached_centred_intercept(
        cache["mean_z"], cache["centre_offset_mean"], beta_cached
    )
    # The centred intercept is the full fit's: for a Gaussian identity fit the
    # working weights and response are the prior weights and y at every
    # iterate, so both form mean_z - (mean_x - c)' beta from the same means
    # and differ only through beta, by (mean_x - c)' d beta, plus the
    # rounding of the two fsum'd sums, u (|mean_z| + |mean_x - c|' |beta|)
    # each.  Through the raw intercept, (mean_z - mean_x' beta) + c' beta, the
    # trial's alpha erred by ~u |mean_x|' |beta| ~ 2e-6 (4.2e-6 on OpenBLAS's
    # Haswell kernels), which moved its REML objective by 1.1e-7.
    np.testing.assert_array_equal(full_result.state_center, centre)
    u = np.finfo(float).eps / 2
    spread = np.abs(np.asarray(cache["mean_x"]) - centre)
    alpha_bound = spread @ np.abs(beta_cached - full_result.beta) + 2.0 * u * (
        abs(float(cache["mean_z"]))
        + spread @ np.maximum(np.abs(beta_cached), np.abs(full_result.beta))
    )
    assert abs(alpha_cached - full_result.centred_intercept) <= alpha_bound
    eta_cached = alpha_cached + centred_matvec(dm, beta_cached, centre)
    mu_cached = link.inverse(eta_cached)
    cached_result = PIRLSResult(
        beta=beta_cached,
        intercept=intercept_cached,
        n_iter=0,
        deviance=float(np.sum(weights * family.deviance_unit(y, mu_cached))),
        converged=True,
        phi=1.0,
        effective_df=0.0,
        log_det_H=log_det_cached,
        reml_hessian_rank=hessian_rank_cached,
        centred_intercept=alpha_cached,
        state_center=centre,
    )
    cached_objective = reml_laml_objective(
        dm,
        family,
        link,
        [group],
        y,
        cached_result,
        {},
        weights,
        np.zeros_like(y),
        XtWX=cache["XtWX"],
        log_det_H=log_det_cached,
        S_override=trial_penalty,
        weight_semantics="frequency",
    )
    full_objective = reml_laml_objective(
        dm,
        family,
        link,
        [group],
        y,
        full_result,
        {},
        weights,
        np.zeros_like(y),
        XtWX=full_xtwx,
        log_det_H=full_result.log_det_H,
        S_override=trial_penalty,
        weight_semantics="frequency",
    )
    assert cached_objective == pytest.approx(full_objective, rel=2.0e-11, abs=2.0e-11)


def test_cached_profiled_solve_keeps_well_conditioned_trials_on_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import superglm.reml.discrete as discrete_reml

    def fail_spectral_fallback(*args, **kwargs):
        raise AssertionError("well-conditioned cached trial used the rank fallback")

    monkeypatch.setattr(discrete_reml, "decompose_gram", fail_spectral_fallback)
    centered_gram = np.diag([2.0, 3.0, 5.0])
    penalty = np.diag([0.2, 0.4, 0.8])
    rhs = np.array([1.0, -2.0, 0.5])
    beta, intercept, log_det_h, rank = discrete_reml._solve_cached_profiled_system(
        centered_gram,
        penalty,
        rhs,
        np.array([0.3, -0.5, 0.1]),
        12.0,
        -0.7,
    )

    np.testing.assert_allclose(beta, np.linalg.solve(centered_gram + penalty, rhs))
    assert np.isfinite(intercept)
    assert np.isfinite(log_det_h)
    assert rank == 4


def test_a_discrete_trial_forms_eta_about_the_prior_weighted_centre(monkeypatch) -> None:
    """A cached discrete REML trial scores the deviance at ``alpha + X~ beta`` (design §3.8).

    With a column translated by 1e10, ``X beta + intercept`` cancels about
    ``1e10 |beta|`` per row: an eta error near ``1e10 u |beta| ~ 1e-6``, which
    moves the Poisson deviance the line search compares by ``~ sum |mu - y|
    1e-6``.  Every recorded trial's deviance must instead match the deviance
    at the centred predictor within the rounding both evaluations carry:
    ``gamma_{p+2} m_r`` in each row's eta (``m_r = |alpha| + sum |x~||beta| +
    |o|``, Higham 2002 §3.1), through ``|d'| = 2 (mu e^delta + y)``, plus 16 u
    of each unit deviance and ``gamma_n`` of the sum.  Mutation: the trial's
    eta formed raw (``intercept + X beta``).
    """
    import pandas as pd

    import superglm.reml.discrete as discrete_module
    from superglm import Numeric, Spline, SuperGLM

    recorded = []
    original = discrete_module.reml_laml_objective

    def record(dm, distribution, link, groups, y, pirls, lambdas, weights, offset, **kwargs):
        if pirls.n_iter == 0 and pirls.state_center is not None:
            recorded.append(
                (
                    dm,
                    pirls.beta.copy(),
                    float(pirls.centred_intercept),
                    pirls.state_center.copy(),
                    float(pirls.deviance),
                    np.asarray(weights),
                    np.asarray(offset),
                )
            )
        return original(
            dm, distribution, link, groups, y, pirls, lambdas, weights, offset, **kwargs
        )

    monkeypatch.setattr(discrete_module, "reml_laml_objective", record)
    rng = np.random.default_rng(1)
    n = 6000
    x = rng.uniform(size=n)
    z = rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 + np.sin(2 * np.pi * x) + 0.4 * z)).astype(float)
    model = SuperGLM(
        family="poisson",
        features={"x": Spline(n_knots=12), "z": Numeric()},
        selection_penalty=0,
        discrete=True,
    )
    with warnings.catch_warnings():
        # the translated fit's terminal mode is disclosed as not certified
        warnings.simplefilter("ignore")
        model.fit_reml(pd.DataFrame({"x": x, "z": z + 1.0e10}), y)
    assert recorded

    u = np.finfo(float).eps / 2
    for dm, beta, alpha, centre, deviance, weights, offset in recorded:
        X = dm.toarray()
        p = X.shape[1]
        centred = X - centre
        eta = alpha + centred @ beta + offset
        mu = np.exp(eta)
        units = 2.0 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0) - (y - mu))
        reference = float(np.sum(weights * units))
        magnitude = abs(alpha) + np.abs(centred) @ np.abs(beta) + np.abs(offset)
        delta = 2.0 * (p + 2) * u / (1.0 - (p + 2) * u) * magnitude
        slope = 2.0 * (mu * np.exp(delta) + y)
        per_row = slope * delta + 16.0 * u * (np.abs(units) + y + mu)
        bound = float(np.sum(weights * per_row)) + 2.0 * n * u / (1.0 - n * u) * float(
            np.sum(weights * np.abs(units))
        )
        assert abs(deviance - reference) <= bound, (deviance - reference, bound)


def _aliased_profiled_system(residue: float) -> tuple[np.ndarray, ...]:
    """An exactly singular profiled system, its data Gram moved by ``residue``.

    Columns 0 and 1 of the design are equal (a factor level and the constant
    of a smooth that sees that level at one value), and the penalty leaves
    both free, so ``H = B'B + S`` is semidefinite with the null vector ``e0 -
    e1`` in exact arithmetic (small integers: exact in binary64).  ``residue``
    is the formation rounding moved onto that direction, ``-residue (e0 -
    e1)(e0 - e1)'``, the sign a level's two weight sums rounded apart give.
    """
    design = np.array(
        [[1, 1, 2, 0], [1, 1, 0, 1], [2, 2, 1, 1], [0, 0, 1, 3], [1, 1, 3, 2], [0, 0, 2, 1]],
        dtype=float,
    )
    exact_gram = design.T @ design
    penalty = np.diag([0.0, 0.0, 1.0, 2.0])
    alias = np.array([1.0, -1.0, 0.0, 0.0])
    formed_gram = exact_gram - residue * np.outer(alias, alias)
    rhs = (exact_gram + penalty) @ np.array([0.5, 0.5, -1.0, 2.0])
    return exact_gram, formed_gram, penalty, rhs


def test_cached_trial_solves_formation_rounding_negativity_as_the_semidefinite_system() -> None:
    """A trial Hessian negative only within its formation rounding is solved, not refused.

    The freMTPL2 Density x Area fit (PR #485's default ``cr``) refused here at
    ``-1.8e-12`` against the eigensolver's bar of ``2.4e-13``: one level's
    103,957 rows rounded its weight sums ``5.5e4 u`` apart, far inside the
    ``5 gamma_{n+4}`` the formation can reach.  ``1e-10`` on a diagonal of
    ``7`` is inside that bound at ``n = 1e5`` rows (``g / d ~ 5.6e-11``) and
    gives a scaled eigenvalue of ``-2.9e-11``.  Solved as the exact
    semidefinite matrix, the trial agrees with that matrix's own
    decomposition within first-order perturbation of its retained spectrum,
    ``eta = (||dE||_2 + p eps ||E||_2) / w_r``, ``w_r`` the smallest retained
    scaled eigenvalue.  Mutation: without the formation bound (no
    ``n_rows``, as at 6eb42f3f) the same call raises.
    """
    from superglm.solvers.rank import decompose_gram

    exact_gram, formed_gram, penalty, rhs = _aliased_profiled_system(1.0e-10)
    mean_x = np.zeros(4)
    sum_w, mean_z = 50.0, 0.3
    with pytest.raises(ValueError, match="materially indefinite"):
        _solve_cached_profiled_system(formed_gram, penalty, rhs, mean_x, sum_w, mean_z)

    disclosure: dict[str, int] = {}
    beta, intercept, log_det_h, rank = _solve_cached_profiled_system(
        formed_gram, penalty, rhs, mean_x, sum_w, mean_z, n_rows=10**5, disclosure=disclosure
    )

    exact = exact_gram + penalty
    reference = decompose_gram(exact)
    scale = np.sqrt(np.diag(exact))
    scaled = exact / np.outer(scale, scale)
    spectrum = np.linalg.eigvalsh(scaled)
    smallest_retained = spectrum[spectrum > np.sqrt(np.finfo(float).eps)].min()
    eta = (
        np.linalg.norm((formed_gram - exact_gram) / np.outer(scale, scale), 2)
        + 4 * np.finfo(float).eps * spectrum[-1]
    ) / smallest_retained
    beta_reference = reference.solve(rhs)
    assert disclosure == {"formation_limited": 1}
    assert rank == 1 + reference.rank == 4
    assert abs(log_det_h - (np.log(sum_w) + reference.log_pdet)) <= 2 * reference.rank * eta
    assert np.linalg.norm(scale * (beta - beta_reference)) <= 4 * eta * np.linalg.norm(
        scale * beta_reference
    )
    assert intercept == pytest.approx(mean_z - mean_x @ beta)


def test_cached_trial_still_refuses_curvature_its_formation_cannot_explain() -> None:
    """Negative curvature past the formation bound is material and still raises.

    ``1e-3`` on the alias (a scaled eigenvalue of ``-2.9e-4``) is seven orders
    past ``5 gamma_{n+4}`` at ``n = 1e5``, and an indefinite penalty is not a
    rounding at all.  Mutation: an unbounded formation slack accepts both.
    """
    exact_gram, formed_gram, penalty, rhs = _aliased_profiled_system(1.0e-3)
    with pytest.raises(ValueError, match="materially indefinite"):
        _solve_cached_profiled_system(
            formed_gram, penalty, rhs, np.zeros(4), 50.0, 0.3, n_rows=10**5
        )
    indefinite_penalty = np.diag([0.0, 0.0, 1.0, -0.5]) - 1.0e-2 * np.eye(4)
    with pytest.raises(ValueError, match="materially indefinite"):
        _solve_cached_profiled_system(
            exact_gram, indefinite_penalty, rhs, np.zeros(4), 50.0, 0.3, n_rows=10**5
        )


def test_identified_part_of_a_trial_restricts_under_the_same_formation_bound() -> None:
    """A trial's identified part decomposes ``H_c[I, I]`` under ``H_c``'s formation bound.

    With a slope left out of the Laplace term, the cached trial decomposes
    its Hessian a second time, restricted to the kept slopes.  The alias
    (columns 0 and 1) is kept when column 3 is left out, so the block shows the
    same ``-2.9e-11`` and, without the bound (the two-entry ``dense`` the trial
    passed at 94879f33), raises "materially indefinite".  A principal block
    inherits ``|dH_ij| <= sqrt(g_i g_j)`` on its kept indices: with it the
    block decomposes as the exact semidefinite one, within first-order
    perturbation of its retained spectrum (``eta`` as above).
    """
    from superglm.reml.discrete import _profiled_formation_error
    from superglm.reml.identified import IdentifiedLaplace
    from superglm.solvers.rank import decompose_gram

    exact_gram, formed_gram, penalty, _ = _aliased_profiled_system(1.0e-10)
    hessian = formed_gram + penalty
    with pytest.raises(ValueError, match="materially indefinite"):
        IdentifiedLaplace(np.array([3])).log_det(None, 0.0, (hessian, 50.0))

    bound = _profiled_formation_error(formed_gram, penalty, np.zeros(4), 50.0, 10**5)
    identified = IdentifiedLaplace(np.array([3]))
    dense = (hessian, 50.0, bound)
    log_det = identified.log_det(None, 0.0, dense)
    rank = identified.rank(0, None, dense)

    kept = np.array([0, 1, 2])
    exact = (exact_gram + penalty)[np.ix_(kept, kept)]
    reference = decompose_gram(exact)
    scale = np.sqrt(np.diag(exact))
    spectrum = np.linalg.eigvalsh(exact / np.outer(scale, scale))
    smallest_retained = spectrum[spectrum > np.sqrt(np.finfo(float).eps)].min()
    residue = (formed_gram - exact_gram)[np.ix_(kept, kept)] / np.outer(scale, scale)
    eta = (np.linalg.norm(residue, 2) + 3 * np.finfo(float).eps * spectrum[-1]) / smallest_retained
    assert rank == 1 + reference.rank == 3
    assert abs(log_det - (np.log(50.0) + reference.log_pdet)) <= 2 * reference.rank * eta


def _spy_old_refusals(monkeypatch: pytest.MonkeyPatch, module) -> list[bool]:
    """Wrap ``module.decompose_gram``: per call given a formation bound, whether
    the same matrix without one is refused as materially indefinite (the rule
    at 6eb42f3f).  The matrix is decomposed twice; the answer returned is the
    bounded one."""
    original = module.decompose_gram
    refused: list[bool] = []

    def spy(matrix, **kwargs):
        if kwargs.get("formation_error") is not None:
            try:
                original(matrix)
            except ValueError as error:
                if "materially indefinite" not in str(error):
                    raise
                refused.append(True)
            else:
                refused.append(False)
        return original(matrix, **kwargs)

    monkeypatch.setattr(module, "decompose_gram", spy)
    return refused


def _levels_at_one_value(*, weak_level: bool = False):
    """Three levels of ``g`` whose rows share one ``x`` each, as on the book's Area A and B.

    Each such level's constant is aliased with its dummy, so the cached trial
    Hessian is singular in exact arithmetic and its computed null eigenvalue
    falls on either side of zero by the rounding of the level's weight sums
    over its 15,000 rows.  ``weak_level`` adds a factor ``h`` with one level
    whose 50 rows carry prior weight ``1e-20``: a slope the Laplace term
    leaves out (``reml.identified``).
    """
    import pandas as pd

    rng = np.random.default_rng(2)
    n, per_level = 60000, 15000
    x = np.exp(rng.normal(5.0, 1.5, n))
    level = np.where(x < np.quantile(x, 0.5), "B", "C")
    for k, (name, value) in enumerate((("A", 3.0), ("D", 11.0), ("E", 40.0))):
        rows = slice(k * per_level, (k + 1) * per_level)
        level[rows] = name
        x[rows] = value
    exposure = np.full(n, 0.5)
    y = rng.poisson(0.1 * np.exp(0.1 * np.log(x)) * exposure) / exposure
    frame = pd.DataFrame({"x": x, "g": level})
    if weak_level:
        rare = rng.choice(n, size=50, replace=False)
        frame["h"] = np.where(np.isin(np.arange(n), rare), "rare", "common")
        exposure[rare] = 1.0e-20
        y[rare] = 0.0
    return frame, y, exposure


def _fit_levels_at_one_value(frame, y, exposure):
    from superglm import Categorical, Spline, SuperGLM

    features = {"x": Spline(n_knots=10), "g": Categorical()}
    if "h" in frame:
        features["h"] = Categorical()
    return SuperGLM(
        family="poisson",
        discrete=True,
        selection_penalty=0,
        features=features,
        interactions=[("x", "g")],
    ).fit_reml(frame, y, sample_weight=exposure)


@pytest.mark.slow
def test_discrete_spline_by_factor_fits_when_levels_sit_at_one_value(monkeypatch) -> None:
    """The alias residue is gone at its source: no cached trial needs the formation bound.

    At 6eb42f3f this fit raised "materially indefinite" (``-8.7e-13`` against
    a bar of ``1.1e-13``, OpenBLAS on x86-64).  With the formation bound alone
    (94879f33) it completed, solving 8 trials as formation-limited, and the
    alias was dropped where its residue landed negative and kept where it
    landed positive.  The Categorical x spline-by-categorical cross now scales
    the level's binned weight sum, the one the level's other blocks use, so
    the residue is at the rounding of products (``~5e-16``) and inside the
    eigensolver's bar on either sign.  The spy re-decomposes every bounded
    trial without the bound: none may be refused, and none counted.
    """
    import superglm.reml.discrete as discrete_module

    refused = _spy_old_refusals(monkeypatch, discrete_module)
    frame, y, exposure = _levels_at_one_value()
    model = _fit_levels_at_one_value(frame, y, exposure)

    diagnostics = model.reml_diagnostics()
    assert diagnostics["converged"]
    assert np.all(np.isfinite(model.predict(frame)))
    assert not any(refused)
    assert diagnostics["profile"]["reml_n_formation_limited_trials"] == 0


@pytest.mark.slow
def test_identified_trials_restrict_under_the_formation_bound_in_a_fit(monkeypatch) -> None:
    """A weak slope left out of the Laplace term: the fit converges, and every
    restricted decomposition of a cached Hessian carries its formation bound.

    The weak level of ``h`` makes the identified part non-empty, so each
    candidate and each cached trial decomposes ``H_c[I, I]`` again
    (``reml.identified``).  At 94879f33 the trial's restriction had no bound
    and raised "materially indefinite" from ``identified.py`` (``-7.6e-13``).
    Given the bound but with the residue still formed, the candidate's
    restriction kept it at the policy cutoff where it landed positive and
    dropped it where negative: its rank moved between 57 and 59 across
    iterations, its objective by about 22, and the fit stopped at
    ``max_reml_iter``.  With the residue removed at its source
    (``_cross_gram_categorical_spline_categorical``) no restriction is
    refused and the fit converges.
    """
    import superglm.solvers.rank as rank_module

    refused = _spy_old_refusals(monkeypatch, rank_module)
    frame, y, exposure = _levels_at_one_value(weak_level=True)
    model = _fit_levels_at_one_value(frame, y, exposure)

    diagnostics = model.reml_diagnostics()
    assert diagnostics["converged"]
    assert diagnostics["profile"]["reml_laplace_excluded"]
    assert refused, "no restricted decomposition received a formation bound"
    assert not any(refused)
