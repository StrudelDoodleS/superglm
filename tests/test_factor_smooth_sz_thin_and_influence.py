"""``sz`` factor smooths with exhausted levels, and their inference, as complete fits.

The stage-3 verifier's findings on the balance tree:

- one-row and weightless levels exhaust their balance subtrees; the tree must
  defer what they leave (decision 3) and fit as the dense solver does, where
  it declared every border column null and published wrong fits;
- leverage is the influence diagonal with the intercept, ``sum h = edf``, at
  any lambda and any width (design §3.10, definition A), without an
  ``sz``-wide inverse block;
- standard errors are NaN exactly on the coefficients the data leave
  non-estimable (the package's rule, as the dense solver), decided on the
  balance tree of the data, never on the signed rows' indefinite level
  blocks;
- a level whose rows carry only noise-level weight is named in the weak
  identification warning, not the whole model.

Every test fails under the mutation or the unfixed tree its docstring names.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import (
    Categorical,
    FactorSmooth,
    LambdaPolicy,
    Numeric,
    RandomEffect,
    Spline,
    SuperGLM,
)
from superglm.reml.penalty_algebra import (
    _penalty_component_omega_ssp,
    build_penalty_matrix,
    penalty_component_quadratic,
)

EPS = float(np.finfo(np.float64).eps)
_U = EPS / 2.0


def _gamma(count: float) -> float:
    return count * _U / (1.0 - count * _U)


def _frame(K: int = 30, n: int = 4000, seed: int = 3, one_row: int = 0):
    """The block-factor note's base: gamma-popular levels, a smooth deviation per level.

    Levels ``0 .. one_row - 1`` keep one row each; their other rows move to
    the last level.
    """
    rng = np.random.default_rng(seed)
    popularity = rng.gamma(2.0, size=K)
    g = rng.choice(K, size=n, p=popularity / popularity.sum())
    for level in range(one_row):
        rows = np.flatnonzero(g == level)
        g[rows[1:]] = K - 1
    cat = rng.integers(0, 8, n)
    x = rng.uniform(size=n)
    frame = pd.DataFrame(
        {
            "x": x,
            "x1": rng.normal(size=n),
            "cat": np.array([f"c{c}" for c in cat], dtype=object),
            "g": np.array([f"g{c:03d}" for c in g], dtype=object),
        }
    )
    eta = (
        0.2
        + 0.3 * frame["x1"].to_numpy()
        + rng.normal(0, 0.2, 8)[cat]
        + rng.normal(0, 0.3, K)[g]
        + rng.normal(0, 0.3, K)[g] * (x - 0.5)
        + 0.3 * np.sin(2 * np.pi * x) * (1.0 + rng.normal(0, 0.3, K)[g])
    )
    return frame, eta, rng, g


def _response(family: str, eta: np.ndarray, rng) -> np.ndarray:
    if family == "gaussian":
        return eta + rng.normal(0, 0.5, len(eta))
    if family == "gaussian_log":
        return np.exp(0.5 * eta) + rng.normal(0, 0.6, len(eta))
    if family == "poisson":
        return rng.poisson(np.exp(np.clip(eta, -5, 5))).astype(float)
    return rng.gamma(2.0, np.exp(np.clip(eta, -5, 5)) / 2.0)


def _model(
    family: str,
    direct_solve: str,
    *,
    lam: float | None = 0.7,
    k: int = 6,
    discrete: bool = False,
    numerics=("x1",),
    random=(),
    main_lam: float = 1.0,
) -> SuperGLM:
    family_name, link = {
        "gaussian": ("gaussian", None),
        "gaussian_log": ("gaussian", "log"),
        "poisson": ("poisson", None),
        "gamma_log": ("gamma", "log"),
    }[family]
    features = {name: Numeric() for name in numerics}
    features["cat"] = Categorical()
    features.update({name: RandomEffect() for name in random})
    features["x"] = Spline(
        n_knots=6, lambda_policy=None if lam is None else LambdaPolicy.fixed(main_lam)
    )
    policy = None if lam is None else {"wiggle": LambdaPolicy.fixed(lam)}
    kwargs = {} if link is None else {"link": link}
    return SuperGLM(
        family=family_name,
        features=features,
        interactions=[FactorSmooth("x", group="g", basis="sz", k=k, lambda_policy=policy)],
        selection_penalty=0,
        direct_solve=direct_solve,
        discrete=discrete,
        **kwargs,
    )


def _fit(model: SuperGLM, frame, y, weight) -> SuperGLM:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit_reml(frame, y, sample_weight=weight)


def _penalized_objective(model: SuperGLM) -> float:
    dm = model._dm
    S = build_penalty_matrix(
        dm.group_matrices, model._groups, model._reml_lambdas, dm.p, model._reml_penalties
    )
    beta = model.result.beta
    return float(model.result.deviance + beta @ S @ beta)


def _nan_mask(model: SuperGLM, frame, y, weight) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        se = model.metrics(frame, y, sample_weight=weight).coefficient_se
    return np.concatenate(
        [np.isnan(np.atleast_1d(np.asarray(se[g.name], dtype=float))) for g in model._groups]
    )


def _structurally_non_estimable(model: SuperGLM, weight) -> np.ndarray:
    """Slopes a null vector of ``[1, X]`` over the positive-weight rows touches.

    Column-equilibrated SVD: a null singular value is below ``max(n, p) eps``
    of the largest (``dgesdd``'s backward error).
    """
    X = np.asarray(model._dm.toarray(), dtype=np.float64)[weight > 0]
    A = np.column_stack([np.ones(len(X)), X])
    A = A / np.maximum(np.linalg.norm(A, axis=0), np.finfo(float).tiny)
    _, singular, rows = np.linalg.svd(A, full_matrices=False)
    null = rows[singular <= max(A.shape) * EPS * singular[0]]
    return np.any(np.abs(null[:, 1:]) > np.sqrt(EPS), axis=0)


def _assert_fits_as_gram(auto: SuperGLM, gram: SuperGLM) -> None:
    """One route to the same fit: no refusal, the dense solver's rank and penalized objective.

    Each mode is where PIRLS's stop rule ended, not the exact minimizer: the
    rule leaves an error in the REML criterion ``V`` below ``reml_tol (1 +
    |V|)`` (``mode_score``, **The bar**), and the penalized deviance ``D_p``
    enters the profiled criterion at ``dV/dD_p = 1 / (2 phi)`` (the envelope
    theorem on the profiled scale; ``phi = 1`` for a known scale).  One mode's
    ``D_p`` is therefore within ``2 phi reml_tol (1 + |V|)`` of the exact
    minimum and the two solvers' within twice that, however each platform's
    BLAS rounds them.  (``n eps |D_p|``, the bound this held before, is the
    rounding of one evaluation of ``D_p``, not the stop rule's resolution: the
    two solvers' objectives sat at 95% of it on Linux and above it on macOS
    and Windows.)  The unfixed tree declared every border column null (rank
    174 against 190) and stopped at 1.6 to 10 times the dense objective.
    """
    assert auto._reml_profile["direct_backend"] == "structured"
    assert bool(auto._reml_result.converged)
    assert int(auto.result.reml_hessian_rank) == int(gram.result.reml_hessian_rank)
    tree_objective, gram_objective = _penalized_objective(auto), _penalized_objective(gram)
    reml_tol = 1e-9  # fit_reml's default
    criterion = abs(float(gram._reml_result.objective))
    tolerance = 4.0 * float(gram.result.phi) * reml_tol * (1.0 + criterion)
    assert abs(tree_objective - gram_objective) <= tolerance


# ------------------------------------------------------------ exhausted levels
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("one_row", [4, 8])
def test_sz_one_row_levels_fit_as_the_dense_solver_does(one_row, discrete) -> None:
    """Four or eight one-row levels (stage-3 verifier, finding 1).

    A one-row level's deviation trades with the main effect: the data leave
    it free, and the sum-to-zero constraint spreads it over every level, so
    the standard errors are NaN on the design's structural non-estimable set
    exactly, as the dense solver's.  Mutations: the balance tree's deferral
    judged against the stacked matrix's own column norm (``_decide``)
    instead of the unreduced column; estimability read from the fit's own
    (penalized) aliases, which leaves the penalized coordinates finite.
    """
    frame, eta, rng, _ = _frame(one_row=one_row)
    y = _response("gaussian", eta, rng)
    weight = np.ones(len(frame))
    auto = _fit(_model("gaussian", "auto", discrete=discrete), frame, y, weight)
    gram = _fit(_model("gaussian", "gram", discrete=discrete), frame, y, weight)
    _assert_fits_as_gram(auto, gram)
    assert len(auto._reml_profile["structured_thin_levels"]) == one_row
    nan = _nan_mask(auto, frame, y, weight)
    np.testing.assert_array_equal(nan, _structurally_non_estimable(auto, weight))
    np.testing.assert_array_equal(nan, _nan_mask(gram, frame, y, weight))


@pytest.mark.parametrize("family", ["gaussian", "poisson", "gamma_log"])
def test_sz_weightless_levels_fit_as_the_dense_solver_does(family) -> None:
    """Two levels with no weight, in different balance subtrees (finding 1, coupled nulls).

    Mutation: as the previous test.
    """
    frame, eta, rng, g = _frame()
    y = _response(family, eta, rng)
    weight = np.where(np.isin(g, [3, 7]), 0.0, 1.0)
    auto = _fit(_model(family, "auto"), frame, y, weight)
    gram = _fit(_model(family, "gram"), frame, y, weight)
    _assert_fits_as_gram(auto, gram)


def _block_factor_frame(one_row: int, response: str = "signed"):
    """The block-factor note's sz fixture as the stage-3 verifier built it (K30, n 4000).

    ``response="signed"``: a log link's (observed-Newton rows that change
    sign); ``"fisher"``: an identity link's, the same draws to the noise.
    """
    rng = np.random.default_rng(3)
    K, n, n_cat = 30, 4000, 8
    popularity = rng.gamma(2.0, size=K)
    g = rng.choice(K, size=n, p=popularity / popularity.sum())
    for level in range(one_row):
        rows = np.flatnonzero(g == level)
        g[rows[1:]] = K - 1
    cat = rng.integers(0, n_cat, n)
    x = rng.uniform(size=n)
    x1 = rng.normal(size=n)
    x10 = 10.0 + rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "x": x,
            "x1": x1,
            "x10": x10,
            "cat": np.array([f"c{c:02d}" for c in cat], dtype=object),
            "g": np.array([f"g{c:03d}" for c in g], dtype=object),
        }
    )
    eta = (
        0.2
        + 0.3 * x1
        + 0.05 * (x10 - 10)
        + rng.normal(0, 0.2, n_cat)[cat]
        + rng.normal(0, 0.3, K)[g]
        + rng.normal(0, 0.3, K)[g] * (x - 0.5)
        + 0.3 * np.sin(2 * np.pi * x) * (1.0 + rng.normal(0, 0.3, K)[g])
    )
    if response == "fisher":
        return frame, eta + rng.normal(0, 0.5, n)
    return frame, np.exp(0.5 * eta) + rng.normal(0, 0.6, n)


@pytest.mark.parametrize("one_row", [4, 5])
def test_sz_one_row_levels_under_a_log_link_keep_their_border(one_row) -> None:
    """Gaussian with a log link and four or five one-row levels (finding 1, its last cases).

    The levels' exhausted subtrees leave carried coordinates whose Schur
    diagonals are a few times their own rounding.  With the bound's one free
    ``t`` set from them, every border column's bound grew to that share
    (``u_s`` above 1) and the retained factor certified no border column:
    rank 178 against the dense solver's 189, and the border's standard
    errors were zero.  The bound's second majorant takes ``t`` from the
    columns above the deferral's yardstick, and the factor keeping the higher
    certified rank retains the dense solver's rank.  Mutation: the single
    majorant.
    """
    frame, y = _block_factor_frame(one_row)
    weight = np.ones(len(frame))
    models = {
        solve: _fit(_model("gaussian_log", solve, numerics=("x1", "x10")), frame, y, weight)
        for solve in ("auto", "gram")
    }
    auto = models["auto"]
    assert auto._reml_profile["direct_backend"] == "structured"
    assert auto._linear_system_state.augmented_factor.rank == int(
        models["gram"].result.reml_hessian_rank
    )
    groups = ("x1", "x10", "cat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        se = {
            solve: np.concatenate(
                [
                    np.atleast_1d(np.asarray(model.metrics(frame, y).coefficient_se[g], float))
                    for g in groups
                ]
            )
            for solve, model in models.items()
        }
    assert np.all(np.isfinite(se["gram"]) & (se["gram"] > 0.0))
    assert np.all(np.isfinite(se["auto"]) & (se["auto"] > 0.0))


# ------------------------------------------- signed rows beside aliased levels
def _signed_aliased_frame(variant: str, response: str = "signed"):
    """The block-factor fixture with signed rows: two one-row levels, one level at a single
    ``x``, or two weightless levels (the final verifier's thin2, same_x and zero_weight)."""
    if variant == "one_row":
        frame, y = _block_factor_frame(2, response)
        return frame, y, np.ones(len(frame))
    frame, y = _block_factor_frame(0, response)
    g = frame["g"].to_numpy()
    if variant == "same_x":
        frame.loc[g == "g005", "x"] = 0.37
        return frame, y, np.ones(len(frame))
    return frame, y, np.where(np.isin(g, ["g003", "g007"]), 0.0, 1.0)


def _dense_log_pdet(model: SuperGLM, y, weight, rows: str = "signed") -> tuple[float, float]:
    """``log sum W + log pdet(H_c)`` at the published mode and rank, and its float64 bound.

    ``H_c = X'WX + S - X'W1 1'WX / sum W`` in the public coordinates with the
    observed rows of a Gaussian log link, ``W = w mu (2 mu - y)`` (``rows=
    "fisher"``: an identity link's, ``W = w``), assembled and
    diagonalized densely.  Each eigenvalue is within ``delta`` of the exact
    one (Weyl): the assembly's ``gamma_{n+2} || |X|'|W||X| || + gamma_p ||S||``
    (Higham 2002, section 3.5) and the symmetric eigensolver's ``p u ||H_c||``
    (its backward error), so the log of the retained ones is within ``sum_i
    log(lambda_i / (lambda_i - delta))``.
    """
    dm = model._dm
    X = np.asarray(dm.toarray(), dtype=np.float64)
    p = X.shape[1]
    S = build_penalty_matrix(
        dm.group_matrices, model._groups, model._reml_lambdas, p, model._reml_penalties
    )
    S = 0.5 * (S + S.T)
    if rows == "fisher":
        W = np.asarray(weight, dtype=np.float64)
    else:
        eta = np.clip(X @ model.result.beta + model.result.intercept, -80.0, 80.0)
        mu = np.exp(eta)
        W = weight * mu * (2.0 * mu - y)
    total = float(np.sum(W))
    cross = X.T @ W
    H = X.T @ (W[:, None] * X) + S - np.outer(cross, cross) / total
    H = 0.5 * (H + H.T)
    values = np.linalg.eigvalsh(H)
    retained = values[p + 1 - int(model.result.reml_hessian_rank) :]
    magnitude = np.abs(X).T @ (np.abs(W)[:, None] * np.abs(X))
    delta = (
        _gamma(len(y) + 2) * np.linalg.norm(magnitude, 2)
        + _gamma(p) * np.linalg.norm(S, 2)
        + p * _U * float(np.max(np.abs(values)))
    )
    assert np.all(retained > delta)
    bound = float(np.sum(np.log(retained / (retained - delta))))
    return math.log(total) + float(np.sum(np.log(retained))), bound


def test_sz_one_row_level_below_a_log_links_range_reaches_its_mode() -> None:
    """A one-row level whose response (-0.357) lies below the log link's range (stage 4, open).

    The level's eta is unpenalized and the likelihood drives its mean to the
    boundary: no finite mode along it (a direction of recession, Geyer 2009,
    Theorem 4).  Once the row's working weight is down to its rounding the
    tree truncates the direction, and the normal-equations solution of a
    truncated factor is its minimum-norm representative: it reset the
    direction every step (the row's eta from -20.7 to +2844 on one step of
    the stage-4 trace), so the fit never reached the boundary, kept the
    row's vanishing curvature (rank 192 against the dense solver's 191) and
    published an objective 16.8 below the dense solver's here.  The
    minimum-norm increment (Pes & Rodriguez 2021, arXiv 2101.07560, eq. 1.4)
    leaves the direction where it is, and the fit is the dense solver's.
    Mutation: the ``sz`` PIRLS step back to the normal-equations solution.
    """
    frame, y, weight = _signed_aliased_frame("one_row")
    assert float(y[frame["g"].to_numpy() == "g000"][0]) < 0.0
    models = {
        solve: _fit(
            _model("gaussian_log", solve, lam=128.0, numerics=("x1", "x10"), main_lam=1e4),
            frame,
            y,
            weight,
        )
        for solve in ("auto", "gram")
    }
    _assert_fits_as_gram(models["auto"], models["gram"])


@pytest.mark.parametrize("variant", ["one_row", "same_x", "weightless", "weightless_fisher"])
def test_sz_aliased_levels_converge_under_reml(variant) -> None:
    """REML beside two one-row, one single-``x`` or two weightless levels.

    The stage-4 tree stopped ``line_search_failed`` on every one (final
    verifier, finding 1, signed rows: objectives 21, 0.34 and 0.15 above the
    dense solver's; Fisher rows likewise).  Besides the one-row level's reset
    (previous test), each aliased level's data-null direction is penalized
    only by the main effect's penalty, which does not annihilate ``x``: its
    curvature, ``lambda_x`` times about ``1e-9``, crossed the border
    certificate's tolerance with ``lambda_x``, so the published rank and
    ``log|H|`` (by about 14) changed between nearby REML evaluations.  That
    direction is now deflated with its exact penalty curvature at every
    lambda, and the fit converges.  Mutations: no penalized alias deflation
    (every variant); the normal-equations PIRLS step (``one_row``); the
    dense penalty quadratic (``weightless_fisher``: lambda_x walks to its
    bound on the objective's rounding, next test).
    """
    if variant == "weightless_fisher":
        frame, y, weight = _signed_aliased_frame("weightless", response="fisher")
        family = "gaussian"
    else:
        frame, y, weight = _signed_aliased_frame(variant)
        family = "gaussian_log"
    model = _fit(_model(family, "auto", lam=None, numerics=("x1", "x10")), frame, y, weight)
    assert model._reml_profile["direct_backend"] == "structured"
    assert bool(model._reml_result.converged)
    assert model._reml_result.termination_reason != "line_search_failed"


_UNIDENTIFIED = {"weightless": ("g003", "g007"), "one_row": ("g000", "g001"), "same_x": ("g005",)}


def _thin_warnings(caught) -> list:
    return [w for w in caught if "follow the population" in str(w.message)]


def _fitted_eta(model: SuperGLM) -> np.ndarray:
    """The fit's own linear predictor on its training rows."""
    from superglm.links import stabilize_eta
    from superglm.solvers.mode_score import linear_predictor

    solver = model._solver_pirls_result()
    return stabilize_eta(linear_predictor(model._dm, solver, model._fit_offset), model._link)


def _sz_rule(spec, beta: np.ndarray):
    """``_sz_rule_blocks`` on the term's free coefficients."""
    return _sz_rule_blocks(spec, spec._level_blocks(beta))


def _sz_rule_blocks(spec, blocks: np.ndarray):
    """``(blocks, predicted blocks, c)`` of the ``sz`` term, and the rule's float64 magnitude.

    ``beta_t' = beta_t - F F' (beta_t - c)`` and ``c = sum_i V_i beta_i``
    (``FactorSmooth._identified_blocks``): each entry is within
    ``gamma_(K k + 4 k)`` of ``|beta_t| + |c| + |F| |F'| (|beta_t| + |c|)``
    with ``|c|`` itself bounded by ``sum_i |V_i| |beta_i|`` (Higham 2002,
    section 3.5); returned per level, ``(K, k)``.
    """
    predicted, offset = spec._identified_blocks(blocks)
    mapping = spec._population_map()
    reach = np.zeros(blocks.shape[1])
    if mapping is not None:
        levels, V = mapping
        stacked = np.broadcast_to(np.abs(V), (len(levels), *V.shape[-2:]))
        reach = np.einsum("lij,lj->i", stacked, np.abs(blocks[levels]))
    magnitude = np.abs(blocks) + np.abs(predicted) + reach[None, :]
    for level, free in zip(spec._unidentified_levels, spec._free_directions, strict=True):
        F = np.abs(np.asarray(free))
        magnitude[level] += F @ (F.T @ (np.abs(blocks[level]) + reach))
    return blocks, predicted, offset, magnitude


def _eta_magnitude(model: SuperGLM, frame: pd.DataFrame) -> np.ndarray:
    """Per row, the magnitudes every float64 operation of ``predict`` and of the fit touches.

    ``|alpha|``, each spline's ``|T(x)| |gamma|``, the ``sz`` term's
    ``|b(x)|`` against ``_sz_rule``'s magnitude of its level, and any other
    term's one product or coefficient: any order of the predictor's ``p``
    products and sums is then within ``gamma_(p + K k + 4 k + 2)`` of it
    (Higham 2002, sections 3.1 and 3.5).
    """
    from superglm._frame import as_eager_frame
    from superglm.features.spline import _SplineBase
    from superglm.model.base import _prediction_plan, _score_prediction_term_exact

    beta = np.asarray(model.result.beta, dtype=np.float64)
    eager = as_eager_frame(frame)
    total = np.full(len(frame), abs(float(model.result.intercept)))
    centred = getattr(model.result, "centred_intercept", None)
    if centred is not None:
        total += abs(float(centred))
    centre = getattr(model.result, "state_center", None)
    if centre is not None:
        total += float(np.abs(np.asarray(centre)) @ np.abs(beta))
    plan = _prediction_plan(model)
    for term in plan["features"] + plan["interactions"]:
        spec = term["spec"]
        coefficients = beta[term["beta_idx"]]
        if isinstance(spec, _SplineBase):
            columns = np.abs(spec.transform(frame[term["name"]].to_numpy(dtype=float)))
            total += columns @ np.abs(coefficients)
        elif isinstance(spec, FactorSmooth):
            _, _, _, magnitude = _sz_rule(spec, coefficients)
            codes = pd.Index(spec._levels).get_indexer(frame["g"].to_numpy())
            basis = np.abs(spec.marginal_basis(frame["x"].to_numpy(dtype=float)))
            total += np.einsum("ij,ij->i", basis, magnitude[np.maximum(codes, 0)])
        else:
            total += np.abs(_score_prediction_term_exact(term, eager, beta))
    return total


def _own_row_misfit(model: SuperGLM, frame: pd.DataFrame, weight: np.ndarray) -> np.ndarray:
    """Per row, how far a thin level's rule may move its own rows in exact arithmetic.

    ``b(x_i)' F F' (beta_t - c)`` is zero where the free directions ``F``
    are the exact null space of the level's distinct rows on ``N_P``; the
    computed ``F`` is that of the rows perturbed by the SVD's backward error,
    ``m u ||E_t||_2`` (this module's eigensolver convention), so the term is
    within that times ``||F' (beta_t - c)||_2``.
    """
    spec = model._interaction_specs["x:g:sz"]
    group = next(g for g in model._groups if g.name == "x:g:sz")
    blocks, _, offset, _ = _sz_rule(spec, np.asarray(model.result.beta[group.sl]))
    nullity = spec._population_null_space.shape[1]
    codes = pd.Index(spec._levels).get_indexer(frame["g"].to_numpy())
    misfit = np.zeros(len(frame))
    for level, free in zip(spec._unidentified_levels, spec._free_directions, strict=True):
        rows = (codes == level) & (weight > 0.0)
        if not np.any(rows):
            continue
        seen = spec.marginal_basis(frame["x"].to_numpy(dtype=float)[rows])
        seen = np.unique(seen, axis=0) @ spec._population_null_space
        size = nullity * _U * float(np.linalg.norm(seen, 2))
        misfit[rows] = size * float(np.linalg.norm(np.asarray(free).T @ (blocks[level] - offset)))
    return misfit


def _rounding_bound(model: SuperGLM, frame: pd.DataFrame) -> np.ndarray:
    spec = model._interaction_specs["x:g:sz"]
    count = model._dm.p + len(spec._levels) * spec.k + 4 * spec.k + 2
    return _gamma(count) * _eta_magnitude(model, frame)


@pytest.mark.parametrize(
    ("variant", "response", "solve"),
    [
        ("weightless", "signed", "auto"),
        ("weightless", "fisher", "auto"),
        ("weightless", "fisher", "gram"),
        ("one_row", "signed", "auto"),
        ("same_x", "fisher", "auto"),
    ],
)
def test_thin_sz_levels_keep_what_their_rows_identify(variant, response, solve) -> None:
    """A thin level keeps the fit at its own rows; one without weight takes the population (#432 a).

    Two weightless levels, two one-row levels or one level at a single ``x``
    leave part of each level's polynomial deviation free: shifting it, every
    level the other way by ``1 / K`` and the main effect with them keeps every
    other level's curve and every level's fit at its own rows, so the fit's
    coefficients hold an arbitrary point of that family.  The level keeps
    what its rows identify and takes the population's part where they say
    nothing, ``beta_t - Pi_t (beta_t - c)`` (owner decision 2026-10-01; the
    Opus and Claude reviews): ``predict`` on the training rows is the fit's
    own predictor to its rounding (``_rounding_bound`` for each predictor,
    ``_own_row_misfit`` for the free directions' backward error), as on
    master, with one warning per call naming the levels; a level without
    weight predicts the population value exactly.  On b5080877 the whole
    deviation went to the population: g005's 132 rows at ``x = 0.37``
    predicted 0.3508 against a mean response and fit of 0.0562.  Recorded
    from the design and prior weights, so gram fits alike.  Mutation: every
    thin level's deviation set to the population offset.
    """
    from superglm.model import base

    frame, y, weight = _signed_aliased_frame(variant, response=response)
    family = "gaussian" if response == "fisher" else "gaussian_log"
    model = _fit(_model(family, solve, lam=None, numerics=("x1", "x10")), frame, y, weight)
    spec = model._interaction_specs["x:g:sz"]
    expected = _UNIDENTIFIED[variant]
    assert spec._unidentified_level_names == expected
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eta = base.predict_eta_exact(model, frame)
    named = _thin_warnings(caught)
    assert len(named) == 1
    assert all(level in str(named[0].message) for level in expected)
    positive = weight > 0.0
    bound = 2.0 * _rounding_bound(model, frame) + _own_row_misfit(model, frame, weight)
    assert np.all(np.abs(eta - _fitted_eta(model))[positive] <= bound[positive])
    population = base.predict_eta_exact(model, frame, random_effects="population", warn=False)
    weightless = np.isin(frame["g"].to_numpy(), expected) & ~positive
    assert np.array_equal(eta[weightless], population[weightless])


def _family_shift(spec, blocks: np.ndarray, levels, scale: float, seed: int = 0) -> tuple:
    """Move every given level along a free direction: ``r_t`` in, ``-R / K`` from every level.

    ``r_t = F_t a_t`` for a thin level (``_free_directions``) and any
    ``N_P a_t`` for a separated one; ``R = sum_t r_t``.  The main effect then
    moves by ``+b(x)' R / K``, so ``predict`` is unchanged when every
    predicted block, and ``c``, moves by ``-R / K``.  Returns the shifted
    blocks and ``R / K``.
    """
    rng = np.random.default_rng(seed)
    free = dict(zip(spec._unidentified_levels, spec._free_directions, strict=True))
    shifted = np.array(blocks, copy=True)
    total = np.zeros(blocks.shape[1])
    for level in levels:
        directions = np.asarray(free.get(level, spec._population_null_space))
        r = directions @ rng.normal(size=directions.shape[1])
        r *= scale / np.linalg.norm(r)
        shifted[level] += r
        total += r
    shifted -= total[None, :] / len(blocks)
    return shifted, total / len(blocks)


def _assert_rule_follows(spec, blocks: np.ndarray, shifted: np.ndarray, step: np.ndarray, keep=()):
    """``predicted' + R / K == predicted`` and ``c' + R / K == c``, to the rule's rounding.

    Each side is within ``gamma_(K k + 4 k)`` of its ``_sz_rule`` magnitude,
    and the shift's own two roundings add ``gamma_2 (|blocks| + |R / K|)``.
    ``keep`` are levels whose own curve legitimately moves (a separated line).
    """
    count = len(blocks) * blocks.shape[1] + 4 * blocks.shape[1] + 2
    _, before, c0, m0 = _sz_rule_blocks(spec, blocks)
    _, after, c1, m1 = _sz_rule_blocks(spec, shifted)
    slack = _gamma(count) * (m0 + m1) + _gamma(2) * (np.abs(blocks) + np.abs(step)[None, :])
    moved = np.abs(after + step[None, :] - before)
    rows = [level for level in range(len(blocks)) if level not in set(keep)]
    assert np.all(moved[rows] <= slack[rows])
    assert np.all(np.abs(c1 + step - c0) <= np.max(slack, axis=0))


@pytest.mark.parametrize("variant", ["same_x", "one_row", "weightless"])
def test_sz_predictions_do_not_move_along_the_alias_family(variant) -> None:
    """Shifting the fit along a thin level's alias moves no prediction (#432 a; Claude review).

    The family: ``r_t`` in a thin level's free directions, ``-R / K`` from
    every level, ``+b(x)' R / K`` into the main effect.  ``predict`` is
    ``main + b(x)' beta'_t`` and the population ``main + b(x)' c``, so neither
    moves when every predicted block and ``c`` move by ``-R / K``: checked to
    the rule's rounding with a shift as large as the coefficients
    themselves.  Mutations: ``c`` the mean over every level (zero by the
    constraint), or no offset at all.
    """
    frame, y, weight = _signed_aliased_frame(variant, response="fisher")
    model = _fit(_model("gaussian", "auto", lam=None, numerics=("x1", "x10")), frame, y, weight)
    spec = model._interaction_specs["x:g:sz"]
    group = next(g for g in model._groups if g.name == "x:g:sz")
    blocks = spec._level_blocks(np.asarray(model.result.beta[group.sl]))
    scale = float(np.max(np.abs(blocks)))
    shifted, step = _family_shift(spec, blocks, spec._unidentified_levels, scale)
    _assert_rule_follows(spec, blocks, shifted, step)


def _all_thin_frame():
    """The Sol review's reproduction: ten levels, each with 100 rows at one ``x``."""
    rng = np.random.default_rng(4)
    x = np.repeat(np.linspace(0.01, 0.99, 10), 100)
    y = np.sin(4 * x) + 0.2 * rng.normal(size=1000)
    frame = pd.DataFrame({"x": x, "g": np.repeat([f"g{i}" for i in range(10)], 100)})
    return frame, y


def _all_thin_model(solve: str = "auto") -> SuperGLM:
    return SuperGLM(
        family="gaussian",
        features={"x": Spline(n_knots=6, lambda_policy=LambdaPolicy.fixed(1.0))},
        interactions=[
            FactorSmooth(
                "x", group="g", basis="sz", lambda_policy={"wiggle": LambdaPolicy.fixed(1.0)}
            )
        ],
        selection_penalty=0,
        direct_solve=solve,
    )


def test_an_sz_term_whose_every_level_is_thin_reproduces_its_fit() -> None:
    """Every level at one ``x``: the canonical population, a warning, and the fit kept (#432 b).

    No level identifies the population curve, so ``_population_offset`` was
    zero and every level's deviation went to it: the conditional training
    predictions spanned -260702 to 260703 on b5080877 against master's -0.726
    to 0.954 (Codex and Sol reviews, P1).  The population is now the family's
    canonical point, where each level's free part is zero, named in a warning
    at fit; the predictions on the training rows are the fit's own to their
    rounding, and the rule follows the family there as well.
    """
    from superglm.model import base

    frame, y = _all_thin_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _all_thin_model().fit_reml(frame, y)
    assert [w for w in caught if "fixed by convention" in str(w.message)]
    spec = model._interaction_specs["x:g:sz"]
    assert spec._population_convention == "canonical"
    eta = base.predict_eta_exact(model, frame, warn=False)
    weight = np.ones(len(frame))
    bound = 2.0 * _rounding_bound(model, frame) + _own_row_misfit(model, frame, weight)
    assert np.all(np.abs(eta - _fitted_eta(model)) <= bound)
    group = next(g for g in model._groups if g.name == "x:g:sz")
    blocks = spec._level_blocks(np.asarray(model.result.beta[group.sl]))
    shifted, step = _family_shift(spec, blocks, (0, 4, 9), float(np.max(np.abs(blocks))))
    _assert_rule_follows(spec, blocks, shifted, step)


def test_metrics_on_a_copy_of_the_training_rows_are_the_fits() -> None:
    """``metrics`` reads the same deviance from the training frame and from a copy (#432; Sol P1).

    The training frame's own object reuses the fit's means; a copy is
    predicted.  On b5080877 the copy predicted the same-x level g00 (80 rows
    at ``x = 0.37``, response 10) at the population value: deviance 6540.76
    against 27.99.  Now ``predict`` keeps that level's fit, and the two agree
    to the predictions' rounding ``rho`` (``_rounding_bound`` twice and
    ``_own_row_misfit``): ``|D_1 - D_2| <= sum w (2 |y - mu| rho + rho^2)``
    plus each sum's ``gamma_n D`` (identity link, ``mu = eta``).  Library
    evaluations do not raise ``predict``'s warning (Opus review, P3).
    """
    rng = np.random.default_rng(4)
    n = 800
    g = np.repeat(np.arange(10), 80)
    x = rng.uniform(size=n)
    x[g == 0] = 0.37
    y = np.sin(4 * x) + rng.normal(0, 0.2, n)
    y[g == 0] = 10.0
    frame = pd.DataFrame({"x": x, "g": np.array([f"g{i:02d}" for i in g], dtype=object)})
    model = _all_thin_model("structured")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        own = model.metrics(frame, y).deviance
        copied = model.metrics(frame.copy(), y.copy()).deviance
    assert not _thin_warnings(caught)
    weight = np.ones(n)
    rho = 2.0 * _rounding_bound(model, frame) + _own_row_misfit(model, frame, weight)
    residual = np.abs(y - _fitted_eta(model))
    bound = float(np.sum(2.0 * residual * rho + rho * rho)) + 2.0 * _gamma(n) * own
    assert abs(own - copied) <= bound


@pytest.mark.parametrize(("variant", "solve"), [("weightless", "auto"), ("same_x", "gram")])
def test_the_sz_population_is_the_identified_levels_mean(variant, solve) -> None:
    """The population curve is the one the identified levels' polynomial deviations sum to zero from.

    On a grid of ``x``, the identified levels' deviations from the population,
    ``sum_l (eta_l - eta_pop) = b(x)' sum_l (beta_l - c)``, are recovered as
    coefficients ``gamma`` by least squares on the term's basis ``b``; their
    part in the penalty's null space ``N_P' gamma`` is zero (``c`` the
    population offset, ``_population_offset``): as if the unidentified levels
    were not in the model.  On a2a909ef it was minus the unidentified levels'
    polynomial deviations, hundreds to thousands on these fits.  Bound: each
    predictor's rounding ``gamma_(T+2)`` times the sum of its terms'
    magnitudes ``M`` (``T`` terms), two predictors per row and ``L`` levels,
    through ``||b^+||_2``, plus the least-squares solve's own backward error
    ``k u kappa(b) ||gamma||``.  Fisher rows (identity link), the auto and
    gram backends.
    """
    from superglm._frame import as_eager_frame
    from superglm.model.base import _prediction_plan, _score_prediction_term_exact

    frame, y, weight = _signed_aliased_frame(variant, response="fisher")
    model = _fit(_model("gaussian", solve, lam=None, numerics=("x1", "x10")), frame, y, weight)
    spec = model._interaction_specs["x:g:sz"]
    identified = [level for level in spec._levels if level not in spec._unidentified_level_names]
    grid = np.linspace(0.02, 0.98, 3 * spec.k)
    rows = pd.DataFrame(
        {
            "x": np.tile(grid, len(identified)),
            "x1": 0.0,
            "x10": 10.0,
            "cat": frame["cat"].iloc[0],
            "g": np.repeat(identified, len(grid)),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deviations = model.predict(rows) - model.predict(rows, random_effects="population")
    total = deviations.reshape(len(identified), len(grid)).sum(axis=0)
    basis = spec.marginal_basis(grid)
    gamma, *_ = np.linalg.lstsq(basis, total, rcond=None)
    null_part = spec._population_null_space.T @ gamma

    plan = _prediction_plan(model)
    terms = plan["features"] + plan["interactions"]
    eager = as_eager_frame(rows)
    magnitude = abs(float(model.result.intercept)) + sum(
        np.abs(_score_prediction_term_exact(term, eager, model.result.beta)) for term in terms
    )
    rounding = _gamma(len(terms) + 2) * float(np.max(magnitude))
    singular = np.linalg.svd(basis, compute_uv=False)
    bound = (2.0 * len(identified) * math.sqrt(len(grid)) * rounding) / singular[-1] + (
        spec.k * _U * singular[0] / singular[-1] * float(np.linalg.norm(gamma))
    )
    assert float(np.max(np.abs(null_part))) <= bound


def _clear_level_records(spec) -> None:
    """The ``sz`` spec as if no level were thin or separated (the fit's own coordinates)."""
    spec._unidentified_levels = ()
    spec._free_directions = ()
    spec._weightless_levels = ()
    spec._separated_levels = ()
    spec._population_null_space = None


def test_library_evaluations_read_the_fits_own_predictor() -> None:
    """Screening, random-effect reporting and the discretization deltas read the fit (Claude review).

    They evaluate the fit on its training rows through the predictor, and on
    b5080877 met the thin-level rule there: the same-x level's rows took the
    population value, so the working residuals at them held the level's whole
    effect, away from the fit.  They now read the fit's own coefficients,
    exactly as with no level recorded, and raise no ``predict`` warning.
    Demonstration: each table moves when the rule is applied to them.
    """
    frame, y, weight = _signed_aliased_frame("same_x", response="fisher")
    rng = np.random.default_rng(11)
    frame["h"] = np.array([f"h{v}" for v in rng.integers(0, 10, len(frame))], dtype=object)
    model = _fit(_random_effect_model("auto", (2462.0, 566.8, 16.28)), frame, y, weight)
    spec = model._interaction_specs["x:g:sz"]
    assert spec._unidentified_level_names == ("g005",)

    def evaluate() -> list[pd.DataFrame]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tables = [
                model.screen_interactions(frame, y, candidates=[("x1", "cat")]),
                model.random_effects("h", X=frame, y=y, sample_weight=weight).table,
                model.discretization_impact(frame, y, weight, features=["x"]).tables["x"],
            ]
        assert not _thin_warnings(caught)
        return tables

    recorded = evaluate()
    held = {
        name: getattr(spec, name)
        for name in (
            "_unidentified_levels",
            "_free_directions",
            "_weightless_levels",
            "_separated_levels",
            "_population_null_space",
        )
    }
    _clear_level_records(spec)
    try:
        cleared = evaluate()
    finally:
        for name, value in held.items():
            setattr(spec, name, value)
    for left, right in zip(recorded, cleared, strict=True):
        pd.testing.assert_frame_equal(left, right, check_exact=True)


def _separated_poisson():
    """A Poisson ``sz`` fit with a zero-claim level, a one-sided level and a two-sided one.

    g000 has no claim; g001's claims all sit at ``x = 0.2`` with its other
    rows above it (the line ``-(x - 0.2)`` separates); g002's claims sit at
    ``x = 0.5`` with rows on both sides (no line separates).
    """
    rng = np.random.default_rng(21)
    K, n = 12, 3600
    g = rng.integers(0, K, n)
    x = rng.uniform(size=n)
    y = rng.poisson(np.exp(0.2 + np.sin(3 * x) + rng.normal(0, 0.3, K)[g])).astype(float)
    y[g == 0] = 0.0
    one, two = np.flatnonzero(g == 1), np.flatnonzero(g == 2)
    x[one] = 0.25 + 0.7 * rng.uniform(size=len(one))
    x[one[:3]] = 0.2
    y[one] = 0.0
    y[one[:3]] = 1.0
    y[two] = 0.0
    x[two[:3]] = 0.5
    y[two[:3]] = 2.0
    frame = pd.DataFrame({"x": x, "g": np.array([f"g{v:03d}" for v in g], dtype=object)})
    return frame, y


def _separated_model(separation: str = "warn") -> SuperGLM:
    return SuperGLM(
        family="poisson",
        features={"x": Spline(n_knots=6, lambda_policy=LambdaPolicy.fixed(1.0))},
        interactions=[
            FactorSmooth(
                "x", group="g", basis="sz", lambda_policy={"wiggle": LambdaPolicy.fixed(1.0)}
            )
        ],
        selection_penalty=0,
        separation=separation,
    )


def test_an_sz_line_that_separates_is_named_and_left_out_of_the_population() -> None:
    """A level whose unpenalized line separates the response is named, and stays out (Opus P2).

    On pg17's make model 26 of 71 identified makes had no claim; their lines
    walked toward ``-inf`` for as long as each fit ran, so the population
    curve, the mean of the levels with them, sat at ``[-37, -9.6]`` (auto)
    against ``[-80, 80]`` (gram).  The scan (``separated_factor_smooth_levels``)
    names a zero-claim level and a one-sided one, not a level with claims
    inside its rows, in a ``SeparationWarning`` (``separation="ignore"``
    silences it and still leaves them out).  Along a separated line, ``d``
    into the level and ``-d / K`` from every level, the population and every
    other level do not move (``_assert_rule_follows``).  The binomial scan,
    on the same design, finds the level whose ones all lie above its zeros
    (the linear program's side).  Mutations: no scan; the separated levels
    kept in the mean.
    """
    from superglm.diagnostics.separation import (
        SeparationWarning,
        separated_factor_smooth_levels,
    )

    frame, y = _separated_poisson()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _separated_model().fit_reml(frame, y)
    spec = model._interaction_specs["x:g:sz"]
    assert [spec._levels[code] for code in spec._separated_levels] == ["g000", "g001"]
    named = [w for w in caught if "unpenalized line" in str(w.message)]
    assert len(named) == 1
    assert issubclass(named[0].category, SeparationWarning)
    assert "'g000'" in str(named[0].message) and "'g001'" in str(named[0].message)
    assert "'g002'" not in str(named[0].message)
    group = next(g for g in model._groups if g.name == "x:g:sz")
    blocks = spec._level_blocks(np.asarray(model.result.beta[group.sl]))
    shifted, step = _family_shift(spec, blocks, (0, 1), float(np.max(np.abs(blocks))))
    _assert_rule_follows(spec, blocks, shifted, step, keep=(0, 1))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet = _separated_model("ignore").fit_reml(frame, y)
    assert not [w for w in caught if "unpenalized line" in str(w.message)]
    assert quiet._interaction_specs["x:g:sz"]._separated_levels == spec._separated_levels

    dm = model._dm.group_matrices[model._groups.index(group)]
    codes = np.asarray(dm.codes)
    binary = (y > 0).astype(float)
    level = np.flatnonzero(codes == 3)
    binary[level] = (frame["x"].to_numpy()[level] > 0.6).astype(float)
    found = separated_factor_smooth_levels(
        dm, spec._population_null_space, None, binary, ("zero", "one")
    )
    assert 3 in found and 2 not in found


def test_an_sz_model_saved_before_the_record_predicts_as_refitted() -> None:
    """A v0.35.0 or master pickle records its thin levels at the first prediction (Opus P3).

    Their specs carry no record, and on b5080877 they predicted bit for bit
    as before, silently (the one-row model's population spanned 1.8e-35 to
    5.5e34).  The fit's design and prior weights, which such a model keeps,
    rebuild the record: the loaded model predicts as the fitted one, bit for
    bit, and names the levels.
    """
    import pickle

    from superglm.model import base

    frame, y, weight = _signed_aliased_frame("one_row", response="fisher")
    model = _fit(_model("gaussian", "auto", lam=None, numerics=("x1", "x10")), frame, y, weight)
    expected = base.predict_eta_exact(model, frame, warn=False)
    population = base.predict_eta_exact(model, frame, random_effects="population", warn=False)
    spec = model._interaction_specs["x:g:sz"]
    for name in (
        "_unidentified_levels",
        "_free_directions",
        "_weightless_levels",
        "_separated_levels",
        "_population_null_space",
    ):
        delattr(spec, name)
    loaded = pickle.loads(pickle.dumps(model))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eta = base.predict_eta_exact(loaded, frame)
    assert _thin_warnings(caught)
    assert np.array_equal(eta, expected)
    assert np.array_equal(
        base.predict_eta_exact(loaded, frame, random_effects="population", warn=False), population
    )


@pytest.mark.parametrize("variant", ["weightless", "one_row"])
def test_sz_reports_agree_with_the_population_prediction(variant) -> None:
    """``factor_smooth``, ``reconstruct_feature`` and ``relativities`` report what ``predict`` does.

    With levels left out of the population a level's curve is ``predict(level)
    - predict(population)`` and the global curve the population's
    (``_population_deviations``, ``with_population_curve``).  On b5080877
    ``factor_smooth`` showed the fit's coordinates: identified g000 from -379
    to +316 where the predictions differ by -0.13 to -0.21, and weightless
    g003 about +-5000 against a predicted deviation of exactly zero (Opus
    review, P2).  Checked on training ``x`` values to the predictions'
    rounding (``_rounding_bound`` for each predictor and ``gamma_k`` for the
    reported product); a weightless level's curve and its error are zero.
    Mutation: the reports unshifted.
    """
    from superglm.model import base

    frame, y, weight = _signed_aliased_frame(variant, response="fisher")
    model = _fit(_model("gaussian", "auto", lam=None, numerics=("x1", "x10")), frame, y, weight)
    spec = model._interaction_specs["x:g:sz"]
    xs = np.sort(frame["x"].to_numpy()[:: len(frame) // 25])
    levels = list(dict.fromkeys(["g000", "g002", *spec._unidentified_level_names]))
    result = model.factor_smooth("x:g:sz", grid=xs, levels=levels)
    reconstructed = model.reconstruct_feature("x:g:sz")["coefficients"]
    k = spec.k
    for level in levels:
        rows = pd.DataFrame(
            {"x": xs, "x1": 0.0, "x10": 10.0, "cat": frame["cat"].iloc[0], "g": level}
        )
        own = base.predict_eta_exact(model, rows, warn=False)
        population = base.predict_eta_exact(model, rows, random_effects="population", warn=False)
        bound = _rounding_bound(model, rows) + _rounding_bound(model, rows.assign(g="unseen-level"))
        curve = result.curves[result.curves["level"] == level]
        effect = curve["effect"].to_numpy()
        basis = spec.marginal_basis(xs)
        product = basis @ reconstructed[level]
        assert np.all(
            np.abs(effect - (own - population))
            <= bound + _gamma(k) * np.abs(basis) @ np.abs(reconstructed[level])
        )
        assert np.all(
            np.abs(product - effect)
            <= 2.0 * _gamma(k) * np.abs(basis) @ np.abs(reconstructed[level])
        )
        if level in spec._levels and spec._levels.index(level) in spec._weightless_levels:
            assert np.all(effect == 0.0)
            assert np.all(curve["posterior_se"].to_numpy() == 0.0)

    relativity = model.relativities()["x"]
    grid = relativity["x"].to_numpy()
    rows = pd.DataFrame(
        {"x": grid, "x1": 0.0, "x10": 10.0, "cat": frame["cat"].iloc[0], "g": "unseen-level"}
    )
    population = base.predict_eta_exact(model, rows, random_effects="population", warn=False)
    reported = relativity["log_relativity"].to_numpy()
    drift = (reported - reported[0]) - (population - population[0])
    bound = _rounding_bound(model, rows)
    assert np.all(np.abs(drift) <= 2.0 * (bound + bound[0]))


def test_the_reported_curves_errors_are_those_of_their_contrasts() -> None:
    """The reported curves' errors are those of the contrasts they report (#432; Opus P2).

    ``factor_smooth``'s level curve ``b(x)' Q_t (beta_t - c)``
    (``_population_deviations``) and the global curve ``T(x) gamma + b(x)'
    c`` (``_population_curve_se``) are linear in the coefficients, so each
    error is ``sqrt(g' V g)`` for its contrast ``g``.  On a gram fit the
    covariance is dense, and both routes are sums of products of its entries:
    each within ``gamma_(2 p + 4 k)`` of ``|g|' |V| |g|`` (Higham 2002,
    section 3.5).  A weightless level's curve has no error.  Mutation: the
    unshifted curves' errors.
    """
    frame, y, weight = _signed_aliased_frame("weightless", response="fisher")
    model = _fit(_model("gaussian", "gram", lam=None, numerics=("x1", "x10")), frame, y, weight)
    spec = model._interaction_specs["x:g:sz"]
    contrast = spec._population_contrast()
    K, k = len(spec._levels), spec.k

    def check(G: np.ndarray, V: np.ndarray, se: np.ndarray) -> None:
        quad = np.einsum("ij,jk,ik->i", G, V, G)
        size = np.einsum("ij,jk,ik->i", np.abs(G), np.abs(V), np.abs(G))
        assert np.all(np.abs(se * se - quad) <= 2.0 * _gamma(2 * len(V) + 4 * k) * size)

    covariance, active = model._coef_covariance
    V = np.asarray(covariance, dtype=np.float64)
    main = next(group for group in active if group.feature_name == "x")
    term = next(group for group in active if group.feature_name == "x:g:sz")
    relativity = model.relativities(with_se=True)["x"]
    grid = relativity["x"].to_numpy()
    G = np.zeros((len(grid), len(V)))
    G[:, main.start : main.end] = model._specs["x"].transform(grid)
    G[:, term.start : term.end] = spec.marginal_basis(grid) @ contrast
    check(G, V, relativity["se_log_relativity"].to_numpy())

    inference = model._fit_inference_info
    augmented = float(model.result.phi) * np.asarray(inference["XtWX_inv_aug"], dtype=np.float64)
    group = next(a for a in inference["active_groups"] if a.name == "x:g:sz")
    xs = grid[::25]
    levels = ["g000", "g002", *spec._unidentified_level_names, spec._levels[-1]]
    result = model.factor_smooth("x:g:sz", grid=xs, levels=levels)
    free = dict(zip(spec._unidentified_levels, spec._free_directions, strict=True))
    basis = spec.marginal_basis(xs)
    for level in levels:
        code = spec._levels.index(level)
        select = np.zeros((k, (K - 1) * k))
        if code < K - 1:
            select[:, code * k : (code + 1) * k] = np.eye(k)
        else:
            select[:] = -np.tile(np.eye(k), (1, K - 1))
        keep = np.eye(k)
        if code in free:
            F = np.asarray(free[code])
            keep = np.zeros((k, k)) if code in spec._weightless_levels else keep - F @ F.T
        G = np.zeros((len(xs), len(augmented)))
        G[:, 1 + group.start : 1 + group.end] = basis @ keep @ (select - contrast)
        se = result.curves.loc[result.curves["level"] == level, "posterior_se"].to_numpy()
        check(G, augmented, se)
        if code in spec._weightless_levels:
            assert np.all(se == 0.0)


def test_a_holdout_drop_term_starts_from_the_predicted_predictor() -> None:
    """Holdout drop-term deltas start from ``predict``'s predictor (#432; Claude review, Low).

    ``term_drop_diagnostics(mode="holdout")`` summed the terms at the fit's
    coordinates, so a weightless level's rows took the fit's arbitrary point
    along its alias (about +-5000 on this fixture) where ``predict`` gives the
    population.  For the numeric ``x1`` the delta is ``D(eta - x1 b) -
    D(eta)``; with ``eta`` from ``predict``, each deviance is within its rows'
    predictor rounding ``rho`` (``2 |y - mu| rho + rho^2`` a row, identity
    link) and each sum's ``gamma_n D``.
    """
    from superglm.diagnostics.term_diagnostics import term_drop_diagnostics
    from superglm.model import base

    frame, y, weight = _signed_aliased_frame("weightless", response="fisher")
    model = _fit(_model("gaussian", "auto", lam=None, numerics=("x1", "x10")), frame, y, weight)
    rows = frame["g"].isin(["g003", "g007", "g010"]).to_numpy()
    held = frame[rows].reset_index(drop=True)
    target = y[rows]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = term_drop_diagnostics(
            model,
            frame,
            y,
            weight,
            mode="holdout",
            X_val=held,
            y_val=target,
            sample_weight_val=np.ones(len(held)),
        )
    reported = float(table.loc[table["feature"] == "x1", "delta_deviance"].iloc[0])
    eta = base.predict_eta_exact(model, held, warn=False)
    x1 = next(g for g in model._groups if g.name == "x1")
    term = held["x1"].to_numpy() * float(model.result.beta[x1.sl][0])
    full = float(np.sum((target - eta) ** 2))
    dropped = float(np.sum((target - (eta - term)) ** 2))
    rho = 2.0 * _rounding_bound(model, held) + _gamma(2) * np.abs(term)
    bound = (
        float(np.sum(2.0 * np.abs(target - eta) * rho + rho * rho))
        + float(np.sum(2.0 * np.abs(target - eta + term) * rho + rho * rho))
        + _gamma(len(held)) * (full + dropped)
    )
    assert abs(reported - (dropped - full)) <= bound


def test_sz_aliased_levels_converge_beside_a_laplace_excluded_column(monkeypatch) -> None:
    """Two weightless levels beside ``xt``, which only two rows of weight 1e-15 move (#432 e).

    ``xt`` is left out of the Laplace term, so REML reads ``log|H_II|`` from
    the factor rebuilt without it, which deflates the levels' penalized alias
    only where the alias is exactly zero on ``xt``.  Represented over every
    border column, the alias took a least-squares coefficient on ``xt`` that
    it does not need (2e-16 of the largest in the scaled solve, never 0.0),
    so the rebuild handed it to the pivoted factorization: the rank went
    190 -> 189 between each accepted iterate and every trial 0.7% away
    (``log|H_II|`` +13.4), and REML stopped ``line_search_failed`` 0.0088
    above its optimum.  The rank of ``H_II`` does not depend on positive
    smoothing parameters, so every rebuild has one rank.  Mutation: the
    excluded columns back in the representation (``_penalized_aliases``).
    """
    from superglm.solvers._structured.balance_tree import SumToZeroTreeFactor

    frame, y, weight = _signed_aliased_frame("weightless", response="fisher")
    rows = np.flatnonzero((weight > 0) & (frame["g"].to_numpy() == "g029"))[:2]
    frame["xt"] = 5.0
    frame.loc[frame.index[rows], "xt"] = [6.0, 7.0]
    weight[rows] = 1e-15
    build = SumToZeroTreeFactor.__init__
    ranks: list[int] = []

    def recording(self, *args, excluded=(), **kwargs):
        build(self, *args, excluded=excluded, **kwargs)
        if excluded:
            ranks.append(int(self.rank))

    monkeypatch.setattr(SumToZeroTreeFactor, "__init__", recording)
    model = _fit(
        _model("gaussian", "auto", lam=None, numerics=("x1", "x10", "xt")), frame, y, weight
    )
    assert model._reml_profile["direct_backend"] == "structured"
    xt = next(group.start for group in model._groups if group.name == "xt")
    assert model._reml_profile["reml_laplace_excluded"] == (xt,)
    assert ranks and len(set(ranks)) == 1
    assert bool(model._reml_result.converged)
    assert model._reml_result.termination_reason != "line_search_failed"


def _penalty_root(model: SuperGLM, p: int) -> tuple[np.ndarray, float]:
    """``R`` with ``R'R = S``, and its blocks' eigensolver backward error ``sum 2 p_k u ||S_k||_2``.

    An ``sz`` block is ``kron([I; -1'], omega^1/2)``: its ``omega`` is diagonal
    (the natural parameterization), so its root is exact to ``u`` entrywise.  A
    dense block's root comes from its eigendecomposition, exact for ``S_k``
    perturbed by ``p_k u ||S_k||`` and by as much again for the clipped
    negative eigenvalues.
    """
    rows, backward = [], 0.0
    for component in model._reml_penalties:
        lam = model._reml_lambdas[component.name]
        omega = lam * np.asarray(component.omega_ssp, dtype=np.float64)
        if component.penalty_kind == "sum_to_zero":
            assert np.array_equal(omega, np.diag(np.diag(omega)))
            K = component.repeat_count
            contrast = np.vstack((np.eye(K - 1), -np.ones((1, K - 1))))
            block = np.kron(contrast, np.diag(np.sqrt(np.diag(omega))))
        else:
            assert component.penalty_kind == "dense"
            values, vectors = np.linalg.eigh(omega)
            block = np.sqrt(np.clip(values, 0.0, None))[:, None] * vectors.T
            backward += 2.0 * len(values) * _U * float(np.max(np.abs(values)))
        root = np.zeros((block.shape[0], p))
        root[:, component.group_sl] = block
        rows.append(root)
    return np.vstack(rows), backward


def _dense_cross_trace(model: SuperGLM, X, W, rates, rank: int) -> tuple[float, float]:
    """``t = tr(H^+ C H^+ C)`` from square roots, and a bound ``beta`` on ``sqrt(t)``'s error.

    ``H = A'A``, ``A = [W^1/2 X_c; R]`` (Fisher rows centred on their weighted
    mean ``c``, ``R'R = S``), and ``C = G'JG``, ``G = |a|^1/2 X_c``, ``J =
    sign(a)``.  Householder QR of ``A``, then the SVD ``U s V'`` of its
    triangle, give ``t = ||F'JF||_F^2``, ``F = G V_r / s_r``, at the factor's
    rank ``r``.  ``t`` is invariant under ``H, C -> B'HB, B'CB`` for any
    invertible ``B``, so ``V_r`` may span any complement of ``H``'s exact
    nulls once ``C`` annihilates them too.  No Gram is formed: the alias
    (``s_r`` near ``1e-6``) costs ``kappa(A)``, not ``kappa(A)^2 = 1e16``.

    Rounding perturbs the square roots.  ``dA = (gamma_mp + gamma_2) ||A||_F
    + p u s_1 + sqrt(sum W) ||dc||``: Householder QR column by column (Higham
    2002, Theorem 19.4, its constant taken as 1 as in Connolly & Higham 2022,
    section 7), the entries, the SVD (``p u ||A||_2``, its factors orthogonal
    to ``p u``) and the computed mean, ``||dc|| <= gamma_(2n+1) || |X|'W || /
    sum W``.  ``dG = (gamma_2 + gamma_p sqrt(p)) ||G||_F + sqrt(sum |a|)
    ||dc||``, the second term the product ``G V_r``.  Each costs ``1 /
    sigma``, ``sigma^2 = (s_r - dA)^2 - dS`` a lower bound on the exact
    ``s_r^2`` (Weyl; ``dS`` from ``_penalty_root``): ``rho = dA / sigma``,
    ``rho_G = dG / sigma``.  The weights' relative rounding ``delta`` (the
    predictor's ``gamma_(p+1) (|X||b| + |b_0|)``, with ``|X||b|`` up to
    ``1e6`` here, then ``exp``, the products and ``W^1/2``) costs no ``1 /
    sigma``: ``c`` stays a weighted mean and ``W^1/2 X_c H^-1/2`` has norm 1,
    so ``H`` moves by ``2 delta`` and ``C`` by ``delta ||F||_F^2 + 2 w phi +
    w^2`` (``w = delta kappa / (1 - delta)``, ``kappa^2 = sum |a| / sum W``,
    ``phi >= ||G H^-1/2||_2``) in the scaling below.  So the computed ``H``
    is ``H^1/2 (I + E) H^1/2`` with ``||E|| <= eta = (1 + 2 delta)(1 + 2 rho
    + rho^2 + 2 p u + dS / sigma^2) - 1``, and ``M = H^-1/2 C H^-1/2`` moves by
    ``dM <=`` the weights' term plus ``(1 + 2 delta)(2 phi rho_G + rho_G^2 +
    gamma_(n+4) ||F||_F^2)`` (the last ``F'JF``).  With ``K = (I + E)^-1/2``
    the computed trace is ``||K (M + dM) K||_F^2``, so its square root is
    within ``(eta sqrt(t) + dM) / (1 - eta)`` of ``sqrt(t)``; with ``sqrt(t)``
    taken from the computed one, ``beta = (eta sqrt(t~) + dM) / (1 - 2 eta)``,
    plus the final sum's ``2 gamma_(r^2) sqrt(t~)``.
    """
    n, p = X.shape
    total = float(np.sum(W))
    Xc = X - (X.T @ W) / total
    root, dS = _penalty_root(model, p)
    A = np.vstack((np.sqrt(W)[:, None] * Xc, root))
    G = np.sqrt(np.abs(rates))[:, None] * Xc
    _, s, Vt = np.linalg.svd(np.linalg.qr(A, mode="r"))
    F = (G @ Vt[:rank].T) / s[:rank]
    M = F.T @ (np.sign(rates)[:, None] * F)
    reference = float(np.sum(M * M))

    predictor = np.abs(X) @ np.abs(model.result.beta) + abs(float(model.result.intercept))
    delta = math.expm1(2.0 * _gamma(p + 1) * float(np.max(predictor))) * (1.0 + _gamma(9))
    delta += _gamma(9)
    dc = _gamma(2 * n + 1) * float(np.linalg.norm(np.abs(X).T @ W)) / total
    absolute = float(np.sum(np.abs(rates)))
    dA = (_gamma(A.shape[0] * p) + _gamma(2)) * float(np.linalg.norm(A)) + p * _U * s[0]
    dA += math.sqrt(total) * dc
    dG = (_gamma(2) + _gamma(p) * math.sqrt(p)) * float(np.linalg.norm(G))
    dG += math.sqrt(absolute) * dc
    assert np.all(s[rank:] <= dA)  # the factor's truncated directions: exact nulls
    assert s[rank - 1] > dA and (s[rank - 1] - dA) ** 2 > dS
    sigma = math.sqrt((s[rank - 1] - dA) ** 2 - dS)
    rho, rho_G = dA / sigma, dG / sigma
    eta = (1.0 + 2.0 * delta) * (1.0 + 2.0 * rho + rho * rho + 2.0 * p * _U + dS / sigma**2) - 1.0
    assert eta < 0.5
    phi = (float(np.linalg.norm(F, 2)) + rho_G) * math.sqrt(1.0 + eta)
    frobenius = (float(np.linalg.norm(F)) + rho_G) ** 2 * (1.0 + eta)
    w = delta * math.sqrt(absolute / total) / (1.0 - delta)
    dM = delta * frobenius + 2.0 * w * phi + w * w
    dM += (1.0 + 2.0 * delta) * (2.0 * phi * rho_G + rho_G**2 + _gamma(n + 4) * frobenius)
    root_t = math.sqrt(reference)
    beta = (eta * root_t + dM) / (1.0 - 2.0 * eta) + 2.0 * _gamma(rank * rank) * root_t
    return reference, beta


@pytest.mark.parametrize(
    ("variant", "lam_x"), [("same_x", 8e-4), ("same_x", 1e-4), ("one_row", 1e-4)]
)
def test_an_sz_weight_derivative_cross_trace_survives_the_alias_variance(variant, lam_x) -> None:
    """``tr(H^+ C H^+ C)`` of a centred weight-derivative operator is the dense trace (#432 d).

    The REML Hessian traces each weight-derivative operator ``C`` (centred:
    ``J' O J`` over ``[1, X]``) against the profiled sz factor.  With
    ``lambda_x`` small a thin level's penalized alias has a variance of
    ``1e11`` to ``1e12``, and ``C`` vanishes along it; held as raw moments
    plus a rank-two centring, the two parts met that variance separately and
    the trace cancelled to their rounding.  ``_trace_form`` (the intercept
    column per level, each part annihilating the alias) keeps the dense
    trace.  The reference (``_dense_cross_trace``) is within ``beta`` of the
    exact trace in its square root and the factor is allowed as much, so
    ``|t - t_ref| <= 2 beta (2 sqrt(t_ref) + 2 beta)``: here ``7e-5``,
    ``1e-4`` and ``2e-3`` against ``0.48``, ``1.8`` and ``1.8``.  Mutation
    (``_trace_form`` returning ``self._form(operator)``): ``-1.0e5``,
    ``-1.05e7`` and ``+4.7e7``; origin/master: ``-3.0e5``, ``+1.3e7`` and
    ``-2.5e7``: the positive two would pass a check of the sign alone.  The
    rates vanish where ``W`` does, as ``dW/drho = 2 W deta/drho`` does on a
    log link's Fisher rows: ``C`` then annihilates every exact null of ``H``
    (``one_row``'s level below the link's range has ``W = 0``), so every
    generalized inverse gives the one trace.
    """
    from superglm.solvers._structured.block_leaves import factor_smooth_moment_operators
    from superglm.solvers._structured.operators import CenteredBlockOperator
    from superglm.solvers._structured.state import centred_data_operator
    from superglm.solvers.structured import get_structured_layout

    frame, y, weight = _signed_aliased_frame(variant)
    model = _model("gaussian_log", "auto", lam=1.29, main_lam=lam_x, numerics=("x1", "x10"))
    model = _fit(model, frame, y, weight)
    factor = model._linear_system_state.profiled_factor
    system = factor.augmented_factor.system
    assert not system.leaf.signed  # the terminal refit's rows are Fisher's, W = w mu^2
    X = np.asarray(model._dm.toarray(), dtype=np.float64)
    mu = np.exp(X @ model.result.beta + model.result.intercept)
    W = weight * mu * mu
    rates = np.random.default_rng(0).normal(size=len(W)) * W
    layout = get_structured_layout(
        model._dm, model._groups, dominant_group_index=system.dominant_group_index
    )
    ((raw, cross, total, level),) = factor_smooth_moment_operators(
        layout, [rates], center=system.leaf.center, level_cross=True
    )
    operator = CenteredBlockOperator(
        raw=raw,
        cross=cross,
        total=total,
        center=centred_data_operator(system).center,
        raw_structured_cross=level,
    )
    reference, beta = _dense_cross_trace(model, X, W, rates, factor.rank)
    both = 2.0 * beta
    error = factor.operator_cross_trace(operator, operator) - reference
    assert abs(error) <= both * (2.0 * math.sqrt(reference) + both)


@pytest.mark.threads
def test_sz_same_x_reml_decisions_do_not_follow_the_blas_thread_count(monkeypatch) -> None:
    """REML's decisions beside a same-x level are the same at 1 and N BLAS threads (#432 d).

    The level's penalized alias ``(a_0, v)`` is a null of ``[1, X]``, so every
    centred weight-derivative operator vanishes along it, while its variance is
    ``1 / (lambda_x a'Sa)`` (1.8e8 in the border at the first iterate).  Held
    as raw moments plus a rank-two centring, the two parts met that variance
    separately and the outer Hessian's wiggle entry cancelled to its rounding,
    so the first Newton step, and then the iteration count (14 at one BLAS
    thread against 13 at eight on a2a909ef), followed the thread count.
    ``ProfiledSumToZeroTreeFactor._trace_form`` holds the
    same operator as ``K' O K`` with the intercept column per level, each part
    annihilating the alias.  ``native`` keeps the pools live inside the fit
    (the default caps them to one thread below 1500 columns, which hides it).
    Mutation: ``_trace_form`` returning ``self._form(operator)``.
    """
    from threadpoolctl import ThreadpoolController, threadpool_limits

    monkeypatch.setenv("SUPERGLM_BLAS_THREADS", "native")
    native = max(
        (
            pool["num_threads"]
            for pool in ThreadpoolController().info()
            if pool.get("user_api") == "blas"
        ),
        default=1,
    )
    if native < 2:
        pytest.skip("one BLAS thread: nothing to compare")
    frame, y, weight = _signed_aliased_frame("same_x")
    runs = []
    for threads in (1, min(native, 8)):
        with threadpool_limits(threads, user_api="blas"):
            model = _fit(
                _model("gaussian_log", "auto", lam=None, numerics=("x1", "x10")), frame, y, weight
            )
        result = model._reml_result
        runs.append(
            (int(result.n_reml_iter), result.termination_reason, len(result.lambda_history))
        )
    assert runs[0] == runs[1]


def test_a_dense_penalty_charges_nothing_along_its_null_space() -> None:
    """The main spline's penalty quadratic over its range: a null-space coefficient adds nothing.

    Beside weightless levels REML drives ``lambda_x`` up, and the main
    effect's coefficient along its reparametrised penalty's null eigenvector
    reached 222.  The dense product charged ``lambda_x`` times that
    eigenvalue's rounding (``1.6e-15``) times its square: about one unit of
    the objective at ``lambda_x`` 1e10 (``lambda_x beta' Omega beta`` was
    -0.19 against a rounding bound of 1.57), so the objective turned to noise
    and REML wandered to its bound (``weightless_fisher`` above).  Over the
    range of the penalty, with its rank fixed as ``log |S|_+`` fixes it
    (Wood, Pya & Safken 2016, section 3.1.1), a null-space shift ``t v``
    moves the quadratic only through the computed eigenvectors' departure
    from orthogonality, ``p u`` each: at most ``2 ||Omega|| ||beta|| t p u +
    ||Omega|| (t p u)^2``.  Mutation: the dense product
    (``penalty_component_quadratic``).
    """
    frame, eta, rng, _ = _frame()
    y = _response("gaussian", eta, rng)
    model = _fit(_model("gaussian", "auto"), frame, y, np.ones(len(frame)))
    group = next(group for group in model._groups if group.name == "x")
    matrix = model._dm.group_matrices[model._groups.index(group)]
    component = next(pc for pc in model._reml_penalties if pc.group_name == "x")
    omega = _penalty_component_omega_ssp(component, matrix)
    assert omega is not None
    width = omega.shape[0]
    assert 0 < int(round(component.rank)) < width
    values, vectors = np.linalg.eigh(0.5 * (omega + omega.T))
    null = vectors[:, 0]
    beta = np.asarray(model.result.beta[group.sl], dtype=np.float64)
    shift = 1e6
    base = penalty_component_quadratic(component, beta, matrix)
    moved = penalty_component_quadratic(component, beta + shift * null, matrix)
    norm = float(np.max(np.abs(values)))
    drift = shift * width * _U
    bound = 2.0 * norm * float(np.linalg.norm(beta)) * drift + norm * drift * drift
    assert base >= 0.0
    assert abs(moved - base) <= bound


def test_sz_deflated_alias_beside_exact_nulls_keeps_the_exact_pseudo_determinant() -> None:
    """``log|H|`` with the penalized alias deflated beside the levels' exact nulls.

    Two one-row levels leave one penalized alias, one exact alias and the
    boundary row's direction.  The deflation's change of basis is unimodular
    but not orthogonal, so the pseudo-determinant of the congruence carries
    ``det(V'V) / det(Z'Z)`` over the mapped null vectors (one only when the
    nulls vanish on the deflated block); without it the published
    ``log|H|`` was off by 0.6 to 3.3 on this fixture.  The published value is the dense
    pseudo-determinant within both sides' bounds.  Mutation: the factor
    dropped (``border.factor_border``).
    """
    frame, y, weight = _signed_aliased_frame("one_row")
    model = _fit(
        _model("gaussian_log", "auto", lam=128.0, numerics=("x1", "x10"), main_lam=1e5),
        frame,
        y,
        weight,
    )
    factor = model._linear_system_state.augmented_factor
    assert factor._border.deflated
    assert factor.rank_truncated
    reference, bound = _dense_log_pdet(model, y, weight)
    certificate = factor.border_certificate.logdet_bound
    assert abs(float(model.result.log_det_H) - reference) <= bound + certificate


def _random_effect_model(direct_solve: str, lam: tuple[float, float, float]) -> SuperGLM:
    """The aliased fixture's model with a ten-level random effect in the border, fixed lambdas."""
    return SuperGLM(
        family="gaussian",
        features={
            "x1": Numeric(),
            "x10": Numeric(),
            "cat": Categorical(),
            "h": RandomEffect(lambda_policy=LambdaPolicy.fixed(lam[0])),
            "x": Spline(n_knots=6, lambda_policy=LambdaPolicy.fixed(lam[1])),
        },
        interactions=[
            FactorSmooth(
                "x",
                group="g",
                basis="sz",
                k=6,
                lambda_policy={"wiggle": LambdaPolicy.fixed(lam[2])},
            )
        ],
        selection_penalty=0,
        direct_solve=direct_solve,
    )


def test_sz_thin_levels_beside_a_random_effect_keep_the_dense_rank() -> None:
    """Two one-row levels beside a random effect: an exact alias is not deflated as curvature.

    A random effect's levels sum to the intercept, so the least-squares
    representation of a thin level's alias spread its constant onto them, and
    their ridge read as the alias's penalty curvature.  The deflation then
    reduced the alias against the random effect's generator, which cancels
    that curvature, and factored an exactly singular ``a_NN``: rank +1 and
    ``log|H|`` 64 below the dense value at fixed lambdas, silently (round-2
    review, P1; under Poisson REML the smoothing parameters went to their
    lower bounds with edf -927).  The generator's columns now stay out of the
    representation, so the deflated block is ``diag(G'SG, A'SA)`` and the
    alias certificate holds for it.  Demonstration: 6544d2bc publishes rank
    203 here against the dense 202.
    """
    frame, y, weight = _signed_aliased_frame("one_row", response="fisher")
    rng = np.random.default_rng(11)
    frame["h"] = np.array([f"h{v}" for v in rng.integers(0, 10, len(frame))], dtype=object)
    lam = (2462.0, 566.8, 16.28)
    models = {
        solve: _fit(_random_effect_model(solve, lam), frame, y, weight)
        for solve in ("auto", "gram")
    }
    auto = models["auto"]
    assert auto._reml_profile["direct_backend"] == "structured"
    assert int(auto.result.reml_hessian_rank) == int(models["gram"].result.reml_hessian_rank)
    reference, bound = _dense_log_pdet(auto, y, weight, rows="fisher")
    certificate = auto._linear_system_state.augmented_factor.border_certificate.logdet_bound
    assert abs(float(auto.result.log_det_H) - reference) <= bound + certificate
    assert float(auto.result.effective_df) > 0.0


def test_gram_counts_no_rounding_curvature_along_an_sz_alias() -> None:
    """gram's factor route on two one-row levels: the exact alias stays a null.

    The augmented factor stacked ``sqrt`` of every positive eigenvalue of the
    whole penalty, rounding included.  sz's natural coordinates leave its
    unpenalized polynomial rows exactly zero, and one ``eigh`` of the whole
    matrix returned them at up to ``+5e-12`` against ``||S||_2 = 4e4``: that
    rounding curvature identified the thin levels' exact data-null alias, so
    gram's rank was one above the structured solver's and the dense
    pseudo-determinant's (round-2 review, P1; REML's smoothing parameters
    ended up to 800x off, and auto picks gram for small sz models).
    ``penalty_factor`` now keeps each block's eigenpairs above its
    eigensolver resolution.  Demonstration: 6544d2bc's gram publishes rank
    193 here against 192.
    """
    frame, y, weight = _signed_aliased_frame("one_row", response="fisher")
    models = {
        solve: _fit(
            _model("gaussian", solve, lam=128.0, numerics=("x1", "x10"), main_lam=1e4),
            frame,
            y,
            weight,
        )
        for solve in ("gram", "structured")
    }
    gram = models["gram"]
    assert gram._reml_profile["direct_backend"] == "gram"
    assert int(gram.result.reml_hessian_rank) == int(models["structured"].result.reml_hessian_rank)
    reference, bound = _dense_log_pdet(gram, y, weight, rows="fisher")
    assert abs(float(gram.result.log_det_H) - reference) <= bound


# ------------------------------------------------------------------ leverage
@pytest.mark.parametrize(("lam", "scale"), [(None, 1.0), (1e-7, 1e4)], ids=["reml", "tiny_lambda"])
def test_sz_leverage_of_a_wide_term_sums_to_edf(lam, scale) -> None:
    """``sum_i h_i = edf`` with ``h_i = w_i a_i' H_aug^+ a_i`` (design §3.10, A; finding 3).

    A K60 k5 sz term has 295 coefficients, above the 256-coefficient block
    limit: leverage and Cook's distance raised ``Refusing to materialize a
    295 x 295 inverse block`` on the stage-3 tree.  The row forms never form
    a block.  Each ``h_i`` sums nonnegative squares of whitened innovations
    up the row's path and of the border, each at most 1 and rounding at
    ``gamma_{4p}``; the edf's identity route sums ``p`` such terms, so ``|sum
    h - edf| <= (n + 2p) gamma_{4p}``, at a REML lambda and at lambda 1e-7
    beside weights 1e4.  Mutation: no ``row_quadratic_forms`` on the sz
    factor.
    """
    frame, eta, rng, _ = _frame(K=60)
    y = _response("gaussian", eta, rng)
    weight = np.full(len(frame), scale)
    model = _fit(_model("gaussian", "auto", lam=lam, k=5), frame, y, weight)
    assert model._reml_profile["direct_backend"] == "structured"
    assert model._dm.p > 256
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = model.metrics(frame, y, sample_weight=weight)
        leverage = np.asarray(metrics.leverage)
        cooks = np.asarray(metrics.cooks_distance)
    p = model._dm.p + 1
    assert np.all(np.isfinite(cooks))
    assert abs(float(np.sum(leverage)) - float(model.result.effective_df)) <= (
        len(frame) + 2 * p
    ) * _gamma(4 * p)


# ------------------------------------------------------------ standard errors
@pytest.mark.parametrize(
    "numerics", [("x1", "x10"), ("x1", "x10", "x1dup")], ids=["signed_rows", "duplicated_column"]
)
def test_sz_standard_errors_are_nan_only_on_exact_aliases(numerics) -> None:
    """The NaN set is the design's structural set and the dense solver's (finding 2).

    Gaussian with a log link on the block-factor note's fixture: its
    structured standard errors were NaN on 183 of 192 estimable coefficients
    (the range-space estimability solve of the retired sum-to-zero rule, on
    ill-conditioned level blocks).  A duplicated column is an exact alias:
    NaN on exactly its two columns.  Demonstration: the stage-4 builder's
    tree reports 183 and 185 NaN here.
    """
    frame, y = _block_factor_frame(0)
    frame["x1dup"] = frame["x1"].to_numpy().copy()
    weight = np.ones(len(frame))
    models = {
        solve: _fit(_model("gaussian_log", solve, numerics=numerics), frame, y, weight)
        for solve in ("auto", "gram")
    }
    assert models["auto"]._reml_profile["direct_backend"] == "structured"
    assert models["gram"]._reml_profile["direct_backend"] == "gram"
    mask = _nan_mask(models["auto"], frame, y, weight)
    np.testing.assert_array_equal(mask, _nan_mask(models["gram"], frame, y, weight))
    np.testing.assert_array_equal(mask, _structurally_non_estimable(models["auto"], weight))


def test_sz_random_effect_levels_are_non_estimable_as_on_gram() -> None:
    """A random effect's levels sum to the intercept: NaN on every one, as the dense solver.

    The ridge penalty identifies that direction, so it is no alias of the
    fit's Hessian, but estimability reads the data alone (the package's rule,
    and v0.35.0's on the models saved with this shape).  Mutation:
    estimability read from the fit's own (penalized) aliases, which reported
    the twelve levels' standard errors finite.
    """
    frame, eta, rng, _ = _frame()
    frame["h"] = np.array([f"h{v:02d}" for v in rng.integers(0, 12, len(frame))], dtype=object)
    y = _response("gaussian", eta, rng)
    weight = np.ones(len(frame))
    models = {
        solve: _fit(_model("gaussian", solve, random=("h",)), frame, y, weight)
        for solve in ("auto", "gram")
    }
    assert models["auto"]._reml_profile["direct_backend"] == "structured"
    mask = _nan_mask(models["auto"], frame, y, weight)
    h = next(group for group in models["auto"]._groups if group.name == "h")
    assert np.all(mask[h.sl]) and h.end - h.start == 12
    np.testing.assert_array_equal(mask, _nan_mask(models["gram"], frame, y, weight))
    np.testing.assert_array_equal(mask, _structurally_non_estimable(models["auto"], weight))


# ------------------------------------------------------ weak identification
def test_a_tiny_weight_level_is_named_not_the_whole_model() -> None:
    """One level's rows weighted 1e-15: its deviation is flat to within float64 (finding 4).

    The sum-to-zero constraint spreads its free deviation over every level
    and the main effect, so every coordinate of the truncated direction was
    flagged, noise included (71 coefficients under REML, among them
    ``cat``).  The warning names the level's own coordinates and the
    columns that move with it, and never an unrelated column; the fit still
    converges (the certificate leaves every moved column ungated), and the
    standard errors follow the data rule as the dense solver's do (the
    level's weighted columns sit below ``sqrt(eps)`` of the data's scale).
    Mutations: the disclosure reading every column the direction moves; the
    direction's entries kept within its Davis-Kahan resolution; the levels'
    entries read without removing the shift the constraint gives them all.
    """
    frame, eta, rng, g = _frame()
    y = _response("gaussian", eta, rng)
    weight = np.where(g == 4, 1e-15, 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _model("gaussian", "auto", lam=None).fit_reml(frame, y, sample_weight=weight)
    assert model._reml_profile["direct_backend"] == "structured"
    assert bool(model._reml_result.converged)
    flagged = set(model._reml_profile["reml_weakly_identified"])
    groups = {group.name: group for group in model._groups}
    sz, main = groups["x:g:sz"], groups["x"]
    position = list(model._dm.group_matrices[model._groups.index(sz)].levels).index("g004")
    level = sz.start + position * 6 + np.arange(6)
    allowed = set(level) | set(range(main.start, main.end))
    assert flagged
    assert flagged <= allowed
    assert flagged & set(level)
    assert not flagged & set(range(groups["cat"].start, groups["cat"].end))
    names = [str(item.message) for item in caught if "noise level" in str(item.message)]
    assert len(names) == 1 and "x:g:sz[" in names[0]
    gram = _fit(_model("gaussian", "gram", lam=None), frame, y, weight)
    np.testing.assert_array_equal(
        _nan_mask(model, frame, y, weight), _nan_mask(gram, frame, y, weight)
    )
