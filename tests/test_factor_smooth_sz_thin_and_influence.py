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
from superglm.reml.penalty_algebra import build_penalty_matrix

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
        n_knots=6, lambda_policy=None if lam is None else LambdaPolicy.fixed(1.0)
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


def _assert_fits_as_gram(auto: SuperGLM, gram: SuperGLM, n: int) -> None:
    """One route to the same fit: no refusal, the dense solver's rank and penalized objective.

    The penalized objective of the tree's mode is at most the dense solver's
    within the objective's own rounding, ``n eps |objective|`` (Higham 2002,
    section 3.1), and the dense solver's at most the tree's likewise: an
    exhausted level's aliases are truncated alike, and the fitted rows are the
    same.  The unfixed tree declared every border column null (rank 174
    against 190) and stopped at 1.6 to 10 times the dense objective.
    """
    assert auto._reml_profile["direct_backend"] == "structured"
    assert bool(auto._reml_result.converged)
    assert int(auto.result.reml_hessian_rank) == int(gram.result.reml_hessian_rank)
    tree_objective, gram_objective = _penalized_objective(auto), _penalized_objective(gram)
    tolerance = n * EPS * abs(gram_objective)
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
    _assert_fits_as_gram(auto, gram, len(frame))
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
    _assert_fits_as_gram(auto, gram, int(np.count_nonzero(weight)))


def _block_factor_frame(one_row: int):
    """The block-factor note's sz fixture as the stage-3 verifier built it (K30, n 4000)."""
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
