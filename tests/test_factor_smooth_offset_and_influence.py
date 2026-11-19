"""Factor smooths beside a large-offset column, and their influence diagonal.

The stage-2 verifier's findings on the ``fs`` leaf route, as complete fits:

- a numeric column near ``1e8`` (an epoch time, an ID) must not move an ``fs``
  REML fit, which the same model with the column translated back exactly fits
  (the centred operators are formed on the ``c0``-shifted rows, design §3.2);
- standard errors are NaN exactly on the coefficients the design cannot
  estimate, whatever a column's offset;
- leverage is the influence diagonal with the intercept, ``sum h = edf``, at
  any lambda and any width (design §3.10, definition A);
- a signed (observed-curvature) ``fs`` fit publishes the observed Hessian's
  ``log|H|``;
- one dispatch decision per model class for the two classes design §3.13's
  size rule will change (T1), invariant under the data.

Every test fails under the mutation its docstring names.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, FactorSmooth, LambdaPolicy, Numeric, RandomEffect, SuperGLM
from superglm.solvers.mode_score import linear_predictor

EPS = float(np.finfo(np.float64).eps)
_U = EPS / 2.0
_COMPONENTS = ("wiggle", "null_0", "null_1")


def _gamma(count: float) -> float:
    return count * _U / (1.0 - count * _U)


def _frame(K: int = 30, n: int = 4000, seed: int = 3):
    """The block-factor note's ``fs`` base: gamma-popular levels, a smooth per level."""
    rng = np.random.default_rng(seed)
    popularity = rng.gamma(2.0, size=K)
    g = rng.choice(K, size=n, p=popularity / popularity.sum())
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
        + 0.3 * np.sin(2 * np.pi * x) * (1.0 + rng.normal(0, 0.3, K)[g])
    )
    return frame, eta, rng, g


def _model(family, link=None, *, k=6, lam=None, discrete=False, numerics=("x1",)) -> SuperGLM:
    policy = None if lam is None else {c: LambdaPolicy.fixed(lam) for c in _COMPONENTS}
    features = {name: Numeric() for name in numerics}
    features["cat"] = Categorical()
    kwargs = {} if link is None else {"link": link}
    return SuperGLM(
        family=family,
        features=features,
        interactions=[FactorSmooth("x", group="g", basis="fs", k=k, lambda_policy=policy)],
        selection_penalty=0,
        discrete=discrete,
        **kwargs,
    )


def _fit(model, frame, y, weight):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit_reml(frame, y, sample_weight=weight)


def _nan_mask(model, frame, y, weight) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        se = model.metrics(frame, y, sample_weight=weight).coefficient_se
    return np.concatenate(
        [
            np.isnan(np.atleast_1d(np.asarray(se[group.name], dtype=float)))
            for group in model._groups
        ]
    )


def _structurally_non_estimable(model, weight) -> np.ndarray:
    """Slopes a null vector of ``[1, X]`` over the positive-weight rows touches.

    Column-equilibrated SVD: a null singular value is below ``max(n, p) eps``
    of the largest (``dgesdd``'s backward error); on these designs the next
    one is about 1e-3, a gap of twelve decades.
    """
    X = np.asarray(model._dm.toarray(), dtype=np.float64)[weight > 0]
    A = np.column_stack([np.ones(len(X)), X])
    A = A / np.maximum(np.linalg.norm(A, axis=0), np.finfo(float).tiny)
    _, singular, rows = np.linalg.svd(A, full_matrices=False)
    null = rows[singular <= max(A.shape) * EPS * singular[0]]
    return np.any(np.abs(null[:, 1:]) > np.sqrt(EPS), axis=0)


# ------------------------------------------------------------------ offsets
@pytest.mark.slow
@pytest.mark.parametrize(("family", "link"), [("gaussian", "log"), ("poisson", None)])
def test_fs_reml_does_not_move_with_a_large_column_offset(family, link) -> None:
    """A column at ``1e8`` against the same column translated back exactly.

    The two designs are equal up to a multiple of the intercept column, so the
    REML problems are identical.  Both fits must converge, reach the same
    objective within the stop rule's resolution ``2 reml_tol (1 + |V|)``
    (``reml.convergence``), keep the same rank and report NaN standard errors
    on the same coefficients.  On the stage-2 route the raw fit ended
    ``line_search_failed`` with a smoothing parameter 27% away and every
    standard error NaN (190 against 64).  Mutation: the centred data
    operator, the mode score's centred scale or the REML weight-derivative
    operators formed from the raw moments.
    """
    frame, eta, rng, _ = _frame()
    n = len(frame)
    big = 1e8 + rng.normal(size=n)
    eta = eta + 0.05 * (big - 1e8)
    if family == "gaussian":
        y = np.exp(0.5 * eta) + rng.normal(0, 0.6, n)
    else:
        y = rng.poisson(np.exp(np.clip(eta, -5, 5))).astype(float)
    weight = np.ones(n)
    fits = {}
    for name, column in (("raw", big), ("translated", big - 1e8)):
        data = frame.assign(xbig=column)
        model = _fit(_model(family, link, numerics=("x1", "xbig")), data, y, weight)
        fits[name] = (model, _nan_mask(model, data, y, weight))
    raw, translated = fits["raw"][0], fits["translated"][0]
    for model in (raw, translated):
        assert model._reml_result.converged, model._reml_result.termination_reason
    objective = float(translated._reml_result.objective)
    resolution = 2.0 * 1e-9 * (1.0 + abs(objective))
    assert abs(float(raw._reml_result.objective) - objective) <= resolution
    assert raw.result.reml_hessian_rank == translated.result.reml_hessian_rank
    np.testing.assert_array_equal(fits["raw"][1], fits["translated"][1])


@pytest.mark.parametrize("discrete", [False, True])
def test_fs_standard_errors_are_nan_only_where_the_design_cannot_estimate(discrete) -> None:
    """Poisson ``fs`` with a column at ``1e6``, lambda 1e-7 beside weights 1e4.

    Estimability for standard errors was read off the raw moments centred by
    subtraction, which cancel to noise for the offset column: every one of the
    190 standard errors came back NaN.  The NaN set must be the design's
    structural non-estimable set (each level's constant, aliased with the
    intercept), the same with the column translated back.  Mutation: the
    retained centred data operator on the raw moments.
    """
    frame, eta, rng, _ = _frame()
    n = len(frame)
    column = 1e6 + rng.normal(size=n)
    eta = eta + 0.1 * (column - 1e6)
    y = rng.poisson(np.exp(np.clip(eta, -5, 5))).astype(float)
    weight = np.full(n, 1e4)
    masks = []
    for values in (column, column - 1e6):
        data = frame.assign(xoff=values)
        model = _fit(
            _model("poisson", lam=1e-7, discrete=discrete, numerics=("x1", "xoff")),
            data,
            y,
            weight,
        )
        assert model._reml_profile["direct_backend"] == "structured"
        mask = _nan_mask(model, data, y, weight)
        np.testing.assert_array_equal(mask, _structurally_non_estimable(model, weight))
        masks.append(mask)
    np.testing.assert_array_equal(masks[0], masks[1])
    assert 0 < int(masks[0].sum()) < len(masks[0])


# ---------------------------------------------------------------- leverage
@pytest.mark.parametrize(
    ("fixture", "lam", "scale"), [("clean", 1e-7, 1e6), ("level_const", 1e-9, 1e4)]
)
def test_fs_leverage_sums_to_edf_at_tiny_lambda(fixture, lam, scale) -> None:
    """``sum_i h_i = edf`` with ``h_i = w_i a_i' H_aug^+ a_i`` (design §3.10, A; T2).

    Each ``h_i`` sums nonnegative squares of a row of the level's orthogonal
    factor and of the border, each at most 1 and rounding at ``gamma_{4p}``;
    the edf's identity route sums ``p`` such terms.  So ``|sum h - edf| <= (n
    + 2p) gamma_{4p}``.  The dense slope inverse on centred rows gave 0.065
    and 0.23 here (rows off by up to 2.4e-3 against the exact inverse).
    Mutation: leverage from the materialized slope block (no
    ``row_quadratic_forms`` on the fs factor).
    """
    frame, eta, rng, g = _frame()
    n = len(frame)
    numerics = ["x1"]
    if fixture == "level_const":
        frame["attr"] = rng.normal(size=30)[g]
        numerics.append("attr")
    y = eta + rng.normal(0, 0.5, n)
    weight = np.full(n, scale)
    model = _fit(_model("gaussian", lam=lam, numerics=tuple(numerics)), frame, y, weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        leverage = np.asarray(model.metrics(frame, y, sample_weight=weight).leverage)
    p = model._dm.p + 1
    edf = float(model.result.effective_df)
    assert abs(float(np.sum(leverage)) - edf) <= (n + 2 * p) * _gamma(4 * p)


def test_fs_leverage_of_a_term_wider_than_the_inverse_block_limit() -> None:
    """A K60 k5 term (300 coefficients, above the 256-coefficient block limit).

    Leverage and Cook's distance raised ``Refusing to materialize a 300 x 300
    inverse block`` on master, v0.35.0 and stage 2; the row forms never form
    a block.  Mutation: as the previous test.
    """
    frame, eta, rng, _ = _frame(K=60)
    y = eta + rng.normal(0, 0.5, len(frame))
    weight = np.ones(len(frame))
    model = _fit(_model("gaussian", k=5), frame, y, weight)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = model.metrics(frame, y, sample_weight=weight)
        leverage = np.asarray(metrics.leverage)
        cooks = np.asarray(metrics.cooks_distance)
    p = model._dm.p + 1
    assert model._dm.p > 300
    assert np.all(np.isfinite(cooks))
    assert abs(float(np.sum(leverage)) - float(model.result.effective_df)) <= (
        len(frame) + 2 * p
    ) * _gamma(4 * p)


# ------------------------------------------------------------ signed rows
def test_signed_fs_publishes_the_observed_hessian_log_determinant() -> None:
    """Gaussian/log is non-canonical: its REML Hessian is the observed one, with signed rows.

    The published ``log|H|`` against the dense observed Hessian at the
    published mode, ``A' W_obs A + S`` with ``W_obs = w mu (2 mu - y)``:
    within the factor's certified border bound plus the dense ``slogdet``'s
    own first-order error ``size kappa(H) gamma_size``.  Mutation: signed rows
    factored as Fisher rows with ``|w|`` (the stage-2 verifier's V2, which only
    the memo tests caught).
    """
    frame, eta, rng, _ = _frame()
    n = len(frame)
    y = np.exp(0.5 * eta) + rng.normal(0, 0.6, n)
    weight = np.ones(n)
    model = _fit(_model("gaussian", "log", lam=0.7), frame, y, weight)
    dm = model._dm
    mu = np.exp(linear_predictor(dm, model.result, None))
    observed = weight * mu * (2.0 * mu - y)
    assert np.any(observed < 0.0)
    A = np.column_stack([np.ones(n), np.asarray(dm.toarray(), dtype=np.float64)])
    from superglm.reml.penalty_algebra import build_penalty_matrix

    S = build_penalty_matrix(
        dm.group_matrices, model._groups, model._reml_lambdas, dm.p, model._reml_penalties
    )
    H = A.T @ (observed[:, None] * A)
    H[1:, 1:] += 0.5 * (S + S.T)
    H = 0.5 * (H + H.T)
    sign, logdet = np.linalg.slogdet(H)
    assert sign > 0
    size = H.shape[0]
    certificate = model._linear_system_state.augmented_factor.border_certificate
    tolerance = size * np.linalg.cond(H) * _gamma(size) + certificate.logdet_bound
    assert abs(float(model.result.log_det_H) - logdet) <= tolerance


# ------------------------------------------------------------- dispatch (T1)
def _t1_data(variant: str):
    rng = np.random.default_rng(1701)
    n = 1500
    frame = pd.DataFrame(
        {
            "x1": rng.uniform(size=n),
            "x2": rng.uniform(size=n),
            "z": rng.normal(size=n),
            "g1": np.array([f"a{c}" for c in rng.integers(0, 12, n)], dtype=object),
            "g2": np.array([f"b{c}" for c in rng.integers(0, 8, n)], dtype=object),
            "re": np.array([f"r{c:03d}" for c in rng.integers(0, 250, n)], dtype=object),
        }
    )
    y = np.sin(3 * frame["x1"].to_numpy()) + 0.3 * frame["z"].to_numpy() + rng.normal(0, 0.4, n)
    weight = np.full(n, 1e6 if variant == "w1e6" else 1.0)
    if variant == "offset":
        frame["z"] = 1e8 + frame["z"]
    numerics = ["z", "zdup"] if variant == "dup" else ["z"]
    if variant == "dup":
        frame["zdup"] = frame["z"].to_numpy().copy()
    return frame, y, weight, numerics


@pytest.mark.parametrize("variant", ["base", "w1e6", "offset", "lam1e-7", "dup"])
@pytest.mark.parametrize("shape", ["two_fs", "fs_beside_larger_re"])
def test_the_size_rule_classes_take_one_decision_whatever_the_data(shape, variant) -> None:
    """T1 pins for the two classes design §3.13's size rule is to change (deferred, see notes).

    ``fs`` beside a crossed random effect with more levels than the ``fs`` term
    has coefficients, and two FactorSmooths: today both take the dense
    solver by structure (the cost rule on ``K``, ``k`` and ``q``; the
    one-FactorSmooth limit).  The decision, read from the fit, must not move
    with weights x1e6, a 1e8 offset, lambda 1e-7 or a duplicated column.
    Mutation: a route that reads the data (for example a weight or lambda
    test in ``selection``).
    """
    frame, y, weight, numerics = _t1_data(variant)
    lam = 1e-7 if variant == "lam1e-7" else 0.9
    policy = {c: LambdaPolicy.fixed(lam) for c in _COMPONENTS}
    features = {name: Numeric() for name in numerics}
    interactions = [FactorSmooth("x1", group="g1", k=5, lambda_policy=policy)]
    if shape == "two_fs":
        interactions.append(FactorSmooth("x2", group="g2", k=5, lambda_policy=policy))
        expected = "at most one FactorSmooth"
    else:
        features["re"] = RandomEffect(lambda_policy=LambdaPolicy.fixed(lam))
        expected = "crossover"
    model = SuperGLM(
        family="gaussian",
        features=features,
        interactions=interactions,
        selection_penalty=0.0,
        direct_solve="auto",
    )
    _fit(model, frame, y, weight)
    assert model.result.direct_backend == "gram"
    assert expected in model.result.direct_fallback_reason
