"""Analytic fourth derivatives: the second curvature direction of every LSS family.

Each family's ``predictor_curvature_second_directional_derivative`` is the
curvature's derivative along ``a`` then ``b``, from the row log-likelihood's
fourth derivatives in the predictors.  It is held here to the Richardson
difference of the family's analytic third derivative (its first curvature
direction) along ``b``, at that difference's resolution; the symbolic oracle
(``test_family_derivatives_symbolic``) holds it to ``sympy.diff`` at the edge
points.  The LAML Hessian takes it in place of differencing the curvature
twice, and falls back to differences wherever a family refuses.
"""

from __future__ import annotations

import numpy as np
import pytest

from superglm.distributional.families import (
    GammaLS,
    GaussianLS,
    GeneralizedGammaLSS,
    GeneralizedParetoLSS,
    LogNormalLS,
    NegativeBinomialLS,
    TweedieLSS,
    TwoPieceLogNormalLSS,
    TwoPieceNormalLSS,
)
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    PredictorCurvatureSecondDirectionalFamily,
)
from superglm.distributional.smoothing.endpoint_direction import (
    _curvature_packed,
    analytic_curvature_second_direction,
    finite_difference_curvature_second_direction,
)
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.links import IdentityLink, LogLink

_EPS = np.finfo(np.float64).eps


def _plan(family, y, weights=None, semantics="prior"):
    resolved = resolve_likelihood_weights(
        weights, n_observations=len(y), contract=WeightContract(semantics)
    )
    return family.bind_likelihood(y, resolved, COMPLETE_OBSERVATION)


def _links(family):
    return tuple(spec.default_link for spec in family.parameters)


def _rows(name: str, n_rows: int, seed: int):
    """``(family, y, eta)`` for one family on representative rows."""
    rng = np.random.default_rng(seed)
    severity = np.exp(rng.normal(7.0, 1.2, n_rows))
    if name == "gaussian":
        y = np.log(severity)
        return (
            GaussianLS(),
            y,
            np.column_stack((y + rng.normal(0, 0.5, n_rows), rng.normal(-0.3, 0.5, n_rows))),
        )
    if name == "gamma":
        eta = np.column_stack(
            (np.log(severity) + rng.normal(0, 0.5, n_rows), rng.normal(-0.3, 0.6, n_rows))
        )
        return GammaLS(), severity, eta
    if name in ("lognormal_mean", "lognormal_location"):
        family = LogNormalLS(parametrisation=name.removeprefix("lognormal_"))
        first = np.log(severity) + rng.normal(0, 0.5, n_rows)
        if name == "lognormal_mean":
            first = first + 0.5
        return family, severity, np.column_stack((first, rng.normal(-0.3, 0.6, n_rows)))
    if name == "nb":
        mean = np.exp(rng.normal(-1.0, 1.0, n_rows))
        theta = np.exp(rng.normal(0.5, 1.5, n_rows))
        y = rng.negative_binomial(theta, theta / (theta + mean)).astype(float)
        return NegativeBinomialLS(), y, np.column_stack((np.log(mean), np.log(theta)))
    if name == "gp":
        y = np.exp(rng.normal(7.0, 2.0, n_rows))
        eta = np.column_stack((rng.normal(7.0, 1.0, n_rows), rng.normal(0.0, 2.0, n_rows)))
        return GeneralizedParetoLSS(), y, eta
    if name.startswith("gengamma"):
        family = GeneralizedGammaLSS(parametrisation=name.removeprefix("gengamma_"))
        eta = np.column_stack(
            (
                np.log(severity) + rng.normal(0, 0.5, n_rows),
                np.minimum(rng.normal(0.0, 0.8, n_rows), 1.5),
                rng.uniform(-0.15, 1.2, n_rows),
            )
        )
        return family, severity, eta
    if name.startswith("twopiece"):
        family = {
            "twopiece_normal": TwoPieceNormalLSS(),
            "twopiece_lognormal_location": TwoPieceLogNormalLSS(parametrisation="location"),
            "twopiece_lognormal_mean": TwoPieceLogNormalLSS(),
        }[name]
        y = np.log(severity) if name == "twopiece_normal" else severity
        first = np.log(severity) + rng.normal(0, 0.5, n_rows)
        eta = np.column_stack((first, rng.normal(0.0, 0.6, n_rows), rng.normal(0.0, 1.0, n_rows)))
        if name == "twopiece_lognormal_mean":
            eta[:, 0] = first + 0.5
        return family, y, eta
    if name == "tweedie":
        family = TweedieLSS()
        mean = np.exp(rng.normal(0.0, 1.0, n_rows))
        y = np.where(rng.uniform(size=n_rows) < 0.6, 0.0, rng.gamma(2.0, mean))
        eta = np.column_stack(
            (np.log(mean), rng.normal(0.0, 0.5, n_rows), rng.normal(0.0, 1.0, n_rows))
        )
        return family, y, eta
    raise AssertionError(name)


_NAMES = (
    "gamma",
    "gaussian",
    "gengamma_location",
    "gengamma_mean",
    "gp",
    "lognormal_location",
    "lognormal_mean",
    "nb",
    "tweedie",
    "twopiece_lognormal_location",
    "twopiece_lognormal_mean",
    "twopiece_normal",
)


def _join_free(family, y, eta, direction, links, step):
    """Two-piece rows whose residual keeps its sign along the whole stencil."""
    from superglm.distributional.kernels.two_piece import location_of_mean

    if not isinstance(family, (TwoPieceNormalLSS, TwoPieceLogNormalLSS)):
        return np.ones(len(y), dtype=bool)
    variate = y if isinstance(family, TwoPieceNormalLSS) else np.log(y)
    unit = direction / np.linalg.norm(direction, axis=1, keepdims=True)

    def left(values):
        theta = np.column_stack([links[k].inverse(values[:, k]) for k in range(3)])
        location = theta[:, 0]
        if getattr(family, "parametrisation", "location") == "mean":
            location = location_of_mean(*theta.T)
        return (variate - location) / theta[:, 1] < 0.0

    centre = left(eta)
    same = np.ones(len(y), dtype=bool)
    for offset in (step, -step, step / 2, -step / 2, 0.0):
        same &= left(eta + offset * unit) == centre
    return same


@pytest.mark.parametrize("name", _NAMES)
def test_second_direction_agrees_with_the_difference_of_the_third(name: str) -> None:
    """``d/db`` of the analytic first direction along ``a``, Richardson-extrapolated.

    The central differences at ``h`` and ``h/2`` of the family's analytic
    third derivative combine to an ``O(h^4)`` estimate whose distance from the
    finer one is its truncation estimate; the analytic fourth derivative must
    sit within four times that plus the differences' round-off,
    ``64 eps S / h`` with ``S`` the row's largest third derivative or
    curvature: the chain rule forms a small third derivative from terms of the
    curvature's size, so its rounding follows them (a generalized Pareto row at
    ``z = 5e-6`` has third derivatives of ``1e-5`` built from terms of order one).
    """
    family, y, eta = _rows(name, 1500, 21)
    assert isinstance(family, PredictorCurvatureSecondDirectionalFamily)
    links = _links(family)
    plan = _plan(family, y)
    rng = np.random.default_rng(5)
    a = rng.normal(size=eta.shape)
    b = rng.normal(size=eta.shape)
    step = 1.0e-2
    smooth = _join_free(family, y, eta, b, links, step) & _join_free(family, y, eta, a, links, 0.0)
    y, eta, a, b = y[smooth], eta[smooth], a[smooth], b[smooth]
    plan = _plan(family, y)
    analytic = family.predictor_curvature_second_directional_derivative(y, eta, a, b, links, plan)
    norms = np.linalg.norm(b, axis=1, keepdims=True)
    unit = b / norms

    def third(offset):
        return family.predictor_curvature_directional_derivative(
            y, eta + offset * unit, a, links, plan
        )

    plus, minus = third(step), third(-step)
    plus_half, minus_half = third(step / 2), third(-step / 2)
    coarse = (plus - minus) / (2.0 * step)
    fine = (plus_half - minus_half) / step
    extrapolated = (4.0 * fine - coarse) / 3.0
    curvature = _curvature_packed(family, y, eta, links, plan)
    row_scale = np.maximum(
        np.max(np.abs(np.stack([plus, minus, plus_half, minus_half])), axis=(0, 2)),
        np.max(np.abs(curvature), axis=1),
    )
    resolution = np.abs(extrapolated - fine) + 64.0 * _EPS * row_scale[:, None] / step
    numeric = extrapolated * norms
    gap = np.abs(analytic - numeric)
    scale = np.max(np.abs(analytic), axis=1, keepdims=True) + 1.0e-300
    assert np.all(gap <= 4.0 * resolution * norms + 1.0e-9 * scale), name
    assert np.median(gap / scale) < 1.0e-8


@pytest.mark.parametrize("name", _NAMES)
def test_stacked_pairs_equal_the_single_pair_calls(name: str) -> None:
    """The LAML Hessian's stacked call gives every pair's values bit for bit."""
    family, y, eta = _rows(name, 60, 8)
    links = _links(family)
    plan = _plan(family, y)
    rng = np.random.default_rng(9)
    a = rng.normal(size=(3, *eta.shape))
    b = rng.normal(size=(3, *eta.shape))
    if name.startswith("twopiece"):
        try:
            family.predictor_curvature_second_directional_derivative(y, eta, a, b, links, plan)
        except UnsupportedLikelihoodContractError:  # pragma: no cover - a row on its join
            pytest.skip("a fixture row sits on the join")
    stacked = family.predictor_curvature_second_directional_derivative(y, eta, a, b, links, plan)
    for index in range(3):
        single = family.predictor_curvature_second_directional_derivative(
            y, eta, a[index], b[index], links, plan
        )
        np.testing.assert_array_equal(stacked[index], single)
    swapped = family.predictor_curvature_second_directional_derivative(y, eta, b, a, links, plan)
    tolerance = 64.0 * _EPS * (np.max(np.abs(stacked), axis=-1, keepdims=True) + 1e-300)
    assert np.all(np.abs(swapped - stacked) <= tolerance)


def test_a_link_without_a_fourth_derivative_is_differenced() -> None:
    """A link with three inverse derivatives but no fourth sends the caller to differences."""

    class ThreeDerivativeLog(LogLink):
        deriv4_inverse = None

    family = GeneralizedParetoLSS()
    y = np.array([3.0, 40.0])
    eta = np.array([[1.0, 0.2], [2.0, -0.3]])
    plan = _plan(family, y)
    links = (ThreeDerivativeLog(), _links(family)[1])
    a = np.ones_like(eta)
    with pytest.raises(UnsupportedLikelihoodContractError, match="deriv4_inverse"):
        family.predictor_curvature_second_directional_derivative(y, eta, a, a, links, plan)
    assert analytic_curvature_second_direction(family, y, eta, a, a, links, plan) is None
    differenced = finite_difference_curvature_second_direction(family, y, eta, a, a, links, plan)
    assert np.all(np.isfinite(differenced.values))


def test_two_piece_second_direction_refuses_a_path_through_its_join() -> None:
    """At ``t == mu`` the second direction is refused whenever either direction moves ``mu``."""
    y = np.array([1.0, 1.3, 0.7])
    family = TwoPieceNormalLSS()
    links = _links(family)
    plan = _plan(family, y)
    eta = np.column_stack(
        (np.ones(3), links[1].link(np.full(3, 0.8)), links[2].link(np.full(3, 0.3)))
    )
    moving = np.tile([0.3, -0.5, 0.8], (3, 1))
    fixed = np.tile([0.0, 1.0, -0.4], (3, 1))
    for a, b in ((moving, fixed), (fixed, moving), (moving, moving)):
        with pytest.raises(UnsupportedLikelihoodContractError, match="join at row 0"):
            family.predictor_curvature_second_directional_derivative(y, eta, a, b, links, plan)
        assert analytic_curvature_second_direction(family, y, eta, a, b, links, plan) is None
    family.predictor_curvature_second_directional_derivative(y, eta, fixed, fixed, links, plan)


def test_negative_binomial_certifies_only_its_bbbb_channel() -> None:
    """``l_bbbb`` enters the ``bb`` channel times ``a_b b_b``; the rest are closed forms."""
    family, y, eta = _rows("nb", 400, 13)
    links = _links(family)
    plan = _plan(family, y)
    rng = np.random.default_rng(2)
    a = rng.normal(size=eta.shape)
    b = rng.normal(size=eta.shape)
    values, certificate = family.certified_predictor_curvature_second_directional_derivative(
        y, eta, a, b, links, plan
    )
    np.testing.assert_array_equal(
        values, family.predictor_curvature_second_directional_derivative(y, eta, a, b, links, plan)
    )
    assert np.all(certificate[:, :2] == 0.0)
    assert np.all(certificate[:, 2] >= 0.0)
    assert np.all(certificate[:, 2] <= 1e-6 * (np.abs(values[:, 2]) + np.abs(a[:, 1] * b[:, 1])))
    with pytest.raises(UnsupportedLikelihoodContractError):
        family.predictor_curvature_second_directional_derivative(
            y, eta, a, b, (LogLink(), IdentityLink()), plan
        )


def test_the_tweedie_kernel_carries_the_generated_digest() -> None:
    """The compiled Tweedie kernel's digest names the generated module it was built against.

    Numba's on-disk cache keys a caller on its own file, so the generator
    stamps a digest of its output into ``_tweedie_numba.py``; a stale stamp
    would leave cached callers running superseded generated code.
    """
    import hashlib
    from pathlib import Path

    import superglm.distributional.kernels._fourth_generated as generated
    import superglm.distributional.kernels._tweedie_numba as kernel

    text = Path(generated.__file__).read_text(encoding="utf-8")
    assert kernel._FOURTH_GENERATED_DIGEST == hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.slow
def test_the_generated_module_is_current() -> None:
    """``tools/generate_lss_fourth_derivatives.py --check`` regenerates the committed module."""
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("sympy")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "tools" / "generate_lss_fourth_derivatives.py"), "--check"],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _stacked_axis_pairs(k: int):
    pairs = [(q, r) for q in range(k) for r in range(q, k)]
    first = np.zeros((len(pairs), 1, k))
    second = np.zeros_like(first)
    for index, (q, r) in enumerate(pairs):
        first[index, 0, q] = 1.0
        second[index, 0, r] = 1.0
    return pairs, first, second


def test_two_piece_certifies_rows_within_reach_of_its_join() -> None:
    """A row the derivative reach carries across its join carries the curvature's jump.

    The analytic Hessian is exact on each piece; as a local model across the
    join it is off by about ``|jump of C| / r^2``, which is what second
    differences over ``r`` recorded there.  In the location form only pairs
    that move ``mu`` reach the join; a far row and the other pairs carry none,
    and the values are the uncertified ones.
    """
    from superglm.distributional.kernels.two_piece import JOIN_REACH, location_join_jumps

    family = TwoPieceNormalLSS()
    links = _links(family)
    y = np.array([1.0 + 0.5 * JOIN_REACH, 1.5, 1.0 - 0.2 * JOIN_REACH])
    plan = _plan(family, y)
    sigma, skew = 0.8, 0.3
    eta = np.column_stack(
        (np.ones(3), links[1].link(np.full(3, sigma)), links[2].link(np.full(3, skew)))
    )
    pairs, first, second = _stacked_axis_pairs(3)
    values, certificate = family.certified_predictor_curvature_second_directional_derivative(
        y, eta, first, second, links, plan
    )
    np.testing.assert_array_equal(
        values,
        family.predictor_curvature_second_directional_derivative(
            y, eta, first, second, links, plan
        ),
    )
    moves_mu = np.array([0 in pair for pair in pairs])
    assert np.all(certificate[:, 1] == 0.0)
    assert np.all(certificate[~moves_mu] == 0.0)
    hessian_jump, _ = location_join_jumps(np.full(3, sigma), np.full(3, skew), np.ones(3))
    floor = hessian_jump[0, 0] / JOIN_REACH**2
    assert np.all(certificate[moves_mu][:, [0, 2], 0] >= floor * (1.0 - 1e-12))
    resolved = analytic_curvature_second_direction(family, y, eta, first, second, links, plan)
    assert resolved is not None and resolved[1] is not None
    np.testing.assert_array_equal(resolved[1], certificate)


def test_two_piece_mean_form_certifies_a_row_near_its_join_on_every_pair() -> None:
    """In the mean form every direction moves ``mu = log m - log K``, so a near row is certified on every pair, even at zero skew, where only the third derivative jumps."""
    from superglm.distributional.kernels.two_piece import JOIN_REACH, log_mean_loading

    family = TwoPieceLogNormalLSS()
    links = _links(family)
    y = np.array([2.0, 2.0])
    sigma, skew = np.array([0.6, 0.6]), np.array([0.0, 0.0])
    logk = log_mean_loading(sigma, skew)[0]
    offsets = np.array([0.3 * JOIN_REACH, 0.5])
    mean = np.exp(np.log(y) - offsets + logk)
    eta = np.column_stack((np.log(mean), links[1].link(sigma), links[2].link(skew)))
    plan = _plan(family, y)
    _, first, second = _stacked_axis_pairs(3)
    _, certificate = family.certified_predictor_curvature_second_directional_derivative(
        y, eta, first, second, links, plan
    )
    assert np.all(np.max(certificate[:, 0, :], axis=-1) > 0.0)
    assert np.all(certificate[:, 1, :] == 0.0)


def test_two_piece_join_reach_covers_a_mixed_pair_along_its_combined_direction() -> None:
    """A mixed pair's second differences step along ``(a + b) / |a + b|``.

    So a row that direction carries across the join is certified on that
    pair even when neither axis alone reaches it: here both axes move the
    location at unit rate and the row sits ``1.2`` reaches from the join,
    beyond either axis and within ``sqrt 2``.  The diagonal pairs stay
    uncertified.
    """
    from superglm.distributional.kernels.two_piece import (
        JOIN_REACH,
        join_certificate,
        location_join_jumps,
    )

    jumps = location_join_jumps(np.array([0.8]), np.array([0.3]), np.ones(1))
    ones, zeros = np.ones((1, 3)), np.zeros((1, 3))
    pairs, first, second = _stacked_axis_pairs(3)
    certificate = join_certificate(
        np.array([1.2 * JOIN_REACH]),
        np.array([[1.0, 1.0, 0.0]]),
        jumps,
        (ones, zeros, zeros),
        first,
        second,
    )
    floor = jumps[0][0, 0] / JOIN_REACH**2
    assert certificate[pairs.index((0, 1)), 0, 0] >= floor * (1.0 - 1e-12)
    assert np.all(certificate[pairs.index((0, 0))] == 0.0)
    assert np.all(certificate[pairs.index((1, 1))] == 0.0)


def test_two_piece_join_certificate_follows_the_engine_derivative_step() -> None:
    """The near-join reach is the derivative step the engine differences with.

    A row ``5e-3`` from the join is outside the default step's reach and
    inside a ``1e-2`` step's, where its certificate is at least the Hessian
    jump over that step squared; the engine passes its step through
    :func:`analytic_curvature_second_direction`.
    """
    from superglm.distributional.kernels.two_piece import JOIN_REACH, location_join_jumps
    from superglm.distributional.smoothing.endpoint_direction import DEFAULT_STEP

    assert JOIN_REACH == DEFAULT_STEP
    family = TwoPieceNormalLSS()
    links = _links(family)
    y = np.array([1.0 + 5e-3])
    plan = _plan(family, y)
    sigma, skew = 0.8, 0.3
    eta = np.column_stack(
        (np.ones(1), links[1].link(np.full(1, sigma)), links[2].link(np.full(1, skew)))
    )
    pairs, first, second = _stacked_axis_pairs(3)
    _, default = family.certified_predictor_curvature_second_directional_derivative(
        y, eta, first, second, links, plan
    )
    assert np.all(default == 0.0)
    step = 1e-2
    _, wide = family.certified_predictor_curvature_second_directional_derivative(
        y, eta, first, second, links, plan, step=step
    )
    hessian_jump, _ = location_join_jumps(np.full(1, sigma), np.full(1, skew), np.ones(1))
    assert wide[pairs.index((0, 0)), 0, 0] >= hessian_jump[0, 0] / step**2 * (1.0 - 1e-12)
    resolved = analytic_curvature_second_direction(
        family, y, eta, first, second, links, plan, step=step
    )
    assert resolved is not None and resolved[1] is not None
    np.testing.assert_array_equal(resolved[1], wide)
