"""Uniform prior-weight rescaling through each structured route (issue #433).

Multiplying every prior weight and every smoothing parameter by ``c`` multiplies
the penalized log-likelihood by ``c``: for any family the score
``A' w (y - mu) mu' / V - S theta`` and the Hessian ``A' W A + S`` scale by
``c`` together, so the mode, the linear predictor and the edf are invariant,
the deviance scales by ``c``, and so does the dispersion that keeps
``w / phi`` fixed.  The REML criterion of the rescaled problem read at
dispersion ``c phi`` is the original one plus a constant, so its gradient and
Hessian in ``log lambda`` are invariant too.  ``c = 2^k`` makes the rescaled
data exact, and every scale-free step (a Jacobi-scaled pivot, a rank cut on a
scaled form, a relative stop rule) decides the same at both scales.  An
absolute floor, an unscaled constant, or a square of a weight-sized float
(``sum_w ** 2`` leaves float64 past ``sum_w ~ 1.3e154`` and rounds to zero
below ``1.6e-162``) does not.

The routes are the three structured systems ``build_augmented_structured_factor``
dispatches: ``NestedStructuredSystem`` (a lone random effect, a chain of one,
and a nested chain), ``FactorSmoothLeafSystem`` (``fs``) and
``SumToZeroLeafSystem`` (``sz``).  Every smoothing parameter is held at ``c``
times the reference's, and two tests compare each route with its fit at
``c = 1``:

- the route's factor on the reference design: the PIRLS mode, and the REML
  outer gradient and W-corrected Hessian through the route's profiled factor
  (``derivative_cross_traces``, whose cross matrix squared ``sum_w`` before
  #425), read at inverse dispersion ``1 / c``;
- the complete ``fit_reml`` with ``direct_solve="structured"``: the backend,
  the route, the rank, the linear predictor, the edf and the deviance.

Scales.  ``2^-600`` and ``2^520`` lie beyond the retired factors' ``2^+-400``
(``test_sum_to_zero_scale_invariance.py`` and ``test_structured_factor_extreme.py``,
deleted with them in #425) and put ``sum_w`` (1200 rows) below the square's
underflow and above its overflow.  The nested factor's REML Hessian is in
range only between them, so its passing cases are ``2^-500`` and ``2^502``,
where ``sum_w = 1.6e154`` is past the overflow #425 removed.  Known
non-equivariance on master, each a strict xfail naming its cause (issue #433
follow-ups):

- the nested per-level cross traces multiply two weight-sized quantities
  (``_operator_pair``, ``_rho ** 4``): the Hessian is finite and wrong from
  about ``2^506`` and not finite from ``2^512`` or below ``2^-520``;
- ``reml_w_correction``'s second-order term divides by the Python float
  ``sum_w ** 2`` (``OverflowError`` or ``ZeroDivisionError``, every backend);
- ``fit_reml``'s bootstrap PIRLS runs at absolute lambdas even when every
  lambda is fixed, and at ``2^520`` the ``fs`` factor refuses it.

The ``sz`` main-effect spline is unpenalized: the terminal design rebuild
absorbs a fixed smoothing parameter into the SSP basis in absolute units
beside a weight-normalized Gram, which breaks the rescaling past about
``2^48`` on every backend (a follow-up outside the structured routes).

Where the tolerances come from.

- The mode.  To first order a fit's distance to the exact mode is
  ``H^-1 g`` at its own state, ``g`` the penalized score (``_gap``): the
  score is summed exactly (``math.fsum``) over row products rounded once,
  and the rounding of each score row and of ``eta`` is added through
  ``|H^-1|``, as ``test_nested_structured_fit._fixed_point_gap`` does.
  This bounds the solves' rounding and the stop rule's resolution together,
  whichever iteration stopped either fit.  The two fits' coefficients
  therefore differ by at most the sum of their gaps, and their linear
  predictors by ``|A|`` times it plus each one's evaluation rounding,
  ``(m + 1) eps |A| |theta|`` over ``m`` nonzeros per row.
- The factor.  A backward-stable factorization of the Jacobi-scaled ``H``,
  whose diagonal is one, perturbs it by at most ``eta = p gamma_{n+p+1}
  max_ij (|A|' W |A| + |S|)_ij / sqrt(H_ii H_jj)`` in the 2-norm (Higham
  2002, Theorems 10.3 and 19.4, the row accumulations included; ``||.||_2 <=
  p ||.||_max``), so with ``kappa`` the Jacobi-scaled condition number a
  trace ``tr(H^-1 B)`` of a positive semidefinite ``B`` moves by at most
  ``kappa eta / (1 - kappa eta)`` relatively per ``H^-1`` it contains
  (Higham 2002, Theorem 7.2, first order; ``_agreement`` in
  ``test_factor_smooth_leaf_factor.py``).
- What the mode moves.  Poisson/log weights are ``W = w mu``, so a mode
  within ``delta`` of the exact one in ``eta`` moves every ``W_i`` by at most
  ``expm1(delta)`` relatively, the deviance by ``2 sum w |y - mu| delta``,
  and the edf, ``sum_i h_i``, by ``edf expm1(delta)`` (``|d tr(H^-1 S)| <=
  sum_i h_i |dW_i| / W_i``).
- The REML derivatives.  Each entry is a sum of traces with at most two
  ``H^-1``, two W-dependent operators and quadratic forms in ``beta``; its
  pre-cancellation size is bounded by ``M_kl`` (``_derivative_bounds``), and
  each factor carries the relative budget above, at most four per entry per
  fit.
"""

from __future__ import annotations

import math
import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest

from superglm import FactorSmooth, LambdaPolicy, Numeric, RandomEffect, Spline, SuperGLM
from superglm.model.reml_setup import collect_reml_groups
from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian
from superglm.reml.penalty_algebra import build_penalty_context, build_penalty_matrix
from superglm.reml.w_derivatives import reml_w_correction
from superglm.solvers.irls_direct import StructuredSolverError, fit_irls_direct
from superglm.solvers.structured import (
    ProfiledFactorSmoothLeafFactor,
    ProfiledNestedSchurFactor,
    ProfiledSumToZeroTreeFactor,
)

EPS = float(np.finfo(np.float64).eps)
_U = EPS / 2.0
# route: (profiled factor, the route's structured terms)
ROUTES = {
    "nested-lone": (ProfiledNestedSchurFactor, ("country",)),
    "nested-chain": (ProfiledNestedSchurFactor, ("country", "region")),
    "fs": (ProfiledFactorSmoothLeafFactor, ("x:country:fs",)),
    "sz": (ProfiledSumToZeroTreeFactor, ("x:country:sz",)),
}


def _gamma(count: float) -> float:
    return count * _U / (1.0 - count * _U)


@cache
def _data(n: int = 1200, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Poisson counts: 12 countries with six regions each, a numeric, and a
    smooth in ``x`` whose shape varies by country."""
    rng = np.random.default_rng(seed)
    country = rng.integers(0, 12, n)
    region = rng.integers(0, 6, n)
    x, x1 = rng.uniform(size=n), rng.normal(size=n)
    eta = (
        0.2
        + 0.3 * x1
        + rng.normal(0.0, 0.3, 12)[country]
        + rng.normal(0.0, 0.2, 72)[6 * country + region]
        + 0.3 * np.sin(2.0 * np.pi * x) * (1.0 + rng.normal(0.0, 0.3, 12)[country])
    )
    frame = pd.DataFrame(
        {
            "x": x,
            "x1": x1,
            "country": [f"c{c:02d}" for c in country],
            "region": [f"c{c:02d}r{r}" for c, r in zip(country, region, strict=True)],
        }
    )
    return frame, rng.poisson(np.exp(eta)).astype(float)


def _model(route: str, scale: float) -> SuperGLM:
    def fixed(value: float) -> LambdaPolicy:
        return LambdaPolicy.fixed(scale * value)

    features: dict = {"x1": Numeric()}
    interactions = []
    if route.startswith("nested"):
        features["country"] = RandomEffect(lambda_policy=fixed(3.0))
        if route == "nested-chain":
            features["region"] = RandomEffect(lambda_policy=fixed(5.0))
    else:
        if route == "sz":
            features["x"] = Spline(n_knots=6, lambda_policy=LambdaPolicy.off())
        interactions.append(
            FactorSmooth("x", group="country", basis=route, k=6, lambda_policy=fixed(0.7))
        )
    return SuperGLM(
        family="poisson",
        features=features,
        interactions=interactions,
        selection_penalty=0,
        direct_solve="structured",
    )


def _fit(route: str, exponent: int) -> SuperGLM:
    frame, y = _data()
    scale = math.ldexp(1.0, exponent)
    model = _model(route, scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y, sample_weight=np.full(len(y), scale))
    return model


@cache
def _reference(route: str) -> SuperGLM:
    return _fit(route, 0)


# ── Bounds ────────────────────────────────────────────────────────────────


class _Mode:
    """A Poisson/log mode in solver coordinates: ``A = [X, 1]``, ``theta = (beta, alpha)``."""

    def __init__(self, dm, groups, penalties, lambdas, beta, alpha, weight):
        _, y = _data()
        p = dm.shape[1]
        self.A = np.hstack([dm.toarray(), np.ones((dm.shape[0], 1))])
        self.S = np.zeros((p + 1, p + 1))
        self.S[:p, :p] = build_penalty_matrix(dm.group_matrices, groups, lambdas, p, penalties)
        self.theta = np.append(beta, alpha)
        self.weight, self.y = weight, y
        self.eta = self.A @ self.theta
        self.mu = np.exp(self.eta)
        self.W = weight * self.mu
        self.H = self.A.T @ (self.W[:, None] * self.A) + self.S

    @classmethod
    def of_model(cls, model: SuperGLM, weight) -> _Mode:
        shift = model._runtime_canonical_state["intercept_shift"]
        return cls(
            model._dm,
            model._groups,
            model._reml_penalties,
            model._reml_lambdas,
            model.result.beta,
            model.result.intercept - shift,
            weight,
        )

    def gap(self) -> np.ndarray:
        """``|theta - theta*|`` componentwise, to first order (module docstring)."""
        A, theta, y, weight = self.A, self.theta, self.y, self.weight
        inverse = np.linalg.inv(self.H)
        score = weight * (y - self.mu)
        terms = np.vstack([A * score[:, None], -(self.S * theta).T])
        g = np.array([math.fsum(column) for column in terms.T])
        d_eta = (np.count_nonzero(A, axis=1) + 1) * EPS * (np.abs(A) @ np.abs(theta))
        rows = self.W * d_eta + 16 * EPS * weight * (np.abs(y) + self.mu)
        error = np.abs(A).T @ (rows + 2 * EPS * np.abs(score))
        error = error + 2 * EPS * np.abs(self.S) @ np.abs(theta)
        return np.abs(inverse @ g) + np.abs(inverse) @ error

    def kappa_eta(self) -> float:
        """``kappa eta`` of the Jacobi-scaled ``H`` (module docstring, the factor)."""
        n, p = self.A.shape
        scale = 1.0 / np.sqrt(np.diag(self.H))
        magnitude = np.abs(self.A).T @ (self.W[:, None] * np.abs(self.A)) + np.abs(self.S)
        eta = p * _gamma(n + p + 1) * float(np.max(scale[:, None] * magnitude * scale[None, :]))
        kappa = float(np.linalg.cond(scale[:, None] * self.H * scale[None, :]))
        value = kappa * eta
        assert value < 0.5
        return value / (1.0 - value)

    def deviance_rounding(self) -> float:
        """Each term's few operations and the sum: ``gamma_{n+8}`` of the term magnitudes."""
        y, mu = self.y, self.mu
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.where(y > 0, np.abs(y * np.log(y / mu)), 0.0)
        return _gamma(len(y) + 8) * float(np.sum(2 * self.weight * (log_term + y + mu)))


def _eta_bound(mode: _Mode, gap: np.ndarray) -> np.ndarray:
    """``|A| gap`` plus the evaluation rounding of ``A theta``."""
    A = np.abs(mode.A)
    rounding = (np.count_nonzero(A, axis=1) + 1) * EPS * (A @ np.abs(mode.theta))
    return A @ gap + rounding


def _assert_same_mode(reference: _Mode, scaled: _Mode) -> np.ndarray:
    """The two linear predictors within the gaps' bound; returns that bound."""
    bound = _eta_bound(reference, reference.gap()) + _eta_bound(scaled, scaled.gap())
    np.testing.assert_array_less(np.abs(scaled.eta - reference.eta), bound)
    return bound


def _penalty_directions(dm, groups, penalties, lambdas) -> list[np.ndarray]:
    """``lambda_k S_k`` in the augmented coordinates, one per penalty component."""
    p = dm.shape[1]
    out = []
    for component in penalties:
        single = {
            name: (value if name == component.name else 0.0) for name, value in lambdas.items()
        }
        block = np.zeros((p + 1, p + 1))
        block[:p, :p] = build_penalty_matrix(dm.group_matrices, groups, single, p, penalties)
        out.append(block)
    return out


def _derivative_bounds(layer: dict, moved: np.ndarray, tau: float):
    """``(gradient, Hessian)`` bounds on the REML derivatives' rescaling difference.

    Pre-cancellation sizes (``D_k = lambda_k S_k``, ``r_k`` its rank, ``phi =
    1`` at the reference): ``-1/2 tr(H^-1 (D_k + dH_k) H^-1 (D_l + dH_l))`` is
    at most ``a_k a_l / 2`` with ``a_k = sqrt(r_k) + sqrt(p) d_k``, because
    ``H^-1/2 D_k H^-1/2`` has rank ``r_k`` and eigenvalues in ``[0, 1]`` and
    ``dH_k = A' diag(W deta_k) A`` lies between ``-+ d_k A' W A``, ``d_k =
    max |deta_k|``, ``deta_k = -A H^-1 D_k theta``; the quadratic forms
    ``theta' D_k H^-1 D_l theta`` are at most ``b_k b_l``, ``b_k^2 = q_k =
    theta' D_k theta <= 2 |g_k| + r_k`` (``D_k H^-1 D_k <= D_k`` and ``0 <=
    tr(H^-1 D_k) <= r_k``); the diagonal's ``g_k + r_k / 2`` is at most
    ``|g_k| + 2 r_k``, and a shared block's ``d2 log|S|_+ / 2`` at most
    ``(sqrt(r_k r_l) + delta_kl r_k) / 2``.  The modes differ by ``moved``
    (componentwise), which moves ``d_k`` by at most ``f_k = max (|A| |H^-1| |D_k| moved)`` and
    ``b_k`` by at most ``sqrt(moved' |D_k| moved)``: the bilinear terms move
    by at most ``M(a + sqrt(p) f, b + db) - M(a, b)``.  The rounding is
    ``tau`` relative to ``M`` (module docstring).
    """
    mode, gradient, ranks = layer["mode"], np.abs(layer["gradient"]), layer["ranks"]
    inverse = np.linalg.inv(mode.H)
    p = mode.H.shape[0]
    A = mode.A
    d = np.array([np.max(np.abs(A @ (inverse @ (D @ mode.theta)))) for D in layer["directions"]])
    f = np.array(
        [np.max(np.abs(A) @ (np.abs(inverse) @ (np.abs(D) @ moved))) for D in layer["directions"]]
    )
    db = np.sqrt([moved @ np.abs(D) @ moved for D in layer["directions"]])
    a = np.sqrt(ranks) + math.sqrt(p) * d
    b = np.sqrt(2.0 * gradient + ranks)
    logdet = 0.5 * (np.sqrt(np.outer(ranks, ranks)) + np.diag(ranks))

    def size(a, b):
        return 0.5 * np.outer(a, a) + np.outer(b, b) + np.diag(gradient + 2.0 * ranks) + logdet

    exact, perturbed = size(a, b), size(a + math.sqrt(p) * f, b + db)
    gradient_bound = tau * (gradient + 2.0 * ranks) + 0.5 * ((b + db) ** 2 - b**2)
    return gradient_bound, tau * perturbed + (perturbed - exact)


# ── Layers ────────────────────────────────────────────────────────────────


def _factor_layer(reference: SuperGLM, exponent: int, order: int, factor_type) -> dict:
    """The route's PIRLS mode and REML derivatives at the reference design, scale ``2^exponent``."""
    scale = math.ldexp(1.0, exponent)
    _, y = _data()
    dm, groups = reference._dm, reference._groups
    matrices = list(dm.group_matrices)
    penalties, _, _ = build_penalty_context(matrices, collect_reml_groups(groups, matrices))
    lambdas = {name: scale * value for name, value in reference._reml_lambdas.items()}
    weight = np.full(len(y), scale)
    offset = np.zeros(len(y))
    family, link = reference._distribution, reference._link
    result, factor = fit_irls_direct(
        X=dm,
        y=y,
        weights=weight,
        family=family,
        link=link,
        groups=groups,
        lambda2=lambdas,
        offset=offset,
        direct_solve="structured",
        reml_penalties=penalties,
        convergence="mode_score",
        weight_semantics="prior",
    )
    assert result.direct_backend == "structured"
    assert isinstance(factor, factor_type)
    gradient = reml_direct_gradient(
        matrices, result, factor, lambdas, reml_penalties=penalties, inverse_phi=1.0 / scale
    )
    correction = reml_w_correction(
        dm,
        link,
        groups,
        result,
        factor,
        lambdas,
        sample_weight=weight,
        offset_arr=offset,
        distribution=family,
        w_correction_order=order,
        reml_penalties=penalties,
    )
    hessian = reml_direct_hessian(
        matrices,
        family,
        factor,
        lambdas,
        gradient=gradient,
        pirls_result=result,
        inverse_phi=1.0 / scale,
        dH_extra=correction[1],
        dH2_cross=correction[2] if order == 2 else None,
        reml_penalties=penalties,
    )
    mode = _Mode(dm, groups, penalties, lambdas, result.beta, result.intercept, weight)
    directions = _penalty_directions(dm, groups, penalties, lambdas)
    ranks = np.array([float(component.rank) for component in penalties])
    return {
        "mode": mode,
        "gradient": gradient,
        "correction": correction[0],
        "hessian": hessian,
        "directions": directions,
        "ranks": ranks,
        "rank": int(result.reml_hessian_rank),
    }


@cache
def _reference_layer(route: str, order: int) -> dict:
    return _factor_layer(_reference(route), 0, order, ROUTES[route][0])


def _assert_factor_layer(reference: dict, scaled: dict) -> None:
    """Same rank, mode and REML derivatives; the W correction is finite."""
    ref_mode, mode = reference["mode"], scaled["mode"]
    assert scaled["rank"] == reference["rank"]
    eta_bound = _assert_same_mode(ref_mode, mode)
    for key in ("gradient", "correction", "hessian"):
        assert np.all(np.isfinite(scaled[key])), key
    # two fits, at most four budget-carrying factors per entry each
    tau = 8.0 * (ref_mode.kappa_eta() + math.expm1(float(np.max(eta_bound))))
    moved = ref_mode.gap() + mode.gap()
    gradient_bound, hessian_bound = _derivative_bounds(reference, moved, tau)
    np.testing.assert_array_less(np.abs(scaled["gradient"] - reference["gradient"]), gradient_bound)
    np.testing.assert_array_less(np.abs(scaled["hessian"] - reference["hessian"]), hessian_bound)


def _assert_complete_fit(reference: SuperGLM, scaled: SuperGLM, route: str, scale: float):
    for model in (reference, scaled):
        profile = model._reml_profile
        assert profile["direct_backend"] == "structured"
        assert tuple(profile["structured_chain"]) == ROUTES[route][1]
        assert isinstance(model._linear_system_state.profiled_factor, ROUTES[route][0])
        assert bool(model._reml_result.converged)
    assert int(scaled.result.reml_hessian_rank) == int(reference.result.reml_hessian_rank)
    _, y = _data()
    ref_mode = _Mode.of_model(reference, np.ones(len(y)))
    mode = _Mode.of_model(scaled, np.full(len(y), scale))
    eta_bound = _assert_same_mode(ref_mode, mode)
    delta = float(np.max(eta_bound))
    p = ref_mode.H.shape[0]
    edf = float(reference.result.effective_df)
    edf_bound = 2.0 * (edf * math.expm1(delta) + p * ref_mode.kappa_eta())
    assert abs(float(scaled.result.effective_df) - edf) <= edf_bound
    moved = 2.0 * float(np.sum(np.abs(y - ref_mode.mu) * eta_bound))
    deviance_bound = moved + ref_mode.deviance_rounding() + mode.deviance_rounding() / scale
    assert abs(float(scaled.result.deviance) / scale - float(reference.result.deviance)) <= (
        deviance_bound
    )
    assert float(scaled.result.phi) == float(reference.result.phi) == 1.0


# ── The tests ─────────────────────────────────────────────────────────────


def _xfail(reason: str, raises) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, raises=raises, reason=f"issue #433 follow-up: {reason}")


_NESTED_TRACES = (
    _xfail(
        "the nested per-level cross traces multiply two weight-sized quantities "
        "(_operator_pair, _rho ** 4), so the REML Hessian leaves float64",
        AssertionError,
    ),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
)
_SECOND_ORDER = _xfail(
    "reml_w_correction's second-order term divides by the Python float sum_w ** 2",
    (OverflowError, ZeroDivisionError),
)
_BOOTSTRAP = _xfail(
    "fit_reml's bootstrap PIRLS runs at absolute lambdas (1.0 and 1e-4) with every "
    "lambda fixed; at prior weights 2^520 they leave the fs level intercepts "
    "unpenalized against the data and the fs factor refuses the super-root pivot "
    "(gram fits)",
    StructuredSolverError,
)
_NESTED = ("nested-lone", "nested-chain")


def _case(route: str, exponent: int, order: int | None = None, marks=()):
    suffix = "" if order in (None, 1) else f"-order{order}"
    values = (route, exponent) if order is None else (route, exponent, order)
    return pytest.param(*values, id=f"{route}-2^{exponent}{suffix}", marks=marks)


@pytest.mark.parametrize(
    ("route", "exponent", "order"),
    [
        *(_case(route, k, 1) for route in _NESTED for k in (-500, 502)),
        *(_case(route, k, 1, _NESTED_TRACES) for route in _NESTED for k in (-600, 520)),
        *(_case(route, k, 1) for route in ("fs", "sz") for k in (-600, 520)),
        _case("fs", 200, 2),
        *(_case("fs", k, 2, _SECOND_ORDER) for k in (-600, 520)),
    ],
)
def test_route_factor_is_invariant_under_weight_rescaling(
    route: str, exponent: int, order: int
) -> None:
    """The route's mode and REML derivatives with weights and lambdas times ``2^exponent``.

    Mutations: restoring ``np.outer(totals, totals) / self.sum_w ** 2`` with a
    Python-float ``sum_w`` in ``ProfiledNestedSchurFactor._cross_matrix``
    raises ``OverflowError`` at ``2^502``; an absolute floor in a route's
    pivot or rank decision (the nested tree pivot, the ``fs`` super-root, the
    ``sz`` border rank) refuses the fit or changes its rank at the small scale.
    """
    scaled = _factor_layer(_reference(route), exponent, order, ROUTES[route][0])
    _assert_factor_layer(_reference_layer(route, order), scaled)


@pytest.mark.parametrize(
    ("route", "exponent"),
    [
        *(_case(route, k) for route in (*_NESTED, "sz") for k in (-600, 520)),
        _case("fs", -600),
        _case("fs", 48),
        _case("fs", 520, marks=_BOOTSTRAP),
    ],
)
def test_route_fit_is_invariant_under_weight_rescaling(route: str, exponent: int) -> None:
    """``fit_reml`` with weights and fixed lambdas times ``2^exponent``: the same fit.

    Mutation: an absolute floor in a route's pivot or rank decision refuses
    the fit or stops it unconverged at ``2^-600``.
    """
    scale = math.ldexp(1.0, exponent)
    _assert_complete_fit(_reference(route), _fit(route, exponent), route, scale)
