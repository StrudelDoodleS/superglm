"""Family-correct profiling of REML dispersion terms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar
from scipy.special import digamma, gammaln, i0e, i1e, polygamma, zeta

from superglm.profiling.tweedie import _P15_BESSEL_ASYMPTOTIC_MIN_ARGUMENT
from superglm.solvers.dispersion import FREQUENCY_WEIGHTS, PRIOR_WEIGHTS

_GAMMA_ASYMPTOTIC_SHAPE = 100.0

_LOG_TWO_PI = float(np.log(2.0 * np.pi))

# Initial log-phi search window for the Tweedie profile, expanded on demand up
# to the hard limit. The window is a numerical bracket, not a model bound: a
# profile optimum outside the representable window raises rather than clamps.
_TWEEDIE_LOG_PHI_WINDOW = 12.0
_TWEEDIE_LOG_PHI_STEP = 15.0
_TWEEDIE_LOG_PHI_LIMIT = 45.0
_TWEEDIE_LOG_PHI_XATOL = 1e-9
_TWEEDIE_LOG_PHI_EDGE_TOL = 1e-6
# Central-difference step (in log phi) for the profile curvature. The
# preferred arm differences the family's ANALYTIC log-phi score, so
# evaluation noise eps enters the curvature at O(eps/step); the value-form
# fallback (analytic scores unavailable on some saddlepoint rows) pays
# O(eps/step^2) and uses the smaller step to bound its truncation instead.
_TWEEDIE_SCORE_CURVATURE_STEP = 1e-3


class _ScoreUnavailableInBracketError(Exception):
    """An analytic log-phi score branch dropped out inside a polish bracket."""


_TWEEDIE_CURVATURE_STEP = 1e-4


@dataclass(frozen=True)
class ProfiledScaleTerm:
    """Minimized scale-dependent part of Wood's REML criterion."""

    phi: float
    inverse_phi: float
    criterion: float
    d_inverse_phi_d_penalized_deviance: float


class _LazyGammaScaleTerm(ProfiledScaleTerm):
    """Gamma prior-arm term computing ``d(1/phi)/d(Dp)`` on first read.

    The derivative needs the profile curvature, whose trigamma pass over the
    distinct weights is the single most expensive array operation in the
    Gamma profiler -- and the outer loop never reads it for rejected
    line-search trials, the boot evaluation, or the post-fit phi recompute,
    all of which consume only ``phi`` and ``criterion``.  Deferring it skips
    those passes entirely while leaving the value, and the
    ``FloatingPointError`` contract on an unrepresentable derivative, exactly
    as the eager path states them for every term actually read.

    The subclass keeps the parent's frozen field surface (attribute reads,
    dataclass ``repr``/``eq`` through the property) and reduces to a plain
    eager ``ProfiledScaleTerm`` under pickle and ``deepcopy`` so the retained
    weight arrays never travel with a serialized term.
    """

    # Set through ``object.__setattr__`` below (the parent dataclass is
    # frozen); declared here so the attribute surface is statically visible.
    _profile_data: GammaScaleProfileData
    _penalty_nullity: float
    _derivative: float | None

    def __init__(
        self,
        *,
        phi: float,
        inverse_phi: float,
        criterion: float,
        profile_data: GammaScaleProfileData,
        penalty_nullity: float,
    ) -> None:
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "inverse_phi", inverse_phi)
        object.__setattr__(self, "criterion", criterion)
        object.__setattr__(self, "_profile_data", profile_data)
        object.__setattr__(self, "_penalty_nullity", penalty_nullity)
        object.__setattr__(self, "_derivative", None)

    @property  # type: ignore[override]
    def d_inverse_phi_d_penalized_deviance(self) -> float:
        value = self._derivative
        if value is None:
            value = float(
                _gamma_inverse_shape_derivative(
                    self.inverse_phi,
                    self._profile_data.scaled_curvature(self.inverse_phi, self._penalty_nullity),
                )
            )
            if not np.isfinite(value):
                raise FloatingPointError("Gamma REML scale derivative is not representable")
            object.__setattr__(self, "_derivative", value)
        return value

    def __reduce__(self):
        return (
            ProfiledScaleTerm,
            (
                self.phi,
                self.inverse_phi,
                self.criterion,
                self.d_inverse_phi_d_penalized_deviance,
            ),
        )


@dataclass(frozen=True)
class GammaScaleProfileData:
    """Fit-invariant sufficient statistics for saturated Gamma likelihoods.

    Under ``"frequency"`` the saturated log-likelihood is ``sum(w)`` copies of
    one scalar function of the shape, so two scalars are sufficient and the
    root find never rescans rows.  Under ``"prior"`` row ``i`` has its own
    shape ``w_i / phi``, and the saturated arm becomes ``sum_i G(w_i k)`` with
    ``G(a) = a log a - a - log Gamma(a)`` -- the same function, evaluated at a
    per-row argument.  The weights are therefore retained, collapsed onto
    their distinct values with multiplicities: the collapse is exact (the
    summand depends on the row only through its weight) and it makes the
    unweighted case evaluate one scalar, reproducing the frequency arithmetic
    to the bit rather than to a tolerance.
    """

    sum_weight: float
    sum_weight_log_y: float
    weight_semantics: str = FREQUENCY_WEIGHTS
    distinct_weight: NDArray | None = field(default=None, repr=False, compare=False)
    weight_multiplicity: NDArray | None = field(default=None, repr=False, compare=False)
    # Per-fit solver state, prior arm only.  ``_profile_cache`` memoizes whole
    # ``ProfiledScaleTerm`` results by exact ``(Dp, Mp)`` key: the outer loop
    # re-evaluates accepted line-search points at bitwise-identical inputs, so
    # a hit returns the identical object a recomputation would rebuild.
    # ``_warm_history`` retains ``(Dp, Mp, log_shape_root)`` of recent solves
    # so the next root find can start from a secant prediction instead of the
    # fixed +-30 window.  Neither is consulted under ``"frequency"``.
    _profile_cache: dict = field(default_factory=dict, repr=False, compare=False)
    _warm_history: list = field(default_factory=list, repr=False, compare=False)
    _uniform_multiplicity: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.sum_weight, self.sum_weight_log_y],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or self.sum_weight <= 0.0:
            raise ValueError(
                "Gamma scale sufficient statistics must be finite with positive weight"
            )
        if self.weight_semantics not in (PRIOR_WEIGHTS, FREQUENCY_WEIGHTS):
            raise ValueError(
                f"weight_semantics must be 'prior' or 'frequency', got {self.weight_semantics!r}"
            )
        if self.weight_semantics == PRIOR_WEIGHTS and (
            self.distinct_weight is None or self.weight_multiplicity is None
        ):
            raise ValueError("prior-weight Gamma scale data requires the retained row weights")
        object.__setattr__(self, "sum_weight", float(self.sum_weight))
        object.__setattr__(self, "sum_weight_log_y", float(self.sum_weight_log_y))
        if self.weight_semantics == PRIOR_WEIGHTS and self.weight_multiplicity is not None:
            # All-distinct weights carry multiplicity exactly 1.0 everywhere;
            # ``sum(1.0 * f)`` and ``sum(f)`` are the same bits, so the scale
            # methods may skip that multiply pass when this holds.
            object.__setattr__(
                self,
                "_uniform_multiplicity",
                bool(np.all(self.weight_multiplicity == 1.0)),
            )

    def _row_arguments(self, shape: float) -> tuple[NDArray, NDArray]:
        """Return the per-row shapes and their multiplicities.

        ``__post_init__`` refuses prior-weight data without them, so this
        narrowing cannot fail; it is written out so the retained arrays carry
        their non-optional type to every arithmetic site below.
        """
        assert self.distinct_weight is not None
        assert self.weight_multiplicity is not None
        return self.distinct_weight * shape, self.weight_multiplicity

    def _weighted_total(self, values: NDArray) -> float:
        """Sum per-distinct-weight contributions with their multiplicities.

        When every multiplicity is exactly ``1.0`` the multiply is the
        identity on every element, so skipping it returns the same bits for
        one fewer full-array pass.
        """
        assert self.weight_multiplicity is not None
        if self._uniform_multiplicity:
            return float(np.sum(values))
        return float(np.sum(self.weight_multiplicity * values))

    def saturated_normalizer(self, shape: float) -> float:
        """Return the shape-dependent part of the saturated log-likelihood."""
        if self.weight_semantics == FREQUENCY_WEIGHTS:
            return self.sum_weight * _gamma_saturated_normalizer(shape)
        argument, _ = self._row_arguments(shape)
        return self._weighted_total(_gamma_saturated_normalizer_array(argument, assume_sorted=True))

    def saturated_log_shape_score(self, shape: float) -> float:
        """Return ``k d(l_sat)/dk``, the saturated arm of the log-shape score."""
        if self.weight_semantics == FREQUENCY_WEIGHTS:
            return self.sum_weight * _shape_times_log_minus_digamma(shape)
        argument, _ = self._row_arguments(shape)
        return self._weighted_total(
            _shape_times_log_minus_digamma_array(argument, assume_sorted=True)
        )

    def scaled_curvature(self, shape: float, penalty_nullity: float) -> float:
        """Return ``k**2 d2Q/dk2`` without squaring extreme shapes.

        The profile curvature can overflow when ``shape**2`` underflows, even
        though its reciprocal derivative is representable as signed zero.  Work
        instead with ``shape**2 * curvature``, which stays near ``1`` as the
        shape vanishes and near ``sum(w)/2`` as it grows.

        The frequency arm keeps its own association of the three expansions
        verbatim, so this refactor cannot move a number on the contract that
        was already shipping.
        """
        if self.weight_semantics == FREQUENCY_WEIGHTS:
            sum_weight = self.sum_weight
            if shape < 1.0e-4:
                zeta_2 = np.pi**2 / 6.0
                zeta_3 = 1.2020569031595942
                zeta_4 = np.pi**4 / 90.0
                return float(
                    sum_weight
                    - 0.5 * penalty_nullity
                    - sum_weight * shape
                    + sum_weight
                    * (zeta_2 * shape**2 - 2.0 * zeta_3 * shape**3 + 3.0 * zeta_4 * shape**4)
                )
            if shape < _GAMMA_ASYMPTOTIC_SHAPE:
                return float(
                    sum_weight * shape**2 * _trigamma_minus_inverse(shape) - 0.5 * penalty_nullity
                )
            inverse = 1.0 / shape
            return float(
                0.5 * (sum_weight - penalty_nullity)
                + sum_weight * (inverse / 6.0 - inverse**3 / 30.0 + inverse**5 / 42.0)
            )
        argument, _ = self._row_arguments(shape)
        return (
            self._weighted_total(_scaled_trigamma_minus_inverse_array(argument, assume_sorted=True))
            - 0.5 * penalty_nullity
        )


def prepare_gamma_reml_scale_data(
    y: NDArray,
    sample_weight: NDArray,
    *,
    weight_semantics: str,
) -> GammaScaleProfileData:
    """Validate rows once and reduce them to Gamma saturated-likelihood statistics."""
    if weight_semantics not in (PRIOR_WEIGHTS, FREQUENCY_WEIGHTS):
        raise ValueError(
            f"weight_semantics must be 'prior' or 'frequency', got {weight_semantics!r}",
        )
    y = np.asarray(y, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if y.ndim != 1 or sample_weight.shape != y.shape:
        raise ValueError("y and sample_weight must be one-dimensional with matching shape")
    if (
        not np.all(np.isfinite(y))
        or np.any(y <= 0.0)
        or not np.all(np.isfinite(sample_weight))
        or np.any(sample_weight < 0.0)
    ):
        raise ValueError("Gamma scale profiling requires positive y and non-negative weights")
    if weight_semantics == FREQUENCY_WEIGHTS:
        with np.errstate(over="ignore", invalid="ignore"):
            sum_weight = float(np.sum(sample_weight, dtype=np.float64))
            sum_weight_log_y = float(np.sum(sample_weight * np.log(y), dtype=np.float64))
        if sum_weight <= 0.0:
            raise ValueError("Gamma scale profiling requires positive total weight")
        return GammaScaleProfileData(
            sum_weight=sum_weight,
            # Elementwise reduction avoids BLAS thread-launch overhead for this
            # one-vector sufficient statistic (materially slower in measured fits).
            sum_weight_log_y=sum_weight_log_y,
            weight_semantics=FREQUENCY_WEIGHTS,
        )
    # A zero prior weight is a row observed with infinite variance, so it
    # leaves the likelihood entirely rather than contributing a saturated term
    # of its own -- log Gamma(0) is not finite, and the row is not part of the
    # size the residual d.f. corrects.
    carried = sample_weight > 0.0
    likelihood_size = float(np.count_nonzero(carried))
    if likelihood_size <= 0.0:
        raise ValueError("Gamma scale profiling requires at least one positive prior weight")
    distinct_weight, multiplicity = np.unique(sample_weight[carried], return_counts=True)
    with np.errstate(over="ignore", invalid="ignore"):
        sum_log_y = float(np.sum(np.log(y[carried]), dtype=np.float64))
    return GammaScaleProfileData(
        sum_weight=likelihood_size,
        sum_weight_log_y=sum_log_y,
        weight_semantics=PRIOR_WEIGHTS,
        distinct_weight=distinct_weight,
        weight_multiplicity=multiplicity.astype(np.float64),
    )


def gaussian_reml_scale_terms(
    sample_weight: NDArray,
    *,
    weight_semantics: str,
) -> tuple[float, float]:
    """Return the Gaussian likelihood size and its weight-only saturated term.

    Under ``"prior"`` each row's saturated density is ``N(y_i; y_i, phi/w_i)``,
    whose normalizer is ``0.5 log(w_i / (2 pi phi))``.  The ``0.5 log w_i`` part
    carries no ``phi`` and no ``lambda``, so it cannot move the profiled
    dispersion or the selected smoothing parameters; it is returned and applied
    anyway so that the published criterion is the restricted log-likelihood
    rather than the restricted log-likelihood plus an unnamed constant, which
    is what makes a cross-package comparison meaningful.
    """
    if weight_semantics not in (PRIOR_WEIGHTS, FREQUENCY_WEIGHTS):
        raise ValueError(
            f"weight_semantics must be 'prior' or 'frequency', got {weight_semantics!r}",
        )
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weight_semantics == FREQUENCY_WEIGHTS:
        return float(np.sum(weights, dtype=np.float64)), 0.0
    carried = weights > 0.0
    likelihood_size = float(np.count_nonzero(carried))
    saturated_log_weight = float(0.5 * np.sum(np.log(weights[carried]), dtype=np.float64))
    return likelihood_size, saturated_log_weight


def profile_gaussian_reml_scale(
    penalized_deviance: float,
    likelihood_size: float,
    penalty_nullity: float,
    *,
    saturated_log_weight: float = 0.0,
) -> ProfiledScaleTerm:
    """Return the full closed-form Gaussian scale term from Wood's criterion."""
    penalized_deviance = float(penalized_deviance)
    likelihood_size = float(likelihood_size)
    penalty_nullity = float(penalty_nullity)
    saturated_log_weight = float(saturated_log_weight)
    values = np.asarray(
        [penalized_deviance, likelihood_size, penalty_nullity, saturated_log_weight],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or penalized_deviance <= 0.0:
        raise ValueError("Gaussian profile inputs must be finite with positive deviance")
    residual_size = likelihood_size - penalty_nullity
    if penalty_nullity < 0.0 or residual_size <= 0.0:
        raise ValueError("Gaussian REML profile requires positive residual likelihood size")
    log_phi = float(np.log(penalized_deviance) - np.log(residual_size))
    log_float_max = float(np.log(np.finfo(np.float64).max))
    log_derivative_magnitude = float(np.log(residual_size) - 2.0 * np.log(penalized_deviance))
    if abs(log_phi) > log_float_max or log_derivative_magnitude > log_float_max:
        raise FloatingPointError("Gaussian REML scale profile is not representable")
    phi = float(np.exp(log_phi))
    inverse_phi = float(np.exp(-log_phi))
    log_smallest = float(np.log(np.nextafter(0.0, 1.0)))
    derivative = (
        -0.0
        if log_derivative_magnitude < log_smallest
        else float(-np.exp(log_derivative_magnitude))
    )
    criterion = float(
        0.5 * residual_size * (1.0 + np.log(2.0 * np.pi) + log_phi) - saturated_log_weight
    )
    if not np.isfinite(criterion):
        raise FloatingPointError("Gaussian REML scale criterion is not representable")
    return ProfiledScaleTerm(
        phi=phi,
        inverse_phi=inverse_phi,
        criterion=criterion,
        d_inverse_phi_d_penalized_deviance=derivative,
    )


def _log_minus_digamma(shape: float) -> float:
    """Evaluate ``log(shape) - digamma(shape)`` without large-shape cancellation."""
    if shape < _GAMMA_ASYMPTOTIC_SHAPE:
        return float(np.log(shape) - digamma(shape))
    inverse = 1.0 / shape
    inverse2 = inverse * inverse
    return float(
        0.5 * inverse
        + inverse2 / 12.0
        - inverse2 * inverse2 / 120.0
        + inverse2 * inverse2 * inverse2 / 252.0
    )


def _shape_times_log_minus_digamma(shape: float) -> float:
    """Evaluate ``shape * (log(shape) - digamma(shape))`` stably."""
    if shape < 1.0e-4:
        euler_gamma = 0.5772156649015329
        zeta_2 = np.pi**2 / 6.0
        zeta_3 = 1.2020569031595942
        zeta_4 = np.pi**4 / 90.0
        return float(
            1.0
            + shape * (np.log(shape) + euler_gamma)
            - zeta_2 * shape**2
            + zeta_3 * shape**3
            - zeta_4 * shape**4
        )
    if shape < _GAMMA_ASYMPTOTIC_SHAPE:
        return float(shape * _log_minus_digamma(shape))
    inverse = 1.0 / shape
    return float(0.5 + inverse / 12.0 - inverse**3 / 120.0 + inverse**5 / 252.0)


def _gamma_saturated_normalizer(shape: float) -> float:
    """Return ``k log(k) - k - log Gamma(k)`` stably."""
    if shape < _GAMMA_ASYMPTOTIC_SHAPE:
        return float(shape * np.log(shape) - shape - gammaln(shape))
    inverse = 1.0 / shape
    return float(
        0.5 * (np.log(shape) - np.log(2.0 * np.pi))
        - inverse / 12.0
        + inverse**3 / 360.0
        - inverse**5 / 1260.0
    )


def _trigamma_minus_inverse(shape: float) -> float:
    """Evaluate ``trigamma(shape) - 1 / shape`` stably."""
    if shape < _GAMMA_ASYMPTOTIC_SHAPE:
        return float(polygamma(1, shape) - 1.0 / shape)
    inverse = 1.0 / shape
    inverse2 = inverse * inverse
    return float(
        0.5 * inverse2
        + inverse2 * inverse / 6.0
        - inverse2 * inverse**3 / 30.0
        + inverse2 * inverse**5 / 42.0
    )


def _branch_selectors(
    argument: NDArray,
    *,
    assume_sorted: bool,
) -> tuple[Any, Any, Any]:
    """Return (small, middle, large) row selectors for the three expansions.

    With ``assume_sorted`` the argument is ascending (the prior contract
    evaluates ``w_i * k`` on ``np.unique``-sorted weights at ``k > 0``, and
    multiplying by a positive scalar is monotone under round-to-nearest), so
    the branch boundaries are two binary searches and the selectors are
    slices -- views, not gather/scatter copies.  A boolean mask over a sorted
    array selects the same elements in the same order as the slice, and the
    per-branch arithmetic is elementwise, so both selector kinds produce
    bitwise-identical results.  Empty selectors are returned as ``None``.
    """
    if assume_sorted:
        i_small = int(np.searchsorted(argument, 1.0e-4, side="left"))
        i_large = int(np.searchsorted(argument, _GAMMA_ASYMPTOTIC_SHAPE, side="left"))
        small = slice(0, i_small) if i_small > 0 else None
        middle = slice(i_small, i_large) if i_large > i_small else None
        large = slice(i_large, argument.size) if i_large < argument.size else None
        return small, middle, large
    is_small = argument < 1.0e-4
    is_large = argument >= _GAMMA_ASYMPTOTIC_SHAPE
    is_middle = ~is_small & ~is_large
    return (
        is_small if np.any(is_small) else None,
        is_middle if np.any(is_middle) else None,
        is_large if np.any(is_large) else None,
    )


def _shape_times_log_minus_digamma_array(
    argument: NDArray,
    *,
    assume_sorted: bool = False,
) -> NDArray:
    """Vectorized ``a (log a - digamma(a))``, branch for branch.

    The prior-weight contract evaluates this at a per-row ``a = w_i k``, and a
    single fit's weights can straddle all three of the scalar helper's
    branches.  Each branch is applied to its own rows and nowhere else: the
    small-shape series diverges at large argument and the asymptotic series is
    meaningless at small argument, so a ``np.where`` over all three would
    compute both wrong answers before discarding them, complete with their
    overflow warnings.
    """
    result = np.empty_like(argument)
    small, middle, large = _branch_selectors(argument, assume_sorted=assume_sorted)
    if small is not None:
        a = argument[small]
        euler_gamma = 0.5772156649015329
        zeta_2 = np.pi**2 / 6.0
        zeta_3 = 1.2020569031595942
        zeta_4 = np.pi**4 / 90.0
        result[small] = (
            1.0 + a * (np.log(a) + euler_gamma) - zeta_2 * a**2 + zeta_3 * a**3 - zeta_4 * a**4
        )
    if middle is not None:
        a = argument[middle]
        result[middle] = a * (np.log(a) - digamma(a))
    if large is not None:
        inverse = 1.0 / argument[large]
        result[large] = 0.5 + inverse / 12.0 - inverse**3 / 120.0 + inverse**5 / 252.0
    return result


def _gamma_saturated_normalizer_array(
    argument: NDArray,
    *,
    assume_sorted: bool = False,
) -> NDArray:
    """Vectorized ``a log a - a - log Gamma(a)``.

    Two branches, matching the scalar helper: nothing cancels as ``a -> 0``
    (the expression tends to ``log a`` with every term evaluated at its own
    magnitude), so only the large-shape rows need the Stirling expansion.
    """
    result = np.empty_like(argument)
    if assume_sorted:
        i_large = int(np.searchsorted(argument, _GAMMA_ASYMPTOTIC_SHAPE, side="left"))
        below: Any = slice(0, i_large) if i_large > 0 else None
        above: Any = slice(i_large, argument.size) if i_large < argument.size else None
    else:
        is_large = argument >= _GAMMA_ASYMPTOTIC_SHAPE
        below = ~is_large if not np.all(is_large) else None
        above = is_large if np.any(is_large) else None
    if below is not None:
        a = argument[below]
        result[below] = a * np.log(a) - a - gammaln(a)
    if above is not None:
        a = argument[above]
        inverse = 1.0 / a
        result[above] = (
            0.5 * (np.log(a) - np.log(2.0 * np.pi))
            - inverse / 12.0
            + inverse**3 / 360.0
            - inverse**5 / 1260.0
        )
    return result


def _scaled_trigamma_minus_inverse_array(
    argument: NDArray,
    *,
    assume_sorted: bool = False,
) -> NDArray:
    """Vectorized ``a**2 (trigamma(a) - 1/a)``.

    The product is what the curvature needs, and forming it as a product would
    overflow long before the shape itself does: ``trigamma(a) ~ a**-2``, so a
    weight small enough to put ``a`` near ``1e-160`` sends the factor past the
    representable range while their product is still exactly ``1``.  The three
    branches are the same expansions the scalar curvature uses, lifted out of
    it so that the per-row and the scalar path evaluate one function.

    The middle branch calls ``zeta(2, a)`` where the scalar helper spells
    ``polygamma(1, a)``.  These are the same numbers bit for bit: scipy's
    ``polygamma`` computes ``(-1)**(n+1) * gamma(n+1) * zeta(n+1, x)`` and at
    ``n = 1`` the prefactor is exactly ``1.0``, whose multiply is the IEEE
    identity -- but ``polygamma`` also evaluates ``psi(x)`` over the whole
    array only to discard it in a ``where``, and this branch runs at every
    distinct weight, so the direct call skips two full-array passes.
    """
    result = np.empty_like(argument)
    small, middle, large = _branch_selectors(argument, assume_sorted=assume_sorted)
    if small is not None:
        a = argument[small]
        zeta_2 = np.pi**2 / 6.0
        zeta_3 = 1.2020569031595942
        zeta_4 = np.pi**4 / 90.0
        result[small] = 1.0 - a + zeta_2 * a**2 - 2.0 * zeta_3 * a**3 + 3.0 * zeta_4 * a**4
    if middle is not None:
        a = argument[middle]
        result[middle] = a**2 * (zeta(2.0, a) - 1.0 / a)
    if large is not None:
        inverse = 1.0 / argument[large]
        result[large] = 0.5 + inverse / 6.0 - inverse**3 / 30.0 + inverse**5 / 42.0
    return result


def _gamma_inverse_shape_derivative(
    shape: float,
    scaled_curvature: float,
) -> float:
    """Return ``d(shape) / d(Dp)`` from the scaled profile curvature.

    Implicit differentiation of the profile score at its root gives
    ``d(shape)/d(Dp) = -1/2 / Q''(shape)``, and the caller supplies
    ``shape**2 Q''`` rather than ``Q''`` so that the ratio survives shapes whose
    square is not representable.  The division is taken in log space for the
    same reason.
    """
    if not np.isfinite(scaled_curvature) or scaled_curvature <= 0.0:
        raise FloatingPointError("Gamma REML scale profile has non-positive curvature")

    log_magnitude = float(np.log(0.5) + 2.0 * np.log(shape) - np.log(scaled_curvature))
    if log_magnitude < np.log(np.nextafter(0.0, 1.0)):
        return -0.0
    if log_magnitude > np.log(np.finfo(np.float64).max):
        return float("-inf")
    return float(-np.exp(log_magnitude))


def _make_gamma_shape_score(
    profile_data: GammaScaleProfileData,
    penalized_deviance: float,
    penalty_nullity: float,
    stash: dict[float, float] | None,
):
    """Build the profile score ``S(u) = Dp e^u / 2 - k dl_sat/dk + Mp / 2``.

    ``stash`` (prior arm only) memoizes the expensive saturated-score sum by
    exact log-shape key within one solve: ``brentq`` re-evaluates its bracket
    endpoints and returns its last evaluated point, so without the stash the
    same full-array reduction runs again at bitwise-identical arguments.  A
    hit returns the identical float a recomputation would produce.
    """

    def shape_score(log_shape: float) -> float:
        shape = float(np.exp(log_shape))
        if stash is None:
            saturated = profile_data.saturated_log_shape_score(shape)
        else:
            saturated = stash.get(log_shape)
            if saturated is None:
                saturated = profile_data.saturated_log_shape_score(shape)
                stash[log_shape] = saturated
        return float(0.5 * penalized_deviance * shape - saturated + 0.5 * penalty_nullity)

    return shape_score


def _solve_gamma_profile_root_cold(shape_score) -> float:
    """Bracket from the fixed window and solve; the historical root find.

    This is the shipped solve, verbatim: the +-30 log-shape window, the
    widening loop toward the representable limits, and the same ``brentq``
    tolerances.  The frequency arm always solves here, and the prior arm
    falls back here whenever it has no usable warm state.
    """
    log_shape_lo = -30.0
    log_shape_hi = 30.0
    score_lo = shape_score(log_shape_lo)
    score_hi = shape_score(log_shape_hi)
    log_shape_step = 30.0
    log_shape_min = float(np.log(np.nextafter(0.0, 1.0)))
    log_shape_max = float(np.log(np.finfo(np.float64).max))
    while score_lo >= 0.0 and log_shape_lo > log_shape_min:
        log_shape_lo = max(log_shape_lo - log_shape_step, log_shape_min)
        score_lo = shape_score(log_shape_lo)
    while score_hi <= 0.0 and log_shape_hi < log_shape_max:
        log_shape_hi = min(log_shape_hi + log_shape_step, log_shape_max)
        score_hi = shape_score(log_shape_hi)
    if not score_lo < 0.0 or not score_hi > 0.0:
        raise ValueError("Gamma REML scale profile could not bracket a finite optimum")
    return float(
        brentq(
            shape_score,
            log_shape_lo,
            log_shape_hi,
            xtol=1.0e-12,
            rtol=4.0 * np.finfo(float).eps,
            maxiter=100,
        )
    )


# Warm-start bracket policy: initial half-width floor, widening factor, and
# attempts before falling back to the cold window.  The floor covers the
# secant prediction's own error near convergence; the ladder covers early
# iterations where the penalized deviance still moves the root materially.
_GAMMA_WARM_DELTA_FLOOR = 1.0e-5
_GAMMA_WARM_DELTA_GROWTH = 64.0
_GAMMA_WARM_ATTEMPTS = 3


def _solve_gamma_profile_root_warm(
    shape_score,
    history: list,
    penalized_deviance: float,
    penalty_nullity: float,
) -> float | None:
    """Solve from the previous root, or return None to use the cold window.

    The profile score is ``S(u) = Dp e^u / 2 - F(e^u) + Mp / 2`` with ``F``
    independent of ``Dp``, so the root moves smoothly (and, once the outer
    loop is converging, almost linearly) in ``Dp``.  The predictor is the
    secant through the last two roots in ``Dp``; the guess is then bracketed
    by a widening ladder and handed to the same ``brentq`` tolerances as the
    cold solve.  Failure at every ladder rung returns None -- correctness
    never depends on the warm attempt succeeding.
    """
    u_prev = history[-1][2]
    u_guess = u_prev
    if len(history) >= 2:
        dp_1, nullity_1, u_1 = history[-2]
        dp_2, nullity_2, u_2 = history[-1]
        if nullity_1 == nullity_2 == penalty_nullity and dp_2 != dp_1:
            slope = (u_2 - u_1) / (dp_2 - dp_1)
            candidate = u_2 + slope * (penalized_deviance - dp_2)
            if np.isfinite(candidate) and abs(candidate - u_2) <= 1.0:
                u_guess = candidate
    delta = max(_GAMMA_WARM_DELTA_GROWTH * abs(u_guess - u_prev), _GAMMA_WARM_DELTA_FLOOR)
    for _ in range(_GAMMA_WARM_ATTEMPTS):
        bracket_lo = u_guess - delta
        bracket_hi = u_guess + delta
        score_lo = shape_score(bracket_lo)
        score_hi = shape_score(bracket_hi)
        if score_lo < 0.0 < score_hi:
            return float(
                brentq(
                    shape_score,
                    bracket_lo,
                    bracket_hi,
                    xtol=1.0e-12,
                    rtol=4.0 * np.finfo(float).eps,
                    maxiter=100,
                )
            )
        delta *= _GAMMA_WARM_DELTA_GROWTH
    return None


def profile_gamma_reml_scale(
    profile_data: GammaScaleProfileData,
    penalized_deviance: float,
    penalty_nullity: float,
) -> ProfiledScaleTerm:
    """Profile Gamma dispersion while retaining Wood's saturated likelihood.

    Under the frequency contract ``sum(weights)`` is the likelihood observation
    count, the saturated arm is that many copies of one scalar function of the
    shape, and the root find never rescans rows.  Under the prior contract row
    ``i`` carries shape ``w_i / phi``, so the saturated arm is a sum over the
    distinct weights; the deviance arm ``Dp k / 2`` is unchanged, because the
    weighted deviance ``sum_i w_i d_i`` is already what both contracts put
    there.

    The likelihood-size guard is the same statement in each: as ``k -> 0`` the
    saturated arm behaves like ``N log k`` and the criterion's remaining
    ``log k`` coefficient is ``Mp / 2``, so a finite interior optimum needs
    ``2 N > Mp`` with ``N`` the contract's own likelihood size.

    The prior arm's per-evaluation cost is a full pass over the distinct
    weights, so it keeps two per-fit accelerations the scalar frequency arm
    has no use for: an exact memo of whole results by ``(Dp, Mp)`` key (the
    outer loop re-evaluates accepted line-search points bitwise-identically),
    and a warm-started root find from the previous solves' roots with the
    cold window as fallback.  The frequency arm runs the shipped code path
    unconditionally.
    """
    if not isinstance(profile_data, GammaScaleProfileData):
        raise TypeError("profile_data must be GammaScaleProfileData")
    penalized_deviance = float(penalized_deviance)
    penalty_nullity = float(penalty_nullity)
    if not np.isfinite(penalized_deviance) or penalized_deviance <= 0.0:
        raise ValueError("penalized_deviance must be positive and finite")
    if not np.isfinite(penalty_nullity) or penalty_nullity < 0.0:
        raise ValueError("penalty_nullity must be finite and non-negative")

    likelihood_size = profile_data.sum_weight
    if 2.0 * likelihood_size <= penalty_nullity:
        raise ValueError("Gamma REML scale profile has no finite interior optimum")

    is_prior = profile_data.weight_semantics == PRIOR_WEIGHTS
    if is_prior:
        cache_key = (penalized_deviance, penalty_nullity)
        cached = profile_data._profile_cache.get(cache_key)
        if cached is not None:
            return cached

    stash: dict[float, float] | None = {} if is_prior else None
    shape_score = _make_gamma_shape_score(profile_data, penalized_deviance, penalty_nullity, stash)
    log_shape: float | None = None
    if is_prior and profile_data._warm_history:
        log_shape = _solve_gamma_profile_root_warm(
            shape_score,
            profile_data._warm_history,
            penalized_deviance,
            penalty_nullity,
        )
    if log_shape is None:
        log_shape = _solve_gamma_profile_root_cold(shape_score)
    if is_prior:
        history = profile_data._warm_history
        history.append((penalized_deviance, penalty_nullity, log_shape))
        if len(history) > 2:
            del history[:-2]
    shape = float(np.exp(log_shape))
    phi = 1.0 / shape
    saturated_log_likelihood = (
        profile_data.saturated_normalizer(shape) - profile_data.sum_weight_log_y
    )
    criterion = (
        0.5 * penalized_deviance * shape
        - saturated_log_likelihood
        + 0.5 * penalty_nullity * log_shape
        - 0.5 * penalty_nullity * np.log(2.0 * np.pi)
    )
    if not np.isfinite(phi) or not np.isfinite(criterion):
        raise FloatingPointError("Gamma REML scale profile produced a non-finite result")
    if is_prior:
        # The curvature behind ``d(1/phi)/d(Dp)`` is the profiler's most
        # expensive array pass and is unread for rejected line-search trials,
        # the boot evaluation, and the post-fit phi recompute; defer it to
        # first read.  The memo hands every later hit the same term, so an
        # accepted point's curvature is computed at most once per solve.
        lazy_term = _LazyGammaScaleTerm(
            phi=phi,
            inverse_phi=shape,
            criterion=float(criterion),
            profile_data=profile_data,
            penalty_nullity=penalty_nullity,
        )
        profile_data._profile_cache[cache_key] = lazy_term
        return lazy_term
    d_inverse_phi_d_penalized_deviance = _gamma_inverse_shape_derivative(
        shape,
        profile_data.scaled_curvature(shape, penalty_nullity),
    )
    if not np.isfinite(d_inverse_phi_d_penalized_deviance):
        raise FloatingPointError("Gamma REML scale derivative is not representable")
    return ProfiledScaleTerm(
        phi=phi,
        inverse_phi=shape,
        criterion=float(criterion),
        d_inverse_phi_d_penalized_deviance=float(d_inverse_phi_d_penalized_deviance),
    )


@dataclass(frozen=True)
class TweedieScaleProfileData:
    """Fit-invariant state for exact Tweedie saturated log-likelihoods.

    The Tweedie saturated log-likelihood decomposes exactly over the
    zero/positive split of the response (Joergensen 1997): a zero row's
    saturated contribution is the log of an atom probability at ``mu = y = 0``,
    which is exactly ``0`` for every dispersion, while a positive row
    contributes the log-density normalizer evaluated at ``mu = y`` (its unit
    deviance vanishes there). Only the positive rows therefore vary with
    ``phi``, and their prepared density state (validation, base measure) is
    hoisted here once per fit so each profile evaluation is a single series
    pass over positive rows.
    """

    power: float
    n_positive: int
    prepared_positive: Any = field(repr=False)
    # Under ``"prior"`` the weight is already inside ``prepared_positive`` --
    # the compound-Poisson normalizer carries it -- and every positive row
    # contributes once.  Under ``"frequency"`` the prepared state is built at
    # unit weight and the replication count multiplies each row's contribution
    # instead, so the weights are retained here and the effective positive size
    # is their total rather than the row count.
    weight_semantics: str = PRIOR_WEIGHTS
    row_weight: NDArray | None = field(default=None, repr=False, compare=False)
    # Closed-form p = 1.5 state, or None at every other power.
    #
    # At p = 1.5 the Wright parameter a = (2-p)/(p-1) is exactly 1, and DLMF
    # 10.46.2 -- I_nu(z) = (z/2)^nu phi(1, nu+1; z^2/4) -- collapses Wright's
    # function to a modified Bessel function:
    #
    #     Phi(1, 2; t) = I_1(2 sqrt(t)) / sqrt(t),   Phi(1, 1; t) = I_0(2 sqrt(t))
    #
    # This is the case Dunn & Smyth (2005) single out as Siegel's (1979)
    # noncentral chi-squared with zero degrees of freedom, and whose Bessel form
    # their Fourier-inversion companion (2008, Table 2) uses as its reference
    # truth.  The saturated profile is simpler still: this state is prepared
    # with mu = y, so the unit deviance is identically zero and t = (K/(2 phi))^2
    # with K = 4 w sqrt(y) fit-invariant.  The whole positive-row saturated
    # log-likelihood is then
    #
    #     l_sat(phi) = C0 - N_pos log phi + sum_i w_i log i1e(K_i / phi)
    #     T(phi)     = sum_i z_i (i0e(z_i) - i1e(z_i)) / i1e(z_i),  z = K / phi
    #
    # with C0 = sum_i (log 2 + log w_i - log(y_i)/2).  ``i1e``/``i0e`` are the
    # d->d Cephes Chebyshev evaluators, not the dd->d AMOS ``ive`` the density
    # evaluator's overflow fallback reaches for.
    bessel_scale: NDArray | None = field(default=None, repr=False, compare=False)
    bessel_log_constant: float = field(default=0.0, repr=False, compare=False)
    _saturated_cache: dict[float, float] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _saturated_score_cache: dict[float, float] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @property
    def positive_size(self) -> float:
        """Return the likelihood size the positive rows carry."""
        if self.weight_semantics == PRIOR_WEIGHTS:
            return float(self.n_positive)
        assert self.row_weight is not None
        return float(np.sum(self.row_weight, dtype=np.float64))

    def _row_total(self, values: NDArray) -> float:
        """Sum per-row contributions the way this contract accumulates them."""
        if self.weight_semantics == PRIOR_WEIGHTS:
            return float(np.sum(values, dtype=np.float64))
        assert self.row_weight is not None
        return float(np.sum(self.row_weight * values, dtype=np.float64))

    def _bessel_saturated_log_likelihood(self, phi: float) -> float | None:
        """Closed-form saturated log-likelihood at p = 1.5, or None.

        Returns None whenever the closed form is unavailable (any other power)
        or unrepresentable at this ``phi``, in which case the caller keeps the
        general Wright-Bessel route and its ``FloatingPointError`` contract.
        """
        scale = self.bessel_scale
        if scale is None:
            return None
        with np.errstate(all="ignore"):
            argument = scale / phi
            scaled_bessel_one = i1e(argument)
            log_scaled = np.log(scaled_bessel_one)
            if not np.all(np.isfinite(log_scaled)):
                return None
            value = float(
                self.bessel_log_constant
                - self.positive_size * float(np.log(phi))
                + self._row_total(log_scaled)
            )
        return value if np.isfinite(value) else None

    def _bessel_saturated_score(self, phi: float) -> tuple[float, float] | None:
        """Closed-form ``(T(phi), l_sat(phi))`` at p = 1.5, or None.

        ``T = d(-l_sat)/d log phi``.  Both come out of one ``i1e`` pass, so the
        value is returned alongside and cross-fills the value cache for free.
        The large-argument rule is the density evaluator's, verbatim, so this
        path introduces no score behaviour of its own.
        """
        scale = self.bessel_scale
        if scale is None:
            return None
        with np.errstate(all="ignore"):
            argument = scale / phi
            scaled_bessel_one = i1e(argument)
            log_scaled = np.log(scaled_bessel_one)
            if not np.all(np.isfinite(log_scaled)):
                return None
            scaled_bessel_zero = i0e(argument)
            score_component = (
                argument * (scaled_bessel_zero - scaled_bessel_one) / scaled_bessel_one
            )
            large_argument = np.isfinite(argument) & (
                argument >= _P15_BESSEL_ASYMPTOTIC_MIN_ARGUMENT
            )
            asymptotic_score = ~np.isfinite(score_component) & large_argument
            if np.any(asymptotic_score):
                inverse_z = 1.0 / argument[asymptotic_score]
                score_component[asymptotic_score] = (
                    0.5
                    + 3.0 * inverse_z / 8.0
                    + 3.0 * np.square(inverse_z) / 8.0
                    + 63.0 * np.power(inverse_z, 3) / 128.0
                )
            if not np.all(np.isfinite(score_component)):
                return None
            score = self._row_total(score_component)
            value = float(
                self.bessel_log_constant
                - self.positive_size * float(np.log(phi))
                + self._row_total(log_scaled)
            )
        if not np.isfinite(score):
            return None
        return score, value

    def saturated_log_likelihood(self, phi: float) -> float:
        """Exact saturated log-likelihood at dispersion ``phi``.

        Zero rows contribute exactly zero and are not evaluated; the returned
        value is the positive-row sum through the adaptive Wright-Bessel density
        evaluation the Tweedie likelihood uses everywhere else (Dunn & Smyth
        2005; Wood, Pya & Saefken 2016, supplementary App. J), or -- at p = 1.5,
        where Wright's function reduces to a modified Bessel function in closed
        form -- through that reduction.  The two agree to 4e-13 relative across
        eleven decades of phi, and mpmath at 45 digits puts the Bessel form on
        the accurate side wherever they differ.
        """
        key = float(phi)
        cached = self._saturated_cache.get(key)
        if cached is not None:
            return cached
        value = self._bessel_saturated_log_likelihood(key)
        if value is None:
            from superglm.profiling.tweedie import _evaluate_tweedie_density

            evaluation = _evaluate_tweedie_density(self.prepared_positive, key)
            value = self._row_total(evaluation.logpdf)
        if not np.isfinite(value):
            raise FloatingPointError(
                f"Tweedie saturated log-likelihood is not finite at phi={key:g}"
            )
        self._saturated_cache[key] = value
        return value

    def saturated_nll_log_phi_score(self, phi: float) -> float | None:
        """Analytic d(-l_sat)/d(log phi) at ``phi``, or None if unavailable.

        The density evaluator's per-row log-phi score is the closed-form
        derivative of the negative log-density (it agrees with numerical
        differentiation of the log-density to ~1e-10 relative); it is
        unavailable only when a row's evaluation lands on a branch without
        an analytic score, in which case the caller falls back to
        differencing the criterion itself.
        """
        key = float(phi)
        cached = self._saturated_score_cache.get(key)
        if cached is not None:
            return cached if np.isfinite(cached) else None
        bessel = self._bessel_saturated_score(key)
        if bessel is not None:
            score_value, saturated_value = bessel
            if np.isfinite(saturated_value):
                self._saturated_cache.setdefault(key, saturated_value)
            self._saturated_score_cache[key] = score_value
            return score_value if np.isfinite(score_value) else None
        from superglm.profiling.tweedie import _evaluate_tweedie_density

        evaluation = _evaluate_tweedie_density(
            self.prepared_positive,
            key,
            compute_score=True,
        )
        # The score pass fills ``logpdf`` too and ``compute_score`` does not
        # touch it, so the saturated VALUE at this phi is already computed and
        # would otherwise be thrown away.  brentq returns its last evaluated
        # point, so the criterion evaluation at the polished optimum lands on
        # exactly one of these keys: cross-filling turns it into a cache hit
        # for the cost of one log-sum.  Only finite values are stored, so
        # ``saturated_log_likelihood``'s FloatingPointError contract is intact.
        saturated_value = self._row_total(evaluation.logpdf)
        if np.isfinite(saturated_value):
            self._saturated_cache.setdefault(key, saturated_value)
        if not evaluation.score_valid or evaluation.log_phi_score is None:
            self._saturated_score_cache[key] = float("nan")
            return None
        value = self._row_total(evaluation.log_phi_score)
        if not np.isfinite(value):
            self._saturated_score_cache[key] = float("nan")
            return None
        self._saturated_score_cache[key] = value
        return value


def prepare_tweedie_reml_scale_data(
    y: NDArray,
    sample_weight: NDArray,
    power: float,
    *,
    weight_semantics: str,
) -> TweedieScaleProfileData:
    """Validate rows once and hoist the phi-invariant Tweedie density state.

    Under ``"prior"`` the weight is an EDM precision (observation-specific
    dispersion ``phi / w``) and the prepared state applies it inside the
    density evaluation exactly as the fitted likelihood does.  Strictly
    positive weights are required there and only there: the compound-Poisson
    normalizer carries ``log w``, so ``w = 0`` is not a row with no
    information but an unevaluable density.

    Under ``"frequency"`` the density is the unit-weight one and the weight is
    a replication count applied outside it, so a zero weight is simply a row
    that appears no times and drops out with the rest of the arithmetic
    untouched.

    At ``power == 1.5`` the closed-form Bessel state is prepared alongside; see
    ``TweedieScaleProfileData``.  Every other power carries ``bessel_scale =
    None``.
    """
    from superglm.profiling.tweedie import _prepare_tweedie_density

    if weight_semantics not in (PRIOR_WEIGHTS, FREQUENCY_WEIGHTS):
        raise ValueError(
            f"weight_semantics must be 'prior' or 'frequency', got {weight_semantics!r}",
        )
    y = np.asarray(y, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if y.ndim != 1 or sample_weight.shape != y.shape or y.size == 0:
        raise ValueError("y and sample_weight must be one-dimensional with matching shape")
    if not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("Tweedie scale profiling requires finite non-negative y")
    if weight_semantics == PRIOR_WEIGHTS:
        if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight <= 0.0):
            raise ValueError("Tweedie scale profiling requires strictly positive prior weights")
    elif not np.all(np.isfinite(sample_weight)) or np.any(sample_weight < 0.0):
        raise ValueError("Tweedie scale profiling requires finite non-negative frequency weights")
    positive = y > 0.0
    if weight_semantics == FREQUENCY_WEIGHTS:
        positive = positive & (sample_weight > 0.0)
    n_positive = int(np.count_nonzero(positive))
    if n_positive == 0:
        raise ValueError(
            "Tweedie scale profiling requires at least one positive response; "
            "an all-zero response has no estimable dispersion"
        )
    y_positive = y[positive]
    weights_positive = sample_weight[positive]
    density_weights = (
        weights_positive if weight_semantics == PRIOR_WEIGHTS else np.ones_like(weights_positive)
    )
    prepared = _prepare_tweedie_density(
        y_positive,
        y_positive,
        float(power),
        weights=density_weights,
    )
    bessel_scale: NDArray | None = None
    bessel_log_constant = 0.0
    if float(power) == 1.5:
        # Associated exactly as the density evaluator's own p = 1.5 branch
        # associates its Bessel argument: (4 w) * sqrt(y), then / phi.  The
        # replication count never reaches the argument -- it multiplies the
        # unit-weight row's contribution instead -- so the frequency arm sends
        # w = 1 through here and carries the count in the constant and in the
        # per-row totals.
        bessel_scale = (4.0 * density_weights) * np.sqrt(y_positive)
        bessel_scale.setflags(write=False)
        row_constant = np.log(2.0) + np.log(density_weights) - 0.5 * np.log(y_positive)
        if weight_semantics == FREQUENCY_WEIGHTS:
            row_constant = weights_positive * row_constant
        bessel_log_constant = float(np.sum(row_constant, dtype=np.float64))
        if not np.isfinite(bessel_log_constant) or not np.all(np.isfinite(bessel_scale)):
            bessel_scale = None
            bessel_log_constant = 0.0
    return TweedieScaleProfileData(
        power=float(power),
        n_positive=n_positive,
        prepared_positive=prepared,
        weight_semantics=weight_semantics,
        row_weight=None if weight_semantics == PRIOR_WEIGHTS else weights_positive,
        bessel_scale=bessel_scale,
        bessel_log_constant=bessel_log_constant,
    )


def profile_tweedie_reml_scale(
    profile_data: TweedieScaleProfileData,
    penalized_deviance: float,
    penalty_nullity: float,
) -> ProfiledScaleTerm:
    """Profile Tweedie dispersion while retaining Wood's saturated likelihood.

    Minimizes the exact scale-dependent part of the Wood (2011) Eq. (4) /
    Wood, Pya & Saefken (2016) Sec. 3.3 criterion over ``log(phi)``:

        Q(phi) = Dp / (2 phi) - l_sat(phi) - (Mp / 2) log(2 pi phi)

    with ``l_sat`` the exact compound Poisson-gamma saturated log-likelihood
    (zero rows are an atom and contribute a phi-free 0; positive rows carry
    the Dunn-Smyth series normalizer). This replaces the Gaussian-shaped
    substitution ``0.5 (n - Mp) log(Dp)``, which charges every zero row a
    ``log phi`` the exact saturated likelihood does not contain and thereby
    overweights the deviance arm in proportion to the zero fraction.

    The solve is a bounded scalar minimization in ``log(phi)`` with an
    expanding bracket; the ``d(1/phi)/d(Dp)`` contract required by the outer
    REML Newton follows from implicit differentiation of the profile score,
    with the log-phi curvature measured by a central difference of ``Q``
    around the optimum (the same quantity the Gaussian and Gamma profilers
    obtain in closed form).
    """
    if not isinstance(profile_data, TweedieScaleProfileData):
        raise TypeError("profile_data must be TweedieScaleProfileData")
    penalized_deviance = float(penalized_deviance)
    penalty_nullity = float(penalty_nullity)
    if not np.isfinite(penalized_deviance) or penalized_deviance <= 0.0:
        raise ValueError("penalized_deviance must be positive and finite")
    if not np.isfinite(penalty_nullity) or penalty_nullity < 0.0:
        raise ValueError("penalty_nullity must be finite and non-negative")
    # Each positive row's saturated density decays like phi**(-1/(p-1)) as
    # phi grows (the Dunn-Smyth series is dominated by its single-event term,
    # whose weight carries phi**(-(alpha+1)) with alpha+1 = 1/(p-1); verified
    # numerically at p in {1.2, 1.5, 1.8} to 1e-6), NOT like 1/phi - assuming
    # the Gaussian-shaped 1/phi tail here is the same substitution this
    # profiler exists to remove, one level down. Q's upper-tail slope in
    # log(phi) is therefore N_pos/(p-1) - Mp/2, and a finite interior
    # optimum needs that positive.
    if 2.0 * profile_data.positive_size <= (profile_data.power - 1.0) * penalty_nullity:
        raise ValueError("Tweedie REML scale profile has no finite interior optimum")

    def criterion(log_phi: float) -> float:
        phi = float(np.exp(log_phi))
        return float(
            0.5 * penalized_deviance / phi
            - profile_data.saturated_log_likelihood(phi)
            - 0.5 * penalty_nullity * (_LOG_TWO_PI + log_phi)
        )

    log_phi_lo = -_TWEEDIE_LOG_PHI_WINDOW
    log_phi_hi = _TWEEDIE_LOG_PHI_WINDOW
    while True:
        solution = minimize_scalar(
            criterion,
            bounds=(log_phi_lo, log_phi_hi),
            method="bounded",
            options={"xatol": _TWEEDIE_LOG_PHI_XATOL},
        )
        log_phi = float(solution.x)
        if (
            log_phi - log_phi_lo < _TWEEDIE_LOG_PHI_EDGE_TOL
            and log_phi_lo > -_TWEEDIE_LOG_PHI_LIMIT
        ):
            log_phi_lo = max(log_phi_lo - _TWEEDIE_LOG_PHI_STEP, -_TWEEDIE_LOG_PHI_LIMIT)
            continue
        if log_phi_hi - log_phi < _TWEEDIE_LOG_PHI_EDGE_TOL and log_phi_hi < _TWEEDIE_LOG_PHI_LIMIT:
            log_phi_hi = min(log_phi_hi + _TWEEDIE_LOG_PHI_STEP, _TWEEDIE_LOG_PHI_LIMIT)
            continue
        break
    if (
        log_phi - log_phi_lo < _TWEEDIE_LOG_PHI_EDGE_TOL
        or log_phi_hi - log_phi < _TWEEDIE_LOG_PHI_EDGE_TOL
    ):
        raise FloatingPointError("Tweedie REML scale profile is not representable")

    # Polish the bounded minimizer to a root of the ANALYTIC profile score
    # S(u) = -Dp e^{-u}/2 + T(u) - Mp/2 (T = d(-l_sat)/d log phi, closed
    # form). Bounded Brent leaves O(xatol) placement freedom in WHERE inside
    # its final bracket it stops, and which side it stops on is decided by
    # late golden-section comparisons that can flip on machine-classed
    # summation rounding: measured placement scatter across trivially
    # equivalent solver configurations is ~2e-8 in log phi around a score
    # residual of 1e-13. Downstream consumers difference gradients built on
    # this optimum at O(1e-4) steps, amplifying that freedom ~2500x into
    # their comparisons. A root of the analytic score has no placement
    # freedom: cross-machine variation reduces to the score evaluation's
    # own rounding divided by the score slope. When the analytic score is
    # unavailable (saddlepoint rows without scores) the bounded minimizer
    # stands, with its documented tolerance.
    def profile_score(u: float) -> float | None:
        t_value = profile_data.saturated_nll_log_phi_score(float(np.exp(u)))
        if t_value is None:
            return None
        return -0.5 * penalized_deviance * float(np.exp(-u)) + t_value - 0.5 * penalty_nullity

    polish_window = 64.0 * _TWEEDIE_LOG_PHI_XATOL
    while polish_window <= 1e-2:
        bracket_lo = max(log_phi - polish_window, log_phi_lo)
        bracket_hi = min(log_phi + polish_window, log_phi_hi)
        score_bracket_lo = profile_score(bracket_lo)
        score_bracket_hi = profile_score(bracket_hi)
        if score_bracket_lo is None or score_bracket_hi is None:
            break
        if score_bracket_lo < 0.0 < score_bracket_hi:

            def bracketed_score(u: float) -> float:
                value = profile_score(u)
                if value is None:  # pragma: no cover - branch flip inside a tiny bracket
                    raise _ScoreUnavailableInBracketError
                return value

            try:
                log_phi = float(
                    brentq(
                        bracketed_score,
                        bracket_lo,
                        bracket_hi,
                        xtol=1e-15,
                        rtol=4.0 * np.finfo(np.float64).eps,
                        maxiter=100,
                    )
                )
            except _ScoreUnavailableInBracketError:  # pragma: no cover - see above
                pass
            break
        polish_window *= 8.0

    criterion_value = float(criterion(log_phi))
    # Profile curvature in log(phi). The deviance arm Dp/(2 phi) is analytic;
    # the saturated arm is a central difference of the family's ANALYTIC
    # log-phi score T(u) = sum d(-log f)/d(log phi), so d2(l_sat)/du2 = -T'(u)
    # and Q''(u) = Dp e^{-u}/2 + T'(u). Differencing an analytic first
    # derivative keeps evaluation noise eps at O(eps/step) in the curvature;
    # differencing the criterion value amplifies it by O(eps/step^2), which
    # is stack-sensitive (an older special-function stack's eps reached the
    # published d(1/phi)/d(Dp) at test-visible size). Truncation is
    # O(step^2 * T'''/6), a curvature relative error around 1e-6 at the 1e-3
    # step. The value-difference fallback below runs only when a saddlepoint
    # row carries no analytic score.
    score_step = _TWEEDIE_SCORE_CURVATURE_STEP
    score_hi = profile_data.saturated_nll_log_phi_score(float(np.exp(log_phi + score_step)))
    score_lo = profile_data.saturated_nll_log_phi_score(float(np.exp(log_phi - score_step)))
    if score_hi is not None and score_lo is not None:
        log_phi_curvature = 0.5 * penalized_deviance * float(np.exp(-log_phi)) + (
            score_hi - score_lo
        ) / (2.0 * score_step)
    else:
        step = _TWEEDIE_CURVATURE_STEP
        log_phi_curvature = (
            criterion(log_phi - step) - 2.0 * criterion_value + criterion(log_phi + step)
        ) / (step * step)
    if not np.isfinite(log_phi_curvature) or log_phi_curvature <= 0.0:
        raise FloatingPointError("Tweedie REML scale profile has non-positive curvature")

    phi = float(np.exp(log_phi))
    inverse_phi = float(np.exp(-log_phi))
    # At the optimum Q'(xi) = 0, so d2Q/d(log phi)2 = xi^2 * Q''(xi) with
    # xi = 1/phi, and implicit differentiation of the profile score gives
    # d(xi)/d(Dp) = -1/2 / Q''(xi). (For the Gaussian profile this reduces to
    # the closed form -(n - Mp)/Dp^2 published by profile_gaussian_reml_scale.)
    log_derivative_magnitude = float(np.log(0.5) - 2.0 * log_phi - np.log(log_phi_curvature))
    log_smallest = float(np.log(np.nextafter(0.0, 1.0)))
    if log_derivative_magnitude > float(np.log(np.finfo(np.float64).max)):
        raise FloatingPointError("Tweedie REML scale derivative is not representable")
    derivative = (
        -0.0
        if log_derivative_magnitude < log_smallest
        else float(-np.exp(log_derivative_magnitude))
    )
    if not np.isfinite(phi) or not np.isfinite(criterion_value):
        raise FloatingPointError("Tweedie REML scale profile produced a non-finite result")
    return ProfiledScaleTerm(
        phi=phi,
        inverse_phi=inverse_phi,
        criterion=criterion_value,
        d_inverse_phi_d_penalized_deviance=derivative,
    )


def prepare_reml_scale_data(
    distribution: Any,
    y: NDArray,
    sample_weight: NDArray,
    *,
    weight_semantics: str,
) -> tuple[
    float | None, float | None, GammaScaleProfileData | None, TweedieScaleProfileData | None
]:
    """Hoist a fit's scale-profiling state once, under the declared contract.

    Returns the Gaussian pair and the Gamma and Tweedie prepared states, of
    which at most one arm is populated.  Every REML optimizer prepares through
    here so that the contract enters the scale machinery in exactly one place
    and cannot be re-derived per evaluation.
    """
    from superglm.distributions import Gamma, Gaussian, Tweedie

    if isinstance(distribution, Gaussian):
        likelihood_size, saturated_log_weight = gaussian_reml_scale_terms(
            sample_weight,
            weight_semantics=weight_semantics,
        )
        return likelihood_size, saturated_log_weight, None, None
    if isinstance(distribution, Gamma):
        return (
            None,
            None,
            prepare_gamma_reml_scale_data(y, sample_weight, weight_semantics=weight_semantics),
            None,
        )
    if isinstance(distribution, Tweedie):
        return (
            None,
            None,
            None,
            prepare_tweedie_reml_scale_data(
                y,
                sample_weight,
                distribution.p,
                weight_semantics=weight_semantics,
            ),
        )
    return None, None, None, None


__all__ = [
    "GammaScaleProfileData",
    "ProfiledScaleTerm",
    "TweedieScaleProfileData",
    "gaussian_reml_scale_terms",
    "prepare_gamma_reml_scale_data",
    "prepare_reml_scale_data",
    "prepare_tweedie_reml_scale_data",
    "profile_gamma_reml_scale",
    "profile_gaussian_reml_scale",
    "profile_tweedie_reml_scale",
]
