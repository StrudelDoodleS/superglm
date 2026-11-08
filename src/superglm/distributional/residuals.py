"""PIT and randomised quantile residuals for a fitted distributional model.

``u_i = F(y_i | theta_hat_i)`` and ``r_i = Phi^{-1}(u_i)``, the randomised
quantile residual of Dunn and Smyth (1996), *JCGS* 5(3); under a correct model
``u`` is uniform and ``r`` standard normal (calibration reading: Gneiting,
Balabdaoui and Raftery 2007, *JRSS B* 69(2)). A point mass takes
``u ~ U(F(y-), F(y))``, counted in ``randomised_rows``; ``u`` is clipped into
``[1e-12, 1 - 1e-12]`` before the inverse, and the clipped count is reported
as misfit. A truncated row with window ``(a, b]`` and ``D = F(b) - F(a)`` has
``u = (F(y) - F(a)) / D``; a right-censored row at ``c`` is uniform on
``((F(c) - F(a)) / D, 1)`` and a left-censored one on
``(0, (F(c) - F(a)) / D)``, both randomised.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import special

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional._common import json_floats, readonly_float
from superglm.distributional.family import (
    AtomFamily,
    DistributionFunctionFamily,
    PriorWeightedDistributionFunctionFamily,
    ResponseWindowFamily,
    bounded_observation,
    plan_observation,
)

# The response and offset shapes are checked before zero-weight rows are
# selected away, exactly as the fit checks them; restating those rules here is
# how a contract drifts between two entry points that have to agree.
from superglm.distributional.model import (
    _unvalidated_response_shape,
)
from superglm.distributional.observation import (
    LEFT_CENSORED,
    RIGHT_CENSORED,
    BoundPair,
    CensoredRows,
    ResponseBounds,
    resolve_response_bounds,
    unvalidated_bound_arrays,
)
from superglm.distributional.offsets import prediction_offsets
from superglm.distributional.weights import resolve_likelihood_weights

#: A residual is the transform itself or its normal inverse.
ResidualKind = Literal["pit", "quantile"]

#: ``u`` is clipped into this closed interval before the normal inverse.
_PROBABILITY_FLOOR = 1.0e-12
_PROBABILITY_CEILING = 1.0 - _PROBABILITY_FLOOR
#: Row cap on the literal replication a frequency-weighted check expands into.
_MAX_REPLICATION_ROWS = 100_000


def _validated_kind(kind: str) -> ResidualKind:
    if kind not in ("pit", "quantile"):
        raise ValueError(f"kind must be 'pit' or 'quantile'; got {kind!r}")
    return kind


@dataclass(frozen=True)
class ResidualSet:
    """The transform, its normal inverse, and everything a check needs beside them.

    ``weights`` are the replication weights the checks group by (frequency
    counts, or ones under the prior contract); ``prior_weights`` are the row
    law's own weights (ones under the frequency contract). Zero-weight rows
    are absent. ``pit_lower`` and ``pit_upper``, when present, are the
    unclipped bounds ``F(y-)`` and ``F(y)``; ``bounds`` are the resolved
    truncation and censoring of the retained rows, or ``None``. Everything
    describes physical rows.
    """

    pit: NDArray[np.float64]
    quantile: NDArray[np.float64]
    theta: NDArray[np.float64]
    eta: NDArray[np.float64]
    y: NDArray[np.float64]
    weights: NDArray[np.float64]
    prior_weights: NDArray[np.float64]
    clipped_rows: int
    randomised_rows: int
    weight_semantics: str
    pit_lower: NDArray[np.float64] | None = None
    pit_upper: NDArray[np.float64] | None = None
    bounds: ResponseBounds | None = None

    def __post_init__(self) -> None:
        pit = readonly_float(self.pit)
        if pit.ndim != 1 or len(pit) < 1:
            raise ValueError("pit, quantile, y and both weight vectors give one value per row")
        n_rows = len(pit)
        rowwise = {
            "quantile": self.quantile,
            "y": self.y,
            "weights": self.weights,
            "prior_weights": self.prior_weights,
        }
        for name, values in rowwise.items():
            array = readonly_float(values)
            if array.shape != (n_rows,):
                raise ValueError("pit, quantile, y and both weight vectors give one value per row")
            object.__setattr__(self, name, array)
        for name in ("theta", "eta"):
            array = readonly_float(getattr(self, name))
            if array.ndim != 2 or array.shape[0] != n_rows or array.shape[1] < 1:
                raise ValueError(f"{name} must be a (rows, parameters) parameter matrix")
            object.__setattr__(self, name, array)
        for name in ("weights", "prior_weights"):
            array = getattr(self, name)
            if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("clipped_rows", "randomised_rows"):
            count = int(getattr(self, name))
            if count < 0 or count > n_rows:
                raise ValueError(f"clipped_rows and randomised_rows must be row counts of {n_rows}")
            object.__setattr__(self, name, count)
        if self.weight_semantics not in ("prior", "frequency"):
            raise ValueError("weight_semantics must be 'prior' or 'frequency'")
        if (self.pit_lower is None) != (self.pit_upper is None):
            raise ValueError("pit_lower and pit_upper must be supplied together")
        if self.pit_lower is not None:
            lower, upper = readonly_float(self.pit_lower), readonly_float(self.pit_upper)
            if (
                lower.shape != (n_rows,)
                or upper.shape != (n_rows,)
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower < 0.0)
                or np.any(upper > 1.0)
                or np.any(lower > upper)
            ):
                raise ValueError("CDF intervals must give finite ordered bounds in [0, 1] per row")
            object.__setattr__(self, "pit_lower", lower)
            object.__setattr__(self, "pit_upper", upper)
        object.__setattr__(self, "pit", pit)

    @property
    def n_rows(self) -> int:
        """Rows carried by these residuals, after any zero-weight row left."""
        return len(self.pit)

    def values(self, kind: ResidualKind = "quantile") -> NDArray[np.float64]:
        """Return the transform (``"pit"``) or its normal inverse (``"quantile"``)."""
        return self.pit if _validated_kind(kind) == "pit" else self.quantile

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe payload; every non-finite value becomes ``None``."""
        return {
            "n_rows": int(self.n_rows),
            "n_parameters": int(self.theta.shape[1]),
            "pit": json_floats(self.pit),
            "quantile": json_floats(self.quantile),
            "theta": json_floats(self.theta),
            "eta": json_floats(self.eta),
            "y": json_floats(self.y),
            "weights": json_floats(self.weights),
            "prior_weights": json_floats(self.prior_weights),
            "clipped_rows": int(self.clipped_rows),
            "randomised_rows": int(self.randomised_rows),
            "weight_semantics": str(self.weight_semantics),
            "pit_lower": None if self.pit_lower is None else json_floats(self.pit_lower),
            "pit_upper": None if self.pit_upper is None else json_floats(self.pit_upper),
        }


def compute_residuals(
    fitted: Any,
    X: FrameLike | EagerFrame,
    y: NDArray,
    *,
    sample_weight: NDArray | None = None,
    offsets: Mapping[str, NDArray] | None = None,
    seed: int = 42,
    kind: ResidualKind = "quantile",
    truncation: BoundPair = None,
    censored: CensoredRows = None,
) -> ResidualSet:
    """Return the residual payload of ``y`` under the fit's plug-in parameters.

    ``kind`` is validated here; the payload always carries both ``pit`` and
    ``quantile``. Frequency weights leave the transform alone and travel on
    the payload for :func:`replication_sample`. Prior weights are validated as
    the fit validates them and read through the family's prior-weighted
    distribution function; a family without one refuses. ``truncation`` and
    ``censored`` are read as ``SuperLSS.fit`` reads them, and a censored row
    is randomised.
    """
    _validated_kind(kind)
    evaluation = _evaluate_residuals(
        fitted,
        X,
        y,
        sample_weight=sample_weight,
        offsets=offsets,
        truncation=truncation,
        censored=censored,
    )
    lower, upper = evaluation.pit_lower, evaluation.pit_upper
    atoms = lower < upper
    u = np.where(atoms, _residual_rng(seed, 0).uniform(lower, upper), upper)
    clipped_rows = int(np.count_nonzero((u < _PROBABILITY_FLOOR) | (u > _PROBABILITY_CEILING)))
    u = np.clip(u, _PROBABILITY_FLOOR, _PROBABILITY_CEILING)
    return ResidualSet(
        pit=u,
        quantile=special.ndtri(u),
        clipped_rows=clipped_rows,
        randomised_rows=int(np.count_nonzero(atoms)),
        **evaluation._asdict(),
    )


class _ResidualEvaluation(NamedTuple):
    y: NDArray
    eta: NDArray
    theta: NDArray
    weights: NDArray
    prior_weights: NDArray
    weight_semantics: str
    pit_lower: NDArray
    pit_upper: NDArray
    bounds: ResponseBounds | None


def _bounded_intervals(
    values: NDArray,
    lower: NDArray,
    upper: NDArray,
    bounds: ResponseBounds,
    log_window: Callable[[NDArray, NDArray, NDArray], NDArray],
) -> tuple[NDArray, NDArray]:
    """Map the parent CDF interval of each bounded row to its truncated or censored one.

    Probabilities are ratios of tail-safe log window probabilities; a window
    with no fitted mass puts the row at the top, where the clipped count shows it.
    """
    new_lower = np.array(lower, dtype=np.float64, copy=True)
    new_upper = np.array(upper, dtype=np.float64, copy=True)
    rows = np.union1d(bounds.truncated_rows(), bounds.censored_rows())
    a, b, y = bounds.lower[rows], bounds.upper[rows], np.asarray(values)[rows]
    log_mass = log_window(a, b, rows)
    has_mass = np.isfinite(log_mass)
    with np.errstate(invalid="ignore"):
        edge = np.where(has_mass, np.exp(log_window(a, y, rows) - log_mass), 1.0)
    side = bounds.side[rows]
    # An observed row's interval collapses to its edge; a censored row's
    # interval runs from its edge to the end beyond its bound.
    new_lower[rows] = np.where(has_mass & (side == LEFT_CENSORED), 0.0, edge)
    new_upper[rows] = np.where(has_mass & (side == RIGHT_CENSORED), 1.0, edge)
    return np.clip(new_lower, 0.0, 1.0), np.clip(new_upper, 0.0, 1.0)


def _evaluate_residuals(
    fitted: Any,
    X: FrameLike | EagerFrame,
    y: NDArray,
    *,
    sample_weight: NDArray | None = None,
    offsets: Mapping[str, NDArray] | None = None,
    truncation: BoundPair = None,
    censored: CensoredRows = None,
    bounds: ResponseBounds | None = None,
) -> _ResidualEvaluation:
    """Evaluate the retained row law without randomization or retained designs.

    Held-out responses must be finite after zero-weight selection.
    ``truncation`` and ``censored`` resolve as the fit resolves them;
    ``bounds``, the retained rows' already resolved bounds, replaces both.
    """
    family = fitted.family
    if not isinstance(family, DistributionFunctionFamily):
        raise NotImplementedError(
            "residuals need a family with a distribution function; this one has none"
        )

    frame = as_eager_frame(X)
    n_observations = len(frame)
    response = _unvalidated_response_shape(y, n_observations)
    original_bounds = unvalidated_bound_arrays(truncation, censored, n_observations)
    contract = fitted.fit_state.weight_contract
    resolved = resolve_likelihood_weights(
        sample_weight,
        n_observations=n_observations,
        contract=contract,
    )
    positions = resolved.input_positions
    resolved_offsets = prediction_offsets(
        fitted.fit_state.predictor_templates, frame, offsets, positions=positions
    )
    if len(positions) != n_observations:
        frame = as_eager_frame(frame.take_rows(positions))
        response = response[positions]
    values = np.asarray(response, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("retained residual responses must be finite")
    if bounds is None:
        bounds = resolve_response_bounds(
            values,
            None if original_bounds is None else tuple(a[positions] for a in original_bounds),
            positions,
        )

    prior = contract.semantics == "prior"
    resolved_weights = np.asarray(resolved.values, dtype=np.float64)
    unit = np.ones(len(values), dtype=np.float64)
    if prior or bounds is not None:
        # The family owns the refusal: this is the call the fit makes, and the
        # families that cannot represent a prior weight or bounded rows raise
        # from it.  The plan returns the bounds on the family's support.
        plan = family.bind_likelihood(values, resolved, bounded_observation(bounds))
        bounds = plan_observation(plan).bounds
    if prior:
        weights, prior_weights = unit, resolved_weights
    else:
        weights, prior_weights = resolved_weights, unit

    eta = np.asarray(
        fitted._predict_eta_resolved(fitted.prediction_coefficients, frame, resolved_offsets),
        dtype=np.float64,
    )
    theta = np.asarray(
        fitted.parameters_from_eta(eta),
        dtype=np.float64,
    )

    prior_law = None if not prior or resolved.provenance.all_unit else resolved.values
    if prior_law is not None and not isinstance(family, PriorWeightedDistributionFunctionFamily):
        raise NotImplementedError(
            f"{type(family).__name__} has no prior-weighted distribution function, so the "
            "row law under non-unit prior weights is unavailable; fit with unit weights, "
            "declare frequency semantics, or implement "
            "PriorWeightedDistributionFunctionFamily"
        )

    upper = np.asarray(
        family.cdf(values, theta)
        if prior_law is None
        else family.cdf_prior_weighted(values, theta, prior_law),
        dtype=np.float64,
    )

    lower = upper
    if isinstance(family, AtomFamily):
        # A point mass makes F discontinuous at y, and only a uniform draw
        # across the jump keeps the transform uniform (Dunn and Smyth 1996).
        lower = np.asarray(
            family.cdf_left_limit(values, theta, weights=prior_law),
            dtype=np.float64,
        )
    if bounds is not None:
        if not isinstance(family, ResponseWindowFamily) or isinstance(family, AtomFamily):
            raise NotImplementedError(
                f"{type(family).__name__} has no tail-safe window probability for continuous "
                "rows, so residuals of its truncated or censored rows are unavailable"
            )

        def log_window(low: NDArray, high: NDArray, rows: NDArray) -> NDArray:
            return np.asarray(
                family.log_window_probability(
                    low, high, theta[rows], None if prior_law is None else prior_law[rows]
                ),
                dtype=np.float64,
            )

        lower, upper = _bounded_intervals(values, lower, upper, bounds, log_window)
    return _ResidualEvaluation(
        theta=theta,
        eta=eta,
        y=values,
        weights=weights,
        prior_weights=prior_weights,
        weight_semantics=contract.semantics,
        pit_lower=lower,
        pit_upper=upper,
        bounds=bounds,
    )


def _validate_residual_evaluation(
    residuals: ResidualSet,
    fitted: Any,
    X: FrameLike | EagerFrame,
    y: NDArray,
    *,
    sample_weight: NDArray | None = None,
    offsets: Mapping[str, NDArray] | None = None,
    truncation: BoundPair = None,
    censored: CensoredRows = None,
    bounds: ResponseBounds | None = None,
) -> None:
    """Allow reuse of an equivalent deterministic evaluation, independent of seed.

    This checks the residual transform, not covariance or design identity.
    ``bounds`` are the retained rows' resolved bounds when ``X`` and ``y`` are
    already the retained rows, as in the Q-Q envelope.
    """
    current = _evaluate_residuals(
        fitted,
        X,
        y,
        sample_weight=sample_weight,
        offsets=offsets,
        truncation=truncation,
        censored=censored,
        bounds=bounds,
    )
    for name, expected in current._asdict().items():
        stored = getattr(residuals, name)
        if name == "bounds":
            if stored != expected:
                raise ValueError(
                    "residuals must come from the same rows and deterministic evaluation"
                )
            continue
        # Legacy continuous payloads do not carry intervals. Their transform
        # is still checked below against the current deterministic CDF.
        if stored is None and name in ("pit_lower", "pit_upper"):
            continue
        if not np.array_equal(stored, expected):
            raise ValueError("residuals must come from the same rows and deterministic evaluation")
    lower = np.clip(current.pit_lower, _PROBABILITY_FLOOR, _PROBABILITY_CEILING)
    upper = np.clip(current.pit_upper, _PROBABILITY_FLOOR, _PROBABILITY_CEILING)
    if (
        not np.all((residuals.pit >= lower) & (residuals.pit <= upper))
        or not np.array_equal(residuals.quantile, special.ndtri(residuals.pit))
        or residuals.randomised_rows != np.count_nonzero(current.pit_lower < current.pit_upper)
    ):
        raise ValueError("residuals must come from the same rows and deterministic evaluation")


def _residual_rng(seed: int, domain: int) -> np.random.Generator:
    """Independent streams: physical atoms, sample atoms, QQ selection/envelope,
    binned bootstrap and reliability bootstrap use domains 0 through 5.
    The legacy replication selection retains the unmodified seed stream.
    """
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(domain,)))


class _ResidualSample(NamedTuple):
    rows: NDArray[np.intp]
    pit: NDArray[np.float64]
    quantile: NDArray[np.float64]


def _sample_residuals(
    residuals: ResidualSet,
    *,
    max_rows: int = _MAX_REPLICATION_ROWS,
    seed: int = 42,
) -> _ResidualSample:
    """Return aligned physical rows and occurrence PIT/quantile values.

    Integer frequency counts below the cap expand literally; above it (or for
    fractional counts) rows are a weighted mixture with fresh atom draws.
    Prior and continuous observations keep their stored transforms. Selection
    and sampled atoms use independent streams of ``seed``.
    """
    rows = replication_sample(residuals, max_rows=max_rows, seed=seed)
    pit, quantile = residuals.pit[rows], residuals.quantile[rows]
    if residuals.weight_semantics == "frequency" and np.any(residuals.weights != 1.0):
        if residuals.pit_lower is None:
            if residuals.randomised_rows:
                raise ValueError("frequency atom replication needs retained CDF intervals")
        else:
            lower, upper = residuals.pit_lower[rows], residuals.pit_upper[rows]
            atoms = lower < upper
            pit[atoms] = np.clip(
                _residual_rng(seed, 1).uniform(lower[atoms], upper[atoms]),
                _PROBABILITY_FLOOR,
                _PROBABILITY_CEILING,
            )
            quantile[atoms] = special.ndtri(pit[atoms])
    return _ResidualSample(rows, pit, quantile)


def residual_values(
    fitted: Any,
    X: FrameLike | EagerFrame,
    y: NDArray,
    *,
    kind: ResidualKind = "quantile",
    **kwargs: Any,
) -> NDArray[np.float64]:
    """Return just the residual array of ``kind``, discarding the rest of the payload."""
    return compute_residuals(fitted, X, y, kind=kind, **kwargs).values(kind)


def replication_sample(
    residuals: ResidualSet,
    *,
    max_rows: int = _MAX_REPLICATION_ROWS,
    seed: int = 42,
) -> NDArray[np.intp]:
    """Return the row indices a row-level check should read.

    Integer frequency weights become literal repeated rows; above ``max_rows``
    (and for fractional weights) a seeded weight-proportional sample stands in.
    Under the prior contract this is the identity.
    """
    if not isinstance(residuals, ResidualSet):
        raise TypeError("residuals must be a ResidualSet")
    cap = int(max_rows)
    if cap < 1:
        raise ValueError("max_rows must be at least one row")

    weights = residuals.weights
    n_rows = len(weights)
    counts = np.rint(weights)
    total = float(np.sum(weights, dtype=np.float64))
    if total <= cap and np.array_equal(weights, counts):
        return np.repeat(np.arange(n_rows, dtype=np.intp), counts.astype(np.intp))
    generator = np.random.default_rng(seed)
    size = cap if total > cap else n_rows
    chosen = generator.choice(n_rows, size=size, replace=True, p=weights / total)
    return np.asarray(chosen, dtype=np.intp)


__all__ = [
    "ResidualKind",
    "ResidualSet",
    "compute_residuals",
    "replication_sample",
    "residual_values",
]
