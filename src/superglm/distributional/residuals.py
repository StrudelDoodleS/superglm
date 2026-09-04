"""PIT and randomised quantile residuals for a fitted distributional model.

One transform serves every fitted family that implements a distribution
function: the probability-integral transform ``u_i = F(y_i | theta_hat_i)``
and its normal inverse ``r_i = Phi^{-1}(u_i)``, the randomised quantile
residual of Dunn and Smyth (1996), *Journal of Computational and Graphical
Statistics* 5(3), 236-244. Under a correctly specified supported family ``u``
is uniform and ``r`` is standard normal, which is what lets one Q-Q envelope,
one worm plot and one PIT histogram check a Gaussian, a gamma and a
generalised Pareto alike; the calibration reading of the uniform histogram is
Gneiting, Balabdaoui and Raftery (2007), *Journal of the Royal Statistical
Society B* 69(2), 243-268.

Two departures from the plain transform are recorded rather than hidden.  A
family with a point mass takes the randomised construction ``u ~ U(F(y-),
F(y))`` over the atom, which is what makes the transform uniform on a
discontinuous CDF, and the number of randomised rows is on the payload.  A row
the fitted model gives essentially no mass has ``u`` clipped into
``[1e-12, 1 - 1e-12]`` before the inverse; the clipped count is a misfit
statement about the fit, not a rounding note, so it travels with the residuals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import special

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    AtomFamily,
    DistributionFunctionFamily,
    PriorWeightedDistributionFunctionFamily,
)

# The response and offset shapes are checked before zero-weight rows are
# selected away, exactly as the fit checks them; restating those rules here is
# how a contract drifts between two entry points that have to agree.
from superglm.distributional.model import (
    _take_unvalidated_offsets,
    _unvalidated_offset_shapes,
    _unvalidated_response_shape,
)
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


def _readonly(values: NDArray, *, dtype: type = np.float64) -> NDArray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _json_values(values: NDArray) -> list[Any]:
    """Return nested lists of floats in which every non-finite value is ``None``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim > 1:
        return [_json_values(row) for row in array]
    return [None if not np.isfinite(value) else float(value) for value in array]


@dataclass(frozen=True)
class ResidualSet:
    """The transform, its normal inverse, and everything a check needs beside them.

    ``weights`` are the *replication* weights the checks group and average by:
    the declared counts under the frequency contract and ones under the prior
    contract, where the weight has already entered each row's own distribution
    function instead.  ``prior_weights`` are that row law's own weights -- the
    resolved prior weights under the prior contract and ones under the
    frequency one -- so a builder that has to simulate or invert the row's
    distribution can hand them straight to the posterior primitive.  Rows whose
    weight is zero are not here at all -- they leave the diagnostics the way
    they leave the likelihood.
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

    def __post_init__(self) -> None:
        pit = _readonly(self.pit)
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
            array = _readonly(values)
            if array.shape != (n_rows,):
                raise ValueError("pit, quantile, y and both weight vectors give one value per row")
            object.__setattr__(self, name, array)
        for name in ("theta", "eta"):
            array = _readonly(getattr(self, name))
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
            "pit": _json_values(self.pit),
            "quantile": _json_values(self.quantile),
            "theta": _json_values(self.theta),
            "eta": _json_values(self.eta),
            "y": _json_values(self.y),
            "weights": _json_values(self.weights),
            "prior_weights": _json_values(self.prior_weights),
            "clipped_rows": int(self.clipped_rows),
            "randomised_rows": int(self.randomised_rows),
            "weight_semantics": str(self.weight_semantics),
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
) -> ResidualSet:
    """Return the residual payload of ``y`` under the fit's plug-in parameters.

    ``kind`` names the residual the caller is after and is validated here; the
    payload always carries both, because ``quantile`` is one normal inverse
    away from ``pit`` and every check wants one of the two.

    Weights follow the contract the model was fitted under.  Frequency weights
    are literal replication, so they leave the transform alone and travel on
    the payload for :func:`replication_sample` to expand.  Prior weights are
    part of the row's own law: the family validates them exactly as the fit
    does -- the families that refuse prior weights raise from that call -- and
    a non-unit prior weight is then read through the family's prior-weighted
    distribution function.  A family without one refuses rather than quietly
    inverting the unit-weight distribution.
    """
    _validated_kind(kind)
    family = fitted.family
    if not isinstance(family, DistributionFunctionFamily):
        raise NotImplementedError(
            "residuals need a family with a distribution function; this one has none"
        )

    frame = as_eager_frame(X)
    n_observations = len(frame)
    response = _unvalidated_response_shape(y, n_observations)
    contract = fitted.fit_state.weight_contract
    resolved = resolve_likelihood_weights(
        sample_weight,
        n_observations=n_observations,
        contract=contract,
    )
    positions = resolved.input_positions
    resolved_offsets = offsets
    if len(positions) != n_observations:
        frame = as_eager_frame(frame.take_rows(positions))
        response = response[positions]
        resolved_offsets = _take_unvalidated_offsets(
            _unvalidated_offset_shapes(offsets, n_observations),
            positions,
        )
    values = np.asarray(response, dtype=np.float64)

    prior = contract.semantics == "prior"
    resolved_weights = np.asarray(resolved.values, dtype=np.float64)
    unit = np.ones(len(values), dtype=np.float64)
    if prior:
        # The family owns the refusal: this is the call the fit makes, and the
        # families that cannot represent a prior weight raise from it.
        family.bind_likelihood(values, resolved, COMPLETE_OBSERVATION)
        weights, prior_weights = unit, resolved_weights
    else:
        weights, prior_weights = resolved_weights, unit

    eta = np.asarray(fitted.predict_eta(frame, offsets=resolved_offsets), dtype=np.float64)
    theta = np.asarray(
        fitted.predict_parameters(frame, offsets=resolved_offsets),
        dtype=np.float64,
    )

    prior_law = None if not prior or resolved.provenance.all_unit else resolved.values
    if prior_law is None:
        upper = np.asarray(family.cdf(values, theta), dtype=np.float64)
    else:
        if not isinstance(family, PriorWeightedDistributionFunctionFamily):
            raise NotImplementedError(
                f"{type(family).__name__} has no prior-weighted distribution function, so the "
                "row law under non-unit prior weights is unavailable; fit with unit weights, "
                "declare frequency semantics, or implement "
                "PriorWeightedDistributionFunctionFamily"
            )
        upper = np.asarray(family.cdf_prior_weighted(values, theta, prior_law), dtype=np.float64)

    randomised_rows = 0
    u = upper
    if isinstance(family, AtomFamily):
        # A point mass makes F discontinuous at y, and only a uniform draw
        # across the jump keeps the transform uniform (Dunn and Smyth 1996).
        lower = np.asarray(
            family.cdf_left_limit(values, theta, weights=prior_law),
            dtype=np.float64,
        )
        atoms = lower < upper
        randomised_rows = int(np.count_nonzero(atoms))
        u = np.where(atoms, np.random.default_rng(seed).uniform(lower, upper), upper)

    clipped_rows = int(np.count_nonzero((u < _PROBABILITY_FLOOR) | (u > _PROBABILITY_CEILING)))
    u = np.clip(u, _PROBABILITY_FLOOR, _PROBABILITY_CEILING)
    return ResidualSet(
        pit=u,
        quantile=special.ndtri(u),
        theta=theta,
        eta=eta,
        y=values,
        weights=weights,
        prior_weights=prior_weights,
        clipped_rows=clipped_rows,
        randomised_rows=randomised_rows,
        weight_semantics=contract.semantics,
    )


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

    Replication is what the frequency contract means, so integer weights become
    literal repeated rows.  Above ``max_rows`` repeated rows -- and for the
    fractional weights that have no literal expansion -- a seeded
    weight-proportional sample stands in for them, as the scalar diagnostics
    do.  Under the prior contract every weight is one and this is the identity.
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
