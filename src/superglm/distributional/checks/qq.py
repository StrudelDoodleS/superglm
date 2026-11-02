"""Q-Q plot of the quantile residuals against a simulated envelope.

Randomised quantile residuals (Dunn and Smyth 1996, *JCGS* 5(3)) are standard
normal under a correct family, so their order statistics plot on
``Phi^{-1}((i - 0.5) / n)``. The envelope simulates ``n_sim`` responses from
the fit, transforms and sorts them, and reads the pointwise 2.5–97.5 percent
band (Fasiolo, Nedellec, Goude and Wood 2020, *JCGS* 29(1)), on a seeded
subsample above ``max_points`` rows. ``parameter_uncertainty=True`` draws
``theta`` from the posterior per replicate, widening the band.

Simulated responses (:func:`~superglm.distributional.posterior.posterior_predictive`)
are transformed through the law they were drawn from, the row's own
prior-weighted one, and atoms take ``u ~ U(F(y-), F(y))`` exactly as
:mod:`superglm.distributional.residuals` does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import special

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional._common import json_floats, readonly_float
from superglm.distributional.family import AtomFamily, DistributionFunctionFamily

# ``_unvalidated_offset_shapes`` and ``_take_unvalidated_offsets`` are the fit's
# own offset rules, and the residual payload was built through them; restating
# them here is how an offset contract drifts between two entry points.
from superglm.distributional.model import (
    _take_unvalidated_offsets,
    _unvalidated_offset_shapes,
)
from superglm.distributional.posterior import posterior_predictive
from superglm.distributional.residuals import (
    ResidualSet,
    _residual_rng,
    _sample_residuals,
    _validate_residual_evaluation,
)

#: Pointwise 95 % envelope percentiles per order statistic.
_LOWER_PERCENTILE = 2.5
_UPPER_PERCENTILE = 97.5
#: The simulated transform is clipped as :mod:`superglm.distributional.residuals` clips it.
_PROBABILITY_FLOOR = 1.0e-12
_PROBABILITY_CEILING = 1.0 - _PROBABILITY_FLOOR
#: Simulated ``(draw, row)`` cells transformed in one block.
_BLOCK_CELLS = 4_000_000


def order_statistic_grid(n: int) -> NDArray[np.float64]:
    """Return the ``n`` standard-normal plotting positions ``Phi^{-1}((i - 0.5) / n)``."""
    count = int(n)
    if count < 1:
        raise ValueError("an order-statistic grid needs at least one point")
    return special.ndtri((np.arange(count, dtype=np.float64) + 0.5) / count)


@dataclass(frozen=True)
class QQPayload:
    """Sorted quantile residuals against the normal grid, with a simulated band."""

    theoretical: NDArray[np.float64]
    observed: NDArray[np.float64]
    envelope_lower: NDArray[np.float64]
    envelope_upper: NDArray[np.float64]
    n_sim: int
    n_rows: int
    subsampled: bool
    seed: int
    kind: str = "qq"
    schema_version: int = 1

    def __post_init__(self) -> None:
        grid = readonly_float(self.theoretical)
        if grid.ndim != 1 or len(grid) < 1:
            raise ValueError("theoretical must be a non-empty one-dimensional grid")
        for name in ("observed", "envelope_lower", "envelope_upper"):
            curve = readonly_float(getattr(self, name))
            if curve.shape != grid.shape:
                raise ValueError(f"{name} must carry one value per theoretical grid point")
            object.__setattr__(self, name, curve)
        object.__setattr__(self, "theoretical", grid)
        object.__setattr__(self, "n_sim", int(self.n_sim))
        object.__setattr__(self, "n_rows", int(self.n_rows))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "subsampled", bool(self.subsampled))

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe payload of lists, numbers, strings and ``None``."""
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "n_rows": self.n_rows,
            "n_sim": self.n_sim,
            "seed": self.seed,
            "subsampled": self.subsampled,
            "theoretical": json_floats(self.theoretical),
            "observed": json_floats(self.observed),
            "envelope_lower": json_floats(self.envelope_lower),
            "envelope_upper": json_floats(self.envelope_upper),
        }


def _simulated_envelope(
    fitted: Any,
    frame: EagerFrame,
    residuals: ResidualSet,
    envelope_rows: NDArray[np.intp],
    *,
    n_sim: int,
    parameter_uncertainty: bool,
    seed: int,
    offsets: Mapping[str, NDArray] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the pointwise band of ``n_sim`` simulated residual vectors."""
    family = fitted.family
    theta = np.asarray(residuals.theta[envelope_rows], dtype=np.float64)
    row_weights = np.asarray(residuals.prior_weights[envelope_rows], dtype=np.float64)
    weighted = not bool(np.all(row_weights == 1.0))
    width = len(envelope_rows)
    simulated = posterior_predictive(
        fitted,
        as_eager_frame(frame.take_rows(envelope_rows)),
        n_sim,
        parameter_uncertainty=parameter_uncertainty,
        offsets=_take_unvalidated_offsets(
            _unvalidated_offset_shapes(offsets, residuals.n_rows), envelope_rows
        ),
        seed=seed,
        weights=row_weights,
    )

    # The atom draws take a third stream from ``seed``: the coefficient draws
    # consume the seed itself and the predictive uniforms its second child, so
    # a third child is what leaves all three independent.
    atom_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])
    sorted_draws = np.empty((n_sim, width), dtype=np.float64)
    block = max(1, _BLOCK_CELLS // width)
    for start in range(0, n_sim, block):
        stop = min(start + block, n_sim)
        stacked = np.tile(theta, (stop - start, 1))
        stacked_weights = np.tile(row_weights, stop - start) if weighted else None
        values = np.asarray(simulated[start:stop], dtype=np.float64).reshape(-1)
        transform = np.asarray(
            family.cdf_prior_weighted(values, stacked, stacked_weights)
            if weighted
            else family.cdf(values, stacked),
            dtype=np.float64,
        )
        if isinstance(family, AtomFamily):
            left = np.asarray(
                family.cdf_left_limit(values, stacked, weights=stacked_weights),
                dtype=np.float64,
            )
            atoms = left < transform
            transform = np.where(atoms, atom_rng.uniform(left, transform), transform)
        block_residuals = special.ndtri(
            np.clip(transform, _PROBABILITY_FLOOR, _PROBABILITY_CEILING)
        ).reshape(stop - start, width)
        sorted_draws[start:stop] = np.sort(block_residuals, axis=1)

    lower, upper = np.percentile(sorted_draws, [_LOWER_PERCENTILE, _UPPER_PERCENTILE], axis=0)
    return lower, upper


def qq_payload(
    fitted: Any,
    residuals: ResidualSet,
    *,
    n_sim: int = 100,
    max_points: int = 50_000,
    seed: int = 42,
    parameter_uncertainty: bool = False,
    X: FrameLike | EagerFrame | None = None,
    offsets: Mapping[str, NDArray] | None = None,
) -> QQPayload:
    """Return the Q-Q payload of ``residuals`` with a simulated envelope.

    ``X`` and ``offsets`` must be the design the residuals were computed on.
    Rows are those :func:`superglm.distributional.residuals.replication_sample`
    gives. Above ``max_points`` rows the envelope uses a seeded subsample and
    the observed curve is interpolated onto its grid (``subsampled`` records it).
    """
    if not isinstance(residuals, ResidualSet):
        raise TypeError("residuals must be a ResidualSet")
    family = fitted.family
    if not isinstance(family, DistributionFunctionFamily):
        raise NotImplementedError(
            "a Q-Q envelope needs a family with a distribution function; this one has none"
        )
    simulations = int(n_sim)
    if simulations < 1:
        raise ValueError("n_sim must simulate at least one residual vector")
    cap = int(max_points)
    if cap < 1:
        raise ValueError("max_points must allow at least one envelope point")
    if X is None:
        raise ValueError(
            "the envelope simulates responses for the fitted rows, so qq_payload needs the "
            "design frame X the residuals were computed on"
        )
    frame = as_eager_frame(X)
    if len(frame) != residuals.n_rows:
        raise ValueError(
            f"X must hold the same rows the residuals were computed on: it has {len(frame)} "
            f"rows against {residuals.n_rows} residual rows (zero-weight rows leave the "
            "residual set)"
        )

    if residuals.weight_semantics != fitted.fit_state.weight_contract.semantics:
        raise ValueError("residuals must come from the same rows and deterministic evaluation")
    _validate_residual_evaluation(
        residuals,
        fitted,
        frame,
        residuals.y,
        sample_weight=(
            residuals.weights
            if residuals.weight_semantics == "frequency"
            else residuals.prior_weights
        ),
        offsets=offsets,
        bounds=residuals.bounds,
    )
    if parameter_uncertainty and residuals.bounds is not None:
        # The envelope simulates complete responses; its parameter spread is
        # not that of truncated or censored observations.
        raise NotImplementedError(
            "parameter_uncertainty=True simulates complete responses, so its envelope does not "
            "describe truncated or censored rows; use the default parameter_uncertainty=False, "
            "whose envelope does"
        )
    sample = _sample_residuals(residuals, seed=seed)
    rows = sample.rows
    n_rows = len(rows)
    observed = np.sort(sample.quantile)
    theoretical = order_statistic_grid(n_rows)

    subsampled = n_rows > cap
    envelope_rows = rows
    if subsampled:
        chosen = _residual_rng(seed, 2).choice(n_rows, size=cap, replace=False)
        envelope_rows = rows[np.sort(chosen)]
        grid = order_statistic_grid(cap)
        observed = np.interp(grid, theoretical, observed)
        theoretical = grid

    lower, upper = _simulated_envelope(
        fitted,
        frame,
        residuals,
        envelope_rows,
        n_sim=simulations,
        parameter_uncertainty=bool(parameter_uncertainty),
        seed=int(_residual_rng(seed, 3).integers(0, 2**63)),
        offsets=offsets,
    )
    return QQPayload(
        theoretical=theoretical,
        observed=observed,
        envelope_lower=lower,
        envelope_upper=upper,
        n_sim=simulations,
        n_rows=n_rows,
        subsampled=subsampled,
        seed=int(seed),
    )


__all__ = ["QQPayload", "order_statistic_grid", "qq_payload"]
