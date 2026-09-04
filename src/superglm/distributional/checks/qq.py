"""Q-Q plot of the quantile residuals against a simulated envelope.

Under a correctly specified family the randomised quantile residuals of Dunn
and Smyth (1996), *Journal of Computational and Graphical Statistics* 5(3),
236-244, are standard normal, so their order statistics plot on the line
``Phi^{-1}((i - 0.5) / n)``.  What that plot needs to be readable is a
statement of how far the order statistics wander under the fitted model itself,
and the envelope is that statement: ``n_sim`` response vectors simulated from
the fit, each passed through the same probability-integral transform and
sorted, give the pointwise 2.5 and 97.5 percent band per order statistic.  That
is the simulation envelope of Fasiolo, Nedellec, Goude and Wood (2020),
*Journal of Computational and Graphical Statistics* 29(1), 78-86, which also
supplies the subsampling that keeps the construction affordable on a long book:
above ``max_points`` rows the envelope is built on a seeded subsample and the
observed curve is interpolated onto its grid, so the figure stays the same size
whatever the row count.

``parameter_uncertainty=False`` -- the default -- simulates at the plug-in
``theta_hat``, and the band is then the pure order-statistic band of the fitted
model.  ``True`` draws ``theta`` from the coefficient posterior per replicate,
which widens the band by the estimation uncertainty of the fit.

The simulated responses come from :func:`superglm.distributional.posterior.posterior_predictive`
and are mapped back through the law they were drawn from: simulating one law
and transforming through another would not produce a uniform transform, and the
band would then be wrong rather than merely approximate.  That law is the row's
own one.  A prior weight -- an exposure on a burn-cost model -- is part of the
row's distribution, so a policy at a fifth of a year is simulated and inverted
on its own law through ``quantile_prior_weighted`` and ``cdf_prior_weighted``,
the same pair the observed residuals read.  A family with a point mass takes
the randomised construction ``u ~ U(F(y-), F(y))`` here exactly as
:mod:`superglm.distributional.residuals` takes it, because a band drawn from
one transform is no band for order statistics of another.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import special

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.distributional.family import AtomFamily, DistributionFunctionFamily

# ``_unvalidated_offset_shapes`` and ``_take_unvalidated_offsets`` are the fit's
# own offset rules, and the residual payload was built through them; restating
# them here is how an offset contract drifts between two entry points.
from superglm.distributional.model import (
    _take_unvalidated_offsets,
    _unvalidated_offset_shapes,
)
from superglm.distributional.posterior import posterior_predictive
from superglm.distributional.residuals import ResidualSet, replication_sample

#: Pointwise 95 % envelope percentiles per order statistic.
_LOWER_PERCENTILE = 2.5
_UPPER_PERCENTILE = 97.5
#: The simulated transform is clipped as :mod:`superglm.distributional.residuals` clips it.
_PROBABILITY_FLOOR = 1.0e-12
_PROBABILITY_CEILING = 1.0 - _PROBABILITY_FLOOR
#: Simulated ``(draw, row)`` cells transformed in one block.
_BLOCK_CELLS = 4_000_000


def _readonly(values: NDArray) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _json_numbers(values: NDArray) -> list[float | None]:
    """Emit a float list with every non-finite entry as ``null``."""
    return [None if not np.isfinite(value) else float(value) for value in np.asarray(values)]


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
        grid = _readonly(self.theoretical)
        if grid.ndim != 1 or len(grid) < 1:
            raise ValueError("theoretical must be a non-empty one-dimensional grid")
        for name in ("observed", "envelope_lower", "envelope_upper"):
            curve = _readonly(getattr(self, name))
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
            "theoretical": _json_numbers(self.theoretical),
            "observed": _json_numbers(self.observed),
            "envelope_lower": _json_numbers(self.envelope_lower),
            "envelope_upper": _json_numbers(self.envelope_upper),
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

    ``X`` and ``offsets`` are the design the residuals were computed on: the
    envelope simulates responses for those same rows, so passing a different
    frame would band a different model.  Rows are the ones
    :func:`superglm.distributional.residuals.replication_sample` gives, which is
    the identity under the prior-weight contract and literal replication under
    the frequency one.  Above ``max_points`` rows the envelope is built on a
    seeded subsample of them and the observed curve is interpolated onto the
    subsample's grid; ``subsampled`` records that this happened.
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

    rows = replication_sample(residuals, seed=seed)
    n_rows = len(rows)
    observed = np.sort(residuals.quantile[rows])
    theoretical = order_statistic_grid(n_rows)

    subsampled = n_rows > cap
    envelope_rows = rows
    if subsampled:
        chosen = np.random.default_rng(seed).choice(n_rows, size=cap, replace=False)
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
        seed=int(seed),
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
