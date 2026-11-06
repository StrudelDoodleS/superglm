"""Histogram of the probability-integral transform with its consistency band.

``u = F(y | theta_hat)`` is uniform under a calibrated forecast (Gneiting,
Balabdaoui and Raftery 2007, *JRSS B* 69(2)). Flat is calibrated; a U means
too little spread (scale predictor low); a central hump too much; a slope a
biased location; one end alone a tail (shape) failure. The pointwise band is
the ``alpha / 2`` and ``1 - alpha / 2`` quantiles of
``Binomial(n, 1 / n_bins)``, so about ``alpha`` of bins leave it on a
calibrated model; a shape across bins is the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.distributional._common import readonly, readonly_float
from superglm.distributional.residuals import ResidualSet, _sample_residuals

#: Row set for a check without a seed of its own: the replication default.
_REPLICATION_SEED = 42


@dataclass(frozen=True)
class PITPayload:
    """Binned probability-integral transform against the uniform band."""

    edges: NDArray[np.float64]
    counts: NDArray[np.int64]
    expected: float
    band_lower: float
    band_upper: float
    n_bins: int
    n_rows: int
    kind: str = "pit"
    schema_version: int = 1

    def __post_init__(self) -> None:
        bins = int(self.n_bins)
        if bins < 1:
            raise ValueError("n_bins must be at least one bin")
        counts = readonly(self.counts, np.int64)
        edges = readonly_float(self.edges)
        if counts.shape != (bins,) or edges.shape != (bins + 1,):
            raise ValueError("counts must hold one count per bin and edges one more edge")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "n_bins", bins)
        object.__setattr__(self, "n_rows", int(self.n_rows))
        for name in ("expected", "band_lower", "band_upper"):
            object.__setattr__(self, name, float(getattr(self, name)))

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe payload of lists, numbers, strings and ``None``."""
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "n_bins": self.n_bins,
            "n_rows": self.n_rows,
            "expected": self.expected,
            "band_lower": self.band_lower,
            "band_upper": self.band_upper,
            "edges": [float(edge) for edge in self.edges],
            "counts": [int(count) for count in self.counts],
        }


def pit_payload(residuals: ResidualSet, *, n_bins: int = 20, alpha: float = 0.05) -> PITPayload:
    """Return the PIT histogram of ``residuals`` with its ``1 - alpha`` uniform band.

    Rows are the ones
    :func:`superglm.distributional.residuals.replication_sample` gives, so a
    frequency-weighted row contributes its count of observations to the
    histogram and to the band's ``n``, exactly as literal replication would.
    """
    from scipy import stats

    if not isinstance(residuals, ResidualSet):
        raise TypeError("residuals must be a ResidualSet")
    bins = int(n_bins)
    if bins < 1:
        raise ValueError("n_bins must be at least one bin")
    level = float(alpha)
    if not 0.0 < level < 1.0:
        raise ValueError("alpha must lie strictly inside (0, 1)")

    sample = _sample_residuals(residuals, seed=_REPLICATION_SEED)
    counts, edges = np.histogram(sample.pit, bins=bins, range=(0.0, 1.0))
    n_rows = len(sample.rows)
    lower, upper = stats.binom.ppf([level / 2.0, 1.0 - level / 2.0], n_rows, 1.0 / bins)
    return PITPayload(
        edges=edges,
        counts=counts,
        expected=n_rows / bins,
        band_lower=lower,
        band_upper=upper,
        n_bins=bins,
        n_rows=n_rows,
    )


__all__ = ["PITPayload", "pit_payload"]
