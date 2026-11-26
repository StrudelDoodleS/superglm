"""Rows for the neighbourhood cross-validation engine: centred design rows and the row loss.

**Centred rows.**  The engine's coordinates are ``theta = (alpha, beta)`` with rows
``z = (1, (x - hi) - lo)``, where ``(hi, lo)`` is the pair the fit's Hessian was centred about
(``CenteredSystem.centre_pair``).  Rows are centred by that pair, in that order of operations, as
``reml._leverage_rows`` centres them: the difference then rounds at a column's spread and not at
``u |mean|``, so a column far from zero (a year, a timestamp) keeps the rows consistent with the
Hessian they are solved against.

**The row loss.**  ``l(eta) = w d(y, mu(eta)) / 2``, half the weighted unit deviance, the loss
PIRLS minimises; the scored loss is ``D = 2 l``.  For the two canonical pairs the derivatives in
``eta`` are direct formulas, from ``d = (y - mu)^2`` with ``mu = eta`` and from
``d = 2 [y log(y / mu) - (y - mu)]`` with ``mu = exp(eta)``:

    Gaussian/identity   l' = w (eta - y)    l'' = w       l''' = 0
    Poisson/log         l' = w (mu - y)     l'' = w mu    l''' = w mu

A row of zero weight is an exact zero in each and its response is never read.  The mean of a
predictor is the one ``predict`` reports: the inverse link of the stabilised predictor, clipped to
the family's range (``links.stabilize_eta``, ``distributions.clip_mu``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._parallel import resolve_max_memory
from superglm.distributions import Distribution, Gaussian, Poisson, clip_mu
from superglm.group_matrix import DesignMatrix
from superglm.links import IdentityLink, Link, LogLink, stabilize_eta

CHUNK_BYTES = 32 * 1024 * 1024
"""What one block of dense rows and its two products may hold, under the default allowance."""


def chunk_rows(width: int) -> int:
    """Rows per dense block: three ``rows x width`` float64 arrays, at least 256 rows.

    The block's budget is ``CHUNK_BYTES``, or a quarter of ``max_memory`` when that is smaller.
    """
    budget = min(CHUNK_BYTES, resolve_max_memory() // 4)
    return max(256, budget // (3 * 8 * max(width, 1)))


def row_chunks(rows: NDArray[np.intp], width: int) -> list[NDArray[np.intp]]:
    """``rows`` in consecutive pieces of at most ``chunk_rows(width)``."""
    size = chunk_rows(width)
    return [rows[start : start + size] for start in range(0, len(rows), size)]


def augmented_rows(
    dm: DesignMatrix,
    rows: NDArray[np.intp],
    centre_hi: NDArray[np.float64],
    centre_lo: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """``(1, (x - hi) - lo)`` for ``rows``: dense, ``(len(rows), p + 1)``."""
    block = np.empty((len(rows), dm.p + 1), dtype=np.float64)
    block[:, 0] = 1.0
    slopes = block[:, 1:]
    slopes[:] = dm.row_subset(rows).toarray()
    slopes -= centre_hi
    if centre_lo is not None:
        slopes -= centre_lo
    return block


@dataclass(frozen=True)
class RowLoss:
    """The half-deviance row loss of Gaussian/identity (``log_mean`` False) or Poisson/log."""

    distribution: Distribution
    link: Link
    log_mean: bool

    @property
    def has_third_derivative(self) -> bool:
        """Whether ``l'''`` is not identically zero."""
        return self.log_mean

    def mean(self, eta: NDArray[np.float64]) -> NDArray[np.float64]:
        """The mean ``predict`` reports for the predictor ``eta``."""
        return np.asarray(
            clip_mu(self.link.inverse(stabilize_eta(eta, self.link)), self.distribution),
            dtype=np.float64,
        )

    def off_scale(self, eta: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Rows whose reported mean is not the inverse link of ``eta``: not finite, or clipped."""
        return ~np.isfinite(eta) | (stabilize_eta(eta, self.link) != eta)

    def _residual(
        self, y: NDArray[np.float64], eta: NDArray[np.float64], mu: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return (mu if self.log_mean else eta) - y

    def derivatives(
        self, y: NDArray[np.float64], w: NDArray[np.float64], eta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """``(l', l'', l''')`` at ``eta``; a row of zero weight is zero in all three."""
        score = np.zeros(len(eta), dtype=np.float64)
        curvature = np.zeros(len(eta), dtype=np.float64)
        on = w > 0.0
        mu = self.mean(eta[on])
        score[on] = w[on] * self._residual(y[on], eta[on], mu)
        curvature[on] = w[on] * mu if self.log_mean else w[on]
        third = curvature.copy() if self.log_mean else np.zeros(len(eta), dtype=np.float64)
        return score, curvature, third

    def scored(
        self, y: NDArray[np.float64], w: NDArray[np.float64], eta: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], bool]:
        """``(mean, D, D', usable)`` of rows scored at the predictor ``eta``.

        ``D = w d(y, mean)`` and ``D' = 2 l'`` are zero on a row of zero weight, whose mean is
        still reported.  ``usable`` is False when a weighted row's predictor is off the scale on
        which ``D'`` is the derivative of ``D`` (:meth:`off_scale`).
        """
        mean = self.mean(eta)
        deviance = np.zeros(len(eta), dtype=np.float64)
        slope = np.zeros(len(eta), dtype=np.float64)
        on = w > 0.0
        if bool(np.any(self.off_scale(eta[on]))):
            return mean, deviance, slope, False
        deviance[on] = w[on] * self.distribution.deviance_unit(y[on], mean[on])
        slope[on] = 2.0 * w[on] * self._residual(y[on], eta[on], mean[on])
        return mean, deviance, slope, True


def row_loss(distribution: Distribution, link: Link) -> RowLoss:
    """The row loss of a supported pair; any other pair is refused."""
    pair = (type(distribution), type(link))
    if pair == (Gaussian, IdentityLink):
        return RowLoss(distribution, link, log_mean=False)
    if pair == (Poisson, LogLink):
        return RowLoss(distribution, link, log_mean=True)
    raise ValueError(
        "Neighbourhood cross-validation supports a Gaussian family with an identity link and a "
        f"Poisson family with a log link, not {pair[0].__name__} with {pair[1].__name__}; "
        "choose the smoothing of this model with fit_reml."
    )
