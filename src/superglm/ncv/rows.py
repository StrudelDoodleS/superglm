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

**Rows without dense blocks.**  :class:`CentredRows` serves ``Z v``, ``Z' v``, ``Z' W Z`` and
``z' M z`` through the design's stored form, as the solver's own products run, and a set of rows
through a row subset of the design.  Setting one up has a fixed cost, so the dense blocks keep
every set of rows below ``COMPACT_ENTRIES`` entries (:func:`serves_compact`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._lookup_tables import lookup_plan, table_leverages
from superglm._parallel import resolve_max_memory
from superglm.distributions import Distribution, Gaussian, Poisson, clip_mu
from superglm.group_matrix import DesignMatrix
from superglm.links import IdentityLink, Link, LogLink, stabilize_eta
from superglm.solvers.centered_system import build_centered_system
from superglm.solvers.mode_score import centred_matvec, dense_centred_rmatvec, dense_columns

CHUNK_BYTES = 32 * 1024 * 1024
"""What one block of dense rows and its two products may hold, under the default allowance."""

COMPACT_ENTRIES = 1 << 18
"""Dense entries (rows times ``P``) from which a set of rows is served by compact products.

A row subset of the design and the solver's plan for it cost about what reading this many
dense entries does, whatever the rows; from here on the compact products are the cheaper route.
"""


def chunk_rows(width: int) -> int:
    """Rows per dense block: three ``rows x width`` float64 arrays, at least 256 rows.

    The block's budget is ``CHUNK_BYTES``, or a quarter of ``max_memory`` when that is smaller.
    """
    budget = min(CHUNK_BYTES, resolve_max_memory() // 4)
    return max(256, budget // (3 * 8 * max(width, 1)))


def cache_allowance() -> int:
    """Bytes a problem may keep between evaluations: a quarter of ``max_memory``.

    Separate from an evaluation's working blocks (``chunk_rows``) and from its results.
    """
    return resolve_max_memory() // 4


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
    """``(1, (x - hi) - lo)`` for ``rows``: dense, ``(len(rows), p + 1)``.

    A design without slopes (``p == 0``) has the intercept's column alone.
    """
    block = np.empty((len(rows), dm.p + 1), dtype=np.float64)
    block[:, 0] = 1.0
    if dm.p == 0:
        return block
    slopes = block[:, 1:]
    slopes[:] = dm.row_subset(rows).toarray()
    slopes -= centre_hi
    if centre_lo is not None:
        slopes -= centre_lo
    return block


def serves_compact(rows: int, width: int) -> bool:
    """Whether ``rows`` rows of ``width`` entries go through compact products, not dense blocks."""
    return rows * width >= COMPACT_ENTRIES


@dataclass(frozen=True)
class CentredRows:
    """The rows ``Z = (1, (x - hi) - lo)`` of a design, as products with its stored form.

    No row is made dense.  A ``DenseGroupMatrix`` column, whose entries its type does not
    bound, is centred row by row about the pair before its product; every other column takes
    its own product less the centre's (``mode_score.centred_matvec``), as the solver's
    predictor and right-hand side do.
    """

    dm: DesignMatrix
    centre_hi: NDArray[np.float64]
    centre_lo: NDArray[np.float64] | None

    @property
    def has_tables(self) -> bool:
        """Whether the design's groups are lookups and dense columns, with term-pair tables."""
        return lookup_plan(self.dm) is not None

    def subset(self, rows: NDArray[np.intp]) -> CentredRows:
        """The rows ``rows`` as a design of their own, about the same centre."""
        return CentredRows(self.dm.row_subset(rows), self.centre_hi, self.centre_lo)

    def matvec(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """``Z theta``, one entry per row."""
        slopes = theta[1:]
        level = float(theta[0])
        if self.centre_lo is not None:
            level -= float(self.centre_lo @ slopes)
        return centred_matvec(self.dm, slopes, self.centre_hi) + level

    def rmatvec(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """``Z' values``."""
        total = float(np.sum(values))
        slopes = self.dm.rmatvec(values) - self.centre_hi * total
        if self.centre_lo is not None:
            slopes -= self.centre_lo * total
        dense = dense_columns(self.dm)
        if dense.any():
            centred = dense_centred_rmatvec(self.dm, values, self.centre_hi, self.centre_lo)
            slopes[dense] = centred[dense]
        return np.concatenate(([total], slopes))

    def gram(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """``Z' diag(weights) Z`` for non-negative ``weights``.

        The solver's centred system (``build_centered_system``) is the Gram ``G_m`` about the
        weights' own mean ``m``.  With ``s = Z' weights = (w, w (m - c))`` for the centre
        ``c``, the Gram about ``c`` has ``s`` as its intercept row and column and
        ``G_m + s_1 s_1' / w`` on the slopes: a sum of two positive semidefinite terms.
        """
        width = self.dm.p + 1
        total = float(np.sum(weights))
        if not total > 0.0:
            return np.zeros((width, width))
        about_mean = build_centered_system(
            dm=self.dm,
            W=weights,
            z_off=np.zeros(self.dm.n),
            penalty=np.zeros((self.dm.p, self.dm.p)),
        ).data_gram
        first = self.rmatvec(weights)
        gram = np.empty((width, width))
        gram[0] = gram[:, 0] = first
        gram[1:, 1:] = about_mean + np.outer(first[1:], first[1:]) / first[0]
        return gram

    def quadratic_forms(
        self, matrix: NDArray[np.float64], active: NDArray[np.bool_]
    ) -> NDArray[np.float64] | None:
        """``z' matrix z`` per row for a symmetric ``matrix``, or ``None`` without lookup tables.

        A design whose groups are lookups or dense columns reads the slopes' part back from
        its term-pair tables (``_lookup_tables.table_leverages``), whatever the rank of
        ``matrix``; the intercept's part is one product.  Rows outside ``active`` are not
        read and hold that product alone.
        """
        plan = lookup_plan(self.dm)
        if plan is None:
            return None
        slopes = table_leverages(plan, matrix[1:, 1:], self.centre_hi, self.centre_lo, active)
        return slopes + self.matvec(np.concatenate(([matrix[0, 0]], 2.0 * matrix[1:, 0])))


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

    @property
    def quadratic(self) -> bool:
        """Whether ``l`` is quadratic in ``eta``, so that ``l''`` is the weight at every fit."""
        return not self.log_mean

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
