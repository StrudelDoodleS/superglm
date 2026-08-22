"""Orthogonal polynomial feature, orthonormalized against the training weights.

Stable alternative to P-splines for features with simple monotone or
quadratic shapes.  Group lasso selects or removes the entire polynomial
as a unit.  Degree 2-3 is the typical insurance choice.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.linalg as la
from numpy.polynomial.legendre import legvander
from numpy.typing import NDArray

from superglm.types import GroupInfo

# Rank guard threshold: a diagonal entry of R below this fraction of the
# largest diagonal entry means the weighted measure cannot identify that
# degree component.  Refuse loudly rather than let QR-based least squares
# regularize the fit silently (Brubeck, Nakatsukasa & Trefethen 2021).
_RANK_RTOL = 1e-10


class Polynomial:
    """Orthogonal polynomial feature (orthonormal in the training-weight measure).

    Scales x to [-1, 1] using training-data min/max, seeds a Legendre
    basis (including the constant), and orthonormalizes it against the
    training ``sample_weight`` by weighted thin QR.  The constant column
    is dropped from the emitted block — the model intercept carries it.

    Normalization convention: with training weights ``w``, the emitted
    components ``phi_j`` (one per stated power) satisfy

        (1 / sum(w)) * sum_i w_i * phi_j(x_i) * phi_k(x_i) = delta_jk

    i.e. they are orthonormal in the *mean* empirical inner product of the
    training ``sample_weight``, and each is orthogonal to the constant in
    the same inner product.  When exposure enters through an offset (the
    documented count workflow) ``sample_weight`` stays at ones, so the
    basis is orthonormalized against the row-count measure.  The weights
    are followed under either weight contract — including ``"prior"``, where
    they are precisions: orthonormalization is inference/selection geometry
    (the spanned column space is weight-invariant), not model geometry,
    so the spline physical-rows rule deliberately does not apply.  Under
    Gaussian/fixed-weight fitting this makes the per-power coefficient
    estimates exactly uncorrelated, and it gives the group penalty the
    within-group orthonormal geometry that the group lasso assumes (Yuan
    & Lin 2006, JRSS-B 68:49-67; Simon & Tibshirani 2012, Statistica
    Sinica 22(3):983-1001 — orthonormalizing within the group is exactly
    equivalent to their standardized group lasso).

    The triangular factor of the weighted QR is stored as fitted state
    beside the min/max scaling.  ``transform``/``score`` push new x
    through the same seed basis and the *stored* factor — they never
    re-orthogonalize against new data or new weights.  Out-of-range x is
    plain polynomial evaluation on the scaled seed basis (unbounded
    growth); every orthogonality and uncorrelatedness statement is a
    property of the *training* measure only.

    Honest caveats:

    - Exact uncorrelatedness of the per-power estimates holds under the
      training weights (the fixed-weight/Gaussian world).  In a GLM it is
      approximate at the IRLS working weights; no published result
      quantifies that gap, and published group-penalty practice makes the
      same fixed spherical approximation of the working Hessian (Simon &
      Tibshirani 2012, section 5.3).
    - Dropping powers by their z-statistics is response-driven selection.
      Validate out-of-fold, or state ``powers`` from the plan.

    Parameters
    ----------
    degree : int, optional
        Maximum polynomial degree; sugar for ``powers=range(1, degree+1)``.
        2 (quadratic) or 3 (cubic) are the standard insurance choices.
        Defaults to 3 when neither ``degree`` nor ``powers`` is given.
        Mutually exclusive with *powers*.
    powers : sequence of int, optional
        Distinct integers >= 1 naming the orthogonal components to keep,
        e.g. ``powers=[1, 2, 4]``.  The orthogonal basis is built up to
        ``max(powers)`` and the stated components are selected, so under
        fixed weights dropping a middle power leaves the retained
        components' fitted coefficients unchanged.  Excluding "power 3"
        excludes the degree-3 *orthogonal component*, not the raw ``x**3``
        monomial — on asymmetric exposure the degree-4 orthogonal
        polynomial carries ``x**3`` monomial content.  API precedent for
        a degree list: ``numpy.polynomial.Polynomial.fit(deg=[...])``.

    Notes
    -----
    Group size = ``len(powers)``.  Columns are emitted in ascending power
    order and summary rows are labelled by the stated power.

    The QR-on-Legendre-seed build is the published standardized-group-
    lasso algorithm and is well conditioned at degree <= 8 on min/max-
    scaled data.  If the degree ceiling ever rises, the upgrade path is
    the three-term recurrence for weighted discrete measures (Forsythe
    1957; Gautschi 2004, *Orthogonal Polynomials: Computation and
    Approximation*): store the recurrence coefficients instead of the
    triangular factor and evaluate new x by re-running the recurrence.

    The rank guard admits pivots down to 1e-10 of the largest (which is
    exactly 1 — the constant column under normalized weights), so a build
    just clearing it has cond(R) near 1e10 and its computed columns are
    orthonormal only to roughly eps * cond(R) ~ 1e-6: near the guard
    boundary the orthonormality and uncorrelatedness statements hold to
    that reduced precision, not machine precision.
    """

    def __init__(
        self,
        degree: int | None = None,
        powers: Sequence[int] | None = None,
    ):
        if degree is not None and powers is not None:
            raise ValueError("Pass either degree= or powers=, not both.")
        if powers is None:
            if degree is None:
                degree = 3
            if isinstance(degree, bool) or not isinstance(degree, int | np.integer):
                raise ValueError(f"degree must be an integer, got {degree!r}")
            if degree < 1:
                raise ValueError(f"degree must be >= 1, got {degree}")
            resolved = tuple(range(1, int(degree) + 1))
        else:
            resolved = _validate_powers(powers)
        self.powers: tuple[int, ...] = resolved
        self.degree: int = resolved[-1]
        self._lo: float = 0.0
        self._hi: float = 1.0
        self._R: NDArray | None = None

    def __repr__(self) -> str:
        if self.powers == tuple(range(1, self.degree + 1)):
            return f"Polynomial(degree={self.degree})"
        return f"Polynomial(powers={list(self.powers)})"

    def __setstate__(self, state: dict) -> None:
        # Pre-0.22 pickles predate the data-orthogonal basis: no stored
        # factor, no powers.  Default them so restore succeeds and
        # _components can refuse with a migration message instead of an
        # AttributeError.
        state = dict(state)
        state.setdefault("_R", None)
        if "powers" not in state:
            state["powers"] = tuple(range(1, int(state.get("degree", 3)) + 1))
        state.setdefault("degree", max(state["powers"]))
        self.__dict__.update(state)

    def _scale(self, x: NDArray) -> NDArray:
        """Scale x to [-1, 1] using stored min/max."""
        span = self._hi - self._lo
        if span < 1e-12:
            return np.zeros_like(x)
        return 2.0 * (x - self._lo) / span - 1.0

    def _seed_basis(self, x_scaled: NDArray) -> NDArray:
        """Legendre seed for degrees 0..degree (constant column included)."""
        return legvander(x_scaled, self.degree)

    def _components(self, seed: NDArray) -> NDArray:
        """Push a seed-basis matrix through the stored triangular factor.

        Returns the stated powers' orthonormal components; never
        re-orthogonalizes.  The constant component (column 0) is dropped.
        """
        if getattr(self, "_R", None) is None:
            raise ValueError(
                f"{self!r} is not fitted: no stored orthonormalization factor. "
                "Call build() first — or, if this spec was restored from a model "
                "fitted before the data-orthogonal basis (0.22.0), refit the "
                "model to migrate."
            )
        full = la.solve_triangular(self._R, seed.T, trans="T", lower=False).T
        return np.ascontiguousarray(full[:, list(self.powers)])

    def build(
        self,
        x: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> GroupInfo:
        """Learn min/max and the weighted orthonormalization from *x*."""
        x = np.asarray(x, dtype=np.float64).ravel()
        w = (
            np.ones_like(x)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64).ravel()
        )
        if w.shape != x.shape:
            raise ValueError(
                f"{self!r}: sample_weight length {w.shape[0]} != x length {x.shape[0]}"
            )
        if np.any(w < 0):
            raise ValueError(f"{self!r}: sample_weight must be nonnegative.")

        # Distinct-support guard: a discrete measure with r distinct
        # positive-weight support points identifies orthogonal polynomials
        # only up to degree r - 1.
        n_distinct = np.unique(x[w > 0]).size
        if n_distinct <= self.degree:
            raise ValueError(
                f"{self!r} needs more than {self.degree} distinct x values with "
                f"positive weight, got {n_distinct}: the weighted data cannot "
                f"identify a degree-{self.degree} orthogonal component."
            )

        # Scale bounds come from the positive-weight support only: a
        # zero-weight outlier must not stretch the [-1, 1] mapping and
        # degrade the seed's conditioning.
        x_support = x[w > 0]
        self._lo, self._hi = float(x_support.min()), float(x_support.max())
        seed = self._seed_basis(self._scale(x))

        # Weighted thin QR of the seed under the mean-normalized weights.
        # Rows are presorted by descending weight so unpivoted Householder
        # QR stays backward stable under extreme weight ratios (Powell &
        # Reid 1969; Cox & Higham 1998).
        w_norm = w / float(w.sum())
        order = np.argsort(-w, kind="stable")
        R = np.linalg.qr(np.sqrt(w_norm[order])[:, None] * seed[order], mode="r")
        signs = np.sign(np.diag(R))
        signs[signs == 0.0] = 1.0
        R = R * signs[:, None]

        # Rank guard on the pivots: refuse rather than silently regularize.
        diag = np.abs(np.diag(R))
        bad = np.flatnonzero(diag <= _RANK_RTOL * diag.max())
        if bad.size:
            raise ValueError(
                f"{self!r}: weighted basis is numerically rank-deficient "
                f"(pivot ratio {diag[bad[0]] / diag.max():.2e} at degree "
                f"{int(bad[0])}); the training weights cannot support the "
                f"requested degrees."
            )

        self._R = R
        cols = self._components(seed)
        return GroupInfo(columns=cols, n_cols=len(self.powers))

    def transform(self, x: NDArray) -> NDArray:
        """Evaluate the fitted orthonormal components at new *x*.

        Pushes x through the stored min/max scaling, the Legendre seed,
        and the stored triangular factor.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        return self._components(self._seed_basis(self._scale(x)))

    def score(self, x: NDArray, beta: NDArray) -> NDArray:
        """Score the fitted polynomial contribution directly on new data."""
        return self.transform(x) @ np.asarray(beta, dtype=np.float64).ravel()

    def reconstruct(self, beta: NDArray, n_points: int = 200) -> dict[str, Any]:
        """Evaluate the fitted polynomial on a grid and return relativities."""
        x_grid = np.linspace(self._lo, self._hi, n_points)
        log_rels = self.transform(x_grid) @ beta
        return {
            "x": x_grid,
            "log_relativity": log_rels,
            "relativity": np.exp(log_rels),
            "degree": self.degree,
            "powers": self.powers,
            "coefficients": beta,
        }


def _validate_powers(powers: Sequence[int]) -> tuple[int, ...]:
    """Validate a powers= sequence at construction time."""
    try:
        items = list(powers)
    except TypeError:
        raise ValueError(
            f"powers must be a sequence of distinct integers >= 1, got {powers!r}"
        ) from None
    if not items:
        raise ValueError("powers must contain at least one power.")
    cleaned: list[int] = []
    for p in items:
        if isinstance(p, bool) or not isinstance(p, int | np.integer):
            raise ValueError(f"powers must be integers >= 1, got {p!r}")
        if p < 1:
            raise ValueError(f"powers must be >= 1, got {p}")
        cleaned.append(int(p))
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"powers must be distinct, got {sorted(cleaned)}")
    return tuple(sorted(cleaned))
