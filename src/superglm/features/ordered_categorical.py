"""OrderedCategorical feature: ordered categories with spline or step basis.

Actuarial pricing data frequently contains continuous variables that have been
pre-binned into ordered categories (e.g. age bands "18-25", "26-35", ...).
This feature type respects the ordering with two modes:

- **spline**: map categories to numeric values, build a spline on those values
- **step**: one-hot encode with a first-difference penalty (D1'D1) so adjacent
  categories are soft-fused
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm.features.categorical import _validate_categorical_levels
from superglm.types import GroupInfo


class OrderedCategorical:
    """Ordered categorical feature with spline or step basis.

    Parameters
    ----------
    values : dict[str, float] or None
        Explicit mapping from category labels to numeric values (e.g.
        midpoints). Mutually exclusive with ``order``.
    order : list[str] or None
        Ordered list of category labels. Numeric values are generated as
        ``linspace(0, 1, len(order))``. Mutually exclusive with ``values``.
    basis : {"spline", "step"}
        "spline" maps categories to numeric values and builds a spline.
        "step" one-hot encodes with a first-difference penalty.
    kind : str
        Spline type (spline mode only). Passed to ``Spline(kind=...)``.
        ``"bs"`` (default), ``"cr"``, ``"ns"``, etc.
    base : str
        Reference level selection for step mode and spline identifiability.
        "most_exposed" (default), "first", or a specific level name.
    n_knots : int
        Number of interior knots (spline mode only). Auto-clamped to
        ``n_levels - 1`` if too large.
    degree : int
        B-spline degree (spline mode only).
    select : bool
        Enable mgcv-style double penalty (spline mode only).
    penalty : str
        Penalty type for internal spline ("ssp" or "none").
    """

    def __init__(
        self,
        values: dict[str, float] | None = None,
        order: list[str] | None = None,
        basis: str = "spline",
        kind: str = "bs",
        base: str = "most_exposed",
        n_knots: int = 5,
        degree: int = 3,
        select: bool = False,
        penalty: str = "ssp",
    ):
        if values is not None and order is not None:
            raise ValueError("Specify exactly one of 'values' or 'order', not both.")
        if values is None and order is None:
            raise ValueError("Must specify either 'values' or 'order'.")
        if basis not in ("spline", "step"):
            raise ValueError(f"basis must be 'spline' or 'step', got {basis!r}")
        if basis == "step" and select:
            raise ValueError("select=True is not supported with basis='step'.")

        self.basis = basis
        self.kind = kind
        self.base = base
        self.select = select
        self.penalty = penalty
        self.degree = degree
        self.n_knots = n_knots
        self._ordered_levels: list[str] = []

        # Derive ordered levels and numeric values
        if values is not None:
            sorted_items = sorted(values.items(), key=lambda kv: kv[1])
            self._ordered_levels = [k for k, _ in sorted_items]
            self._level_to_value = dict(values)
        else:
            self._ordered_levels = list(order)
            n = len(order)
            vals = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])
            self._level_to_value = dict(zip(order, vals.tolist()))

        self._known_levels = set(self._ordered_levels)
        self._n_levels = len(self._ordered_levels)

        # Step mode state
        self._base_level: str = ""
        self._non_base: list[str] = []
        self._R_inv: NDArray | None = None

        # Spline mode: create internal spline (deferred until we know n_levels)
        self._spline = None
        if self.basis == "spline":
            self._init_spline()

    def __repr__(self) -> str:
        n = self._n_levels
        return f"OrderedCategorical(basis={self.basis!r}, {n} levels, n_knots={self.n_knots})"

    def _init_spline(self) -> None:
        """Create the internal Spline object for spline mode."""
        from superglm.features.spline import Spline

        effective_n_knots = min(self.n_knots, self._n_levels - 1)
        if effective_n_knots < self.n_knots:
            warnings.warn(
                f"OrderedCategorical: n_knots={self.n_knots} clamped to "
                f"{effective_n_knots} (n_levels - 1 = {self._n_levels - 1})",
                UserWarning,
                stacklevel=3,
            )
        self._spline = Spline(
            kind=self.kind,
            n_knots=effective_n_knots,
            degree=self.degree,
            penalty=self.penalty,
            select=self.select,
        )

    def _map_to_numeric(self, x: NDArray) -> NDArray:
        """Map categorical values to their numeric representations (vectorized)."""
        return pd.Series(x).map(self._level_to_value).values.astype(np.float64)

    def _choose_base(self, x: NDArray, exposure: NDArray | None) -> None:
        """Choose the base level (step mode)."""
        if self._base_level and self._base_level in self._ordered_levels:
            return

        if self.base == "most_exposed" and exposure is not None:
            exp_by_level = {lev: float(exposure[x == lev].sum()) for lev in self._ordered_levels}
            self._base_level = max(exp_by_level, key=exp_by_level.get)
        elif self.base == "most_exposed" and exposure is None:
            self._base_level = self._ordered_levels[0]
        elif self.base == "first":
            self._base_level = self._ordered_levels[0]
        elif self.base in self._ordered_levels:
            self._base_level = self.base
        else:
            raise ValueError(f"Base '{self.base}' not found in levels: {self._ordered_levels}")

        self._non_base = [lev for lev in self._ordered_levels if lev != self._base_level]

    # ── Build ──────────────────────────────────────────────────────

    def build(
        self,
        x: NDArray,
        exposure: NDArray[np.floating] | None = None,
    ) -> GroupInfo | list[GroupInfo]:
        """Build design columns from ordered categorical data."""
        x = np.asarray(x).ravel()
        _validate_categorical_levels(x, self._known_levels)

        if self.basis == "spline":
            return self._build_spline(x, exposure)
        else:
            return self._build_step(x, exposure)

    def _build_spline(self, x: NDArray, exposure: NDArray | None) -> GroupInfo | list[GroupInfo]:
        """Spline mode: map to numeric, delegate to internal Spline."""
        x_numeric = self._map_to_numeric(x)
        return self._spline.build(x_numeric, exposure=exposure)

    def _build_step(self, x: NDArray, exposure: NDArray | None) -> GroupInfo:
        """Step mode: one-hot with first-difference penalty."""
        self._choose_base(x, exposure)
        n = len(x)
        K = self._n_levels
        n_cols = len(self._non_base)  # K - 1

        # One-hot encode (excluding base) — sparse CSR
        rows = []
        cols = []
        for j, lev in enumerate(self._non_base):
            mask = np.where(x == lev)[0]
            rows.append(mask)
            cols.append(np.full(len(mask), j))
        rows_arr = np.concatenate(rows)
        cols_arr = np.concatenate(cols)
        data = np.ones(len(rows_arr), dtype=np.float64)
        columns = sp.csr_matrix((data, (rows_arr, cols_arr)), shape=(n, n_cols))

        # K=2 edge case: D1 is empty, fall back to unpenalized
        if n_cols <= 1:
            return GroupInfo(columns=columns, n_cols=n_cols)

        # First-difference penalty on the FULL K-level ordering, then project
        # to the (K-1)-dimensional non-base space via base-removal matrix Z.
        # This ensures the penalty respects the original adjacency even when
        # the base level is in the middle of the ordering.
        #
        # The projected penalty Z'D1'D1Z is full rank (K-1) — intentionally.
        # In the treatment-contrast parameterisation (base=0), every direction
        # is penalized including the absolute level of non-base categories
        # relative to base.  This is correct: the constraint beta_base=0
        # breaks the constant null space that a naive (K-2)-rank D1 would have.
        base_idx = self._ordered_levels.index(self._base_level)
        D1_full = np.diff(np.eye(K), n=1, axis=0)  # (K-1, K)
        # Z: (K, K-1) inserts a zero row at base_idx position
        Z = np.zeros((K, n_cols))
        j = 0
        for i in range(K):
            if i != base_idx:
                Z[i, j] = 1.0
                j += 1
        omega = Z.T @ D1_full.T @ D1_full @ Z  # (K-1, K-1)

        return GroupInfo(
            columns=columns,
            n_cols=n_cols,
            penalty_matrix=omega,
            reparametrize=True,
            penalized=True,
        )

    # ── Transform ──────────────────────────────────────────────────

    def transform(self, x: NDArray) -> NDArray:
        """Build design matrix for new data using learned parameters."""
        x = np.asarray(x).ravel()
        _validate_categorical_levels(x, self._known_levels)

        if self.basis == "spline":
            x_numeric = self._map_to_numeric(x)
            return self._spline.transform(x_numeric)
        else:
            # Step mode: one-hot then apply R_inv
            onehot = np.column_stack([(x == lev).astype(np.float64) for lev in self._non_base])
            if self._R_inv is not None:
                return onehot @ self._R_inv
            return onehot

    # ── Reconstruct ────────────────────────────────────────────────

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]:
        """Convert fitted coefficients to interpretable output."""
        if self.basis == "spline":
            return self._reconstruct_spline(beta)
        else:
            return self._reconstruct_step(beta)

    def _reconstruct_spline(self, beta: NDArray) -> dict[str, Any]:
        """Spline mode: delegate to internal spline, add per-level annotations."""
        raw = self._spline.reconstruct(beta)

        # Per-level values on the fitted curve
        level_values = np.array([self._level_to_value[lev] for lev in self._ordered_levels])
        B_levels = self._spline._raw_basis_matrix(level_values)
        beta_orig = self._spline._R_inv @ beta if self._spline._R_inv is not None else beta
        level_log_rels = B_levels @ beta_orig

        raw["levels"] = self._ordered_levels
        raw["level_values"] = dict(zip(self._ordered_levels, level_values.tolist()))
        raw["level_log_relativities"] = dict(zip(self._ordered_levels, level_log_rels.tolist()))
        raw["level_relativities"] = dict(zip(self._ordered_levels, np.exp(level_log_rels).tolist()))
        return raw

    def _reconstruct_step(self, beta: NDArray) -> dict[str, Any]:
        """Step mode: same format as Categorical."""
        # Undo reparametrization
        if self._R_inv is not None:
            beta_orig = self._R_inv @ beta
        else:
            beta_orig = beta

        relativities = {self._base_level: 1.0}
        log_rels = {self._base_level: 0.0}
        for i, lev in enumerate(self._non_base):
            log_rels[lev] = float(beta_orig[i])
            relativities[lev] = float(np.exp(beta_orig[i]))
        return {
            "base_level": self._base_level,
            "levels": self._ordered_levels,
            "log_relativities": log_rels,
            "relativities": relativities,
        }

    # ── Reparametrisation ──────────────────────────────────────────

    def set_reparametrisation(self, R_inv: NDArray) -> None:
        if self.basis == "spline":
            self._spline.set_reparametrisation(R_inv)
        else:
            self._R_inv = R_inv
