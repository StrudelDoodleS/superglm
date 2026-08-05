"""OrderedCategorical feature: ordered categories with a spline basis.

Actuarial pricing data frequently contains continuous variables that have been
pre-binned into ordered categories (e.g. age bands "18-25", "26-35", ...).
This feature type respects the ordering with two modes:

- **spline**: map categories to numeric values, build a spline on those values
- **step**: one-hot encode with a first-difference penalty (D1'D1) so adjacent
  categories are soft-fused (deprecated)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm.features.categorical import _grouping_labels, _validate_categorical_levels
from superglm.types import GroupInfo


def _spline_kind_name(spline: Any) -> str:
    """Return the public factory kind for a spline specification."""
    from superglm.features.spline import (
        BSplineSmooth,
        CardinalCRSpline,
        CubicRegressionSpline,
        NaturalSpline,
        PSpline,
    )

    kind_by_type = (
        (PSpline, "ps"),
        (BSplineSmooth, "bs"),
        (NaturalSpline, "ns"),
        (CubicRegressionSpline, "cr"),
        (CardinalCRSpline, "cr_cardinal"),
    )
    for spline_type, kind in kind_by_type:
        if isinstance(spline, spline_type):
            return kind
    return type(spline).__name__


def _require_two_smooth_levels(smooth_levels: list[str], special_set: set[str]) -> None:
    """The final smooth-level list must retain at least two levels to smooth."""
    if special_set and len(smooth_levels) < 2:
        raise ValueError(
            "OrderedCategorical needs at least two non-special levels to fit a "
            f"smooth; got {smooth_levels!r} after removing {sorted(special_set)!r}. "
            "Use Categorical(...) for independent level effects."
        )


def _require_no_grouped_specials(grouping: Any, special_set: set[str]) -> None:
    """Refuse a grouping that merges a special into any other level.

    Merging a special into an ordered group would smooth it after all, while
    ``_specials`` still reports it free — an inconsistent spec with no error.
    Merging two specials is refused for the same reason the editor refuses it
    (``_require_no_special_members``): the group label would have to replace
    both members in ``specials=``.  Renaming a single special under a new group
    label is refused for the same reason: the renamed level joins the smooth
    with no numeric position while ``specials=`` still names the original.
    Only a group that leaves its level untouched (label identical to its one
    member) is exempt.
    """
    if not special_set or grouping is None:
        return
    for label, originals in grouping.group_to_originals.items():
        members = [str(member) for member in originals]
        if members == [str(label)]:
            continue
        merged = [member for member in members if member in special_set]
        if merged:
            joined = ", ".join(repr(member) for member in merged)
            raise ValueError(
                f"OrderedCategorical grouping merges free level(s) {joined} into group "
                f"{label!r}. Specials are fitted outside the smooth and may not be "
                "grouped; group the ordered levels only."
            )


class OrderedCategorical:
    """Ordered categorical feature with spline or step basis.

    Designed for continuous variables that arrive pre-binned into ordered
    categories (e.g. age bands, mileage bands).  Maps category labels to
    numeric values and fits a smooth function through them, borrowing
    strength between adjacent levels.

    The canonical API passes a :func:`Spline` specification as ``basis``::

        OrderedCategorical(
            order=["low", "medium", "high"],
            basis=Spline(kind="ps", k=6),
        )

    Two modes are currently available:

    - **spline** (default): maps levels to numeric values (midpoints or
      linspace), builds a B-spline through them.  The spline smooths across
      levels and the penalty controls wiggliness.  With ``fit_reml()``,
      REML selects the smoothing parameter automatically — the effective
      degrees of freedom will typically be much less than the number of
      levels.

    - **step** (deprecated): one-hot encodes with a first-difference penalty
      (D1'D1) so adjacent categories are soft-fused. Use ``Spline(...)`` for
      smoothing or :class:`Categorical` for independent level effects.

    Parameters
    ----------
    values : dict[str, float] or None
        Explicit mapping from category labels to numeric values (e.g.
        midpoints: ``{"18-25": 21.5, "26-35": 30.5, ...}``).
        Mutually exclusive with ``order``.
    order : list[str] or None
        Ordered list of category labels.  Numeric values are generated as
        ``linspace(0, 1, len(order))``.  Mutually exclusive with ``values``.
    basis : Spline object, {"spline", "step"}, or None
        Pass a ``Spline(...)`` object for full control over kind, basis size,
        constraints, selection, and penalty::

            OrderedCategorical(order=[...], basis=Spline(kind="cr", k=6))

        Omitting ``basis`` retains the historical default P-spline. The
        string values ``"spline"`` and ``"step"`` are deprecated; step
        smoothing will be removed in a future release.
    kind : str or None
        Deprecated spline shortcut. Configure ``kind`` on ``basis=Spline(...)``.
    base : str
        Reporting reference level. ``"most_exposed"`` (default), ``"first"``,
        or a specific level name. In spline mode this changes only the reported
        relativities and reference-adjusted intercept, not the fitted smooth.
    n_knots : int or None
        Deprecated spline shortcut. Auto-clamped to ``n_levels - 1``.
    degree : int or None
        Deprecated spline shortcut for B-spline degree.
    select : bool or None
        Deprecated spline shortcut for double-penalty shrinkage.
    penalty : str or None
        Deprecated spline shortcut for penalty type.
    specials : list[str] or None
        Level labels held out of the smooth and fitted as free, unpenalized
        level effects — one indicator column and one coefficient each. Use for
        levels that are structurally different rather than merely sparse (a
        ``MISSING`` band, a structural zero); the penalty already handles
        sparse bands better than free levels do. A label listed here is
        removed from ``order``/``values`` if also present there, and never
        receives a numeric position on the smooth's axis.

    Examples
    --------
    Using ordered level names (auto-spaced 0 to 1) with an explicit smooth::

        OrderedCategorical(
            order=["18-25", "26-35", "36-45", "46-55", "56+"],
            basis=Spline(kind="ps", k=6),
        )

    Using explicit midpoints::

        OrderedCategorical(
            values={"18-25": 21.5, "26-35": 30.5, "36-45": 40.5},
            basis=Spline(kind="cr", k=4),
        )

    Independent, unsmoothed level effects should use ``Categorical``::

        Categorical(base="most_exposed")
    """

    def __init__(
        self,
        values: dict[str, float] | None = None,
        order: list[str] | None = None,
        basis: Any | None = None,
        kind: str | None = None,
        base: str = "most_exposed",
        n_knots: int | None = None,
        degree: int | None = None,
        select: bool | None = None,
        penalty: str | None = None,
        grouping: Any = None,
        specials: list[str] | None = None,
    ):
        from superglm.features.spline import _SplineBase

        if values is not None and order is not None:
            raise ValueError("Specify exactly one of 'values' or 'order', not both.")
        if values is None and order is None:
            raise ValueError("Must specify either 'values' or 'order'.")

        self._specials: list[str] = []
        if specials is not None:
            # Level labels live in `str` space throughout this file (grouping
            # coerces too), so coerce here or a non-str special silently fails
            # to pop from order=/values= and gets smoothed as well as claimed.
            for raw_lev in specials:
                lev = str(raw_lev)
                if lev in self._specials:
                    raise ValueError(f"Duplicate special level {lev!r} in 'specials'.")
                self._specials.append(lev)
        special_set = set(self._specials)

        if special_set:
            if values is not None:
                values = {k: v for k, v in values.items() if str(k) not in special_set}
            else:
                order = [lev for lev in order if str(lev) not in special_set]

        basis_was_explicit = basis is not None
        shortcut_values = {
            "kind": kind,
            "n_knots": n_knots,
            "degree": degree,
            "select": select,
            "penalty": penalty,
        }
        used_shortcuts = [name for name, value in shortcut_values.items() if value is not None]

        resolved_basis = "spline" if basis is None else basis
        resolved_kind = "ps" if kind is None else kind
        resolved_n_knots = 5 if n_knots is None else n_knots
        resolved_degree = 3 if degree is None else degree
        resolved_select = False if select is None else select
        resolved_penalty = "ssp" if penalty is None else penalty

        # Accept a Spline object as basis.
        if isinstance(resolved_basis, _SplineBase):
            self._spline_obj = resolved_basis
            self.basis = "spline"
        elif resolved_basis in ("spline", "step"):
            self._spline_obj = None
            self.basis = resolved_basis
        else:
            raise ValueError(
                f"basis must be 'spline', 'step', or a Spline object, got {resolved_basis!r}"
            )

        shortcut_list = ", ".join(f"`{name}`" for name in used_shortcuts)
        shortcut_noun = "shortcut" if len(used_shortcuts) == 1 else "shortcuts"
        shortcut_verb = "is" if len(used_shortcuts) == 1 else "are"
        if self.basis == "step":
            warnings.warn(
                "OrderedCategorical step smoothing (`basis='step'`) is deprecated and "
                "will be removed in a future release. Use `basis=Spline(...)` for "
                "smoothing or `Categorical(...)` for independent level effects.",
                FutureWarning,
                stacklevel=2,
            )
        elif self._spline_obj is not None and used_shortcuts:
            warnings.warn(
                f"OrderedCategorical spline {shortcut_noun} ({shortcut_list}) "
                f"{shortcut_verb} ignored "
                "because basis is a Spline object; configure the Spline object directly.",
                FutureWarning,
                stacklevel=2,
            )
        legacy_spline_string = basis_was_explicit and resolved_basis == "spline"
        if (
            self.basis == "spline"
            and self._spline_obj is None
            and (legacy_spline_string or used_shortcuts)
        ):
            if legacy_spline_string and used_shortcuts:
                deprecated_api = (
                    f"`basis='spline'` and OrderedCategorical spline {shortcut_noun} "
                    f"({shortcut_list}) are"
                )
            elif legacy_spline_string:
                deprecated_api = "`basis='spline'` is"
            else:
                deprecated_api = (
                    f"OrderedCategorical spline {shortcut_noun} ({shortcut_list}) {shortcut_verb}"
                )
            warnings.warn(
                f"{deprecated_api} deprecated; configure the smooth with "
                "`basis=Spline(...)` instead.",
                FutureWarning,
                stacklevel=2,
            )

        if self.basis == "step" and resolved_select:
            raise ValueError("select=True is not supported with basis='step'.")
        if self.basis == "step" and special_set:
            raise ValueError(
                "specials= is not supported with basis='step', which is deprecated. "
                "Use basis=Spline(...) for a smoothed ordinal term with free levels."
            )

        self.kind = resolved_kind
        self.base = base
        self.select = resolved_select
        self.penalty = resolved_penalty
        self.degree = resolved_degree
        self.n_knots = resolved_n_knots
        self._smooth_levels: list[str] = []
        self._ordered_levels: list[str] = []

        # Derive smooth levels and numeric values
        if values is not None:
            sorted_items = sorted(values.items(), key=lambda kv: kv[1])
            self._smooth_levels = [k for k, _ in sorted_items]
            self._level_to_value = dict(values)
        else:
            self._smooth_levels = list(order)
            n = len(order)
            vals = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])
            self._level_to_value = dict(zip(order, vals.tolist()))
        self._ordered_levels = list(self._smooth_levels)

        # Grouping: validate and store
        self._grouping = grouping
        self._original_level_to_value: dict[str, float] | None = None
        if grouping is not None:
            # Preserve original level→value mapping for plot expansion
            orig_ltv = dict(self._level_to_value)
            self._original_level_to_value = orig_ltv
            # Build level_to_value for grouped levels.
            # When values= was used, orig_ltv keys are original level names
            # so we average them per group. When order= was used, orig_ltv
            # keys are already the grouped level names — use them directly
            # if the group name matches, otherwise average originals.
            grouped_ltv = {}
            for glev in grouping.grouped_levels:
                if glev in orig_ltv:
                    grouped_ltv[glev] = orig_ltv[glev]
                else:
                    originals = grouping.group_to_originals[glev]
                    vals = [orig_ltv[o] for o in originals if o in orig_ltv]
                    if vals:
                        grouped_ltv[glev] = float(np.mean(vals))
            self._level_to_value = grouped_ltv
            self._smooth_levels = [lev for lev in grouping.grouped_levels if lev not in special_set]
            # _known_levels includes all *original* levels (for predict-time validation)
            self._known_levels = set(grouping.all_original_levels) | special_set
        else:
            self._known_levels = set(self._smooth_levels) | special_set
        self._ordered_levels = list(self._smooth_levels) + list(self._specials)
        self._n_levels = len(self._smooth_levels)

        _require_no_grouped_specials(grouping, special_set)
        _require_two_smooth_levels(self._smooth_levels, special_set)
        if str(base) in special_set:
            raise ValueError(
                f"OrderedCategorical reporting base {base!r} is a special level. The base "
                "anchors every reported relativity and must lie on the smooth; choose one "
                f"of {self._smooth_levels!r}."
            )

        # Step mode state
        self._base_level: str = ""
        self._non_base: list[str] = []
        self._R_inv: NDArray | None = None

        # Spline mode: create internal spline (deferred until we know n_levels)
        self._spline = None
        if self.basis == "spline":
            self._init_spline()
            if self._spline_obj is not None:
                # The object API is authoritative; ignored legacy shortcuts must
                # not leak contradictory metadata into summaries or editor clones.
                self.kind = _spline_kind_name(self._spline)
                self.select = self._spline.select
                self.penalty = self._spline.penalty
                self.degree = self._spline.degree
                self.n_knots = self._spline.n_knots

    def __repr__(self) -> str:
        n = self._n_levels
        if self._spline is not None:
            return f"OrderedCategorical(basis={self._spline!r}, {n} levels)"
        return f"OrderedCategorical(basis={self.basis!r}, {n} levels, n_knots={self.n_knots})"

    @property
    def has_specials(self) -> bool:
        """True when one or more levels are fitted as free effects."""
        return bool(self._specials)

    def _init_spline(self) -> None:
        """Create the internal Spline object for spline mode."""
        import copy

        if self._spline_obj is not None:
            # User passed a Spline object — deep-copy so we own it,
            # then clamp n_knots if needed.
            self._spline = copy.deepcopy(self._spline_obj)
            if self._spline.n_knots > self._n_levels - 1:
                effective = self._n_levels - 1
                warnings.warn(
                    f"OrderedCategorical: Spline n_knots={self._spline.n_knots} "
                    f"clamped to {effective} (n_levels - 1 = {self._n_levels - 1})",
                    UserWarning,
                    stacklevel=3,
                )
                self._spline.n_knots = effective
            return

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
        """Map categorical values to their numeric representations (vectorized).

        Expects x to already be mapped through grouping if applicable
        (callers build() and transform() handle that).
        """
        return pd.Series(x).map(self._level_to_value).values.astype(np.float64)

    def _choose_base(self, x: NDArray, sample_weight: NDArray | None) -> None:
        """Choose the base level for relativities.

        Specials are excluded: the base anchors every reported relativity and
        must lie on the smooth. On a real book a MISSING band is often the most
        exposed level, so ``most_exposed`` would otherwise select it.
        """
        if self._base_level and self._base_level in self._smooth_levels:
            return

        if self.base == "most_exposed" and sample_weight is not None:
            exp_by_level = {
                lev: float(sample_weight[x == lev].sum()) for lev in self._smooth_levels
            }
            self._base_level = max(exp_by_level, key=exp_by_level.get)
        elif self.base == "most_exposed" and sample_weight is None:
            self._base_level = self._smooth_levels[0]
        elif self.base == "first":
            self._base_level = self._smooth_levels[0]
        elif self.base in self._smooth_levels:
            self._base_level = self.base
        else:
            raise ValueError(f"Base '{self.base}' not found in levels: {self._smooth_levels}")

        self._non_base = [lev for lev in self._smooth_levels if lev != self._base_level]

    # ── Build ──────────────────────────────────────────────────────

    def build(
        self,
        x: NDArray,
        sample_weight: NDArray[np.floating] | None = None,
    ) -> GroupInfo | list[GroupInfo]:
        """Build design columns from ordered categorical data."""
        x = np.asarray(x).ravel()

        if self._grouping is not None:
            x = _grouping_labels(x)
            _validate_categorical_levels(x, self._known_levels)
            x = pd.Series(x).map(self._grouping.original_to_group).values
        else:
            _validate_categorical_levels(x, self._known_levels)

        if self.basis == "spline":
            return self._build_spline(x, sample_weight)
        else:
            return self._build_step(x, sample_weight)

    def _build_spline(
        self, x: NDArray, sample_weight: NDArray | None
    ) -> GroupInfo | list[GroupInfo]:
        """Spline mode: map to numeric, delegate to internal Spline."""
        self._choose_base(x, sample_weight)
        x_numeric = self._map_to_numeric(x)
        return self._spline.build(x_numeric)

    def _build_step(self, x: NDArray, sample_weight: NDArray | None) -> GroupInfo:
        """Step mode: one-hot with first-difference penalty."""
        self._choose_base(x, sample_weight)
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
        if self._grouping is not None:
            x = _grouping_labels(x)
            valid_levels = self._known_levels | set(self._grouping.grouped_levels)
            _validate_categorical_levels(x, valid_levels)
            x = np.array(
                [self._grouping.original_to_group.get(v, v) for v in x],
                dtype=object,
            )
        else:
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

    def score(self, x: NDArray, beta: NDArray[np.floating]) -> NDArray[np.floating]:
        """Score the fitted ordered-categorical contribution directly on new data."""
        x = np.asarray(x).ravel()
        if self._grouping is not None:
            x = _grouping_labels(x)
            valid_levels = self._known_levels | set(self._grouping.grouped_levels)
            _validate_categorical_levels(x, valid_levels)
            x = np.array(
                [self._grouping.original_to_group.get(v, v) for v in x],
                dtype=object,
            )
        else:
            _validate_categorical_levels(x, self._known_levels)

        if self.basis == "spline":
            x_numeric = self._map_to_numeric(x)
            return self._spline.score(x_numeric, beta)

        beta_orig = self._R_inv @ beta if self._R_inv is not None else beta
        level_scores = {self._base_level: 0.0}
        for i, lev in enumerate(self._non_base):
            level_scores[lev] = float(beta_orig[i])
        return np.array([level_scores[lev] for lev in x], dtype=np.float64)

    # ── Reconstruct ────────────────────────────────────────────────

    def _base_log_effect(self, beta: NDArray[np.floating]) -> float:
        """Return the fitted term effect at the reporting reference level."""
        if self.basis != "spline":
            return 0.0
        base_value = np.array([self._level_to_value[self._base_level]], dtype=np.float64)
        return float(self._spline.score(base_value, beta)[0])

    def reconstruct(self, beta: NDArray[np.floating]) -> dict[str, Any]:
        """Convert fitted coefficients to interpretable output."""
        if self.basis == "spline":
            return self._reconstruct_spline(beta)
        else:
            return self._reconstruct_step(beta)

    def _reconstruct_spline(self, beta: NDArray) -> dict[str, Any]:
        """Spline mode: delegate to internal spline, add per-level annotations.

        Shifts the curve so that the base level has log_relativity=0 (relativity=1),
        giving proper categorical-style relativities.
        """
        raw = self._spline.reconstruct(beta)

        # Per-level values on the fitted curve
        level_values = np.array([self._level_to_value[lev] for lev in self._ordered_levels])
        level_log_rels = np.asarray(self._spline.score(level_values, beta), dtype=np.float64)

        # Shift so base level = 0 (relativity = 1)
        base_shift = self._base_log_effect(beta)
        level_log_rels = level_log_rels - base_shift
        raw["log_relativity"] = raw["log_relativity"] - base_shift
        raw["relativity"] = np.exp(raw["log_relativity"])

        raw["base_level"] = self._base_level
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


def resolve_interaction_parent(spec: Any, x: NDArray) -> tuple[Any, NDArray]:
    """Resolve one interaction parent (spec, column) for assembly.

    Identity for every spec — including ``None``, which FactorSmooth group
    columns carry — except spline-mode OrderedCategorical, which
    contributes its inner Spline on the mapped numeric scores, applying
    the same grouping, level validation, and score mapping its own
    ``build``/``transform`` apply.  Step-mode OC cannot parent an
    interaction: the deprecated one-hot geometry has no marginal smooth.
    """
    if not isinstance(spec, OrderedCategorical):
        return spec, x
    if spec.basis != "spline" or spec._spline is None:
        raise NotImplementedError(
            "OrderedCategorical with basis='step' is deprecated and cannot parent "
            "an interaction; use basis=Spline(...) for a smoothed ordinal parent "
            "or a Categorical feature for unsmoothed level effects."
        )
    x = np.asarray(x).ravel()
    if spec._grouping is not None:
        x = _grouping_labels(x)
        valid = spec._known_levels | set(spec._grouping.grouped_levels)
        _validate_categorical_levels(x, valid)
        x = np.array([spec._grouping.original_to_group.get(v, v) for v in x], dtype=object)
    else:
        _validate_categorical_levels(x, spec._known_levels)
    return spec._spline, spec._map_to_numeric(x)


def resolve_interaction_parent_of(ispec: Any, spec: Any, x: NDArray) -> tuple[Any, NDArray]:
    """``resolve_interaction_parent`` for a parent OF a given interaction.

    Identical to it for every interaction that reads both parents as
    *marginal* columns.  ``FactorSmooth`` does not: its second parent is a
    GROUPING column, read as labels and factorized into the term's own level
    set.  Mapping a spline-mode ``OrderedCategorical`` main on that column to
    its level scores would silently re-key that identity — ``_levels`` becomes
    ``[0.0, 0.25, ...]`` instead of ``['18-25', ...]`` — and every by-label
    lookup the term's inference exposes would fail on the fitted labels.  Its
    variable parent is numeric and carries no OC form either, so a
    ``FactorSmooth`` takes both columns as the frame carries them.
    """
    from superglm.features.factor_smooth import FactorSmooth

    if isinstance(ispec, FactorSmooth):
        return spec, x
    return resolve_interaction_parent(spec, x)
