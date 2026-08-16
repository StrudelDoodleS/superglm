"""Discretization impact analysis.

Answers the question: "If I bin this fit's smooth terms into N buckets, how do
my predictions and model metrics change?"

This is a read-only analysis tool — no refitting. It takes a fitted model,
discretizes the smooth contributions analytically — spline and polynomial main
effects into bins, a continuous-by-continuous interaction onto the grid its
rating-table block is sampled on — and reports the impact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import FrameLike, as_eager_frame

if TYPE_CHECKING:
    from superglm.model import SuperGLM


@dataclass
class DiscretizationResult:
    """Result of discretizing smooth spline curves into rating tables.

    Attributes
    ----------
    tables : dict[str, DataFrame]
        Per-MAIN-EFFECT rating tables with columns: bin_from, bin_to,
        relativity, log_relativity, n_obs, sample_weight. ``n_obs`` is always
        the physical row count. ``sample_weight`` is the supplied weight total
        in the bin (frequency mass for non-Tweedie; EDM prior-weight mass for
        Tweedie) and is reported for display rather than reinterpreted as a
        Tweedie replication count.
    interaction_tables : dict[str, DataFrame]
        Per-INTERACTION grids, one row per grid cell, with two axis-value
        columns named for the parents -- suffixed ``(axis 1)``/``(axis 2)``
        when the parent names would collide with each other or with a value
        column -- and then relativity,
        log_relativity, n_obs and sample_weight on the same terms as
        ``tables``. Kept in its own mapping because the two shapes are not
        interchangeable: a main effect is binned into intervals and a
        continuous-by-continuous interaction is SAMPLED at grid nodes, so it
        has axis values where a bin has a half-open interval, and one row per
        cell rather than per bin.
    predictions : NDArray
        Predictions using discretized (binned) curves.
    original_predictions : NDArray
        Original smooth predictions.
    metrics : dict[str, float]
        Comparison metrics between original and discretized predictions. Joint
        over everything discretized in the call, main effects and interactions
        alike, since that is the prediction a consumer of the whole table gets.
    """

    tables: dict[str, pd.DataFrame]
    predictions: NDArray
    original_predictions: NDArray
    metrics: dict[str, float]
    interaction_tables: dict[str, pd.DataFrame] = field(default_factory=dict)


def _validated_discretization_weights(
    model,
    sample_weight,
    n_rows: int,
) -> tuple[NDArray, NDArray]:
    """Return likelihood/display weights and family-appropriate geometry mass."""
    from superglm.distributions import Tweedie

    if sample_weight is None:
        weights = np.ones(n_rows, dtype=np.float64)
    elif isinstance(model._distribution, Tweedie):
        from superglm._utils import _validate_strict_prior_weights

        weights = _validate_strict_prior_weights(sample_weight, n_rows)
    else:
        from superglm.model.input_validation import _finite_vector

        weights = _finite_vector("sample_weight", sample_weight, n_rows)
        if np.any(weights < 0.0):
            raise ValueError("sample_weight must be nonnegative")
        if not np.any(weights > 0.0):
            raise ValueError("sample_weight must not be all zero")

    weight_total = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(weight_total):
        raise ValueError("sample_weight must have a finite sum")

    # Non-Tweedie weights are frequency mass and therefore shape the same
    # support as literal replicated rows. Tweedie weights are prior precision:
    # fit-time spline/discrete geometry remains a function of physical rows.
    geometry_weight = (
        np.ones(n_rows, dtype=np.float64) if isinstance(model._distribution, Tweedie) else weights
    )
    return weights, geometry_weight


def _weighted_quantile_edges(x: NDArray, sample_weight: NDArray, n_bins: int) -> NDArray:
    """Compute edges with roughly equal geometry mass in each bin."""
    positive = sample_weight > 0.0
    x = np.asarray(x[positive], dtype=np.float64)
    sample_weight = np.asarray(sample_weight[positive], dtype=np.float64)
    order = np.argsort(x)
    x_sorted = x[order]
    weight_sorted = sample_weight[order]
    cumulative_weight = np.cumsum(weight_sorted)
    total = cumulative_weight[-1]

    if x_sorted[0] == x_sorted[-1]:
        return np.array([x_sorted[0], x_sorted[0]], dtype=np.float64)

    edges = [x_sorted[0]]
    for i in range(1, n_bins):
        target = total * i / n_bins
        idx = np.searchsorted(cumulative_weight, target, side="right")
        idx = min(idx, len(x_sorted) - 1)
        edges.append(x_sorted[idx])
    edges.append(x_sorted[-1])

    # Deduplicate: if repeated values collapse bins, keep unique edges
    edges = np.unique(edges)
    if len(edges) == 1:
        return np.repeat(edges, 2)
    return edges


def _uniform_edges(x: NDArray, n_bins: int) -> NDArray:
    """Compute equal-width bin edges across the data range."""
    return np.linspace(x.min(), x.max(), n_bins + 1)


def _weighted_percentiles(
    x: NDArray,
    sample_weight: NDArray,
    quantiles: NDArray,
) -> NDArray:
    """Percentiles matching NumPy on literal integer row replication."""
    positive = sample_weight > 0.0
    x_active = np.asarray(x[positive], dtype=np.float64)
    weight_active = np.asarray(sample_weight[positive], dtype=np.float64)
    order = np.argsort(x_active)
    x_sorted = x_active[order]
    cumulative_weight = np.cumsum(weight_active[order])
    total = float(cumulative_weight[-1])
    if total <= 1.0:
        indices = np.searchsorted(
            cumulative_weight,
            total * np.asarray(quantiles, dtype=np.float64),
            side="right",
        )
        return x_sorted[np.clip(indices, 0, len(x_sorted) - 1)]

    positions = (total - 1.0) * np.asarray(quantiles, dtype=np.float64)
    lower_positions = np.floor(positions)
    upper_positions = np.ceil(positions)
    lower_indices = np.searchsorted(cumulative_weight, lower_positions, side="right")
    upper_indices = np.searchsorted(cumulative_weight, upper_positions, side="right")
    lower = x_sorted[np.clip(lower_indices, 0, len(x_sorted) - 1)]
    upper = x_sorted[np.clip(upper_indices, 0, len(x_sorted) - 1)]
    return lower + (positions - lower_positions) * (upper - lower)


def _winsorized_edges(x: NDArray, sample_weight: NDArray, n_bins: int) -> NDArray:
    """Geometry-quantile binning on the [p5, p95] interior, with tail bins."""
    if n_bins < 3:
        # Not enough bins for tail+interior+tail, fall back to weight quantiles.
        return _weighted_quantile_edges(x, sample_weight, n_bins)

    positive = sample_weight > 0.0
    x_geometry = x[positive]
    p5, p95 = _weighted_percentiles(
        x,
        sample_weight,
        np.array([0.05, 0.95], dtype=np.float64),
    )
    x_min, x_max = x_geometry.min(), x_geometry.max()

    # If percentiles collapse (very little spread), fall back
    if p5 >= p95:
        return _weighted_quantile_edges(x, sample_weight, n_bins)

    # Interior: geometry-weight quantiles on observations within [p5, p95].
    interior_mask = (x >= p5) & (x <= p95)
    if not np.any(sample_weight[interior_mask] > 0.0):
        return _weighted_quantile_edges(x, sample_weight, n_bins)
    n_interior = n_bins - 2
    interior_edges = _weighted_quantile_edges(
        x[interior_mask], sample_weight[interior_mask], n_interior
    )

    # Assemble: [x_min, p5, ...interior..., p95, x_max]
    edges = np.concatenate([[x_min], interior_edges, [x_max]])
    edges = np.unique(edges)
    return edges


def _compute_edges(x: NDArray, sample_weight: NDArray, n_bins: int, strategy: str) -> NDArray:
    """Dispatch to the appropriate binning strategy."""
    if strategy == "exposure_quantile":
        return _weighted_quantile_edges(x, sample_weight, n_bins)
    elif strategy == "uniform":
        return _uniform_edges(x[sample_weight > 0.0], n_bins)
    elif strategy == "winsorized":
        return _winsorized_edges(x, sample_weight, n_bins)
    else:
        raise ValueError(
            f"Unknown bin_strategy: {strategy!r}. "
            "Use 'exposure_quantile', 'uniform', or 'winsorized'."
        )


def _is_continuous_feature(model: SuperGLM, name: str) -> bool:
    """Check if a feature is a spline or polynomial (has 'x' in reconstruct)."""
    from superglm.features.polynomial import Polynomial
    from superglm.features.spline import _SplineBase

    return isinstance(model._specs[name], _SplineBase | Polynomial)


_GRID_RECONSTRUCTION_KEYS = frozenset({"x1", "x2", "relativity"})


def reconstruct_interaction(ispec, beta: NDArray, n_points: int) -> dict:
    """Reconstruct an interaction, passing ``n_points`` only if it takes one.

    Shared with the rating-table export so that "is this a grid" is answered
    the same way in both places. Deciding it twice is the shape issue #287
    took, and a signature pre-filter here would have been a third answer: a
    custom spec whose ``reconstruct(beta)`` takes no ``n_points`` and still
    returns a surface is exported as a grid, so it has to be swept as one.
    """
    reconstruct = getattr(ispec, "reconstruct", None)
    if reconstruct is None:
        raise TypeError(f"{type(ispec).__name__} has no reconstruct()")
    try:
        takes_n_points = "n_points" in signature(reconstruct).parameters
    except (TypeError, ValueError):
        takes_n_points = False
    if takes_n_points:
        return reconstruct(beta, n_points=n_points)
    return reconstruct(beta)


def orient_grid_surface(name: str, axis1: NDArray, axis2: NDArray, surface: NDArray) -> NDArray:
    """Normalise a reconstructed surface to ``surface[i, j] = f(x1[i], x2[j])``.

    Shared with ``_continuous_interaction_block`` so the sweep and the exported
    block cannot orient the same grid differently. Two shapes are accepted
    because both built-ins return the meshgrid convention
    ``surface[j, i] = f(x1[i], x2[j])`` while a custom spec may already be in
    the natural order, and a non-square grid distinguishes them.

    A SQUARE grid does not: both built-ins sample ``n_points`` nodes per axis,
    so the first branch always fires there and the shape cannot witness the
    orientation. What decides it is the convention, which
    ``TensorInteraction.reconstruct`` states and ``PolynomialInteraction``
    inherits from ``np.meshgrid``'s default ``indexing="xy"`` -- and what pins
    it is ``test_the_exported_grids_orientation_is_load_bearing``, which checks
    an exported cell against the fitted interaction's own factor on a
    deliberately asymmetric domain.
    """
    surface = np.asarray(surface, dtype=np.float64)
    if surface.shape == (len(axis2), len(axis1)):
        return surface.T
    if surface.shape != (len(axis1), len(axis2)):
        raise ValueError(
            f"Interaction {name!r} returned a {surface.shape} grid, "
            f"expected {(len(axis1), len(axis2))} or {(len(axis2), len(axis1))}."
        )
    return surface


def _non_grid_builtin_interactions() -> tuple[type, ...]:
    """The shipped interaction types whose reconstruction is never a surface.

    A NEGATIVE fast path, and only that: classifying by class is what four
    review rounds had to undo, so this decides nothing on its own -- anything
    not listed still goes through the reconstruction contract, which is what
    the exporter routes on.  What it buys is not reconstructing a term the
    answer is already known for: ``SplineCategorical.reconstruct`` walks every
    level and each call walks them again, so on a high-cardinality parent the
    default diagnostic paid quadratic work to learn the term is a cell table.

    ``test_the_class_fast_path_covers_every_shipped_interaction`` checks that
    this list and the two grid types PARTITION the interaction classes the
    module ships, so a new one cannot land unclassified.
    ``test_the_fast_path_only_skips_work`` then fits a term of EVERY listed
    type and checks each verdict against the contract with the short-circuit
    patched away -- so a listed class that later starts returning a surface,
    which the partition cannot see because membership is still correct, fails
    there instead of silently making the exporter ship a term the sweep will
    refuse.  Adding a type to this list means adding a term of it to that
    fixture; the test asserts the coverage rather than assuming it.
    """
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.interaction import (
        CategoricalInteraction,
        NumericCategorical,
        NumericInteraction,
        PolynomialCategorical,
        SplineCategorical,
    )

    return (
        SplineCategorical,
        PolynomialCategorical,
        NumericCategorical,
        CategoricalInteraction,
        NumericInteraction,
        FactorSmooth,
    )


def unpositionable_grid_parent(ispec, specs: dict) -> str | None:
    """The parent of a would-be grid whose observations have no axis position.

    A grid is a lookup keyed on two NUMERIC axes: a risk is priced by finding
    the nearest printed value on each.  That only means something if the
    parent's own column can be placed on that axis.  A spline, polynomial or
    plain numeric parent can; a spline-mode ``OrderedCategorical`` can, through
    the level scores ``resolve_interaction_parent_of`` maps it to.  A plain
    ``Categorical`` cannot -- its column holds labels, and no label has a
    position among ``0.0, 0.5, 1.0``.

    Shipped grid types never hit this: both have numeric parents by
    construction, and every categorical-parent type is on the non-grid list.
    It is reachable through an explicit spec -- ``interactions=`` accepts any
    object carrying ``parent_names`` and ``name`` -- whose ``reconstruct``
    returns a surface over a categorical parent.  The exporter used to ship
    that as a grid whose axis column a reader could not look anything up in,
    and the sweep then read the label column as float64 and died on
    ``could not convert string to float``.  Both now decline it here, together,
    rather than one shipping what the other refuses.

    Returns the offending parent's name, or ``None`` when both are positionable.
    """
    from superglm.features.categorical import Categorical
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.ordered_categorical import OrderedCategorical

    if isinstance(ispec, FactorSmooth):
        # Its second parent is a GROUPING column read as labels, never an axis.
        return None
    for parent in ispec.parent_names:
        spec = specs.get(parent)
        if isinstance(spec, Categorical) and not isinstance(spec, OrderedCategorical):
            return parent
    return None


def _grid_log_surface(
    name: str, axis1: NDArray, axis2: NDArray, grid: dict, relativity: NDArray
) -> NDArray:
    """The log surface, taking the printed factor wherever it is one.

    Two requirements pull against each other.  The sweep must measure the
    surface the workbook PRINTS, which is ``relativity`` -- a custom
    reconstruction can return two fields that disagree, and measuring the one
    that does not ship reports error for a table nobody holds.  But
    ``relativity`` is ``exp`` of the log surface for both built-ins, and a
    finite log surface outside float64's exponential range comes back as ``0``
    or ``inf`` while the fit's own predictor stays perfectly representable --
    so refusing on the factor would reject a diagnostic with nothing wrong.

    Resolved CELL BY CELL rather than by choosing one field for the whole
    grid: every cell whose printed factor is usable is measured from it, and
    only the cells where it is not fall back to the log field -- and only when
    ``exp`` of the log value reproduces that cell exactly, which is what the
    range losing it looks like.  So the sheet never measures a surface the
    workbook does not carry, at any tolerance, the log field's extra range is
    used exactly where it is the only thing left, and a caller who supplied a
    factor the log contradicts is still refused.

    ``log_relativity`` is not part of the grid contract -- the exporter reads
    ``x1``/``x2``/``relativity`` and nothing else -- so a field that will not
    even normalise is treated as absent rather than allowed to refuse an
    interaction the exporter accepts.
    """
    usable = np.isfinite(relativity) & (relativity > 0.0)
    if np.all(usable):
        return np.log(relativity)

    log_surface = None
    if "log_relativity" in grid:
        try:
            candidate = orient_grid_surface(name, axis1, axis2, grid["log_relativity"])
        except (ValueError, TypeError):
            candidate = None
        if candidate is not None and np.all(np.isfinite(candidate[~usable])):
            # The log field may only speak for a cell whose printed factor the
            # exponential RANGE lost -- ``exp`` of it must reproduce that cell
            # exactly, which an underflow to ``0`` or an overflow to ``inf``
            # does because ``exp`` does the same thing to the same number.  A
            # caller who simply supplied a zero the log contradicts is still
            # refused, since nothing explains the disagreement.
            with np.errstate(over="ignore", under="ignore"):
                round_trip = np.exp(candidate[~usable])
            if np.array_equal(round_trip, relativity[~usable], equal_nan=True):
                log_surface = candidate

    if log_surface is None:
        raise ValueError(
            f"Interaction {name!r} reconstructed a relativity that is not a usable "
            "factor: every cell must be finite and strictly positive, or carry a "
            "finite ``log_relativity`` for the cells that are not."
        )

    surface = np.empty_like(relativity, dtype=np.float64)
    surface[usable] = np.log(relativity[usable])
    surface[~usable] = log_surface[~usable]
    return surface


def _grid_reconstruction(ispec, beta: NDArray, n_points: int) -> dict | None:
    """The spec's reconstruction if it is a sampled surface, else ``None``.

    Classified by the reconstruction CONTRACT rather than by class, because
    that is the rule the rating-table export ships on: ``_interaction_blocks``
    routes any interaction whose reconstruction carries ``x1``, ``x2`` and
    ``relativity`` to the grid block, including an explicit interaction spec a
    caller supplied. An isinstance check against the two built-ins would let
    such a spec be exported approximately and left off the impact sheet, which
    is the same gap this module closes for the built-ins.
    """
    if getattr(ispec, "reconstruct", None) is None:
        return None
    # ``type(...) in``, not ``isinstance``: a SUBCLASS of a listed type is not
    # listed, and it can override ``reconstruct`` to return a surface.  The
    # exporter would ship it as a grid on the contract while an isinstance
    # short-circuit refused it here -- the sixth expression of the same
    # exporter-accepts/sweep-refuses fork, introduced by this very speedup.
    # An exact-type check makes that case unreachable rather than untested,
    # and makes the sentence above literally true.
    if type(ispec) in _non_grid_builtin_interactions():
        return None
    raw = reconstruct_interaction(ispec, beta, n_points)
    # The exporter's test verbatim -- a key-subset check and nothing else.  An
    # ``isinstance(raw, dict)`` beside it is stricter than what ships: a mapping
    # that is not a ``dict`` (a ``UserDict``, say) is exported as a grid, since
    # ``_interaction_blocks`` only iterates keys and subscripts values, and
    # would then be refused here -- taking the whole payload down.
    if not _GRID_RECONSTRUCTION_KEYS <= set(raw):
        return None
    return raw


_INTERACTION_TABLE_RESERVED_COLUMNS = ("relativity", "log_relativity", "n_obs", "sample_weight")


def _axis_column_labels(parent1: str, parent2: str) -> tuple[str, str]:
    """Two distinct, non-reserved names for a grid's axis-value columns.

    Both collisions are reachable and both lose an axis silently, because a
    later key in a ``DataFrame`` dict literal simply overwrites an earlier one.
    ``interactions=[("age", "age")]`` fits and exports, so ``parent1`` and
    ``parent2`` can be the same string; and a feature may be named
    ``relativity`` or ``n_obs``, in which case its axis values would be
    replaced by the value column of that name. Same hazard the offset block
    already guards with ``_OFFSET_SOURCE_RESERVED_COLUMNS``.

    Disambiguated by axis index on BOTH columns rather than on the second
    alone, so a reader never has to know which one moved.
    """
    reserved = set(_INTERACTION_TABLE_RESERVED_COLUMNS)
    if parent1 == parent2 or reserved & {parent1, parent2}:
        return f"{parent1} (axis 1)", f"{parent2} (axis 2)"
    return parent1, parent2


def _ascending_grid(
    axis1: NDArray, axis2: NDArray, surface: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """Put both axes in ascending order, carrying the surface with them.

    ``_nearest_grid_index`` is a binary search, so a descending or unsorted
    axis would silently map a risk onto a non-nearest node -- and the exporter
    preserves whatever order a reconstruction supplies without a monotonicity
    gate. Sorting rather than refusing, because the set of cells is unchanged:
    every factor a reader can look up off the exported block is still here,
    and the lookup that finds it is now the documented one.
    """
    order1 = np.argsort(axis1, kind="stable")
    order2 = np.argsort(axis2, kind="stable")
    return axis1[order1], axis2[order2], surface[np.ix_(order1, order2)]


def _nearest_grid_index(grid: NDArray, x: NDArray) -> NDArray:
    """Index of the closest grid node, ties to the lower index.

    The exported interaction block is keyed on the grid's axis VALUES, not on
    intervals, so a consumer holding a raw risk has no bin to fall into and the
    only lookup the sheet supports is the nearest printed axis value on each
    axis. This is that rule, so what the impact sweep measures is the factor a
    reader will actually apply.
    """
    grid = np.asarray(grid, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if len(grid) < 2:
        return np.zeros(len(x), dtype=np.intp)
    right = np.clip(np.searchsorted(grid, x), 1, len(grid) - 1)
    take_left = (x - grid[right - 1]) <= (grid[right] - x)
    return np.where(take_left, right - 1, right).astype(np.intp)


def _weighted_correlation(x: NDArray, y: NDArray, weights: NDArray) -> float:
    """Correlation under frequency mass or unit physical-row mass."""
    positive = weights > 0.0
    x_active = np.asarray(x[positive], dtype=np.float64)
    y_active = np.asarray(y[positive], dtype=np.float64)
    mass = np.asarray(weights[positive], dtype=np.float64)
    mass /= np.sum(mass, dtype=np.float64)
    x_centered = x_active - float(np.sum(mass * x_active))
    y_centered = y_active - float(np.sum(mass * y_active))
    variance_x = float(np.sum(mass * x_centered**2))
    variance_y = float(np.sum(mass * y_centered**2))
    if variance_x <= 0.0 or variance_y <= 0.0:
        return float("nan")
    correlation = float(np.sum(mass * x_centered * y_centered) / np.sqrt(variance_x * variance_y))
    return float(np.clip(correlation, -1.0, 1.0))


def discretization_impact(
    model: SuperGLM,
    X: FrameLike,
    y: NDArray,
    sample_weight: NDArray | None = None,
    *,
    offset: NDArray | None = None,
    n_bins: int = 100,
    bin_strategy: str = "exposure_quantile",
    features: list[str] | None = None,
) -> DiscretizationResult:
    """Analyse the impact of discretizing the smooth terms of a fit.

    For each spline/polynomial feature, the smooth per-observation
    log-relativity is replaced with a family-appropriate bin average. For each
    continuous-by-continuous interaction it is replaced with the value at the
    nearest node of the ``n_bins``-per-axis grid the rating-table export ships
    -- a SAMPLING rather than an averaging, because that block is keyed on axis
    values and a consumer's only available lookup is the nearest one.
    Predictions are recomputed and compared to the originals.

    Both are covered because both are approximations the exported workbook
    carries, and reporting one without the other understates how far the table
    sits from the model (issue #287). The returned ``metrics`` are joint over
    everything discretized in the call, which is the error a consumer applying
    the whole table actually gets.

    For non-Tweedie families, ``sample_weight`` is case/frequency mass:
    bin geometry, bin averages, mean prediction change, and prediction
    correlation match literal integer row replication. Zero-frequency rows
    retain predictions and physical ``n_obs`` entries but cannot change bin
    geometry or summary metrics. For Tweedie, weights are finite, strictly
    positive EDM prior weights. They weight deviance and the displayed
    ``sample_weight`` totals, while bin geometry, bin averages, and pure
    prediction-comparison summaries use physical rows.

    Parameters
    ----------
    model : SuperGLM
        A fitted SuperGLM model.
    X : pandas or eager Polars DataFrame
        Data used for analysis (typically training data).
    y : NDArray
        Response variable.
    sample_weight : NDArray, optional
        Nonnegative case/frequency weights for non-Tweedie models. For
        Tweedie models, finite, strictly positive EDM prior weights. Defaults
        to ones.
    offset : NDArray, optional
        Link-scale offset aligned to ``X``. Used when comparing original and
        discretized predictions for offset-fitted models.
    n_bins : int
        Number of bins per feature (default 100).
    bin_strategy : str
        Binning strategy: ``"exposure_quantile"`` (the retained public name)
        places edges at equal geometry-weight mass; ``"uniform"`` uses
        equal-width bins; ``"winsorized"`` uses geometry-weight quantiles on
        the interior [p5, p95] with dedicated tail bins. Geometry weight means
        frequency mass for non-Tweedie models and unit physical-row mass for
        Tweedie.
    features : list[str], optional
        Subset of names to discretize: spline/polynomial features, and
        continuous-by-continuous interaction names as they appear in
        ``model._interaction_order``. None means every one of both.

    Returns
    -------
    DiscretizationResult
    """
    frame = as_eager_frame(X)
    result = model.result  # raises if not fitted
    n = len(frame)
    if n == 0:
        raise ValueError("X and y must be non-empty")

    from superglm.distributions import validate_response
    from superglm.model.input_validation import _finite_vector

    y = _finite_vector("y", y, n, require_nonempty=True)
    validate_response(y, model._distribution)
    evaluation_weight, geometry_weight = _validated_discretization_weights(
        model,
        sample_weight,
        n,
    )
    if offset is not None:
        offset = _finite_vector("offset", offset, n)
    if isinstance(n_bins, bool) or not isinstance(n_bins, int | np.integer) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    n_bins = int(n_bins)

    beta = result.beta
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.model import base

    plan = base._prediction_plan(model)
    terms_by_name = {term["name"]: term for term in plan["features"]}
    interaction_terms = {term["name"]: term for term in plan["interactions"]}

    def _interaction_beta(name: str) -> NDArray:
        term = interaction_terms.get(name)
        if term is None:
            raise RuntimeError(f"prediction plan does not define fitted interaction {name!r}")
        return beta[np.asarray(term["beta_idx"], dtype=np.intp)]

    # Classifying an interaction means reconstructing it, and the loop below
    # needs the same surface at the same ``n_bins`` -- so it is kept rather
    # than built twice. On the ``PolynomialInteraction`` path a reconstruction
    # allocates a dense ``(n_points**2, n1*n2)`` array, so the second one was
    # not free.
    grids: dict[str, dict | None] = {}

    def _grid_for(name: str) -> dict | None:
        if name not in grids:
            grids[name] = _grid_reconstruction(
                model._interaction_specs[name], _interaction_beta(name), n_bins
            )
        return grids[name]

    # Determine which terms to discretize. Two namespaces, because a
    # continuous-by-continuous interaction is approximated by the same export
    # and by the same ``n_bins``, and leaving it out understates the answer.
    target_features: list[str] = []
    target_interactions: list[str] = []
    if features is not None:
        # De-duplicated, order preserved. A repeated name used to add its
        # replacement delta once per occurrence while writing one table, so
        # ``features=["age:density", "age:density"]`` reported twice the
        # discretisation error of a block the workbook ships once.
        features = list(dict.fromkeys(features))
        for name in features:
            if name in model._specs:
                if not _is_continuous_feature(model, name):
                    raise ValueError(
                        f"Feature '{name}' is not a spline or polynomial — "
                        "only continuous features can be discretized."
                    )
                target_features.append(name)
            elif name in model._interaction_specs:
                if _grid_for(name) is None:
                    raise ValueError(
                        f"Interaction '{name}' is not continuous-by-continuous — "
                        "only interactions exported as a sampled grid can be discretized."
                    )
                target_interactions.append(name)
            else:
                raise ValueError(f"Unknown feature: {name}")
    else:
        target_features = [
            name for name in model._feature_order if _is_continuous_feature(model, name)
        ]
        target_interactions = [
            name for name in model._interaction_order if _grid_for(name) is not None
        ]

    from superglm.model.input_validation import validate_x_columns

    required = list(
        dict.fromkeys(
            target_features
            + [
                parent
                for name in target_interactions
                for parent in model._interaction_specs[name].parent_names
            ]
        )
    )
    frame.require_columns(tuple(required))
    validate_x_columns(frame, required)
    eta_orig = base.predict_eta_exact(model, frame, offset=offset)
    original_predictions = clip_mu(model._link.inverse(eta_orig), model._distribution)

    # For each target feature, compute the delta (binned - smooth)
    tables: dict[str, pd.DataFrame] = {}
    total_delta = np.zeros(n)

    for name in target_features:
        x_raw = frame.column_array(name, dtype=np.float64)

        # Per-observation smooth log-relativity for this feature
        term = terms_by_name.get(name)
        if term is None:
            raise RuntimeError(f"prediction plan does not define fitted term {name!r}")
        beta_feature = beta[np.asarray(term["beta_idx"], dtype=np.intp)]
        log_rel_smooth = np.asarray(
            base._score_prediction_term_local_exact(term, frame, beta_feature),
            dtype=np.float64,
        ).ravel()

        # Compute bin edges using the selected strategy
        edges = _compute_edges(x_raw, geometry_weight, n_bins, bin_strategy)
        actual_n_bins = len(edges) - 1

        # Assign observations to bins
        bin_idx = np.digitize(x_raw, edges, right=False)
        # digitize returns 1-based; clip to valid range
        bin_idx = np.clip(bin_idx, 1, actual_n_bins) - 1

        # Frequency-weighted for non-Tweedie; physical-row mean for Tweedie.
        bin_log_rel = np.zeros(actual_n_bins)
        bin_weight = np.zeros(actual_n_bins)
        bin_n_obs = np.zeros(actual_n_bins, dtype=int)

        for b in range(actual_n_bins):
            mask = bin_idx == b
            if np.any(mask):
                bin_n_obs[b] = mask.sum()
                bin_weight[b] = float(np.sum(evaluation_weight[mask], dtype=np.float64))
                geometry_mass = geometry_weight[mask]
                if np.any(geometry_mass > 0.0):
                    bin_log_rel[b] = np.average(
                        log_rel_smooth[mask],
                        weights=geometry_mass,
                    )

        # Build rating table
        table_rows = []
        for b in range(actual_n_bins):
            table_rows.append(
                {
                    "bin_from": edges[b],
                    "bin_to": edges[b + 1],
                    "relativity": np.exp(bin_log_rel[b]),
                    "log_relativity": bin_log_rel[b],
                    "n_obs": bin_n_obs[b],
                    "sample_weight": bin_weight[b],
                }
            )
        tables[name] = pd.DataFrame(table_rows)

        # Per-observation delta: replace smooth with bin mean
        binned_log_rel = bin_log_rel[bin_idx]
        total_delta += binned_log_rel - log_rel_smooth

    # Interactions: sampled at grid nodes rather than averaged over bins, so
    # the per-observation replacement is the node a consumer's lookup lands on.
    from superglm.features.ordered_categorical import resolve_interaction_parent_of

    interaction_tables: dict[str, pd.DataFrame] = {}

    for name in target_interactions:
        term = interaction_terms[name]
        beta_term = _interaction_beta(name)
        log_rel_exact = np.asarray(
            base._score_prediction_term_local_exact(term, frame, beta_term),
            dtype=np.float64,
        ).ravel()

        # The same grid the export ships, at the same resolution: ``n_bins``
        # nodes per axis. Reading it back from the spec rather than
        # re-deriving it is what makes the measured error the EXPORTED error.
        ispec = term["spec"]
        grid = _grid_for(name)
        if grid is None:
            raise RuntimeError(
                f"Interaction {name!r} was classified as a sampled grid but its "
                "reconstruction no longer carries one."
            )
        # From ``relativity``, which is the field ``_continuous_interaction_block``
        # prints, rather than from ``log_relativity`` beside it.  For the two
        # built-ins the pair is consistent to an ulp, but a custom reconstructor
        # can return two surfaces that disagree, and then measuring the one the
        # workbook does NOT ship would report error for a table nobody holds.
        axis1 = np.asarray(grid["x1"], dtype=np.float64)
        axis2 = np.asarray(grid["x2"], dtype=np.float64)
        relativity = orient_grid_surface(name, axis1, axis2, grid["relativity"])
        surface = _grid_log_surface(name, axis1, axis2, grid, relativity)
        # Ascending, because the nearest-node search is a binary search and the
        # exporter applies no monotonicity gate to a supplied axis. Sorting the
        # surface with its axes leaves the set of cells -- and so every factor a
        # reader can look up -- unchanged.
        axis1, axis2, surface = _ascending_grid(axis1, axis2, surface)

        # Through the same parent resolution prediction and design assembly
        # use.  A spline-mode ``OrderedCategorical`` parent contributes its
        # inner spline on MAPPED SCORES, so the frame holds level labels while
        # the grid axis is in score space; reading the column as float64
        # directly raised ``could not convert string to float`` and took every
        # rating-table export of such a model down with it.
        parent1, parent2 = term["parent_names"]
        # The exporter's rule for whether a grid is a lookup, applied here so
        # the two decline the same terms.  Without it the label column reached
        # ``np.asarray(..., dtype=np.float64)`` below and the whole export died
        # on ``could not convert string to float``.
        offender = unpositionable_grid_parent(ispec, model._specs)
        if offender is not None:
            raise NotImplementedError(
                f"Interaction {name!r} reconstructs a numeric grid, but its parent "
                f"{offender!r} is a Categorical whose values have no position on a "
                "grid axis, so the surface is not a lookup table for it. Use an "
                "OrderedCategorical parent, whose level scores do have positions, "
                "or reconstruct the term as a cell table."
            )
        left_spec, right_spec = term.get("parent_specs", (None, None))
        _, values1 = resolve_interaction_parent_of(ispec, left_spec, frame.column_array(parent1))
        _, values2 = resolve_interaction_parent_of(ispec, right_spec, frame.column_array(parent2))
        index1 = _nearest_grid_index(axis1, np.asarray(values1, dtype=np.float64))
        index2 = _nearest_grid_index(axis2, np.asarray(values2, dtype=np.float64))
        total_delta += surface[index1, index2] - log_rel_exact

        n_cells = len(axis1) * len(axis2)
        cell = index1 * len(axis2) + index2
        label1, label2 = _axis_column_labels(parent1, parent2)
        interaction_tables[name] = pd.DataFrame(
            {
                label1: np.repeat(axis1, len(axis2)),
                label2: np.tile(axis2, len(axis1)),
                "relativity": np.exp(surface).ravel(),
                "log_relativity": surface.ravel(),
                "n_obs": np.bincount(cell, minlength=n_cells),
                "sample_weight": np.bincount(cell, weights=evaluation_weight, minlength=n_cells),
            }
        )

    # Discretized predictions
    eta_disc = stabilize_eta(eta_orig + total_delta, model._link)
    predictions = clip_mu(model._link.inverse(eta_disc), model._distribution)

    # Compute metrics
    dist = model._distribution
    dev_orig_unit = dist.deviance_unit(y, original_predictions)
    dev_disc_unit = dist.deviance_unit(y, predictions)
    deviance_original = float(np.sum(evaluation_weight * dev_orig_unit))
    deviance_discretized = float(np.sum(evaluation_weight * dev_disc_unit))
    deviance_change = deviance_discretized - deviance_original
    deviance_change_pct = (
        100.0 * deviance_change / deviance_original if deviance_original > 0 else 0.0
    )

    # Prediction-only summaries follow geometry semantics: literal frequency
    # rows for non-Tweedie, physical rows for Tweedie prior weights.
    safe_orig = np.maximum(np.abs(original_predictions), 1e-300)
    abs_pct_change = np.abs(predictions - original_predictions) / safe_orig * 100.0
    summary_active = geometry_weight > 0.0
    summary_weight = geometry_weight[summary_active]
    mean_abs_prediction_change_pct = float(
        np.average(abs_pct_change[summary_active], weights=summary_weight)
    )

    metrics = {
        "deviance_original": deviance_original,
        "deviance_discretized": deviance_discretized,
        "deviance_change": deviance_change,
        "deviance_change_pct": deviance_change_pct,
        "max_abs_prediction_change_pct": float(np.max(abs_pct_change[summary_active])),
        "mean_abs_prediction_change_pct": mean_abs_prediction_change_pct,
        "prediction_correlation": _weighted_correlation(
            original_predictions,
            predictions,
            geometry_weight,
        ),
    }

    return DiscretizationResult(
        tables=tables,
        predictions=predictions,
        original_predictions=original_predictions,
        metrics=metrics,
        interaction_tables=interaction_tables,
    )
