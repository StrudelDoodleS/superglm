"""Design-matrix building: R_inv, discretization, feature auto-detection,
interaction dispatch and the main builder.

Every function takes explicit state (specs, feature_order, lambda2, ...) rather
than a model instance.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm._frame import EagerFrame
from superglm._group_matrix._binned_codes import carry_binned_codes, certify_binned_codes
from superglm._group_matrix._level_partition import carry_level_support, certify_level_partitions
from superglm._predictor_compiler import (
    CompiledPredictorDesign,
    compile_predictor_design,
)
from superglm._roundoff import PENALTY_RANK_CUT, UNIT_ROUNDOFF
from superglm._utils import _validate_strict_prior_weights
from superglm.distributions import Distribution, Tweedie, resolve_distribution
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
    _discretize_column,
)
from superglm.links import Link, resolve_link
from superglm.solvers._structured.selection import carry_nesting_cache
from superglm.solvers.dispersion import PRIOR_WEIGHTS, validate_weight_semantics
from superglm.types import DiscreteTensorBuildResult, FeatureSpec, GroupInfo, GroupSlice

logger = logging.getLogger(__name__)


def validate_term_name_namespace(
    specs: dict[str, FeatureSpec],
    interaction_specs: dict[str, Any],
    *,
    additional_interactions: Iterable[str] = (),
) -> None:
    """Require one public name to identify exactly one model term."""
    interaction_names = set(interaction_specs)
    interaction_names.update(additional_interactions)
    collisions = sorted(set(specs).intersection(interaction_names))
    if collisions:
        names = ", ".join(repr(name) for name in collisions)
        raise ValueError(
            f"Term name(s) {names} are registered as both a main feature and an interaction."
        )


def validate_fitted_group_names(groups: list[GroupSlice]) -> None:
    """Reject fitted coefficient groups whose public names would alias."""
    counts: dict[str, int] = {}
    for group in groups:
        counts[group.name] = counts.get(group.name, 0) + 1
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Generated fitted group names must be unique; found {duplicates!r}.")


# ═══════════════════════════════════════════════════════════════════
# Pure computation helpers
# ═══════════════════════════════════════════════════════════════════


def _resolve_lambda2(lambda2: float | dict) -> float:
    """Resolve lambda2 to a scalar (dict → 1.0 fallback for initial basis)."""
    if isinstance(lambda2, dict):
        return 1.0
    return lambda2


def compute_R_inv(
    B: sp.spmatrix | NDArray,
    omega: NDArray,
    sample_weight: NDArray,
    lambda2: float | dict,
) -> NDArray:
    """Compute SSP reparametrisation matrix R_inv without forming B @ R_inv.

    Wood (2011) Section 3.1 / Section 5: absorb penalty into parameterization.
    R = chol(B'WB/n + λΩ + εI)^T, then R_inv = R^{-1} so that the SSP basis
    X_ssp = B @ R_inv has near-identity X'WX regardless of λ.

    The normaliser is the weight sum over every row, including rows where the
    block is zero, so every block's ``X'WX`` sits on the same scale.
    """
    lam2 = _resolve_lambda2(lambda2)
    total = float(np.sum(sample_weight))
    if total <= 0.0:
        G = np.zeros((omega.shape[0], omega.shape[0]), dtype=np.float64)
    elif sp.issparse(B):
        G = np.asarray((B.multiply(sample_weight[:, None]).T @ B).todense()) / total
    else:
        G = (B * sample_weight[:, None]).T @ B / total
    return _whitening_map(G, omega, lam2)


def compute_projected_R_inv(
    B: sp.spmatrix | NDArray,
    projection: NDArray,
    penalty_sub: NDArray,
    sample_weight: NDArray,
    lambda2: float | dict,
) -> NDArray:
    """Compute SSP R_inv within a projected subspace (linear-split range space).

    Normalised over every row, as in ``compute_R_inv``.
    """
    lam2 = _resolve_lambda2(lambda2)
    total = float(np.sum(sample_weight))
    if total <= 0.0:
        G_full = np.zeros((projection.shape[0], projection.shape[0]), dtype=np.float64)
    elif sp.issparse(B):
        G_full = np.asarray((B.multiply(sample_weight[:, None]).T @ B).todense()) / total
    else:
        G_full = (B * sample_weight[:, None]).T @ B / total
    return _whitening_map(projection.T @ G_full @ projection, penalty_sub, lam2)


def _whitening_map(G: NDArray, omega: NDArray, lam2: float) -> NDArray:
    """``R^-1`` with ``R'R = G + lam2 Omega + 1e-8 I``: the SSP basis.

    Formed, ``lam2 Omega`` rounds by ``u lam2 ||Omega||`` on its null space
    too, where only ``G`` and the ridge hold the sum positive, so near
    ``lam2 = 1e14`` the sum can be indefinite.  It is then factored in
    ``Omega``'s eigenbasis, found without ``lam2``, with the eigenvalues
    under the component's rank cut set to zero: positive definite at any
    ``lam2`` (Wood, Pya & Safken 2016, JASA 111, sec. 3.1.1).
    """
    ridge = np.eye(omega.shape[0]) * 1e-8
    try:
        return np.linalg.inv(np.linalg.cholesky(G + lam2 * omega + ridge).T)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(omega)
        values = np.where(values > PENALTY_RANK_CUT * max(values[-1], 1e-12), values, 0.0)
        rotated = vectors.T @ G @ vectors + np.diag(lam2 * values) + ridge
        return vectors @ np.linalg.inv(np.linalg.cholesky(rotated).T)


def _dominant_penalty_eigenbasis(
    info: GroupInfo, sample_weight: NDArray, lambda2: float | dict
) -> NDArray | None:
    """The eigenbasis of a raw block's penalty where it outweighs the data by ``1/sqrt(u)``.

    In the block's own coordinates ``lambda Omega`` then rounds into its null
    space by ``u lambda ||Omega||``, past ``sqrt(u)`` of the data's curvature
    there.  In the eigenbasis, found without lambda, the null space lies on
    coordinate axes and keeps the formed penalty to second order (Wood, Pya
    & Safken 2016, sec. 3.1.1), and the penalty dominates every range
    direction, so the turn costs the data's column scaling nothing.
    """
    omega = info.penalty_matrix
    if omega is None or info.columns is None or len(info.penalty_components or ()) > 1:
        return None
    penalty = _resolve_lambda2(lambda2) * float(np.max(np.diag(omega)))
    columns = sp.csr_matrix(info.columns)
    data = float(np.max(columns.multiply(columns).T @ sample_weight, initial=0.0))
    if penalty * np.sqrt(UNIT_ROUNDOFF) <= data:
        return None
    return np.linalg.eigh(omega)[1]


def should_discretize(spec: FeatureSpec, model_discrete: bool) -> bool:
    """Check if a feature spec should use fit-time discretization."""
    from superglm.features.spline import _SplineBase

    if not isinstance(spec, _SplineBase):
        return False
    if spec.penalty != "ssp":
        return False
    if spec.discrete is not None:
        return spec.discrete
    return model_discrete


def should_discretize_tensor_interaction(
    ispec: Any, specs: dict[str, FeatureSpec], model_discrete: bool
) -> bool:
    """Check if a tensor interaction should use fit-time discretization."""
    from superglm.features.interaction import TensorInteraction
    from superglm.features.ordered_categorical import OrderedCategorical

    if any(
        isinstance(specs.get(p), OrderedCategorical) for p in getattr(ispec, "parent_names", ())
    ):
        # OC margins live on <= n_levels score points; there is nothing for
        # fit-time discretization to compress, and discretizing
        # reads raw columns as float64, which label data cannot be.
        return False
    if not isinstance(ispec, TensorInteraction):
        return False
    p1, p2 = ispec.parent_names
    return should_discretize(specs[p1], model_discrete) and should_discretize(
        specs[p2], model_discrete
    )


def should_discretize_spline_categorical_interaction(
    ispec: Any, specs: dict[str, FeatureSpec], model_discrete: bool
) -> bool:
    """Check if a spline-categorical interaction should use spline support compression."""
    from superglm.features.interaction import SplineCategorical
    from superglm.features.ordered_categorical import OrderedCategorical

    if any(
        isinstance(specs.get(p), OrderedCategorical) for p in getattr(ispec, "parent_names", ())
    ):
        # OC margins live on <= n_levels score points; there is nothing for
        # fit-time discretization to compress, and discretizing
        # reads raw columns as float64, which label data cannot be.
        return False
    if not isinstance(ispec, SplineCategorical):
        return False
    spline_name, _cat_name = ispec.parent_names
    return should_discretize(specs[spline_name], model_discrete)


def should_discretize_factor_smooth(ispec: Any, model_discrete: bool) -> bool:
    """Use compact marginal support bins for factor smooths in discrete models."""
    from superglm.features.factor_smooth import FactorSmooth

    return isinstance(ispec, FactorSmooth) and model_discrete


def resolve_discrete_n_bins(
    name: str, spec: FeatureSpec, n_bins_config: int | dict[str, int]
) -> int:
    """Resolve the requested bin count for a discretized feature.

    Feature-level ``spec.n_bins`` takes priority. Otherwise the model-level
    ``n_bins_config`` may be a single int or a per-feature dict with a fallback
    of 256 for unspecified features.
    """
    n_bins = getattr(spec, "n_bins", None)
    if n_bins is None:
        if isinstance(n_bins_config, dict):
            n_bins = n_bins_config.get(name, 256)
        else:
            n_bins = n_bins_config

    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError(f"n_bins for feature '{name}' must be >= 1, got {n_bins}")
    return n_bins


def knot_geometry_weight(sample_weight, weight_semantics: str):
    """The weights that place a spline's knots: the rows' own under prior weights.

    Frequency weights are replicated rows and place knots as such. Prior
    weights say how precisely a row was measured, so knots follow the physical
    rows, less those of zero weight, which were not observed at all.
    """
    if sample_weight is None or weight_semantics != PRIOR_WEIGHTS:
        return sample_weight
    return (np.asarray(sample_weight, dtype=np.float64) > 0.0).astype(np.float64)


def _discretize_spline_column(
    x: NDArray,
    n_bins: int,
    geometry_weight: NDArray | None,
    knots: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Discretize on the positive-frequency support and map every fit row.

    ``knots`` are the knots the bins follow (``bin_knots``: the spline's,
    plus its ranges' cuts), so the bins resolve its pieces
    (``discretize_column``), and the active rows count by their geometry
    weight, as they do for the knots. Zero-frequency rows keep a valid bin index (the
    design keeps its physical row count) but do not shape the support: they
    map to the nearest active support point, as clipped spline evaluation
    would.
    """
    values = np.asarray(x, dtype=np.float64).ravel()
    if geometry_weight is None:
        return _discretize_column(values, n_bins, knots)

    weights = np.asarray(geometry_weight, dtype=np.float64).ravel()
    active = weights > 0.0
    if np.all(active):
        return _discretize_column(values, n_bins, knots, weights)

    support, active_index = _discretize_column(values[active], n_bins, knots, weights[active])
    if len(support) == 1:
        full_index = np.zeros(len(values), dtype=np.intp)
    else:
        insertion = np.searchsorted(support, values)
        right = np.clip(insertion, 0, len(support) - 1)
        left = np.clip(insertion - 1, 0, len(support) - 1)
        choose_right = np.abs(values - support[right]) <= np.abs(values - support[left])
        full_index = np.where(choose_right, right, left).astype(np.intp)
    full_index[active] = active_index
    return support, full_index


# ═══════════════════════════════════════════════════════════════════
# Feature auto-detection and interaction dispatch
# ═══════════════════════════════════════════════════════════════════


def auto_detect_features(
    X: EagerFrame,
    sample_weight: NDArray | None,
    *,
    spline_cols: list[str],
    knots_map: dict[str, int],
    degree: int,
    categorical_base: str,
    specs: dict[str, FeatureSpec],
    feature_order: list[str],
) -> None:
    """Auto-detect feature types from native dataframe columns.

    Mutates ``specs`` and ``feature_order`` in place.
    """
    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.spline import PSpline

    lines = ["SuperGLM features:"]
    for raw_col in X.columns:
        col = cast(str, raw_col)
        kind = X.column_kind(col)
        if col in spline_cols:
            nk = knots_map[col]
            spec = PSpline(n_knots=nk, degree=degree, penalty="ssp")
            specs[col] = spec
            feature_order.append(col)
            lines.append(f"  {str(col):<20s} → Spline(kind='ps', n_knots={nk}, degree={degree})")
        elif kind == "categorical":
            base = categorical_base
            if base == "most_exposed" and sample_weight is None:
                base = "first"
            spec = Categorical(base=base)
            specs[col] = spec
            feature_order.append(col)
            lines.append(f"  {str(col):<20s} → Categorical(base={base})")
        elif kind in ("numeric", "boolean"):
            spec = Numeric()
            specs[col] = spec
            feature_order.append(col)
            lines.append(f"  {str(col):<20s} → Numeric()")
        else:
            raise ValueError(
                f"X column {col!r} has unsupported dtype {X.column_dtype(col)!r} "
                "for automatic feature detection; configure the feature explicitly "
                "or convert the column to numeric, boolean, string, or categorical data"
            )
    logger.info("\n".join(lines))


def _spec_kind(spec: FeatureSpec) -> str:
    """Classify a feature spec into one of the four canonical kinds."""
    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.polynomial import Polynomial
    from superglm.features.spline import _SplineBase

    if isinstance(spec, _SplineBase):
        return "spline"
    if isinstance(spec, Polynomial):
        return "polynomial"
    if isinstance(spec, Numeric):
        return "numeric"
    if isinstance(spec, Categorical):
        return "categorical"
    if isinstance(spec, OrderedCategorical):
        # Always the smooth: ``add_interaction`` refuses an OrderedCategorical
        # without an inner spline before it reaches here.
        return "spline"
    return type(spec).__name__


# ── Interaction factories ─────────────────────────────────────────
# Each returns (iname, ispec) given (feat1, feat2, name, **kwargs).
# For asymmetric pairs the factory receives the canonical orientation
# (e.g. spline first, categorical second).


def _make_spline_categorical(f1: str, f2: str, *, name: str | None, **kw: Any) -> tuple[str, Any]:
    from superglm.features.interaction import SplineCategorical

    return name or f"{f1}:{f2}", SplineCategorical(f1, f2)


def _make_polynomial_categorical(
    f1: str, f2: str, *, name: str | None, **kw: Any
) -> tuple[str, Any]:
    from superglm.features.interaction import PolynomialCategorical

    return name or f"{f1}:{f2}", PolynomialCategorical(f1, f2)


def _make_numeric_categorical(f1: str, f2: str, *, name: str | None, **kw: Any) -> tuple[str, Any]:
    from superglm.features.interaction import NumericCategorical

    return name or f"{f1}:{f2}", NumericCategorical(f1, f2)


def _make_categorical_interaction(
    f1: str, f2: str, *, name: str | None, **kw: Any
) -> tuple[str, Any]:
    from superglm.features.interaction import CategoricalInteraction

    return name or f"{f1}:{f2}", CategoricalInteraction(f1, f2)


def _make_numeric_interaction(f1: str, f2: str, *, name: str | None, **kw: Any) -> tuple[str, Any]:
    from superglm.features.interaction import NumericInteraction

    return name or f"{f1}:{f2}", NumericInteraction(f1, f2)


def _make_polynomial_interaction(
    f1: str, f2: str, *, name: str | None, **kw: Any
) -> tuple[str, Any]:
    from superglm.features.interaction import PolynomialInteraction

    return name or f"{f1}:{f2}", PolynomialInteraction(f1, f2)


def _make_tensor_interaction(f1: str, f2: str, *, name: str | None, **kw: Any) -> tuple[str, Any]:
    from superglm.features.interaction import TensorInteraction

    return name or f"{f1}:{f2}", TensorInteraction(f1, f2, **kw)


_INTERACTION_FACTORIES: dict[tuple[str, str], Any] = {
    ("spline", "categorical"): _make_spline_categorical,
    ("polynomial", "categorical"): _make_polynomial_categorical,
    ("numeric", "categorical"): _make_numeric_categorical,
    ("categorical", "categorical"): _make_categorical_interaction,
    ("numeric", "numeric"): _make_numeric_interaction,
    ("polynomial", "polynomial"): _make_polynomial_interaction,
    ("spline", "spline"): _make_tensor_interaction,
}


def _shared_row_support(B_csr, supports: dict | None):
    """``detect_row_support`` once per basis object within one build.

    Groups built over one basis object -- a ``decompose=True`` tensor's
    bilinear and wiggly subgroups, which differ only in ``R_inv`` -- then hold
    one support block and one row index by reference.  Each entry keeps its
    basis alive, so the ``id`` key cannot be reused while ``supports`` lives.
    """
    from superglm._group_matrix._group_matrix_support import detect_row_support, stored_csr_bytes

    entry = None if supports is None else supports.get(id(B_csr))
    if entry is not None and entry[0] is B_csr:
        return entry[1]
    detected = detect_row_support(B_csr, replaced_bytes=stored_csr_bytes(B_csr))
    if supports is not None:
        supports[id(B_csr)] = (B_csr, detected)
    return detected


def _build_ssp_group(B_csr, R_inv, supports: dict | None = None, row_support=None):
    """Cheapest exact representation of a factored SSP block.

    ``row_support`` is what its term found (``GroupInfo.row_support``), else
    ``None`` to scan ``B_csr``.  Compression is lossless row deduplication,
    never binning, independent of ``discrete=True``; it declines when the
    cost model says CSR is cheaper.  The gate charges ``S + 8n`` retained
    (support block, row index) plus ``2 * 8 * n_support * p`` for the Gram's
    projected support and weighted copy.  Groups sharing a basis through
    ``supports`` hold the block and index once, and their widths sum to the
    basis width, so their Gram products total at most ``2S``.
    """
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )

    if row_support is None:
        detected = _shared_row_support(B_csr, supports)
    else:
        detected = row_support if isinstance(row_support, tuple) else None
    if detected is None:
        return SparseSSPGroupMatrix(B_csr, R_inv)
    B_unique_rows, row_index = detected
    return SupportCompressedSSPGroupMatrix(B_unique_rows, R_inv, row_index)


def _build_unpenalized_sparse_group(B_csr, n_cols: int):
    """Cheapest exact representation of an unpenalized sparse block.

    Only for groups that opt in via ``GroupInfo.supports_row_compression``
    (``Piecewise``): the compressed container is a ``DiscretizedSSPGroupMatrix``
    subclass, which would change the tabmat split and design reporting of
    other term types.  The same lossless row-dedup gate as
    ``_build_ssp_group``, with an identity reparameterisation and ``omega``
    ``None`` (unpenalized); declines to the plain CSR wrapper when the cost
    model says compression would not pay.
    """
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )
    from superglm._group_matrix._group_matrix_support import detect_row_support, stored_csr_bytes

    detected = detect_row_support(B_csr, replaced_bytes=stored_csr_bytes(B_csr))
    if detected is None:
        return SparseGroupMatrix(B_csr)
    B_unique_rows, row_index = detected
    return SupportCompressedSSPGroupMatrix(
        B_unique_rows, np.eye(n_cols, dtype=np.float64), row_index
    )


def refuse_unsupported_interaction_parents(
    feat1: str,
    feat2: str,
    specs: Mapping[str, FeatureSpec],
    *,
    ranged_margins: bool = False,
) -> None:
    """Refuse a main-effect spec that cannot parent an interaction.

    ``add_interaction`` runs this when it resolves a ``(left, right)`` pair,
    and the design compiler runs it for every interaction declared as a spec
    (``terms=``, the family helpers' ``ti`` and ``interaction``), except a
    ``FactorSmooth``, which builds its own marginal and reads its group as
    labels.  A parent absent from ``specs`` is left to the caller's own check.

    ``ranged_margins`` (SuperLSS) lets a parent with polynomial ranges through;
    its margin keeps the ranges, so the fitted predictor stays polynomial on
    them.
    """
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.spline import _SplineBase

    # An OrderedCategorical without an inner spline (an old pickle's step mode)
    # has no marginal smooth to cross with, and a specials term's free levels
    # have no spline-axis position: refuse both here, not mid-build.
    for parent in (feat1, feat2):
        spec = specs.get(parent)
        if isinstance(spec, OrderedCategorical) and (
            spec.basis != "spline" or spec._spline is None
        ):
            raise NotImplementedError(
                f"cannot add the interaction ({feat1!r}, {feat2!r}): {parent!r} is an "
                "OrderedCategorical without an inner spline -- a spec built before "
                "step mode was removed in 0.24.0 -- and cannot parent an interaction; "
                "rebuild it with basis=Spline(...) for a smoothed ordinal parent or "
                "a Categorical feature for unsmoothed level effects."
            )
        if isinstance(spec, OrderedCategorical) and spec.has_specials:
            raise NotImplementedError(
                f"cannot add the interaction ({feat1!r}, {feat2!r}): {parent!r} is an "
                f"OrderedCategorical with specials={spec._specials!r}, whose free "
                "levels have no position on the spline axis and so no marginal smooth "
                "to cross with; drop specials= to interact the smoothed ordinal "
                "parent, or use a Categorical feature for unsmoothed level effects."
            )
        if isinstance(spec, OrderedCategorical) and spec.basis_kind != "spline":
            # Same refusal as resolve_interaction_parent, raised HERE at
            # registration so the pair fails where it is written rather than
            # mid design-matrix build after the caller committed a fit.
            raise NotImplementedError(
                f"cannot add the interaction ({feat1!r}, {feat2!r}): {parent!r} is an "
                f"OrderedCategorical(basis={type(spec._basis_spline).__name__}(...)), "
                "an unpenalized parametric block with no marginal smooth to cross "
                "with; use basis=Spline(...) for an interactable ordinal parent, or "
                "a Categorical feature for level-by-level structure."
            )
        # SuperGLM refuses a ranged parent: its interaction screen's probe
        # (keep_ranges=False) has no constrained basis.  SuperLSS passes
        # ranged_margins.
        spline = spec._spline if isinstance(spec, OrderedCategorical) else spec
        if isinstance(spline, _SplineBase) and spline.polynomial_ranges:
            if ranged_margins:
                continue
            raise NotImplementedError(
                f"cannot add the interaction ({feat1!r}, {feat2!r}): {parent!r} has "
                "polynomial_ranges, which shape a main effect only in this version; "
                "declare the interaction on a spline without them."
            )


def add_interaction(
    feat1: str,
    feat2: str,
    specs: dict[str, FeatureSpec],
    interaction_specs: dict[str, Any],
    interaction_order: list[str],
    name: str | None = None,
    **kwargs: Any,
) -> None:
    """Register an interaction between two already-registered features.

    Mutates ``interaction_specs`` and ``interaction_order`` in place.
    """
    if feat1 not in specs:
        raise ValueError(f"Parent feature not found: {feat1}")
    if feat2 not in specs:
        raise ValueError(f"Parent feature not found: {feat2}")
    refuse_unsupported_interaction_parents(feat1, feat2, specs)

    kind1 = _spec_kind(specs[feat1])
    kind2 = _spec_kind(specs[feat2])

    factory = _INTERACTION_FACTORIES.get((kind1, kind2))
    if factory is not None:
        iname, ispec = factory(feat1, feat2, name=name, **kwargs)
    else:
        # Try swapped orientation (asymmetric pairs like categorical+spline)
        factory = _INTERACTION_FACTORIES.get((kind2, kind1))
        if factory is not None:
            iname, ispec = factory(feat2, feat1, name=name, **kwargs)
        else:
            raise TypeError(
                f"Cannot create interaction between {kind1} "
                f"and {kind2}. Supported: {', '.join('+'.join(k) for k in _INTERACTION_FACTORIES)}."
            )

    if iname in interaction_specs:
        raise ValueError(f"Interaction already added: {iname}")
    validate_term_name_namespace(
        specs,
        interaction_specs,
        additional_interactions=(iname,),
    )

    interaction_specs[iname] = ispec
    interaction_order.append(iname)


# ═══════════════════════════════════════════════════════════════════
# Design-matrix builder
# ═══════════════════════════════════════════════════════════════════


def _priced_group_dimension(
    info: GroupInfo, n_cols: int, group_pricing: str
) -> tuple[int, int | None]:
    """``(p_g, penalty_dim)``: what the penalty prices a group at, and its stamp.

    ``info.penalty_width`` is the width the term spans when its spec withholds columns the
    design cannot identify. ``"rank"`` (default) prices at the emitted, identifiable width with
    stamp ``None``, calibrating the ``sqrt(p_g)`` weight to the group's identifiable df (Yuan &
    Lin 2006; Meier, van de Geer & Buhlmann 2008; Breheny & Huang 2015, sec. 2.1).
    ``"spanned"`` prices at the spanned width, so pruning a dead column is a pure
    reparametrisation. The group-lasso weight and the Breheny-Huang df allocation both read it.
    """
    if info.penalty_width is None or group_pricing == "rank":
        return int(n_cols), None
    return int(info.penalty_width), int(info.penalty_width)


def _validate_group_pricing(group_pricing: str) -> None:
    if group_pricing not in ("rank", "spanned"):
        raise ValueError(f"group_pricing must be 'rank' or 'spanned', got {group_pricing!r}")


def _process_info(
    info: GroupInfo,
    *,
    B_unique: NDArray | None = None,
    bin_idx: NDArray | None = None,
    sample_weight: NDArray,
    exposure_agg: NDArray | None = None,
    lambda2: float | dict,
    tensor_build: DiscreteTensorBuildResult | None = None,
    tensor_id: int = -1,
    ssp_supports: dict | None = None,
    eigenbasis_turn: bool = True,
) -> tuple[GroupMatrix, NDArray | None, int]:
    """Compute R_inv and construct a GroupMatrix from a single GroupInfo.

    Returns ``(group_matrix, r_inv_or_none, n_cols)`` where *r_inv_or_none*
    is the R_inv column block (for collecting into combined R_inv) or None
    if no reparametrization was applied.  ``ssp_supports`` is shared by the
    calls of one term, so subgroups over one basis share its row support.
    ``eigenbasis_turn=False`` keeps a raw block in its declared coordinates.
    """
    use_discrete = B_unique is not None
    use_tensor = tensor_build is not None
    R_inv: NDArray | None = None
    # R_inv_local: SSP transform in post-identifiability space (projected -> solver).
    # Used to compose constraints that are already in post-identifiability space.
    R_inv_local: NDArray | None = None
    is_spline_cat = info.spline_cat_mask is not None and (
        info.spline_cat_basis is not None
        or (info.spline_cat_basis_unique is not None and info.spline_cat_bin_idx is not None)
    )

    if is_spline_cat:
        mask = np.asarray(info.spline_cat_mask, dtype=bool)
        row_idx = np.flatnonzero(mask)
        if info.spline_cat_basis_unique is not None and info.spline_cat_bin_idx is not None:
            B_support: sp.spmatrix | NDArray = np.asarray(
                info.spline_cat_basis_unique,
                dtype=np.float64,
            )
            bin_idx_level = np.asarray(info.spline_cat_bin_idx, dtype=np.intp)
            if not info.spline_cat_bin_idx_is_level:
                bin_idx_level = bin_idx_level[row_idx]
            level_weight = np.bincount(
                bin_idx_level,
                weights=sample_weight[row_idx],
                minlength=B_support.shape[0],
            )
            B_for: sp.spmatrix | NDArray = B_support
            use_discrete_spline_cat = True
        else:
            B_full = sp.csr_matrix(info.spline_cat_basis)
            B_level = B_full[row_idx]
            B_for = B_level
            level_weight = sample_weight[row_idx]
            use_discrete_spline_cat = False
        if info.projection is not None:
            P = info.projection
            if info.reparametrize and info.penalty_matrix is not None:
                R_inv_local = compute_projected_R_inv(
                    B_for,
                    P,
                    info.penalty_matrix,
                    level_weight,
                    lambda2,
                )
                R_inv = P @ R_inv_local
            else:
                R_inv = P
                R_inv_local = None
            omega_full = P @ info.penalty_matrix @ P.T if info.penalty_matrix is not None else None
        elif info.reparametrize and info.penalty_matrix is not None:
            R_inv = compute_R_inv(B_for, info.penalty_matrix, level_weight, lambda2)
            R_inv_local = R_inv
            omega_full = info.penalty_matrix
        else:
            R_inv = np.eye(info.n_cols, dtype=np.float64)
            R_inv_local = None
            omega_full = info.penalty_matrix

        n_cols = R_inv.shape[1]
        if use_discrete_spline_cat:
            # Same storage, different contract: the lossless support indexes
            # exact distinct rows, so the class must not claim the binned route.
            spline_cat_cls = (
                SupportCompressedSplineCategoricalGroupMatrix
                if info.spline_cat_support_lossless
                else DiscretizedSplineCategoricalGroupMatrix
            )
            if info.spline_cat_bin_idx_is_level:
                gm = spline_cat_cls(
                    B_support,
                    R_inv,
                    bin_idx_level,
                    row_idx,
                    n_rows=len(mask),
                    bin_idx_is_level=True,
                )
            else:
                gm = spline_cat_cls(
                    B_support,
                    R_inv,
                    info.spline_cat_bin_idx,
                    row_idx,
                )
        else:
            gm = SplineCategoricalGroupMatrix(B_full, R_inv, row_idx)
        gm.spline_cat_level = info.spline_cat_level
        gm.spline_cat_feature = info.spline_cat_feature
        if omega_full is not None:
            gm.omega = omega_full
        if info.projection is not None:
            gm.projection = info.projection

    elif info.projection is not None:
        P = info.projection
        if info.reparametrize and info.penalty_matrix is not None:
            B_for = B_unique if use_discrete else info.columns
            exp_for = exposure_agg if use_discrete else sample_weight
            R_inv_local = compute_projected_R_inv(B_for, P, info.penalty_matrix, exp_for, lambda2)
            R_inv = P @ R_inv_local
        else:
            R_inv_local = (
                _dominant_penalty_eigenbasis(info, sample_weight, lambda2)
                if eigenbasis_turn
                else None
            )
            R_inv = P if R_inv_local is None else P @ R_inv_local
        n_cols = R_inv.shape[1]
        omega_full = P @ info.penalty_matrix @ P.T if info.penalty_matrix is not None else None
        if use_tensor:
            gm: GroupMatrix = DiscretizedTensorGroupMatrix(
                tensor_build.B1_unique,
                tensor_build.B2_unique,
                tensor_build.idx1,
                tensor_build.idx2,
                B_unique,
                R_inv,
                bin_idx,
                tensor_id=tensor_id,
                raw_channels=tensor_build.raw_channels,
            )
        elif use_discrete:
            gm = DiscretizedSSPGroupMatrix(B_unique, R_inv, bin_idx)
        elif info.row_support is not None or sp.issparse(info.columns):
            gm = _build_ssp_group(info.columns, R_inv, ssp_supports, info.row_support)
        else:
            gm = DenseGroupMatrix(info.columns @ R_inv)
        if omega_full is not None and hasattr(gm, "omega"):
            gm.omega = omega_full
        if hasattr(gm, "projection"):
            gm.projection = P
        if info.penalty_components is not None and hasattr(gm, "omega_components"):
            gm.omega_components = [
                (suffix, P @ omega_j @ P.T) for suffix, omega_j in info.penalty_components
            ]
            gm.component_types = info.component_types
            if info.lambda_policies is not None:
                gm.lambda_policies = info.lambda_policies

    elif info.reparametrize and info.penalty_matrix is not None:
        B_for = B_unique if use_discrete else info.columns
        exp_for = exposure_agg if use_discrete else sample_weight
        R_inv = compute_R_inv(B_for, info.penalty_matrix, exp_for, lambda2)
        # No projection — constraints are in raw space, same as R_inv input
        R_inv_local = R_inv
        n_cols = R_inv.shape[1]
        if use_tensor:
            gm = DiscretizedTensorGroupMatrix(
                tensor_build.B1_unique,
                tensor_build.B2_unique,
                tensor_build.idx1,
                tensor_build.idx2,
                B_unique,
                R_inv,
                bin_idx,
                tensor_id=tensor_id,
                raw_channels=tensor_build.raw_channels,
            )
            gm.omega = info.penalty_matrix
        elif use_discrete:
            gm = DiscretizedSSPGroupMatrix(B_unique, R_inv, bin_idx)
            gm.omega = info.penalty_matrix
        elif info.row_support is not None or sp.issparse(info.columns):
            gm = _build_ssp_group(info.columns, R_inv, ssp_supports, info.row_support)
            gm.omega = info.penalty_matrix
        else:
            gm = DenseGroupMatrix(info.columns @ R_inv)
        if info.penalty_components is not None and hasattr(gm, "omega_components"):
            gm.omega_components = info.penalty_components
            gm.component_types = info.component_types
            if info.lambda_policies is not None:
                gm.lambda_policies = info.lambda_policies

    else:
        n_cols = info.n_cols
        R_inv = None
        if use_tensor:
            gm = DiscretizedTensorGroupMatrix(
                tensor_build.B1_unique,
                tensor_build.B2_unique,
                tensor_build.idx1,
                tensor_build.idx2,
                B_unique,
                np.eye(info.n_cols, dtype=np.float64),
                bin_idx,
                tensor_id=tensor_id,
                raw_channels=tensor_build.raw_channels,
            )
            if info.penalty_matrix is not None:
                gm.omega = info.penalty_matrix
            if info.penalty_components is not None:
                gm.omega_components = info.penalty_components
                gm.component_types = info.component_types
                if info.lambda_policies is not None:
                    gm.lambda_policies = info.lambda_policies
        elif info.structured_kind == "factor_smooth":
            factor_basis = (
                info.factor_smooth_basis_unique
                if info.factor_smooth_basis_unique is not None
                else info.factor_smooth_basis
            )
            gm = FactorSmoothGroupMatrix(
                factor_basis,
                info.factor_smooth_codes,
                info.factor_smooth_n_levels,
                natural_map=info.factor_smooth_transform,
                levels=info.factor_smooth_levels,
                repeated_penalty_components=info.repeated_penalty_components,
                factor_basis=info.factor_smooth_factor_basis,
                lambda_policies=info.lambda_policies,
                bin_idx=info.factor_smooth_bin_idx,
            )
        elif info.structured_kind == "random_effect":
            gm = RandomEffectGroupMatrix(
                info.cat_codes,
                info.n_cols,
                lambda_policies=info.lambda_policies,
                nested_in=info.random_effect_nested_in,
            )
        elif info.cat_codes is not None:
            gm = CategoricalGroupMatrix(info.cat_codes, info.n_cols)
        elif info.row_support is not None or sp.issparse(info.columns):
            if info.penalty_matrix is not None or info.penalty_components is not None:
                gm = _build_ssp_group(
                    info.columns,
                    np.eye(info.n_cols, dtype=np.float64),
                    ssp_supports,
                    info.row_support,
                )
                if info.penalty_matrix is not None:
                    gm.omega = info.penalty_matrix
                if info.penalty_components is not None:
                    gm.omega_components = info.penalty_components
                    gm.component_types = info.component_types
                    if info.lambda_policies is not None:
                        gm.lambda_policies = info.lambda_policies
            else:
                # Only a tensor leaves its columns to its row support, and a
                # tensor block is penalised or projected: these are sparse.
                assert info.columns is not None
                if info.supports_row_compression:
                    gm = _build_unpenalized_sparse_group(info.columns, info.n_cols)
                else:
                    # cast, not a runtime check: the checker cannot narrow by
                    # the enclosing branch condition, sp.issparse(info.columns).
                    gm = SparseGroupMatrix(cast(sp.spmatrix, info.columns))
        else:
            gm = DenseGroupMatrix(info.columns)

    if info.structural_ranks is not None and hasattr(gm, "structural_ranks"):
        gm.structural_ranks = info.structural_ranks

    # Compose constraints into solver coordinates: build() gives them after
    # projection, and R_inv_local maps projected -> solver (SSP only).
    # Main-effect QP constraints are rebuilt from raw rows in the feature loop.
    if info.constraints is not None and R_inv_local is not None:
        info.constraints = info.constraints.compose(R_inv_local)
    # raw_to_solver_map from build() is the identifiability projection (raw -> projected).
    # Extend it to the full chain (raw -> solver) by composing with R_inv_local.
    if info.raw_to_solver_map is not None and R_inv_local is not None:
        info.raw_to_solver_map = info.raw_to_solver_map @ R_inv_local

    return gm, R_inv, n_cols


@dataclass
class BuildResult:
    """Return value of build_design_matrix."""

    dm: DesignMatrix
    groups: list[GroupSlice]
    distribution: Distribution
    link: Link
    y: NDArray
    sample_weight: NDArray
    offset: NDArray | None
    compiled: CompiledPredictorDesign


def build_design_matrix(
    X: EagerFrame,
    y: NDArray,
    sample_weight: NDArray | None,
    offset: NDArray | None,
    *,
    family: str | Distribution,
    link_spec: str | Link | None,
    specs: dict[str, FeatureSpec],
    feature_order: list[str],
    interaction_specs: dict[str, Any],
    interaction_order: list[str],
    pending_interactions: list[tuple[str, str]],
    model_discrete: bool,
    n_bins_config: int | dict[str, int],
    lambda2: float | dict,
    level_bindings: dict | None = None,
    alias_prune: bool = True,
    separation: str = "warn",
    selection_penalty: object | None = None,
    group_pricing: str = "rank",
    weight_semantics: str,
) -> BuildResult:
    """Build features, groups, and design matrix from specs.

    Returns a BuildResult. Mutates ``interaction_specs``,
    ``interaction_order`` (resolves pending), ``pending_interactions``
    (empties it), and ``specs`` (via set_reparametrisation calls).

    ``alias_prune`` controls only aliased-cell pruning of categorical
    interactions: dropping one picks a representative of the rank deficiency,
    which a group selection penalty would pick differently, so the caller
    passes False whenever a nonzero lambda1 reaches this block.  Empty cells
    are always pruned: their columns are zero in the weighted geometry.

    ``separation`` controls the build-time separated-cell scan (see
    :mod:`superglm.diagnostics.separation`): ``"warn"`` (default) names the
    offending cells in a ``SeparationWarning``, ``"error"`` refuses the
    design with a ``SeparationError``, ``"ignore"`` skips the scan.
    ``selection_penalty`` is the model's lambda1-style penalty object when a
    nonzero lambda1 will reach this fit, else ``None``; terms that penalty
    targets have bounded optima and are exempt from the scan.

    ``group_pricing`` decides the width the selection penalty and df ledger
    price a group at when its spec emits fewer columns than the term spans:
    ``"rank"`` (default) the emitted width, ``"spanned"`` the spanned width
    (``_priced_group_dimension``).

    ``weight_semantics`` decides what ``sample_weight`` says about a row, and
    reaches the build in two places: which weights a strict-positivity check
    applies to, and whether learned geometry follows weight mass or physical
    rows.
    """
    _validate_group_pricing(group_pricing)
    weight_semantics = validate_weight_semantics(weight_semantics)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    distribution = resolve_distribution(family)
    if sample_weight is None:
        sample_weight = np.ones(n, dtype=np.float64)
    elif isinstance(distribution, Tweedie) and weight_semantics == PRIOR_WEIGHTS:
        # The compound-Poisson normalizer carries ``log w``, so a Tweedie prior
        # weight of zero is not an uninformative row but an unevaluable
        # density.  Read as a replication count the weight never enters that
        # normalizer, and zero means what it means everywhere else.
        sample_weight = _validate_strict_prior_weights(sample_weight, n)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
    link = resolve_link(link_spec, distribution)
    from superglm.diagnostics.separation import response_boundaries

    # Build-time separation scan (issues #340/#341): terms recorded during the
    # loops below, scanned once the GroupSlices exist so an active selection
    # penalty can exempt the blocks it bounds.  An empty boundary tuple means
    # this family/link cannot separate, and the scan costs nothing.
    separation_boundaries = (
        response_boundaries(distribution, link) if separation != "ignore" else ()
    )

    # Frequency weights are replicated rows, so learned geometry follows them.
    # Prior weights are precisions, so model geometry (knot placement,
    # discretized bins, Piecewise quantiles and base='most_exposed') follows
    # the physical rows of positive weight.  Polynomial standardization follows
    # sample_weight under both: it spans the same column space either way.
    physical_rows = weight_semantics == PRIOR_WEIGHTS
    geometry_weight = knot_geometry_weight(sample_weight, weight_semantics)
    # The eigenbasis turn is orthogonal within a block, so it keeps the
    # selection penalty's objective only when that penalty reads each group
    # through its norm; an elementwise L1 part, or a penalty not known to,
    # keeps the declared coordinates.
    from superglm.penalties import GroupElasticNet, GroupLasso, Ridge, SparseGroupLasso

    eigenbasis_turn = (
        selection_penalty is None
        or isinstance(selection_penalty, (Ridge, GroupLasso, GroupElasticNet))
        or (isinstance(selection_penalty, SparseGroupLasso) and selection_penalty.alpha == 0)
    )

    compiled = compile_predictor_design(
        X,
        sample_weight,
        geometry_weight=geometry_weight,
        polynomial_weight=sample_weight,
        categorical_reporting_weight=sample_weight,
        ordered_reporting_weight=sample_weight,
        specs=specs,
        feature_order=feature_order,
        interaction_specs=interaction_specs,
        interaction_order=interaction_order,
        pending_interactions=pending_interactions,
        model_discrete=model_discrete,
        n_bins_config=n_bins_config,
        lambda2=lambda2,
        level_bindings=level_bindings,
        physical_rows=physical_rows,
        separation_boundaries=separation_boundaries,
        group_pricing=group_pricing,
        alias_prune=alias_prune,
        own_specs=False,
        separation_response=y,
        eigenbasis_turn=eigenbasis_turn,
    )
    separation_records = compiled.separation_records
    from superglm.features.categorical import Categorical

    for spec in compiled.specs.values():
        if isinstance(spec, Categorical):
            spec._separated_levels_at_fit = ()
    if separation_records:
        from superglm.diagnostics.separation import (
            emit_curve_separation,
            emit_level_curve_separation,
            emit_separation_findings,
            scan_categorical_term,
            scan_interaction_term,
        )
        from superglm.penalties.base import penalty_targets_group

        findings = []
        curves = []
        level_curves = []
        for record in separation_records:
            term_name = record[1]
            if record[0] == "curve":
                # Already acted on while the term was built (its separated
                # directions are held), so a selection penalty exempts
                # nothing here: it is reported whatever bounds the term.
                curves.extend(record[2])
                continue
            if selection_penalty is not None and any(
                penalty_targets_group(selection_penalty, g)
                for g in compiled.groups
                if g.feature_name == term_name
            ):
                # A nonzero lambda1 reaches this block: the penalty grows
                # without bound along any recession direction, so the
                # penalised optimum is finite and the term is estimable.
                if record[0] == "level_curve":
                    compiled.interaction_specs[term_name]._separated_ranges = {}
                continue
            if record[0] == "level_curve":
                level_curves.append(record[1:])
                continue
            if record[0] == "cat":
                _, name_c, spec_c, x_c = record
                found = scan_categorical_term(
                    name_c,
                    spec_c,
                    x_c,
                    y,
                    sample_weight,
                    separation_boundaries,
                    compiled.scan_codes.pop(name_c, None),
                )
                findings.extend(found)
                # predict names these levels when it meets them (_predict_eta)
                spec_c._separated_levels_at_fit = tuple(
                    label for finding in found for label in finding.labels
                )
            else:
                _, name_i, spec1_i, spec2_i, x1_i, x2_i, p1_i, p2_i = record
                findings.extend(
                    scan_interaction_term(
                        name_i,
                        spec1_i,
                        spec2_i,
                        x1_i,
                        x2_i,
                        p1_i,
                        p2_i,
                        y,
                        sample_weight,
                        separation_boundaries,
                    )
                )
        compiled.scan_codes.clear()
        emit_separation_findings(findings, separation)
        emit_curve_separation(curves, separation)
        emit_level_curve_separation(level_curves)

    certify_level_partitions(compiled.design.group_matrices, compiled.design.n)
    certify_binned_codes(compiled.design.group_matrices, compiled.design.n)
    return BuildResult(
        dm=compiled.design,
        groups=list(compiled.groups),
        distribution=distribution,
        link=link,
        y=y,
        sample_weight=sample_weight,
        offset=offset,
        compiled=compiled,
    )


# ═══════════════════════════════════════════════════════════════════
# Design-matrix rebuild with updated lambdas
# ═══════════════════════════════════════════════════════════════════


def _resolve_group_lambda(gm, g, lambdas):
    """Resolve the effective lambda and omega for a group, handling multi-penalty.

    Returns (effective_lambda, effective_omega, has_components) where:
    - For single-penalty groups: (lambdas[g.name], gm.omega, False)
    - For multi-penalty groups: (1.0, sum(lam_j * omega_j), True)
      The effective lambda is 1.0 because it's already baked into the omega.
    """
    if gm.omega_components is not None:
        effective_omega = sum(
            lambdas[f"{g.name}:{suffix}"] * omega_j for suffix, omega_j in gm.omega_components
        )
        return 1.0, effective_omega, True
    return lambdas[g.name], gm.omega, False


def _group_has_lambda(gm, g, lambdas):
    """Check whether lambdas dict contains entries for this group."""
    if g.name in lambdas:
        return True
    if gm.omega_components is not None:
        first_suffix = gm.omega_components[0][0]
        return f"{g.name}:{first_suffix}" in lambdas
    return False


# Discrete tensor interactions are not constrained against one another: ``ti()``
# marginals are centred, so ``ti(a, b) + ti(a, c)`` span jointly identifiable
# surfaces, and a cross-Gram null space would keep the directions a tensor
# shares least, discarding the fitted surface on a holey joint support.  As on
# the exact path, the penalty and shared rank policy handle degenerate directions.


def rebuild_design_matrix_with_lambdas(
    dm: DesignMatrix,
    groups: list[GroupSlice],
    lambdas: dict[str, float],
    sample_weight: NDArray,
    lambda2: float | dict,
) -> DesignMatrix:
    """Rebuild design matrix with per-group smoothing lambdas.

    Only recomputes R_inv for SSP groups whose lambda changed;
    non-SSP groups are reused unchanged.
    """
    new_gms: list[GroupMatrix] = []
    for gm, g in zip(dm.group_matrices, groups):
        if isinstance(gm, SparseSSPGroupMatrix) and _group_has_lambda(gm, g, lambdas):
            if gm.omega is None:
                new_gms.append(gm)
                continue
            lam, omega_eff, has_comp = _resolve_group_lambda(gm, g, lambdas)
            if gm.projection is not None:
                P = gm.projection
                omega_proj = P.T @ omega_eff @ P
                R_inv_local = compute_projected_R_inv(gm.B, P, omega_proj, sample_weight, lam)
                R_inv_new = P @ R_inv_local
            else:
                R_inv_new = compute_R_inv(gm.B, omega_eff, sample_weight, lam)
            new_gm = SparseSSPGroupMatrix(gm.B, R_inv_new)
            new_gm.omega = gm.omega
            new_gm.projection = gm.projection
            new_gm.omega_components = gm.omega_components
            new_gm.component_types = gm.component_types
            new_gm.structural_ranks = getattr(gm, "structural_ranks", None)
            new_gm.lambda_policies = gm.lambda_policies
            new_gms.append(new_gm)
        elif isinstance(gm, SplineCategoricalGroupMatrix) and _group_has_lambda(gm, g, lambdas):
            if gm.omega is None:
                new_gms.append(gm)
                continue
            lam, omega_eff, has_comp = _resolve_group_lambda(gm, g, lambdas)
            level_weight = sample_weight[gm.row_idx]
            if gm.projection is not None:
                P = gm.projection
                omega_proj = P.T @ omega_eff @ P
                R_inv_local = compute_projected_R_inv(gm.B_level, P, omega_proj, level_weight, lam)
                R_inv_new = P @ R_inv_local
            else:
                R_inv_new = compute_R_inv(gm.B_level, omega_eff, level_weight, lam)
            new_gm = SplineCategoricalGroupMatrix(gm.B, R_inv_new, gm.row_idx)
            new_gm.omega = gm.omega
            new_gm.projection = gm.projection
            new_gm.omega_components = gm.omega_components
            new_gm.component_types = gm.component_types
            new_gm.structural_ranks = getattr(gm, "structural_ranks", None)
            new_gm.lambda_policies = gm.lambda_policies
            new_gm.spline_cat_level = gm.spline_cat_level
            new_gm.spline_cat_feature = gm.spline_cat_feature
            carry_level_support(gm, new_gm)
            carry_binned_codes(gm, new_gm)
            new_gms.append(new_gm)
        elif isinstance(gm, DiscretizedSplineCategoricalGroupMatrix) and _group_has_lambda(
            gm, g, lambdas
        ):
            if gm.omega is None:
                new_gms.append(gm)
                continue
            lam, omega_eff, has_comp = _resolve_group_lambda(gm, g, lambdas)
            level_weight = np.bincount(
                gm.bin_idx_level,
                weights=sample_weight[gm.row_idx],
                minlength=gm.n_bins,
            )
            if gm.projection is not None:
                P = gm.projection
                omega_proj = P.T @ omega_eff @ P
                R_inv_local = compute_projected_R_inv(
                    gm.B_unique,
                    P,
                    omega_proj,
                    level_weight,
                    lam,
                )
                R_inv_new = P @ R_inv_local
            else:
                R_inv_new = compute_R_inv(gm.B_unique, omega_eff, level_weight, lam)
            # type(gm), not the parent: every REML lambda update comes through
            # here, and a lossless block must not be downgraded to a binned one.
            new_gm = type(gm)(
                gm.B_unique,
                R_inv_new,
                gm.bin_idx_level,
                gm.row_idx,
                n_rows=gm.n_rows,
                bin_idx_is_level=True,
            )
            new_gm.omega = gm.omega
            new_gm.projection = gm.projection
            new_gm.omega_components = gm.omega_components
            new_gm.component_types = gm.component_types
            new_gm.structural_ranks = getattr(gm, "structural_ranks", None)
            new_gm.lambda_policies = gm.lambda_policies
            new_gm.spline_cat_level = gm.spline_cat_level
            new_gm.spline_cat_feature = gm.spline_cat_feature
            carry_level_support(gm, new_gm)
            carry_binned_codes(gm, new_gm)
            new_gms.append(new_gm)
        elif isinstance(gm, DiscretizedTensorGroupMatrix) and _group_has_lambda(gm, g, lambdas):
            if gm.omega is None:
                new_gms.append(gm)
                continue
            if gm.omega_components is not None and gm.projection is None:
                # Discrete tensors keep fixed centred coordinates with explicit
                # marginal penalties: a lambda update changes S(lambda), not the
                # packed design, as in mgcv bam(discrete=TRUE).
                new_gms.append(gm)
                continue
            lam, omega_eff, has_comp = _resolve_group_lambda(gm, g, lambdas)
            exposure_agg = np.bincount(gm.bin_idx, weights=sample_weight, minlength=gm.n_bins)
            if gm.projection is not None:
                P = gm.projection
                omega_proj = P.T @ omega_eff @ P
                R_inv_local = compute_projected_R_inv(gm.B_unique, P, omega_proj, exposure_agg, lam)
                R_inv_new = P @ R_inv_local
            else:
                R_inv_new = compute_R_inv(gm.B_unique, omega_eff, exposure_agg, lam)
            new_gm = DiscretizedTensorGroupMatrix(
                gm.B1_unique_t,
                gm.B2_unique_t,
                gm.idx1,
                gm.idx2,
                gm.B_unique,
                R_inv_new,
                gm.bin_idx,
                tensor_id=gm.tensor_id,
                raw_channels=gm._current_raw_channels(),
            )
            carry_binned_codes(gm, new_gm)
            new_gm.omega = gm.omega
            new_gm.projection = gm.projection
            new_gm.omega_components = gm.omega_components
            new_gm.component_types = gm.component_types
            new_gm.structural_ranks = getattr(gm, "structural_ranks", None)
            new_gms.append(new_gm)
        elif isinstance(gm, DiscretizedSSPGroupMatrix) and _group_has_lambda(gm, g, lambdas):
            if gm.omega is None:
                new_gms.append(gm)
                continue
            lam, omega_eff, has_comp = _resolve_group_lambda(gm, g, lambdas)
            exposure_agg = np.bincount(gm.bin_idx, weights=sample_weight, minlength=gm.n_bins)
            if gm.projection is not None:
                P = gm.projection
                omega_proj = P.T @ omega_eff @ P
                R_inv_local = compute_projected_R_inv(gm.B_unique, P, omega_proj, exposure_agg, lam)
                R_inv_new = P @ R_inv_local
            else:
                R_inv_new = compute_R_inv(gm.B_unique, omega_eff, exposure_agg, lam)
            # type(gm) preserves subclass semantics (e.g. lossless support
            # compression) across the per-lambda rebuild.
            new_gm = type(gm)(gm.B_unique, R_inv_new, gm.bin_idx)
            new_gm.omega = gm.omega
            new_gm.projection = gm.projection
            new_gm.omega_components = gm.omega_components
            new_gm.component_types = gm.component_types
            new_gm.structural_ranks = getattr(gm, "structural_ranks", None)
            new_gms.append(new_gm)
        else:
            new_gms.append(gm)
    rebuilt = DesignMatrix(new_gms, dm.n, dm.p)
    rebuilt._centered_pattern_plan = dm._centered_pattern_plan
    # RandomEffect matrices pass through unchanged, so their nesting tests do too.
    carry_nesting_cache(dm._structured_layout_cache, rebuilt._structured_layout_cache)
    return rebuilt
