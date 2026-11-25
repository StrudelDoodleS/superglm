"""Design-matrix building: R_inv, discretization, feature auto-detection,
interaction dispatch and the main builder.

Every function takes explicit state (specs, feature_order, lambda2, ...) rather
than a model instance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm._frame import EagerFrame
from superglm._group_matrix._binned_codes import carry_binned_codes, certify_binned_codes
from superglm._group_matrix._level_partition import carry_level_support, certify_level_partitions
from superglm._predictor_compiler import (
    BinnedRows,
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
)
from superglm.links import Link, resolve_link
from superglm.solvers._structured.selection import carry_nesting_cache
from superglm.solvers.dispersion import PRIOR_WEIGHTS, validate_weight_semantics
from superglm.types import (
    FeatureSpec,
    GroupInfo,
    GroupSlice,
    NullCoordinates,
)

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


def _lambda_beside_penalty(terms: list[tuple[float, NDArray]]) -> tuple[float, NDArray]:
    """``(lam, Omega)`` for a block penalised by the ``(lam_j, Omega_j)`` terms.

    One penalty keeps its lambda beside it, so stating its weight by name does
    not change the arithmetic; several are summed, ``sum(lam_j Omega_j)``,
    under a unit lambda.
    """
    if len(terms) == 1:
        return terms[0]
    return 1.0, cast(NDArray, sum(lam * omega for lam, omega in terms))


def _penalty_terms(
    info: GroupInfo, group_name: str, lambda2: float | dict
) -> list[tuple[float, NDArray]]:
    """The ``(lam_j, Omega_j)`` a fit applies to a block, zero weights dropped.

    A mapping is read as ``build_penalty_matrix`` reads it, each group or
    penalty component weighted by its own entry. The block has a penalty.
    """
    from superglm.reml.penalty_algebra import resolve_component_lambda, resolve_group_lambda

    if not info.penalty_components:
        terms = [(resolve_group_lambda(lambda2, group_name), cast(NDArray, info.penalty_matrix))]
    else:
        terms = [
            (resolve_component_lambda(lambda2, group_name, suffix), omega)
            for suffix, omega in info.penalty_components
        ]
    return [(lam, omega) for lam, omega in terms if lam != 0.0]


def _applied_penalty(
    info: GroupInfo, group_name: str, lambda2: float | dict
) -> tuple[float, NDArray]:
    """``(lam, Omega)`` with ``lam Omega`` the penalty the fit applies to the block.

    A scalar weights the declared penalty; a mapping weights each term
    (``_penalty_terms``). The block has a penalty.
    """
    omega = cast(NDArray, info.penalty_matrix)
    if not isinstance(lambda2, dict):
        return float(lambda2), omega
    terms = _penalty_terms(info, group_name, lambda2)
    if not terms:
        return 0.0, omega
    # Every declared component at one weight is the scalar's penalty, read as the scalar reads it.
    if len(terms) == len(info.penalty_components or ()) > 1 and len({lam for lam, _ in terms}) == 1:
        return terms[0][0], omega
    return _lambda_beside_penalty(terms)


def compute_R_inv(
    B: sp.spmatrix | NDArray,
    omega: NDArray,
    sample_weight: NDArray,
    lambda2: float,
) -> NDArray:
    """Compute SSP reparametrisation matrix R_inv without forming B @ R_inv.

    Meier, van de Geer & Bühlmann (2009, Ann. Statist. 37(6B), sec. 2.2)
    use the sparsity-smoothness penalty. Here R = chol(B'WB/N + λΩ + εI)^T
    and R_inv = R^{-1}, with N the sample weight sum. For raw coefficients
    θ, the group norm of Rθ combines the weighted mean square of Bθ
    with λθ'Ωθ and the numerical ridge. The group lasso in ``fit`` and
    ``fit_path`` uses this norm with loss per unit weight.

    In ``fit_reml``, R_inv changes coordinates for the smoothness penalty.
    REML uses a summed loss. For working weights equal to the sample
    weights, the map built at λ scales the corresponding Hessian block
    to approximately N times the identity at the REML penalty weight N·λ.

    The normaliser is the weight sum over every row, including rows where the
    block is zero, so every block's ``X'WX`` sits on the same scale.
    """
    total = float(np.sum(sample_weight))
    if total <= 0.0:
        G = np.zeros((omega.shape[0], omega.shape[0]), dtype=np.float64)
    elif sp.issparse(B):
        G = np.asarray((B.multiply(sample_weight[:, None]).T @ B).todense()) / total
    else:
        G = (B * sample_weight[:, None]).T @ B / total
    return _whitening_map(G, omega, lambda2)


@dataclass(frozen=True)
class _WhiteningPenalty:
    matrix: NDArray
    weighted_roots: tuple[tuple[float, NDArray], ...]


def _whitening_penalty(matrix, roots, group_name, lambda2):
    from superglm.reml.penalty_algebra import resolve_component_lambda

    if not roots:
        return matrix
    return _WhiteningPenalty(
        matrix,
        tuple(
            (resolve_component_lambda(lambda2, group_name, key), root)
            for key, root in roots.items()
        ),
    )


def compute_projected_R_inv(
    B: sp.spmatrix | NDArray,
    projection: NDArray,
    penalty_sub: NDArray | _WhiteningPenalty,
    sample_weight: NDArray,
    lambda2: float,
) -> NDArray:
    """Compute SSP R_inv within a projected subspace (linear-split range space).

    Normalised over every row, as in ``compute_R_inv``.
    """
    total = float(np.sum(sample_weight))
    if total <= 0.0:
        G_full = np.zeros((projection.shape[0], projection.shape[0]), dtype=np.float64)
    elif sp.issparse(B):
        G_full = np.asarray((B.multiply(sample_weight[:, None]).T @ B).todense()) / total
    else:
        G_full = (B * sample_weight[:, None]).T @ B / total
    return _whitening_map(projection.T @ G_full @ projection, penalty_sub, lambda2)


def _whitening_map(G: NDArray, penalty: NDArray | _WhiteningPenalty, lam2: float) -> NDArray:
    """``R^-1`` with ``R'R = G + lam2 Omega + 1e-8 I``: the SSP basis.

    Formed, ``lam2 Omega`` rounds by ``u lam2 ||Omega||`` on its null space
    too, where only ``G`` and the ridge hold the sum positive, so near
    ``lam2 = 1e14`` the sum can be indefinite.  It is then factored in
    ``Omega``'s eigenbasis, found without ``lam2``, with the eigenvalues
    under the component's rank cut set to zero: positive definite at any
    ``lam2`` (Wood, Pya & Safken 2016, JASA 111, sec. 3.1.1).
    """
    omega = penalty.matrix if isinstance(penalty, _WhiteningPenalty) else penalty
    ridge = np.eye(omega.shape[0]) * 1e-8
    try:
        return np.linalg.inv(np.linalg.cholesky(G + lam2 * omega + ridge).T)
    except np.linalg.LinAlgError:
        if isinstance(penalty, _WhiteningPenalty):
            # QR of the data and penalty rows retains a graded range even
            # when its Gram is too ill-conditioned to recover those rows.
            rows = [np.linalg.cholesky(G + ridge).T]
            rows.extend(np.sqrt(weight) * root for weight, root in penalty.weighted_roots if weight)
            return np.linalg.inv(np.linalg.qr(np.vstack(rows), mode="r"))
        values, vectors = np.linalg.eigh(omega)
        values = np.where(values > PENALTY_RANK_CUT * max(values[-1], 1e-12), values, 0.0)
        rotated = vectors.T @ G @ vectors + np.diag(lam2 * values) + ridge
        return vectors @ np.linalg.inv(np.linalg.cholesky(rotated).T)


def _dominant_penalty_eigenbasis(
    info: GroupInfo,
    columns: NDArray | sp.spmatrix | None,
    weights: NDArray | None,
    group_name: str,
    lambda2: float | dict,
    *,
    term_coordinates: bool = False,
) -> NDArray | None:
    """The eigenbasis of a raw block's penalty where it outweighs the data by ``1/sqrt(u)``.

    In the block's own coordinates ``lambda Omega`` then rounds into its null
    space by ``u lambda ||Omega||``, past ``sqrt(u)`` of the data's curvature
    there.  In the eigenbasis, found without lambda, the null space lies on
    coordinate axes and keeps the formed penalty to second order (Wood, Pya
    & Safken 2016, sec. 3.1.1), and the penalty dominates every range
    direction, so the turn costs the data's column scaling nothing.
    ``columns`` and ``weights`` are the block's raw (or binned) rows and weights.
    It needs one penalty term (``_penalty_terms``): a weighted sum's
    eigenbasis is not found without the weights.
    """
    if info.penalty_matrix is None or columns is None:
        return None
    terms = _penalty_terms(info, group_name, lambda2)
    if len(terms) != 1:
        return None
    lam, omega = terms[0]
    penalty = lam * float(np.max(np.diag(omega)))
    columns = sp.csr_matrix(columns)
    data = float(np.max(columns.multiply(columns).T @ weights, initial=0.0))
    if penalty * np.sqrt(UNIT_ROUNDOFF) <= data:
        return None
    if term_coordinates:
        declared = cast(NullCoordinates, info.null_coordinates)
        parts = info.penalty_components or [("", info.penalty_matrix)]
        index = next(j for j, (_, part) in enumerate(parts) if part is omega)
        # All nested polynomial axes stay in place. Only the common range
        # turns, from the active component's root before squaring.
        d = max(declared.nullities)
        turn = np.eye(omega.shape[0])
        turn[d:, d:] = np.linalg.svd(declared.roots[index][:, d:], full_matrices=False)[2].T
        return turn
    return np.linalg.eigh(omega)[1]


def spline_bins(spec: FeatureSpec, model_discrete: bool) -> bool:
    """Whether a spline asks to bin at fit time: its own ``discrete``, else the model's.

    Every penalty option bins; ``"none"`` holds its raw basis in the binned block.
    """
    from superglm.features.spline import _SplineBase

    if not isinstance(spec, _SplineBase):
        return False
    if spec.discrete is not None:
        return spec.discrete
    return model_discrete


def should_discretize(spec: FeatureSpec, model_discrete: bool) -> bool:
    """Check if a spline bins at fit time: it asks to (``spline_bins``) and is a built-in type.

    A subclass, whatever it overrides, is fitted on its rows, and so are its interactions.
    """
    from superglm.features.spline import _BUILT_IN_SPLINES

    return type(spec) in _BUILT_IN_SPLINES and spline_bins(spec, model_discrete)


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


def interaction_parent_has_ranges(spec: FeatureSpec | None) -> bool:
    """Whether an interaction parent carries polynomial ranges on its spline."""
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.spline import _SplineBase

    spline = spec._spline if isinstance(spec, OrderedCategorical) else spec
    return isinstance(spline, _SplineBase) and bool(spline.polynomial_ranges)


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
        if not ranged_margins and interaction_parent_has_ranges(spec):
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


def _validate_group_pricing(group_pricing: str) -> None:
    if group_pricing not in ("rank", "spanned"):
        raise ValueError(f"group_pricing must be 'rank' or 'spanned', got {group_pricing!r}")


def _in_term_coordinates(info: GroupInfo) -> GroupInfo:
    """``info`` turned into the null-explicit coordinates of its declared null spaces.

    With ``T = [theta, theta_perp]`` (``null_explicit_coordinates``) the
    projection becomes ``P T``. Each penalty keeps its factor with exactly
    zero columns on its declared null space. Factoring the Gram of a dense
    rotation can lose weak range directions at the Gram's resolution
    (Wood 2011, sec. 3.3). Keep the root for transport and use principal
    axes on a graded range for the consumers that require a matrix.
    The QP rows and the raw-to-solver map turn with the coordinates.
    """
    from dataclasses import replace

    from superglm.features._spline_identifiability import null_explicit_coordinates

    declared = cast(NullCoordinates, info.null_coordinates)
    turn = null_explicit_coordinates(declared.basis)
    roots = tuple(root @ turn for root in declared.roots)
    # Rotate only the common range, preserving every nested null space. The
    # Gram's eigensolver has absolute resolution about width * eps * ||R||².
    # Below it store the anchor's singular values on separate coordinate axes.
    anchor = int(np.argmax(declared.nullities))
    d = declared.nullities[anchor]
    _, values, axes = np.linalg.svd(roots[anchor][:, d:], full_matrices=False)
    graded = values.size and values[-1] <= np.sqrt(len(turn) * (2 * UNIT_ROUNDOFF)) * values[0]
    if graded:
        turn[:, d:] = turn[:, d:] @ axes.T
    parts = []
    term_roots = []
    for root, nullity in zip(roots, declared.nullities, strict=True):
        if graded:
            root[:, d:] = root[:, d:] @ axes.T
        block = root[:, nullity:]
        if block.shape[0] > block.shape[1]:
            block = np.linalg.qr(block, mode="r")
        compact = np.zeros((len(block), len(turn)))
        compact[:, nullity:] = block
        term_roots.append(compact)
        parts.append(compact.T @ compact)
    components = info.penalty_components
    return replace(
        info,
        penalty_matrix=parts[0] if len(parts) == 1 else sum(parts),
        projection=turn if info.projection is None else info.projection @ turn,
        penalty_components=(
            None
            if components is None
            else [(suffix, part) for (suffix, _), part in zip(components, parts, strict=True)]
        ),
        constraints=None if info.constraints is None else info.constraints.compose(turn),
        raw_to_solver_map=None if info.raw_to_solver_map is None else info.raw_to_solver_map @ turn,
        null_coordinates=replace(
            declared, basis=np.eye(len(turn))[:, : declared.basis.shape[1]], roots=tuple(term_roots)
        ),
    )


def _term_roots(info: GroupInfo) -> tuple[GroupInfo, dict[str, NDArray] | None]:
    """``info`` in the term coordinates its declared null basis gives, and each penalty's root.

    A block without a declared null basis keeps its coordinates and has no roots.
    """
    if info.null_coordinates is None:
        return info, None
    info = _in_term_coordinates(info)
    keys = [key for key, _ in info.penalty_components] if info.penalty_components else [""]
    roots = cast(NullCoordinates, info.null_coordinates).roots
    return info, dict(zip(keys, roots, strict=True))


def _fixed_qp_lambdas(info: GroupInfo, group_name: str, lambda2: float | dict) -> float | dict:
    """``lambda2``, or a fixed-lambda QP block's own policy values for its first whitening.

    The fixed-lambda QP fits in this first basis without a REML rebuild.
    Its nested null axes keep the weak component when weighting the
    sum. Whiten at the actual weights so the active-set search does
    not take its steps in coordinates graded by the lambda ratio.
    """
    policies = info.lambda_policies
    if (
        info.monotone_engine == "qp"
        and policies
        and all(policy.mode == "fixed" for policy in policies.values())
    ):
        return {f"{group_name}:{key}": policy.value for key, policy in policies.items()}
    return lambda2


def _whitening(info, group_name, lambda2, rows, weights, term_roots, in_term):
    """``(R_inv, R_inv_local)`` of a block that whitens (``info.uses_ssp``) on ``rows``."""
    P = info.projection
    if P is None:
        lam, omega = _applied_penalty(info, group_name, lambda2)
        R_inv = compute_R_inv(rows, omega, weights, lam)
        # No projection: constraints are in raw space, same as R_inv input.
        return R_inv, R_inv
    if in_term:
        lambda2 = _fixed_qp_lambdas(info, group_name, lambda2)
    lam, omega = _applied_penalty(info, group_name, lambda2)
    penalty = _whitening_penalty(omega, term_roots, group_name, lambda2)
    R_inv_local = compute_projected_R_inv(rows, P, penalty, weights, lam)
    return P @ R_inv_local, R_inv_local


def _block_transform(
    info, group_name, sample_weight, lambda2, binned, in_term, term_roots, eigenbasis_turn
):
    """``(R_inv, R_inv_local)``: the block's map to solver coordinates, ``None`` for its own."""
    if info.uses_ssp:
        if binned is None:
            return _whitening(
                info, group_name, lambda2, info.columns, sample_weight, term_roots, in_term
            )
        return _whitening(
            info, group_name, lambda2, binned.B_unique, binned.exposure_agg, term_roots, in_term
        )
    P = info.projection
    if P is None:
        return None, None
    R_inv_local = None
    if eigenbasis_turn:
        # A binned main effect is read on its distinct rows and bin weights.
        # A binned tensor has no columns and one penalty per margin, so the
        # turn never applies to it.
        main = binned is not None and binned.tensor is None
        R_inv_local = _dominant_penalty_eigenbasis(
            info,
            binned.B_unique if main else info.columns,
            binned.exposure_agg if main else sample_weight,
            group_name,
            lambda2,
            # Declared roots turn the range in place. A tensor's fixed
            # coordinates declare none and take the eigenbasis.
            term_coordinates=term_roots is not None,
        )
    return (P if R_inv_local is None else P @ R_inv_local), R_inv_local


def _storage_group(info, R_inv, binned, ssp_supports):
    """The cheapest exact storage of a block's rows at ``R_inv`` (``None``: rows unmapped)."""
    if binned is not None:
        # A binned block without a projection holds its bins' rows unmapped.
        R = np.eye(info.n_cols, dtype=np.float64) if R_inv is None else R_inv
        tensor = binned.tensor
        if tensor is None:
            return DiscretizedSSPGroupMatrix(binned.B_unique, R, binned.bin_idx)
        return DiscretizedTensorGroupMatrix(
            tensor.B1_unique,
            tensor.B2_unique,
            tensor.idx1,
            tensor.idx2,
            binned.B_unique,
            R,
            binned.bin_idx,
            tensor_id=binned.tensor_id,
            raw_channels=tensor.raw_channels,
        )
    if info.row_support is None and not sp.issparse(info.columns):
        return DenseGroupMatrix(info.columns if R_inv is None else info.columns @ R_inv)
    if R_inv is not None or info.penalty_matrix is not None or info.penalty_components is not None:
        R = np.eye(info.n_cols, dtype=np.float64) if R_inv is None else R_inv
        return _build_ssp_group(info.columns, R, ssp_supports, info.row_support)
    # Only a tensor leaves its columns to its row support, and a tensor block
    # is penalised or projected: these are sparse.
    assert info.columns is not None
    if info.supports_row_compression:
        return _build_unpenalized_sparse_group(info.columns, info.n_cols)
    # cast, not a runtime check: the checker cannot narrow by sp.issparse above.
    return SparseGroupMatrix(cast(sp.spmatrix, info.columns))


def _coded_group(info: GroupInfo) -> GroupMatrix:
    """A block held by its level codes: a factor smooth, a random effect or a categorical."""
    if info.structured_kind == "factor_smooth":
        factor_basis = (
            info.factor_smooth_basis_unique
            if info.factor_smooth_basis_unique is not None
            else info.factor_smooth_basis
        )
        return FactorSmoothGroupMatrix(
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
    if info.structured_kind == "random_effect":
        return RandomEffectGroupMatrix(
            info.cat_codes,
            info.n_cols,
            lambda_policies=info.lambda_policies,
            nested_in=info.random_effect_nested_in,
        )
    return CategoricalGroupMatrix(info.cat_codes, info.n_cols)


def _spline_by_level_group(info, group_name, sample_weight, lambda2, in_term, term_roots):
    """``(group, R_inv, R_inv_local)`` of one level of a spline-by-category term.

    The level is held on its own rows, or on their distinct support when the
    term found one (``spline_cat_basis_unique``).
    """
    mask = np.asarray(info.spline_cat_mask, dtype=bool)
    row_idx = np.flatnonzero(mask)
    support = info.spline_cat_basis_unique is not None and info.spline_cat_bin_idx is not None
    if support:
        rows: sp.spmatrix | NDArray = np.asarray(info.spline_cat_basis_unique, dtype=np.float64)
        bin_idx_level = np.asarray(info.spline_cat_bin_idx, dtype=np.intp)
        if not info.spline_cat_bin_idx_is_level:
            bin_idx_level = bin_idx_level[row_idx]
        weights = np.bincount(
            bin_idx_level, weights=sample_weight[row_idx], minlength=rows.shape[0]
        )
    else:
        rows = sp.csr_matrix(info.spline_cat_basis)[row_idx]
        weights = sample_weight[row_idx]
    if info.uses_ssp:
        R_inv, R_inv_local = _whitening(
            info, group_name, lambda2, rows, weights, term_roots, in_term
        )
    else:
        R_inv = (
            np.eye(info.n_cols, dtype=np.float64) if info.projection is None else info.projection
        )
        R_inv_local = None
    if not support:
        gm = SplineCategoricalGroupMatrix._from_level_rows(rows, R_inv, row_idx, len(mask))
    else:
        # Same storage, different contract: the lossless support indexes
        # exact distinct rows, so the class must not claim the binned route.
        level_cls = (
            SupportCompressedSplineCategoricalGroupMatrix
            if info.spline_cat_support_lossless
            else DiscretizedSplineCategoricalGroupMatrix
        )
        if info.spline_cat_bin_idx_is_level:
            gm = level_cls(
                rows, R_inv, bin_idx_level, row_idx, n_rows=len(mask), bin_idx_is_level=True
            )
        else:
            gm = level_cls(rows, R_inv, info.spline_cat_bin_idx, row_idx)
    gm.spline_cat_level = info.spline_cat_level
    gm.spline_cat_feature = info.spline_cat_feature
    return gm, R_inv, R_inv_local


def _attach_penalty(gm, info: GroupInfo, in_term: bool, *, components: bool = True) -> None:
    """Hold the block's penalty on its group, in the coordinates the group's columns span.

    A projected block outside term coordinates states it on the raw basis, ``P omega P'``.
    """
    P = None if in_term else info.projection
    if info.penalty_matrix is not None and hasattr(gm, "omega"):
        gm.omega = info.penalty_matrix if P is None else P @ info.penalty_matrix @ P.T
    if components and info.penalty_components is not None and hasattr(gm, "omega_components"):
        gm.omega_components = (
            info.penalty_components
            if P is None
            else [(suffix, P @ omega_j @ P.T) for suffix, omega_j in info.penalty_components]
        )
        gm.component_types = info.component_types
        if info.lambda_policies is not None:
            gm.lambda_policies = info.lambda_policies


def _attach_coordinates(gm, info: GroupInfo, in_term: bool, R_inv_local, term_roots, n_cols):
    """Hold a projected block's projection, and in term coordinates its term-to-solver map."""
    if info.projection is None:
        return
    if hasattr(gm, "projection"):
        gm.projection = info.projection
    if in_term:
        gm.R_inv_local = np.eye(n_cols) if R_inv_local is None else R_inv_local
        gm.term_roots = term_roots


def _process_info(
    info: GroupInfo,
    *,
    group_name: str,
    sample_weight: NDArray,
    lambda2: float | dict,
    binned: BinnedRows | None = None,
    ssp_supports: dict | None = None,
    eigenbasis_turn: bool = True,
    term_coordinates: bool = False,
) -> tuple[GroupMatrix, NDArray | None, int]:
    """Compute R_inv and construct a GroupMatrix from a single GroupInfo.

    Returns the group, its block of the combined R_inv (None without a
    reparametrization), and its width. ``binned`` holds a binned block's
    distinct rows. Subgroups share row support via ``ssp_supports``.
    ``eigenbasis_turn=False`` keeps a raw block in its declared coordinates.
    ``group_name`` is the block's group, which a ``lambda2`` mapping keys.

    With ``term_coordinates`` a declared null basis keeps the penalty in those
    coordinates beside ``R_inv_local``. Lifting and re-projecting would round
    onto the null space. Consumers read the map via ``penalty_coordinate_map``.
    """
    if isinstance(info.columns, np.ndarray) and (
        info.penalty_matrix is not None or info.penalty_components is not None
    ):
        # Only a sparse block carries a penalty, so dense penalised columns (a
        # build() override's) are held sparse, keeping the penalty they came with.
        info.columns = sp.csr_matrix(info.columns)
    in_term = term_coordinates and (
        info.null_coordinates is not None or info.penalty_in_term_coordinates
    )
    info, term_roots = _term_roots(info) if in_term else (info, None)
    # R_inv_local: SSP transform in post-identifiability space (projected -> solver).
    # Used to compose constraints that are already in post-identifiability space.
    if info.spline_cat_mask is not None and (
        info.spline_cat_basis is not None
        or (info.spline_cat_basis_unique is not None and info.spline_cat_bin_idx is not None)
    ):
        gm, R_inv, R_inv_local = _spline_by_level_group(
            info, group_name, sample_weight, lambda2, in_term, term_roots
        )
        _attach_penalty(gm, info, in_term, components=False)
    elif (
        binned is None
        and info.projection is None
        and not info.uses_ssp
        and (
            info.structured_kind in ("factor_smooth", "random_effect") or info.cat_codes is not None
        )
    ):
        gm, R_inv, R_inv_local = _coded_group(info), None, None
    else:
        R_inv, R_inv_local = _block_transform(
            info, group_name, sample_weight, lambda2, binned, in_term, term_roots, eigenbasis_turn
        )
        gm = _storage_group(info, R_inv, binned, ssp_supports)
        _attach_penalty(gm, info, in_term)
    n_cols = info.n_cols if R_inv is None else R_inv.shape[1]
    _attach_coordinates(gm, info, in_term, R_inv_local, term_roots, n_cols)

    if info.penalty_in_term_coordinates and isinstance(
        gm, (SparseSSPGroupMatrix, DiscretizedSSPGroupMatrix)
    ):
        # A tensor already has fixed marginal coordinates. A later Cholesky
        # rotation would mix the two components' different null axes.
        gm.reparametrize = info.reparametrize
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
    caps_enforced: bool = True,
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
    :mod:`superglm.diagnostics.separation`): ``"warn"`` (default) records the
    offending levels and cells on their terms and holds them for the fit
    call's one ``SeparationWarning``, ``"error"`` refuses the design with a
    ``SeparationError``, ``"ignore"`` skips the scan.
    ``selection_penalty`` is the model's lambda1-style penalty object when a
    nonzero lambda1 will reach this fit, else ``None``; terms that penalty
    targets, and a ``Categorical`` under a fit-time cap, have bounded optima
    and are exempt from the scan; ``caps_enforced`` is False for a solve that
    fits without its constraint rows (``fit_path``), where a cap bounds nothing.

    ``group_pricing`` decides the width the selection penalty and df ledger
    price a group at when its spec emits fewer columns than the term spans:
    ``"rank"`` (default) the emitted width, ``"spanned"`` the spanned width
    (``_predictor_compiler._priced_group_dimension``).

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
        reporting_weight=sample_weight,
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
        term_coordinates=eigenbasis_turn,
    )
    separation_records = compiled.separation_records
    from superglm.features.categorical import Categorical
    from superglm.model.separated_levels import record_separated_levels

    for spec in compiled.specs.values():
        if isinstance(spec, Categorical):
            record_separated_levels(spec)
    for spec in compiled.interaction_specs.values():
        if getattr(spec, "_separated_cells_at_fit", ()):
            spec._separated_cells_at_fit = ()
            spec._occupied_cells_at_fit = None
        if hasattr(spec, "_bounded_groups_at_fit"):
            spec._bounded_groups_at_fit = (False, False, False)
    if separation_records:
        from superglm.diagnostics.separation import (
            SeparatedLevels,
            emit_curve_separation,
            emit_level_curve_separation,
            emit_separation_findings,
            scan_categorical_term,
            scan_interaction_term,
        )
        from superglm.penalties.base import penalty_targets_group

        def _penalised(name: str) -> bool:
            return selection_penalty is not None and any(
                penalty_targets_group(selection_penalty, g)
                for g in compiled.groups
                if g.feature_name == name
            )

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
            if record[0] != "inter" and _penalised(term_name):
                # A nonzero lambda1 reaches this block: the penalty grows
                # without bound along any recession direction, so the
                # penalised optimum is finite and the term is estimable.
                if record[0] == "level_curve":
                    compiled.interaction_specs[term_name]._separated_ranges = {}
                continue
            if record[0] == "level_curve":
                level_curves.append(record[1:])
                continue
            if record[0] == "cat" and caps_enforced and record[2]._constraint_rules:
                # A fit-time cap bounds every level's relativity against the base,
                # so the capped optimum is finite and no level escapes to infinity.
                compiled.scan_codes.pop(term_name, None)
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
                record_separated_levels(spec_c, found)
            else:
                _, name_i, spec1_i, spec2_i, x1_i, x2_i, p1_i, p2_i = record
                cross_spec = compiled.interaction_specs[name_i]
                # An "auto" penalty is calibrated after the build and can come out
                # zero, so only a set value bounds a cell's parents or the cross.
                set_value = getattr(selection_penalty, "lambda1", None) != "auto"
                bounded = (
                    bool(
                        (caps_enforced and spec1_i._constraint_rules)
                        or (set_value and _penalised(p1_i))
                    ),
                    bool(
                        (caps_enforced and spec2_i._constraint_rules)
                        or (set_value and _penalised(p2_i))
                    ),
                    set_value and _penalised(name_i),
                )
                found = scan_interaction_term(
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
                    cross_spec=cross_spec,
                    bounded=bounded,
                )
                findings.extend(found)
                # the standard-error rule reads these cells (state_ops._names_an_unpriced_level)
                cross_spec._separated_cells_at_fit = tuple(
                    label for finding in found for label in finding.labels
                )
                cross_spec._occupied_cells_at_fit = found[0].occupied if found else None
                cross_spec._bounded_groups_at_fit = bounded
        compiled.scan_codes.clear()
        from superglm.links import LogLink

        emit_separation_findings(
            SeparatedLevels(
                findings=findings,
                specs={**compiled.specs, **compiled.interaction_specs},
                sample_weight=sample_weight,
                offset=offset,
                exposure=offset is not None and isinstance(link, LogLink),
                frame=X,
            ),
            separation,
        )
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
    """The lambda and penalty a group's SSP whitening is built from.

    One penalty keeps its lambda beside it, named or not: a lambda policy names
    a single penalty ``"wiggle"``, and stating the default policy must not
    change the arithmetic.  Several penalties are summed, ``sum(lam_j omega_j)``,
    under a unit lambda.  A zero weight drops its penalty, as in the build
    (``_penalty_terms``).
    """
    components = gm.omega_components
    if components is None:
        return lambdas[g.name], gm.omega
    terms = [(lambdas[f"{g.name}:{suffix}"], omega_j) for suffix, omega_j in components]
    terms = [(lam, omega_j) for lam, omega_j in terms if lam != 0.0]
    return _lambda_beside_penalty(terms) if terms else (0.0, gm.omega)


def _projected_penalty(gm, omega: NDArray) -> NDArray:
    """A projected group's penalty ``omega`` in the coordinates of its projection's columns.

    A group held in term coordinates keeps it there already; any other states
    it on the raw basis, ``P' omega P``.
    """
    if getattr(gm, "R_inv_local", None) is not None:
        return omega
    return gm.projection.T @ omega @ gm.projection


def _group_has_lambda(gm, g, lambdas):
    """Check whether lambdas dict contains entries for this group."""
    if g.name in lambdas:
        return True
    if gm.omega_components is not None:
        first_suffix = gm.omega_components[0][0]
        return f"{g.name}:{first_suffix}" in lambdas
    return False


def _sparse_rows(gm, sample_weight):
    return gm.B, sample_weight


def _level_rows(gm, sample_weight):
    return gm.B_level, sample_weight[gm.row_idx]


def _level_bin_rows(gm, sample_weight):
    weights = np.bincount(gm.bin_idx_level, weights=sample_weight[gm.row_idx], minlength=gm.n_bins)
    return gm.B_unique, weights


def _bin_rows(gm, sample_weight):
    return gm.B_unique, np.bincount(gm.bin_idx, weights=sample_weight, minlength=gm.n_bins)


def _tensor_rows(gm, sample_weight):
    if gm.omega_components is not None and gm.projection is None:
        # Discrete tensors keep fixed centred coordinates with explicit
        # marginal penalties: a lambda update changes S(lambda), not the
        # packed design, as in mgcv bam(discrete=TRUE).
        return None
    return _bin_rows(gm, sample_weight)


def _carry_level(gm, new_gm):
    new_gm.spline_cat_level = gm.spline_cat_level
    new_gm.spline_cat_feature = gm.spline_cat_feature
    carry_level_support(gm, new_gm)
    carry_binned_codes(gm, new_gm)
    return new_gm


def _sparse_copy(gm, R_inv):
    return SparseSSPGroupMatrix(gm.B, R_inv)


def _level_copy(gm, R_inv):
    if gm.B is None:
        new_gm = SplineCategoricalGroupMatrix._from_level_rows(
            gm.B_level, R_inv, gm.row_idx, gm.n_rows
        )
    else:
        new_gm = SplineCategoricalGroupMatrix(gm.B, R_inv, gm.row_idx)
    return _carry_level(gm, new_gm)


def _level_bin_copy(gm, R_inv):
    # type(gm), not the parent: every REML lambda update comes through
    # here, and a lossless block must not be downgraded to a binned one.
    new_gm = type(gm)(
        gm.B_unique, R_inv, gm.bin_idx_level, gm.row_idx, n_rows=gm.n_rows, bin_idx_is_level=True
    )
    return _carry_level(gm, new_gm)


def _tensor_copy(gm, R_inv):
    new_gm = DiscretizedTensorGroupMatrix(
        gm.B1_unique_t,
        gm.B2_unique_t,
        gm.idx1,
        gm.idx2,
        gm.B_unique,
        R_inv,
        gm.bin_idx,
        tensor_id=gm.tensor_id,
        raw_channels=gm._current_raw_channels(),
    )
    carry_binned_codes(gm, new_gm)
    return new_gm


def _bin_copy(gm, R_inv):
    # type(gm) preserves subclass semantics (e.g. lossless support
    # compression) across the per-lambda rebuild.
    return type(gm)(gm.B_unique, R_inv, gm.bin_idx)


class _Rebuild(NamedTuple):
    """How one storage class is whitened again at new lambdas.

    ``rows`` gives the basis rows and weights the whitening reads (``None``
    keeps the group), ``copy`` a new group of the same storage at a new
    ``R_inv``, and ``term_roots`` whether the whitening reads the group's
    term roots and the copy carries them.
    """

    storage: type
    rows: Callable
    copy: Callable
    term_roots: bool = True


# Subclasses before their parents. A group of no listed class keeps its design.
# Discrete tensor interactions are not constrained against one another: ``ti()``
# marginals are centred, so ``ti(a, b) + ti(a, c)`` span jointly identifiable
# surfaces, and a cross-Gram null space would keep the directions a tensor
# shares least, discarding the fitted surface on a holey joint support.  As on
# the exact path, the penalty and shared rank policy handle degenerate directions.
_REBUILDS = (
    _Rebuild(SparseSSPGroupMatrix, _sparse_rows, _sparse_copy),
    _Rebuild(SplineCategoricalGroupMatrix, _level_rows, _level_copy),
    _Rebuild(DiscretizedSplineCategoricalGroupMatrix, _level_bin_rows, _level_bin_copy),
    # Unlike every other class, a binned tensor whitens without its term roots
    # and does not carry them to the rebuilt group.
    _Rebuild(DiscretizedTensorGroupMatrix, _tensor_rows, _tensor_copy, term_roots=False),
    _Rebuild(DiscretizedSSPGroupMatrix, _bin_rows, _bin_copy),
)
_CARRIED = ("omega", "projection", "omega_components", "component_types", "lambda_policies")


def _rebuilt_group(gm, g, lambdas: dict[str, float], sample_weight: NDArray):
    """``gm`` whitened again at ``lambdas``, or ``gm`` itself where its design stays."""
    # Earlier saved groups have no flag and retain their original policy.
    if not getattr(gm, "reparametrize", True):
        return gm
    rebuild = next((entry for entry in _REBUILDS if isinstance(gm, entry.storage)), None)
    if rebuild is None or not _group_has_lambda(gm, g, lambdas) or gm.omega is None:
        return gm
    read = rebuild.rows(gm, sample_weight)
    if read is None:
        return gm
    B, weights = read
    lam, omega_eff = _resolve_group_lambda(gm, g, lambdas)
    roots = getattr(gm, "term_roots", None) if rebuild.term_roots else None
    R_inv_local = None
    if gm.projection is None:
        R_inv = compute_R_inv(B, omega_eff, weights, lam)
    else:
        penalty = _whitening_penalty(_projected_penalty(gm, omega_eff), roots, g.name, lambdas)
        R_inv_local = compute_projected_R_inv(B, gm.projection, penalty, weights, lam)
        R_inv = gm.projection @ R_inv_local
    new_gm = rebuild.copy(gm, R_inv)
    for name in _CARRIED:
        setattr(new_gm, name, getattr(gm, name))
    new_gm.R_inv_local = None if getattr(gm, "R_inv_local", None) is None else R_inv_local
    new_gm.term_roots = roots
    new_gm.structural_ranks = getattr(gm, "structural_ranks", None)
    return new_gm


def rebuild_design_matrix_with_lambdas(
    dm: DesignMatrix,
    groups: list[GroupSlice],
    lambdas: dict[str, float],
    sample_weight: NDArray,
) -> DesignMatrix:
    """Rebuild design matrix with per-group smoothing lambdas.

    Recomputes R_inv for groups that use SSP and have an updated lambda.
    Fixed-coordinate groups keep their design as the smoothing penalty changes.
    """
    new_gms = [
        _rebuilt_group(gm, g, lambdas, sample_weight) for gm, g in zip(dm.group_matrices, groups)
    ]
    rebuilt = DesignMatrix(new_gms, dm.n, dm.p)
    rebuilt._centered_pattern_plan = dm._centered_pattern_plan
    # RandomEffect matrices pass through unchanged, so their nesting tests do too.
    carry_nesting_cache(dm._structured_layout_cache, rebuilt._structured_layout_cache)
    return rebuilt
