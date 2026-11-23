"""Family-neutral compilation of one predictor design.

This module owns feature and interaction build state.  It deliberately knows
nothing about responses, distributions, links, or offsets; scalar SuperGLM
adapts those concerns in :mod:`superglm.dm_builder`, which calls
:func:`compile_predictor_design` for its design build.
"""

from __future__ import annotations

import copy
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from superglm._frame import EagerFrame
from superglm._group_matrix._group_matrix_bins import discretize_positive_support
from superglm._parallel import pooled_map
from superglm._warn import feature_build_messages, warn_caller
from superglm.features._spline_build import finish_penalties, qp_geometry, shape_engine
from superglm.features._spline_penalties import structural_penalty_ranks
from superglm.features._spline_runtime import bin_knots
from superglm.features._spline_select import build_select_group_info
from superglm.features.categorical import Categorical
from superglm.features.factor_smooth import FactorSmooth
from superglm.features.interaction import (
    CategoricalInteraction,
    NumericCategorical,
    PolynomialCategorical,
    SplineCategorical,
    TensorInteraction,
)
from superglm.features.ordered_categorical import (
    OrderedCategorical,
    resolve_interaction_parent_of,
)
from superglm.features.piecewise import _PIECEWISE_BUILD, Piecewise
from superglm.features.polynomial import Polynomial
from superglm.features.spline import _BUILT_IN_SPLINES, _SPLINE_BUILD, _SplineBase
from superglm.group_matrix import (
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    GroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.types import DiscreteTensorBuildResult, FeatureSpec, GroupInfo, GroupSlice

# The built-in builds as defined, which the design builder replaces by its own
# (row support found on distinct values or cells). A subclass override, or a
# build() patched in later, is not one of them and keeps its own build.
_SPLINE_KNOT_BUILD = _SplineBase.build_knots_and_penalty
_CATEGORICAL_BUILD = Categorical.build
_TENSOR_BUILD = TensorInteraction.build
_SPLINE_BY_LEVEL_BUILD = SplineCategorical.build
# The term types that bin: exactly these, never a subclass (``_bins_as_built_in``).
_BINNED_TYPES = _BUILT_IN_SPLINES | {TensorInteraction, SplineCategorical, FactorSmooth}


def _bins_as_built_in(name: str, term: Any, asks: bool) -> bool:
    """Whether a term that ``asks`` to bin does: only when its type is exactly a built-in one.

    A binned fit stands in for a term's build and basis evaluation, and its prediction for the
    basis, so a subclass, whatever it overrides, is fitted on its rows and predicted through its
    own methods, with a warning. An interaction on a subclass spline does not ask
    (``should_discretize``), under that spline's warning.
    """
    if not asks or type(term) in _BINNED_TYPES:
        return asks
    remedy = (
        "set discrete=False on it to fit it on its rows without this warning; its "
        "interactions are fitted on their rows too"
        if isinstance(term, _SplineBase)
        else "set discrete=False to fit it on its rows without this warning"
    )
    warn_caller(
        f"Term {name!r} is fitted on its rows, not binned: {type(term).__name__} subclasses a "
        f"built-in term type, and only those bin. Use the built-in type to bin it, or {remedy}."
    )
    return False


def _sz_supports_population_turn(
    info: GroupInfo,
    name: str,
    lambda2: float | dict,
    binned_basis: NDArray | None,
    bins: NDArray | None,
    labels: NDArray,
    weights: NDArray,
    response: NDArray | None,
    boundaries: tuple[str, ...],
) -> bool:
    """Whether every level anchors the population polynomial beside its free sz deviation.

    After profiling the deviations, a thin level can leave only rounding on a population
    axis. Equilibrating that axis would then identify the rounding. Keep the original
    joint coordinates unless every level resolves the constant and the candidate null
    columns at the pooled design's factor-rank resolution. A separated level loses its
    working weight at the likelihood boundary as well. A dominant penalty
    still needs exact null axes: its rounding in mixed coordinates would
    reach the pooled data curvature of the polynomial itself.
    """
    import pandas as pd

    from superglm._roundoff import UNIT_ROUNDOFF
    from superglm.diagnostics.separation import _separated_flags
    from superglm.reml.penalty_algebra import resolve_component_lambda
    from superglm.solvers.rank import SHARED_RANK_POLICY

    declared = info.null_coordinates
    if declared is None or not declared.basis.shape[1]:
        return True
    members = declared.basis
    if info.projection is not None:
        members = info.projection @ members
    if binned_basis is not None:
        assert bins is not None
        columns = (binned_basis @ members)[bins]
    else:
        assert info.columns is not None
        columns = info.columns @ members
    rows = np.column_stack((np.ones(len(weights)), columns)) * np.sqrt(weights)[:, None]
    pooled = np.linalg.svd(rows, compute_uv=False)
    # As in _dominant_penalty_eigenbasis, keep a lambda-scaled penalty from
    # rounding onto a null by more than sqrt(u) of its data curvature.
    assert info.penalty_matrix is not None
    for suffix, omega in info.penalty_components or [("", info.penalty_matrix)]:
        policy = (info.lambda_policies or {}).get(suffix)
        lam = (
            cast(float, policy.value)
            if policy is not None and policy.mode == "fixed"
            else resolve_component_lambda(lambda2, name, suffix)
        )
        if lam * np.sqrt(UNIT_ROUNDOFF) * np.max(np.diag(omega)) > pooled[-1] ** 2:
            return True
    codes, levels = pd.factorize(labels, use_na_sentinel=False)
    if response is not None and any(
        np.any(_separated_flags(codes, len(levels), response, weights, edge)[0])
        for edge in boundaries
    ):
        return False
    cut = SHARED_RANK_POLICY.factor_rcond * pooled[0]
    order = np.argsort(codes, kind="stable")
    stops = np.cumsum(np.bincount(codes, minlength=len(levels)))
    for block in np.split(rows[order], stops[:-1]):
        if len(block) < rows.shape[1] or np.linalg.svd(block, compute_uv=False)[-1] <= cut:
            return False
    return True


@dataclass(frozen=True)
class CompiledPredictorDesign:
    """Owned numerical and learned feature state for one predictor."""

    design: DesignMatrix
    groups: tuple[GroupSlice, ...]
    specs: Mapping[str, FeatureSpec]
    feature_order: tuple[str, ...]
    interaction_specs: Mapping[str, Any]
    interaction_order: tuple[str, ...]
    # Terms the build-time separation scan must look at, in build order; the
    # caller, which owns the response, runs the scan.
    separation_records: tuple[tuple, ...] = ()
    # Each scanned Categorical's row codes from its build, by term name, until
    # that term's scan pops them: scan scratch, left out of pickles, and an init
    # field so ``dataclasses.replace`` carries them.
    scan_codes: dict[str, NDArray] = field(default_factory=dict, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "scan_codes"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        object.__setattr__(self, "scan_codes", {})


def _clone_specification_graph(
    specs: Mapping[str, FeatureSpec],
    feature_order: Sequence[str],
    interaction_specs: Mapping[str, Any],
    interaction_order: Sequence[str],
    pending_interactions: Sequence[tuple[str, str]],
) -> tuple[
    dict[str, FeatureSpec],
    list[str],
    dict[str, Any],
    list[str],
    list[tuple[str, str]],
]:
    """Clone the complete graph once so aliases remain internal to the clone."""
    graph = copy.deepcopy(
        (
            dict(specs),
            tuple(feature_order),
            dict(interaction_specs),
            tuple(interaction_order),
            tuple(tuple(pair) for pair in pending_interactions),
        )
    )
    owned_specs, owned_feature_order, owned_interactions, owned_interaction_order, pending = graph
    return (
        owned_specs,
        list(owned_feature_order),
        owned_interactions,
        list(owned_interaction_order),
        list(pending),
    )


# Transient bytes per data row an interaction's build holds beyond its
# products: weights, gathers and small per-row temporaries (a pool sizing
# estimate, never a result).
_INTERACTION_BUILD_ROW_BYTES = 256


def _interaction_build_row_bytes(ispec: Any, a: int, b: int, binned: bool) -> int:
    """Bytes per data row one interaction's build may hold, from its parents' widths ``a`` and ``b``.

    A binned build forms tables over bins, not rows.  A by-level or
    categorical term holds a basis row per parent.  Any other term (a tensor,
    a numeric or polynomial product) can hold ``a b`` entries a row: the row
    Kronecker's indices and values (12 bytes each) and the weighted copy the
    projection forms (12 more), with room to spare.
    """
    if binned:
        return _INTERACTION_BUILD_ROW_BYTES
    if isinstance(ispec, CategoricalInteraction | SplineCategorical | FactorSmooth):
        return _INTERACTION_BUILD_ROW_BYTES + 32 * (a + b)
    return _INTERACTION_BUILD_ROW_BYTES + 32 * a * b + 8 * (a + b)


def _group_name(term: Any, info: GroupInfo, level: Any = None) -> str:
    """A block's group name: its term, the term's subgroup, or the term's level."""
    if level is not None:
        return f"{term}[{level}]"
    return f"{term}:{info.subgroup_name}" if info.subgroup_name else f"{term}"


@dataclass
class _BuiltInteraction:
    """One interaction's build: its spec, the spec's groups, each group's matrix, its records."""

    spec: Any
    result: Any
    processed: list[tuple[Any, GroupInfo, GroupMatrix, NDArray | None, int]]
    records: list[tuple]


def _term_test_basis(
    info: GroupInfo, matrix: GroupMatrix, transform: NDArray | None
) -> NDArray | None:
    # A penalized block built directly in its declared columns can acquire
    # another whitening during REML. Its matrix retains the current map.
    retained = getattr(matrix, "R_inv", None)
    if transform is None:
        return retained
    if not transform.shape[1]:
        return None
    basis = info.projection
    if basis is None:
        basis = np.eye(transform.shape[0])
    # A plain dense block already multiplied out its fixed transform.
    return basis if retained is not None else np.linalg.lstsq(transform, basis, rcond=None)[0]


def compile_predictor_design(
    X: EagerFrame,
    sample_weight: NDArray,
    *,
    geometry_weight: NDArray | None,
    polynomial_weight: NDArray,
    categorical_reporting_weight: NDArray,
    ordered_reporting_weight: NDArray,
    specs: Mapping[str, FeatureSpec],
    feature_order: Sequence[str],
    interaction_specs: Mapping[str, Any],
    interaction_order: Sequence[str],
    pending_interactions: Sequence[tuple[str, str]],
    model_discrete: bool,
    n_bins_config: int | dict[str, int],
    lambda2: float | dict[str, float],
    level_bindings: dict | None = None,
    physical_rows: bool = False,
    separation_boundaries: tuple = (),
    group_pricing: str = "rank",
    alias_prune: bool = True,
    own_specs: bool = True,
    first_tensor_id: int = 0,
    ranged_margins: bool = False,
    separation_response: NDArray | None = None,
    eigenbasis_turn: bool = True,
    term_coordinates: bool = False,
) -> CompiledPredictorDesign:
    """Compile an owned design without resolving scalar likelihood concerns.

    ``geometry_weight`` is the weights learned geometry follows (an all-rows
    indicator or ``None`` for physical rows, else ``sample_weight``), and
    ``physical_rows`` the same rule as the boolean hosted specs are stamped
    with; the caller's weight contract decides both.  ``polynomial_weight``,
    ``categorical_reporting_weight`` and ``ordered_reporting_weight`` are
    separate because scalar and distributional callers assign those to
    different geometries.

    ``separation_boundaries`` is the caller's response-boundary tuple (empty
    for no scan); the recorded terms come back on
    ``CompiledPredictorDesign.separation_records``.  ``separation_response``
    lets a spline-by-factor term hold a level's separated curve directions at
    their smooth continuation while it builds, recording ``("curve", name,
    curves)``.  ``group_pricing`` and ``alias_prune`` are as
    ``build_design_matrix`` documents them; ``ranged_margins`` as
    ``refuse_unsupported_interaction_parents``.  ``eigenbasis_turn=False``
    keeps every raw block in its declared coordinates (``_process_info``).
    ``term_coordinates=True`` holds each main effect that declares its
    penalty's null basis in null-explicit term coordinates (``_process_info``);
    scalar SuperGLM's penalty algebra reads them.

    ``first_tensor_id`` numbers this design's discrete tensors from that
    value; equal ids share marginals, so a caller pairing two compiled designs
    in one Gram numbers them apart.  ``own_specs=True`` (default) clones the
    spec graph, as a frozen predictor needs; scalar SuperGLM passes ``False``
    for ``build_design_matrix``'s in-place contract.
    """
    # dm_builder imports this module at module scope, so these stay local:
    # hoisted, they raise a partially-initialized-module ImportError.
    from superglm.dm_builder import (
        _priced_group_dimension,
        _process_info,
        add_interaction,
        refuse_unsupported_interaction_parents,
        resolve_discrete_n_bins,
        should_discretize_factor_smooth,
        should_discretize_spline_categorical_interaction,
        should_discretize_tensor_interaction,
        spline_bins,
        validate_fitted_group_names,
        validate_term_name_namespace,
    )

    n = len(sample_weight)
    if own_specs:
        (
            specs,
            feature_order,
            interaction_specs,
            interaction_order,
            pending_interactions,
        ) = _clone_specification_graph(
            specs,
            feature_order,
            interaction_specs,
            interaction_order,
            pending_interactions,
        )
    else:
        # ``own_specs=False`` writes learned state back through these containers
        # and empties ``pending_interactions``, so it needs the caller's own
        # mutable ``dict``/``list``; the casts state that requirement.
        specs = cast("dict[str, FeatureSpec]", specs)
        feature_order = cast("list[str]", feature_order)
        interaction_specs = cast("dict[str, Any]", interaction_specs)
        interaction_order = cast("list[str]", interaction_order)
        pending_interactions = cast("list[tuple[str, str]]", pending_interactions)

    group_matrices: list[GroupMatrix] = []
    col_offset = 0
    groups: list[GroupSlice] = []
    separation_records: list[tuple] = []
    scan_codes: dict[str, NDArray] = {}
    sz_groups: dict[str, list[str]] = {}
    for term in interaction_specs.values():
        if isinstance(term, FactorSmooth) and term.basis == "sz" and not term._selects_lines:
            sz_groups.setdefault(term.variable, []).append(term.group)

    for name in feature_order:
        spec = specs[name]
        # The physical-rows rule reaches hosted model geometry too (an
        # OrderedCategorical's inner basis, a hosted Piecewise); Polynomial's
        # stream stays separate, as the docstring says.
        if isinstance(spec, OrderedCategorical):
            spec._inner_geometry_physical_rows = physical_rows
        x_col = X.column_array(name)
        scan_levels = bool(separation_boundaries) and isinstance(spec, Categorical)
        # Supply level universes the feature cannot discover from its observed
        # rows. A column's declared categorical dtype takes precedence over a
        # caller's full-frame binding. Both hooks decline when the feature
        # already owns a universe; terms without one never gain these hooks.
        declared_categories = X.column_declared_categories(name)
        if declared_categories is not None and hasattr(spec, "adopt_dtype_categories"):
            spec.adopt_dtype_categories(declared_categories)
        if level_bindings is not None and hasattr(spec, "apply_level_binding"):
            binding = level_bindings.get(name)
            if binding is not None:
                spec.apply_level_binding(binding)
        if isinstance(spec, Polynomial) or (
            isinstance(spec, OrderedCategorical) and isinstance(spec._basis_spline, Polynomial)
        ):
            feature_build_weight = polynomial_weight
        elif isinstance(spec, _SplineBase | Piecewise | OrderedCategorical):
            feature_build_weight = geometry_weight
        elif isinstance(spec, Categorical):
            feature_build_weight = categorical_reporting_weight
        else:
            feature_build_weight = sample_weight

        # Check if this feature should use fit-time discretization
        use_discrete = _bins_as_built_in(name, spec, spline_bins(spec, model_discrete))
        B_unique = None
        bin_idx = None
        exposure_agg = None

        engine = shape_engine(spec)

        if use_discrete:
            n_bins_feat = resolve_discrete_n_bins(name, spec, n_bins_config)
            with feature_build_messages(name):
                build_knots = cast(_SplineBase, spec).build_knots_and_penalty
                warning_context = (
                    {"feature_name": name}
                    if getattr(build_knots, "__func__", None) is _SPLINE_KNOT_BUILD
                    else {}
                )
                omega, n_cols_penalty, projection_penalty = build_knots(
                    x_col, feature_build_weight, n_bins_feat, **warning_context
                )
                # ── QP monotone constraint metadata (coefficient-space, survives discretization) ──
                constraints, monotone_engine, raw_to_solver_map = qp_geometry(
                    spec, engine, projection_penalty
                )
            bin_centers, bin_idx = discretize_positive_support(
                x_col,
                n_bins_feat,
                feature_build_weight,
                bin_knots(spec, x_col, feature_build_weight),
            )
            # A bin is read on the basis the unbinned build and prediction read
            # their rows on, under the extrapolation policy. Only splines bin
            # (should_discretize).
            B_unique = cast(_SplineBase, spec)._basis_matrix(bin_centers).toarray()
            exposure_agg = np.bincount(bin_idx, weights=sample_weight, minlength=len(bin_centers))

            if getattr(spec, "select", False):
                info = build_select_group_info(spec, None, projection_penalty)
                info.constraints, info.monotone_engine = constraints, monotone_engine
                info.raw_to_solver_map = raw_to_solver_map
                infos = [info]
                infos[0].structural_ranks = structural_penalty_ranks(spec, infos[0])
            else:
                infos = [
                    finish_penalties(
                        spec,
                        GroupInfo(
                            columns=None,
                            n_cols=n_cols_penalty,
                            penalty_matrix=omega,
                            reparametrize=(spec.penalty == "ssp"),
                            projection=projection_penalty,
                            penalty_components=getattr(spec, "_penalty_components", None),
                            constraints=constraints,
                            monotone_engine=monotone_engine,
                            raw_to_solver_map=raw_to_solver_map,
                        ),
                        B_unique,
                        exposure_agg,
                    )
                ]
        else:
            with feature_build_messages(name):
                if isinstance(spec, OrderedCategorical):
                    result = spec._build_with_geometry(
                        x_col,
                        reporting_weight=ordered_reporting_weight,
                        geometry_weight=feature_build_weight,
                        feature_name=name,
                    )
                elif getattr(spec.build, "__func__", None) is _CATEGORICAL_BUILD:
                    # The scan reads the build's codes. A subclass that
                    # overrides build() keeps its override, and the scan
                    # codes its column itself.
                    result, row_codes = cast(Categorical, spec)._build_with_codes(
                        x_col, sample_weight=feature_build_weight, feature_name=name
                    )
                    # Held until the scan, across every other term's build:
                    # one byte a row for the usual few levels, not eight.
                    if scan_levels:
                        scan_codes[name] = row_codes.astype(np.min_scalar_type(len(spec._levels)))
                    del row_codes
                elif getattr(spec.build, "__func__", None) is _SPLINE_BUILD:
                    # The design keeps the row support the build finds;
                    # any other build() keeps its own, and its basis is
                    # scanned.
                    result = cast(_SplineBase, spec)._build_with_row_support(
                        x_col,
                        sample_weight=feature_build_weight,
                        precision_weight=sample_weight,
                        feature_name=name,
                    )
                elif getattr(spec.build, "__func__", None) is _PIECEWISE_BUILD:
                    result = cast(Piecewise, spec).build(
                        x_col, sample_weight=feature_build_weight, feature_name=name
                    )
                else:
                    result = spec.build(x_col, sample_weight=feature_build_weight)
            infos = result if isinstance(result, list) else [result]
        if scan_levels:
            # Scanned after the loops, once the spec has learned its levels
            # and the GroupSlices exist for the penalty exemption.
            separation_records.append(("cat", name, spec, x_col))

        # The discrete path builds GroupInfo itself, so resolve the spec's
        # lambda_policies onto it as build() does, populating
        # penalty_components for single-penalty terms with a lambda_policy.
        if hasattr(spec, "_lambda_policy") and spec._lambda_policy is not None:
            for info in infos:
                if info.lambda_policies is None:
                    info.lambda_policies = spec._resolve_lambda_policies(info)
                # Single-penalty terms need a synthetic penalty_components
                # for the lambda_policy to flow through to build_penalty_components.
                if info.penalty_components is None and info.penalty_matrix is not None:
                    info.penalty_components = [("wiggle", info.penalty_matrix)]
                    info.component_types = {"wiggle": "wiggle"}

        # Build GroupMatrix + GroupSlice for each subgroup
        r_inv_parts: list[NDArray] = []

        for info in infos:
            gm, r_inv, n_cols = _process_info(
                info,
                group_name=_group_name(name, info),
                B_unique=B_unique,
                bin_idx=bin_idx,
                sample_weight=sample_weight,
                exposure_agg=exposure_agg,
                lambda2=lambda2,
                eigenbasis_turn=eigenbasis_turn,
                term_coordinates=term_coordinates
                and all(
                    _sz_supports_population_turn(
                        info,
                        name,
                        lambda2,
                        B_unique,
                        bin_idx,
                        X.column_array(label),
                        sample_weight,
                        separation_response,
                        separation_boundaries,
                    )
                    for label in sz_groups.get(name, ())
                ),
            )
            # Whether a feature requests SSP must survive REML lambda updates.
            # Interaction blocks have their own coordinate updates.
            if isinstance(gm, (SparseSSPGroupMatrix, DiscretizedSSPGroupMatrix)):
                gm.reparametrize = info.reparametrize
            qp_info = cast(GroupInfo, info)
            if qp_info.monotone_engine == "qp":
                raw_builder = getattr(spec, "_build_monotone_constraints_raw", None)
                if raw_builder is None:
                    raise RuntimeError(
                        f"cannot build QP constraints for feature {name!r}: "
                        "its raw constraint geometry is unavailable"
                    )
                raw_constraints = raw_builder()
                qp_info.constraints = (
                    raw_constraints if r_inv is None else raw_constraints.compose(r_inv)
                )
                if qp_info.constraints.n_params != n_cols:
                    raise RuntimeError(
                        f"QP constraint width for feature {name!r} does not match "
                        "its solver-coordinate group"
                    )
            if r_inv is not None:
                r_inv_parts.append(r_inv)

            group_matrices.append(gm)
            priced, penalty_dim = _priced_group_dimension(info, n_cols, group_pricing)
            groups.append(
                GroupSlice(
                    name=_group_name(name, info),
                    term_test_basis=_term_test_basis(info, gm, r_inv),
                    start=col_offset,
                    end=col_offset + n_cols,
                    weight=np.sqrt(priced),
                    penalty_dim=penalty_dim,
                    penalized=info.penalized,
                    feature_name=name,
                    subgroup_type=info.subgroup_name,
                    ssp_spline=info.uses_ssp,
                    constraints=info.constraints,
                    monotone_engine=info.monotone_engine,
                )
            )
            col_offset += n_cols

        # Set R_inv on spec for transform/reconstruct
        if r_inv_parts and hasattr(spec, "set_reparametrisation"):
            combined = np.hstack(r_inv_parts) if len(r_inv_parts) > 1 else r_inv_parts[0]
            spec.set_reparametrisation(combined)

    # ── Interactions ──────────────────────────────────────────
    # Resolve pending interactions from constructor
    for pair in pending_interactions:
        if f"{pair[0]}:{pair[1]}" not in interaction_specs and (
            f"{pair[1]}:{pair[0]}" not in interaction_specs
        ):
            add_interaction(pair[0], pair[1], specs, interaction_specs, interaction_order)
    pending_interactions.clear()
    _next_tensor_id = first_tensor_id
    scan_curves = separation_response is not None and bool(separation_boundaries)

    # Discrete tensors are numbered in interaction order before any term
    # builds, and every parent column is read here, once, on this thread.
    tensor_ids: dict[str, int] = {}
    # Each interaction's (tensor, by-level, factor-smooth) binned route.
    binned_routes: dict[str, tuple[bool, bool, bool]] = {}
    # Each build is priced from its parents' main-effect widths, or a tensor's
    # override margins; the pool is sized by the largest
    # (``_interaction_build_row_bytes``).
    widths: dict = {}
    for g in groups:
        widths[g.feature_name] = widths.get(g.feature_name, 0) + (g.end - g.start)
    build_row_bytes = _INTERACTION_BUILD_ROW_BYTES
    for iname in interaction_order:
        ispec = interaction_specs[iname]
        if not isinstance(ispec, FactorSmooth):
            # A declared spec skips the refusals a (left, right) pair meets in
            # add_interaction, so they run here before any interaction builds.
            left, right = ispec.parent_names
            refuse_unsupported_interaction_parents(
                left, right, specs, ranged_margins=ranged_margins
            )
        for parent in ispec.parent_names:
            X.column_array(parent)
        # At most one route asks, by the term's type; the builds below read it.
        routes = tuple(
            _bins_as_built_in(iname, ispec, asks)
            for asks in (
                should_discretize_tensor_interaction(ispec, specs, model_discrete),
                should_discretize_spline_categorical_interaction(ispec, specs, model_discrete),
                should_discretize_factor_smooth(ispec, model_discrete),
            )
        )
        binned_routes[iname] = cast("tuple[bool, bool, bool]", routes)
        if routes[0]:
            tensor_ids[iname] = _next_tensor_id
            _next_tensor_id += 1
        binned = any(routes)
        a, b = (max(widths.get(parent, 1), 1) for parent in ispec.parent_names)
        if isinstance(ispec, TensorInteraction) and ispec._n_knots is not None:
            # An n_knots override builds its margins at that knot count, whose
            # B-spline basis has at most n_knots + degree + 1 columns.
            a, b = (
                max(width, knots + getattr(specs.get(parent), "degree", 3) + 1)
                for width, knots, parent in zip(
                    (a, b), ispec._n_knots, ispec.parent_names, strict=True
                )
            )
        build_row_bytes = max(build_row_bytes, _interaction_build_row_bytes(ispec, a, b, binned))

    def build_interaction(iname: str) -> _BuiltInteraction:
        """One interaction's groups: its spec's build and each group matrix (no shared state)."""
        ispec = interaction_specs[iname]
        interaction_build_weight = (
            geometry_weight
            if isinstance(ispec, FactorSmooth | SplineCategorical | TensorInteraction)
            else sample_weight
        )
        p1, p2 = ispec.parent_names
        spec1, x1 = resolve_interaction_parent_of(ispec, specs.get(p1), X.column_array(p1))
        spec2, x2 = resolve_interaction_parent_of(ispec, specs.get(p2), X.column_array(p2))
        records: list[tuple] = []
        if separation_boundaries and isinstance(ispec, CategoricalInteraction):
            records.append(("inter", iname, spec1, spec2, x1, x2, p1, p2))
        if spec1 is specs.get(p1) and spec2 is specs.get(p2):
            parent_specs = specs
        else:
            parent_specs = {**specs, p1: spec1, p2: spec2}
        curve_separation = (
            (
                separation_response,
                sample_weight,
                separation_boundaries,
                isinstance(specs.get(ispec.cat_name), Categorical),
                specs.get(ispec.spline_name)
                if isinstance(specs.get(ispec.spline_name), _SplineBase)
                else None,
            )
            if scan_curves and isinstance(ispec, SplineCategorical)
            else None
        )
        use_discrete_tensor, use_discrete_spline_cat, use_discrete_factor_smooth = binned_routes[
            iname
        ]
        B_unique_inter = None
        bin_idx_inter = None
        exposure_agg_inter = None
        tensor_build: DiscreteTensorBuildResult | None = None
        tensor_id = tensor_ids.get(iname, -1)
        if use_discrete_factor_smooth:
            n_bins_factor_smooth = resolve_discrete_n_bins(p1, ispec, n_bins_config)
            result = ispec.build_discrete(
                x1,
                x2,
                parent_specs,
                n_bins_factor_smooth,
                sample_weight=interaction_build_weight,
            )
        elif use_discrete_tensor:
            n_bins1 = resolve_discrete_n_bins(p1, specs[p1], n_bins_config)
            n_bins2 = resolve_discrete_n_bins(p2, specs[p2], n_bins_config)
            tensor_build = ispec.build_discrete(
                x1,
                x2,
                parent_specs,
                (n_bins1, n_bins2),
                sample_weight=interaction_build_weight,
            )
            result = tensor_build.infos
            B_unique_inter = tensor_build.B_joint
            bin_idx_inter = tensor_build.pair_idx
            exposure_agg_inter = np.bincount(
                bin_idx_inter,
                weights=sample_weight,
                minlength=B_unique_inter.shape[0],
            )
        elif use_discrete_spline_cat:
            n_bins_spline = resolve_discrete_n_bins(p1, specs[p1], n_bins_config)
            result = ispec.build_discrete(
                x1,
                x2,
                parent_specs,
                n_bins_spline,
                sample_weight=interaction_build_weight,
                separation=curve_separation,
                precision_weight=sample_weight,
            )
        else:
            build_kwargs: dict[str, Any] = {"sample_weight": interaction_build_weight}
            if getattr(ispec, "accepts_alias_prune_flag", False):
                build_kwargs["alias_prune"] = alias_prune
            if curve_separation is not None:
                build_kwargs["separation"] = curve_separation
            if getattr(ispec.build, "__func__", None) is _TENSOR_BUILD:
                # The product on its occupied cells, never over every row when
                # it repeats; any other build() keeps its own, and its product
                # is scanned.
                result = cast(TensorInteraction, ispec)._build_exact(
                    x1, x2, parent_specs, on_cells=True, **build_kwargs
                )
            elif getattr(ispec.build, "__func__", None) is _SPLINE_BY_LEVEL_BUILD:
                # A built-in build asks the solver's weights whether a level's
                # rows identify its penalty's null directions.
                result = ispec.build(
                    x1, x2, parent_specs, precision_weight=sample_weight, **build_kwargs
                )
            else:
                result = ispec.build(x1, x2, parent_specs, **build_kwargs)
        if isinstance(ispec, SplineCategorical) and ispec._separated_curves:
            records.append(("curve", iname, ispec._separated_curves))
        if isinstance(ispec, NumericCategorical | PolynomialCategorical):
            from superglm.diagnostics.separation import separated_level_ranges

            ispec._separated_ranges = (
                separated_level_ranges(
                    ispec,
                    x1,
                    x2,
                    sample_weight,
                    cast(NDArray, separation_response),
                    separation_boundaries,
                )
                if scan_curves
                else {}
            )
            if ispec._separated_ranges:
                records.append(("level_curve", iname, list(ispec._separated_ranges)))

        pi_kwargs: dict[str, Any] = dict(
            B_unique=B_unique_inter,
            bin_idx=bin_idx_inter,
            sample_weight=sample_weight,
            exposure_agg=exposure_agg_inter,
            lambda2=lambda2,
            tensor_build=tensor_build,
            tensor_id=tensor_id,
            eigenbasis_turn=eigenbasis_turn,
            term_coordinates=term_coordinates,
        )
        processed: list[tuple[Any, GroupInfo, GroupMatrix, NDArray | None, int]] = []
        if not isinstance(result, list):
            processed.append((None, result, *_process_info(result, group_name=iname, **pi_kwargs)))
        elif any(info.subgroup_name is not None for info in result):
            # A decomposed tensor's subgroups share one basis and its support.
            ssp_supports: dict = {}
            for info in result:
                block = _process_info(
                    info,
                    group_name=_group_name(iname, info),
                    ssp_supports=ssp_supports,
                    **pi_kwargs,
                )
                processed.append((None, info, *block))
        else:
            # Per-level groups (SplineCategorical, PolynomialCategorical)
            for level, info in zip(ispec._non_base, result):
                block = _process_info(info, group_name=_group_name(iname, info, level), **pi_kwargs)
                processed.append((level, info, *block))
        return _BuiltInteraction(ispec, result, processed, records)

    # Interactions read their parents' learned state and write only their
    # own, so they build on the pool (``_parallel.pooled_map``); the design
    # is assembled from them in interaction order, as the serial loop did.
    built = pooled_map(build_interaction, list(interaction_order), build_row_bytes * n)
    try:
        for iname, term in zip(interaction_order, built, strict=True):
            ispec, result = term.spec, term.result
            separation_records.extend(term.records)
            if isinstance(result, list):
                if any(info.subgroup_name is not None for info in result):
                    r_inv_parts_i: list[NDArray] = []
                    for _, info, gm, r_inv, n_cols in term.processed:
                        if r_inv is not None:
                            r_inv_parts_i.append(r_inv)

                        group_matrices.append(gm)
                        priced, penalty_dim = _priced_group_dimension(info, n_cols, group_pricing)
                        groups.append(
                            GroupSlice(
                                name=_group_name(iname, info),
                                term_test_basis=_term_test_basis(info, gm, r_inv),
                                start=col_offset,
                                end=col_offset + n_cols,
                                weight=np.sqrt(priced),
                                penalty_dim=penalty_dim,
                                penalized=info.penalized,
                                feature_name=iname,
                                subgroup_type=info.subgroup_name,
                                interaction=True,
                                ssp_spline=info.uses_ssp,
                                constraints=info.constraints,
                                monotone_engine=info.monotone_engine,
                            )
                        )
                        col_offset += n_cols

                    if r_inv_parts_i and hasattr(ispec, "set_reparametrisation"):
                        ispec.set_reparametrisation(np.hstack(r_inv_parts_i))
                else:
                    r_inv_dict: dict[Hashable, NDArray] = {}
                    for level, info, gm, r_inv, n_cols in term.processed:
                        if r_inv is not None:
                            r_inv_dict[level] = r_inv

                        group_matrices.append(gm)
                        priced, penalty_dim = _priced_group_dimension(info, n_cols, group_pricing)
                        groups.append(
                            GroupSlice(
                                name=_group_name(iname, info, level),
                                term_test_basis=_term_test_basis(info, gm, r_inv),
                                start=col_offset,
                                end=col_offset + n_cols,
                                weight=np.sqrt(priced),
                                penalty_dim=penalty_dim,
                                penalized=True,
                                feature_name=iname,
                                interaction=True,
                                level=level,
                                ssp_spline=info.uses_ssp,
                                constraints=info.constraints,
                                monotone_engine=info.monotone_engine,
                            )
                        )
                        col_offset += n_cols

                    if r_inv_dict and hasattr(ispec, "set_reparametrisation"):
                        ispec.set_reparametrisation(r_inv_dict)
            else:
                # Single group (CategoricalInteraction, NumericCategorical,
                # NumericInteraction, PolynomialInteraction, TensorInteraction)
                _, _, gm, r_inv, n_cols = term.processed[0]
                priced, penalty_dim = _priced_group_dimension(result, n_cols, group_pricing)
                g_new = GroupSlice(
                    name=iname,
                    term_test_basis=_term_test_basis(result, gm, r_inv),
                    start=col_offset,
                    end=col_offset + n_cols,
                    weight=np.sqrt(priced),
                    penalty_dim=penalty_dim,
                    penalized=True,
                    feature_name=iname,
                    interaction=True,
                    ssp_spline=result.uses_ssp,
                    constraints=result.constraints,
                    monotone_engine=result.monotone_engine,
                )
                if hasattr(ispec, "set_reparametrisation") and hasattr(gm, "R_inv"):
                    ispec.set_reparametrisation(gm.R_inv)
                elif r_inv is not None and hasattr(ispec, "set_reparametrisation"):
                    ispec.set_reparametrisation(r_inv)

                group_matrices.append(gm)
                groups.append(g_new)
                col_offset = g_new.end
    finally:
        # A raise in the loop leaves the generator parked inside the pool's
        # BLAS scope; closing it restores the BLAS policy and stops the pool.
        built.close()

    validate_term_name_namespace(specs, interaction_specs)
    validate_fitted_group_names(groups)
    dm = DesignMatrix(group_matrices, n, col_offset)
    compiled = CompiledPredictorDesign(
        design=dm,
        groups=tuple(groups),
        specs=MappingProxyType(specs),
        feature_order=tuple(feature_order),
        interaction_specs=MappingProxyType(interaction_specs),
        interaction_order=tuple(interaction_order),
        separation_records=tuple(separation_records),
    )
    compiled.scan_codes.update(scan_codes)
    return compiled
