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
from typing import Any, NamedTuple, cast

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


@dataclass(frozen=True)
class BinnedRows:
    """A binned block's rows: its basis at the distinct rows, each row's bin, each bin's weight.

    A discrete tensor's block also holds the tensor's build and its id.
    """

    B_unique: NDArray
    bin_idx: NDArray
    exposure_agg: NDArray
    tensor: DiscreteTensorBuildResult | None = None
    tensor_id: int = -1


def _sz_supports_population_turn(
    info: GroupInfo,
    name: str,
    lambda2: float | dict,
    binned: BinnedRows | None,
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
    if binned is not None:
        columns = (binned.B_unique @ members)[binned.bin_idx]
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


def _interaction_build_row_bytes(
    ispec: Any, widths: Mapping, specs: Mapping[str, FeatureSpec], binned: bool
) -> int:
    """Bytes per data row one interaction's build may hold, from its parents' widths ``a`` and ``b``.

    ``widths`` are the main effects' widths; a tensor's ``n_knots`` override
    builds its margins at that knot count instead, whose B-spline basis has at
    most ``n_knots + degree + 1`` columns.  A binned build forms tables over
    bins, not rows.  A by-level or categorical term holds a basis row per
    parent.  Any other term (a tensor, a numeric or polynomial product) can
    hold ``a b`` entries a row: the row Kronecker's indices and values (12
    bytes each) and the weighted copy the projection forms (12 more), with
    room to spare.
    """
    if binned:
        return _INTERACTION_BUILD_ROW_BYTES
    a, b = (max(widths.get(parent, 1), 1) for parent in ispec.parent_names)
    if isinstance(ispec, TensorInteraction) and ispec._n_knots is not None:
        a, b = (
            max(width, knots + getattr(specs.get(parent), "degree", 3) + 1)
            for width, knots, parent in zip((a, b), ispec._n_knots, ispec.parent_names, strict=True)
        )
    if isinstance(ispec, CategoricalInteraction | SplineCategorical | FactorSmooth):
        return _INTERACTION_BUILD_ROW_BYTES + 32 * (a + b)
    return _INTERACTION_BUILD_ROW_BYTES + 32 * a * b + 8 * (a + b)


def _group_name(term: Any, info: GroupInfo, level: Any = None) -> str:
    """A block's group name: its term, the term's subgroup, or the term's level."""
    if level is not None:
        return f"{term}[{level}]"
    return f"{term}:{info.subgroup_name}" if info.subgroup_name else f"{term}"


class _Block(NamedTuple):
    """One built group: its name, info, matrix, solver map, width and its slice's own fields."""

    name: str
    info: GroupInfo
    matrix: GroupMatrix
    r_inv: NDArray | None
    n_cols: int
    fields: dict[str, Any]


@dataclass
class _BuiltInteraction:
    """One interaction's build: its spec, the spec's groups, each group's block, its records."""

    spec: Any
    result: Any
    blocks: list[_Block]
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


@dataclass
class _Assembly:
    """The design as it is assembled: each group's matrix and slice in build order, and the
    terms the separation scan will read."""

    group_pricing: str
    matrices: list[GroupMatrix] = field(default_factory=list)
    groups: list[GroupSlice] = field(default_factory=list)
    width: int = 0
    separation_records: list[tuple] = field(default_factory=list)
    scan_codes: dict[str, NDArray] = field(default_factory=dict)

    def place(self, block: _Block, feature_name: Any) -> None:
        """Append a block's matrix and its slice at the next free columns."""
        info, n_cols = block.info, block.n_cols
        self.matrices.append(block.matrix)
        priced, penalty_dim = _priced_group_dimension(info, n_cols, self.group_pricing)
        self.groups.append(
            GroupSlice(
                name=block.name,
                term_test_basis=_term_test_basis(info, block.matrix, block.r_inv),
                start=self.width,
                end=self.width + n_cols,
                weight=np.sqrt(priced),
                penalty_dim=penalty_dim,
                feature_name=feature_name,
                ssp_spline=info.uses_ssp,
                constraints=info.constraints,
                monotone_engine=info.monotone_engine,
                **block.fields,
            )
        )
        self.width += n_cols


@dataclass(frozen=True)
class _Build:
    """What every term of one compile reads: the frame, its weights and the build settings."""

    X: EagerFrame
    specs: dict[str, FeatureSpec]
    sample_weight: NDArray
    geometry_weight: NDArray | None
    reporting_weight: NDArray
    model_discrete: bool
    n_bins_config: int | dict[str, int]
    lambda2: float | dict[str, float]
    level_bindings: dict | None
    physical_rows: bool
    separation_boundaries: tuple
    separation_response: NDArray | None
    alias_prune: bool
    eigenbasis_turn: bool
    term_coordinates: bool

    @property
    def scan_curves(self) -> bool:
        """Whether varying-coefficient terms hold their levels' separated curves."""
        return self.separation_response is not None and bool(self.separation_boundaries)

    def weight(self, term: Any) -> NDArray | None:
        """The weights a term's build learns from, by the stream its class declares.

        ``"geometry"`` (knots, bins, breaks, centring: learned model geometry),
        ``"reporting"`` (what a parametric or categorical term reports on), and
        the sample weights for every other term.
        """
        stream = getattr(term, "_build_weight", "sample")
        if stream == "geometry":
            return self.geometry_weight
        if stream == "reporting":
            return self.reporting_weight
        return self.sample_weight


def _adopt_level_universe(build: _Build, name: str, spec: Any) -> None:
    """Supply level universes the feature cannot discover from its observed rows.

    A column's declared categorical dtype takes precedence over a caller's
    full-frame binding. Both hooks decline when the feature already owns a
    universe; terms without one never gain these hooks.
    """
    declared_categories = build.X.column_declared_categories(name)
    if declared_categories is not None and hasattr(spec, "adopt_dtype_categories"):
        spec.adopt_dtype_categories(declared_categories)
    if build.level_bindings is not None and hasattr(spec, "apply_level_binding"):
        binding = build.level_bindings.get(name)
        if binding is not None:
            spec.apply_level_binding(binding)


@dataclass
class _MainEffect:
    """One main effect as it compiles: its inputs, then its build's output.

    ``n_bins`` is set for a spline that bins at fit time; ``scan`` says the
    separation scan reads its levels.
    """

    name: str
    spec: Any
    x_col: NDArray
    weight: NDArray | None
    n_bins: int | None
    scan: bool
    built: Any = None


def _main_effect(build: _Build, name: str) -> _MainEffect:
    """A main effect's inputs: its spec, its column, the weights it learns from and its route."""
    from superglm.dm_builder import resolve_discrete_n_bins, spline_bins

    spec = build.specs[name]
    # The physical-rows rule reaches hosted model geometry too (an
    # OrderedCategorical's inner basis, a hosted Piecewise).
    if isinstance(spec, OrderedCategorical):
        spec._inner_geometry_physical_rows = build.physical_rows
    x_col = build.X.column_array(name)
    scan = bool(build.separation_boundaries) and isinstance(spec, Categorical)
    _adopt_level_universe(build, name, spec)
    weight = build.weight(spec)
    bins = _bins_as_built_in(name, spec, spline_bins(spec, build.model_discrete))
    n_bins = resolve_discrete_n_bins(name, spec, build.n_bins_config) if bins else None
    return _MainEffect(name, spec, x_col, weight, n_bins, scan)


def _spline_knots(term: _MainEffect) -> tuple:
    """A binned spline's own build: its knots, penalty and projection, and its QP geometry."""
    spec = cast(_SplineBase, term.spec)
    build_knots = spec.build_knots_and_penalty
    warning_context = (
        {"feature_name": term.name}
        if getattr(build_knots, "__func__", None) is _SPLINE_KNOT_BUILD
        else {}
    )
    omega, n_cols, projection = build_knots(term.x_col, term.weight, term.n_bins, **warning_context)
    # QP monotone constraint metadata (coefficient-space, survives discretization).
    return omega, n_cols, projection, qp_geometry(spec, shape_engine(spec), projection)


def _binned_spline_infos(build: _Build, term: _MainEffect) -> tuple[list[GroupInfo], BinnedRows]:
    """A binned spline's GroupInfos, built on its bin centres, and its binned rows."""
    spec, x_col, weight = cast(_SplineBase, term.spec), term.x_col, term.weight
    omega, n_cols, projection, (constraints, monotone_engine, raw_to_solver_map) = term.built
    bin_centers, bin_idx = discretize_positive_support(
        x_col, cast(int, term.n_bins), weight, bin_knots(spec, x_col, weight)
    )
    # A bin is read on the basis the unbinned build and prediction read
    # their rows on, under the extrapolation policy.
    B_unique = spec._basis_matrix(bin_centers).toarray()
    exposure_agg = np.bincount(bin_idx, weights=build.sample_weight, minlength=len(bin_centers))
    binned = BinnedRows(B_unique, bin_idx, exposure_agg)
    if getattr(spec, "select", False):
        info = build_select_group_info(spec, None, projection)
        info.constraints, info.monotone_engine = constraints, monotone_engine
        info.raw_to_solver_map = raw_to_solver_map
        info.structural_ranks = structural_penalty_ranks(spec, info)
        return [info], binned
    info = GroupInfo(
        columns=None,
        n_cols=n_cols,
        penalty_matrix=omega,
        reparametrize=(spec.penalty == "ssp"),
        projection=projection,
        penalty_components=getattr(spec, "_penalty_components", None),
        constraints=constraints,
        monotone_engine=monotone_engine,
        raw_to_solver_map=raw_to_solver_map,
    )
    return [finish_penalties(spec, info, B_unique, exposure_agg)], binned


def _exact_build(build: _Build, assembly: _Assembly, term: _MainEffect):
    """A main effect's build on its rows: one GroupInfo or a list.

    The built-in builds are replaced by the design's own, which keep the row
    support or codes they find; a subclass override, or a build() patched in
    later, keeps its own build.
    """
    name, spec, x_col, weight = term.name, term.spec, term.x_col, term.weight
    if isinstance(spec, OrderedCategorical):
        return spec._build_with_geometry(
            x_col,
            reporting_weight=build.reporting_weight,
            geometry_weight=weight,
            feature_name=name,
        )
    build_method = getattr(spec.build, "__func__", None)
    if build_method is _CATEGORICAL_BUILD:
        # The scan reads the build's codes; any other build() leaves the scan
        # to code its column itself.
        result, row_codes = cast(Categorical, spec)._build_with_codes(
            x_col, sample_weight=weight, feature_name=name
        )
        # Held until the scan, across every other term's build:
        # one byte a row for the usual few levels, not eight.
        if term.scan:
            assembly.scan_codes[name] = row_codes.astype(np.min_scalar_type(len(spec._levels)))
        return result
    if build_method is _SPLINE_BUILD:
        return cast(_SplineBase, spec)._build_with_row_support(
            x_col, sample_weight=weight, precision_weight=build.sample_weight, feature_name=name
        )
    if build_method is _PIECEWISE_BUILD:
        return cast(Piecewise, spec).build(x_col, sample_weight=weight, feature_name=name)
    return spec.build(x_col, sample_weight=weight)


def _resolve_lambda_policies(spec: Any, infos: list[GroupInfo]) -> None:
    """Resolve the spec's lambda_policies onto its infos, as build() does.

    The discrete path builds GroupInfo itself. A single-penalty term gets a
    synthetic ``penalty_components`` so its policy flows through to
    ``build_penalty_components``.
    """
    if getattr(spec, "_lambda_policy", None) is None:
        return
    for info in infos:
        if info.lambda_policies is None:
            info.lambda_policies = spec._resolve_lambda_policies(info)
        if info.penalty_components is None and info.penalty_matrix is not None:
            info.penalty_components = [("wiggle", info.penalty_matrix)]
            info.component_types = {"wiggle": "wiggle"}


def _solver_qp_constraints(spec: Any, name: str, r_inv: NDArray | None, n_cols: int):
    """A QP main effect's shape rows in solver coordinates, rebuilt from its raw rows."""
    raw_builder = getattr(spec, "_build_monotone_constraints_raw", None)
    if raw_builder is None:
        raise RuntimeError(
            f"cannot build QP constraints for feature {name!r}: "
            "its raw constraint geometry is unavailable"
        )
    raw_constraints = raw_builder()
    constraints = raw_constraints if r_inv is None else raw_constraints.compose(r_inv)
    if constraints.n_params != n_cols:
        raise RuntimeError(
            f"QP constraint width for feature {name!r} does not match its solver-coordinate group"
        )
    return constraints


def _sz_turn(build: _Build, info: GroupInfo, name: str, binned, label: Hashable) -> bool:
    """``_sz_supports_population_turn`` for main effect ``name`` beside the sz term on ``label``."""
    return _sz_supports_population_turn(
        info,
        name,
        build.lambda2,
        binned,
        build.X.column_array(label),
        build.sample_weight,
        build.separation_response,
        build.separation_boundaries,
    )


def _place_main_effect(
    build: _Build, assembly: _Assembly, term: _MainEffect, sz_labels: Sequence[Hashable]
) -> None:
    """Turn a built main effect into groups, place them and hand its spec their solver map.

    ``sz_labels`` are the group columns of the ``sz`` factor smooths on this
    variable, which decide whether its term coordinates hold.
    """
    from superglm.dm_builder import _process_info

    name, spec = term.name, term.spec
    binned = None
    if term.n_bins is not None:
        infos, binned = _binned_spline_infos(build, term)
    else:
        infos = term.built if isinstance(term.built, list) else [term.built]
    if term.scan:
        # Scanned after the loops, once the spec has learned its levels
        # and the GroupSlices exist for the penalty exemption.
        assembly.separation_records.append(("cat", name, spec, term.x_col))
    _resolve_lambda_policies(spec, infos)

    r_inv_parts: list[NDArray] = []
    for info in infos:
        gm, r_inv, n_cols = _process_info(
            info,
            group_name=_group_name(name, info),
            sample_weight=build.sample_weight,
            lambda2=build.lambda2,
            binned=binned,
            eigenbasis_turn=build.eigenbasis_turn,
            term_coordinates=build.term_coordinates
            and all(_sz_turn(build, info, name, binned, label) for label in sz_labels),
        )
        # Whether a feature requests SSP must survive REML lambda updates.
        # Interaction blocks have their own coordinate updates.
        if isinstance(gm, (SparseSSPGroupMatrix, DiscretizedSSPGroupMatrix)):
            gm.reparametrize = info.reparametrize
        if info.monotone_engine == "qp":
            info.constraints = _solver_qp_constraints(spec, name, r_inv, n_cols)
        if r_inv is not None:
            r_inv_parts.append(r_inv)
        fields = {"penalized": info.penalized, "subgroup_type": info.subgroup_name}
        assembly.place(_Block(_group_name(name, info), info, gm, r_inv, n_cols, fields), name)

    # Set R_inv on spec for transform/reconstruct
    if r_inv_parts and hasattr(spec, "set_reparametrisation"):
        combined = np.hstack(r_inv_parts) if len(r_inv_parts) > 1 else r_inv_parts[0]
        spec.set_reparametrisation(combined)


def _plan_interactions(
    build: _Build,
    interaction_specs: Mapping[str, Any],
    interaction_order: Sequence[str],
    groups: list[GroupSlice],
    first_tensor_id: int,
    ranged_margins: bool,
) -> tuple[dict[str, tuple[bool, bool, bool]], dict[str, int], int]:
    """Each interaction's binned routes and tensor id, and the bytes a row its builds may hold.

    Discrete tensors are numbered in interaction order before any term
    builds, and every parent column is read here, once, on this thread. Each
    build is priced from its parents' main-effect widths, or a tensor's
    override margins; the pool is sized by the largest
    (``_interaction_build_row_bytes``).
    """
    from superglm.dm_builder import (
        refuse_unsupported_interaction_parents,
        should_discretize_factor_smooth,
        should_discretize_spline_categorical_interaction,
        should_discretize_tensor_interaction,
    )

    specs, model_discrete = build.specs, build.model_discrete
    # Each interaction's (tensor, by-level, factor-smooth) binned route.
    binned_routes: dict[str, tuple[bool, bool, bool]] = {}
    tensor_ids: dict[str, int] = {}
    next_tensor_id = first_tensor_id
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
            build.X.column_array(parent)
        # At most one route asks, by the term's type; the builds read it.
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
            tensor_ids[iname] = next_tensor_id
            next_tensor_id += 1
        build_row_bytes = max(
            build_row_bytes, _interaction_build_row_bytes(ispec, widths, specs, any(routes))
        )
    return binned_routes, tensor_ids, build_row_bytes


def _curve_separation(build: _Build, ispec: Any) -> tuple | None:
    """What a spline-by-factor build needs to hold its levels' separated curves, or None."""
    if not (build.scan_curves and isinstance(ispec, SplineCategorical)):
        return None
    spline = build.specs.get(ispec.spline_name)
    return (
        build.separation_response,
        build.sample_weight,
        build.separation_boundaries,
        isinstance(build.specs.get(ispec.cat_name), Categorical),
        spline if isinstance(spline, _SplineBase) else None,
    )


def _exact_interaction_build(
    build: _Build, ispec: Any, x1: NDArray, x2: NDArray, parent_specs: dict, weight
):
    """An interaction's GroupInfos built on its rows.

    A built-in build asks the solver's weights whether its rows identify a
    penalty's null directions (``_rows_identify``); any other build() keeps
    its own, and its product is scanned.
    """
    kwargs: dict[str, Any] = {"sample_weight": weight}
    if getattr(ispec, "accepts_alias_prune_flag", False):
        kwargs["alias_prune"] = build.alias_prune
    separation = _curve_separation(build, ispec)
    if separation is not None:
        kwargs["separation"] = separation
    build_method = getattr(ispec.build, "__func__", None)
    if build_method is _TENSOR_BUILD:
        # The product on its occupied cells, never over every row when it repeats.
        return cast(TensorInteraction, ispec)._build_exact(
            x1, x2, parent_specs, on_cells=True, precision_weight=build.sample_weight, **kwargs
        )
    if build_method is _SPLINE_BY_LEVEL_BUILD:
        return ispec.build(x1, x2, parent_specs, precision_weight=build.sample_weight, **kwargs)
    return ispec.build(x1, x2, parent_specs, **kwargs)


def _interaction_blocks(
    build: _Build, iname: str, ispec: Any, result: Any, binned: BinnedRows | None
) -> list[_Block]:
    """An interaction's groups as blocks: its one group, its subgroups, or one per level."""
    from superglm.dm_builder import _process_info

    kwargs: dict[str, Any] = dict(
        sample_weight=build.sample_weight,
        lambda2=build.lambda2,
        binned=binned,
        eigenbasis_turn=build.eigenbasis_turn,
        term_coordinates=build.term_coordinates,
    )
    if not isinstance(result, list):
        # Single group (CategoricalInteraction, NumericCategorical,
        # NumericInteraction, PolynomialInteraction, TensorInteraction)
        block = _process_info(result, group_name=iname, **kwargs)
        return [_Block(iname, result, *block, {"penalized": True, "interaction": True})]
    if any(info.subgroup_name is not None for info in result):
        # A decomposed tensor's subgroups share one basis and its support.
        kwargs["ssp_supports"] = {}
        blocks = []
        for info in result:
            name = _group_name(iname, info)
            fields = {"penalized": info.penalized, "subgroup_type": info.subgroup_name}
            block = _process_info(info, group_name=name, **kwargs)
            blocks.append(_Block(name, info, *block, {**fields, "interaction": True}))
        return blocks
    # Per-level groups (SplineCategorical, PolynomialCategorical)
    blocks = []
    for level, info in zip(ispec._non_base, result):
        name = _group_name(iname, info, level)
        block = _process_info(info, group_name=name, **kwargs)
        fields = {"penalized": True, "interaction": True, "level": level}
        blocks.append(_Block(name, info, *block, fields))
    return blocks


def _built_separation_records(
    build: _Build, iname: str, ispec: Any, x1: NDArray, x2: NDArray
) -> list[tuple]:
    """The separation records a varying-coefficient interaction leaves once built.

    A spline-by-factor term records the curves it held; a numeric or polynomial
    by-factor term finds the level ranges whose slope separates.
    """
    if isinstance(ispec, SplineCategorical) and ispec._separated_curves:
        return [("curve", iname, ispec._separated_curves)]
    if not isinstance(ispec, NumericCategorical | PolynomialCategorical):
        return []
    from superglm.diagnostics.separation import separated_level_ranges

    ispec._separated_ranges = (
        separated_level_ranges(
            ispec,
            x1,
            x2,
            build.sample_weight,
            cast(NDArray, build.separation_response),
            build.separation_boundaries,
        )
        if build.scan_curves
        else {}
    )
    if not ispec._separated_ranges:
        return []
    return [("level_curve", iname, list(ispec._separated_ranges))]


def _build_interaction(
    build: _Build, iname: str, ispec: Any, routes: tuple[bool, bool, bool], tensor_id: int
) -> _BuiltInteraction:
    """One interaction's groups: its spec's build and each group matrix (no shared state)."""
    from superglm.dm_builder import resolve_discrete_n_bins

    specs, sample_weight, n_bins_config = build.specs, build.sample_weight, build.n_bins_config
    weight = build.weight(ispec)
    p1, p2 = ispec.parent_names
    spec1, x1 = resolve_interaction_parent_of(ispec, specs.get(p1), build.X.column_array(p1))
    spec2, x2 = resolve_interaction_parent_of(ispec, specs.get(p2), build.X.column_array(p2))
    records: list[tuple] = []
    if build.separation_boundaries and isinstance(ispec, CategoricalInteraction):
        records.append(("inter", iname, spec1, spec2, x1, x2, p1, p2))
    if spec1 is specs.get(p1) and spec2 is specs.get(p2):
        parent_specs: dict = specs
    else:
        parent_specs = {**specs, p1: spec1, p2: spec2}
    tensor, by_level, factor_smooth = routes
    binned = None
    if factor_smooth:
        n_bins = resolve_discrete_n_bins(p1, ispec, n_bins_config)
        result = ispec.build_discrete(x1, x2, parent_specs, n_bins, sample_weight=weight)
    elif tensor:
        n_bins_pair = (
            resolve_discrete_n_bins(p1, specs[p1], n_bins_config),
            resolve_discrete_n_bins(p2, specs[p2], n_bins_config),
        )
        tensor_build: DiscreteTensorBuildResult = ispec.build_discrete(
            x1, x2, parent_specs, n_bins_pair, sample_weight=weight, precision_weight=sample_weight
        )
        result = tensor_build.infos
        exposure_agg = np.bincount(
            tensor_build.pair_idx, weights=sample_weight, minlength=tensor_build.B_joint.shape[0]
        )
        binned = BinnedRows(
            tensor_build.B_joint,
            tensor_build.pair_idx,
            exposure_agg,
            tensor=tensor_build,
            tensor_id=tensor_id,
        )
    elif by_level:
        result = ispec.build_discrete(
            x1,
            x2,
            parent_specs,
            resolve_discrete_n_bins(p1, specs[p1], n_bins_config),
            sample_weight=weight,
            separation=_curve_separation(build, ispec),
            precision_weight=sample_weight,
        )
    else:
        result = _exact_interaction_build(build, ispec, x1, x2, parent_specs, weight)
    records.extend(_built_separation_records(build, iname, ispec, x1, x2))
    blocks = _interaction_blocks(build, iname, ispec, result, binned)
    return _BuiltInteraction(ispec, result, blocks, records)


def _hold_interaction_transform(term: _BuiltInteraction) -> None:
    """Hand an interaction's spec its groups' maps to solver coordinates, in its own form.

    One matrix of subgroups side by side, one map per level, or a single
    group's own map.
    """
    ispec = term.spec
    if not hasattr(ispec, "set_reparametrisation"):
        return
    if not isinstance(term.result, list):
        block = term.blocks[0]
        if hasattr(block.matrix, "R_inv"):
            ispec.set_reparametrisation(block.matrix.R_inv)
        elif block.r_inv is not None:
            ispec.set_reparametrisation(block.r_inv)
    elif any(info.subgroup_name is not None for info in term.result):
        parts = [block.r_inv for block in term.blocks if block.r_inv is not None]
        if parts:
            ispec.set_reparametrisation(np.hstack(parts))
    else:
        by_level: dict[Hashable, NDArray] = {
            block.fields["level"]: block.r_inv for block in term.blocks if block.r_inv is not None
        }
        if by_level:
            ispec.set_reparametrisation(by_level)


def compile_predictor_design(
    X: EagerFrame,
    sample_weight: NDArray,
    *,
    geometry_weight: NDArray | None,
    reporting_weight: NDArray,
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
    with; the caller's weight contract decides both.  ``reporting_weight`` is
    what parametric and categorical terms report on: a Polynomial's
    standardisation, a Categorical's and an OrderedCategorical's reporting.
    Each term's class declares which of the three streams it learns from
    (``_Build.weight``).

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
        add_interaction,
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

    build = _Build(
        X=X,
        specs=specs,
        sample_weight=sample_weight,
        geometry_weight=geometry_weight,
        reporting_weight=reporting_weight,
        model_discrete=model_discrete,
        n_bins_config=n_bins_config,
        lambda2=lambda2,
        level_bindings=level_bindings,
        physical_rows=physical_rows,
        separation_boundaries=separation_boundaries,
        separation_response=separation_response,
        alias_prune=alias_prune,
        eigenbasis_turn=eigenbasis_turn,
        term_coordinates=term_coordinates,
    )
    assembly = _Assembly(group_pricing)
    sz_groups: dict[str, list[str]] = {}
    for ispec in interaction_specs.values():
        if isinstance(ispec, FactorSmooth) and ispec.basis == "sz" and not ispec._selects_lines:
            sz_groups.setdefault(ispec.variable, []).append(ispec.group)
    for name in feature_order:
        term = _main_effect(build, name)
        # A build's messages are emitted from this frame, so they name the
        # compile's caller (``feature_build_messages``).
        with feature_build_messages(name):
            term.built = (
                _exact_build(build, assembly, term) if term.n_bins is None else _spline_knots(term)
            )
        _place_main_effect(build, assembly, term, sz_groups.get(name, ()))

    # Resolve pending interactions from constructor
    for pair in pending_interactions:
        if f"{pair[0]}:{pair[1]}" not in interaction_specs and (
            f"{pair[1]}:{pair[0]}" not in interaction_specs
        ):
            add_interaction(pair[0], pair[1], specs, interaction_specs, interaction_order)
    pending_interactions.clear()
    routes, tensor_ids, build_row_bytes = _plan_interactions(
        build,
        interaction_specs,
        interaction_order,
        assembly.groups,
        first_tensor_id,
        ranged_margins,
    )

    def build_interaction(iname: str) -> _BuiltInteraction:
        ispec = interaction_specs[iname]
        return _build_interaction(build, iname, ispec, routes[iname], tensor_ids.get(iname, -1))

    # Interactions read their parents' learned state and write only their
    # own, so they build on the pool (``_parallel.pooled_map``); the design
    # is assembled from them in interaction order, as the serial loop did.
    built = pooled_map(build_interaction, list(interaction_order), build_row_bytes * n)
    try:
        for iname, term in zip(interaction_order, built, strict=True):
            assembly.separation_records.extend(term.records)
            for block in term.blocks:
                assembly.place(block, iname)
            _hold_interaction_transform(term)
    finally:
        # A raise in the loop leaves the generator parked inside the pool's
        # BLAS scope; closing it restores the BLAS policy and stops the pool.
        built.close()

    validate_term_name_namespace(specs, interaction_specs)
    validate_fitted_group_names(assembly.groups)
    compiled = CompiledPredictorDesign(
        design=DesignMatrix(assembly.matrices, n, assembly.width),
        groups=tuple(assembly.groups),
        specs=MappingProxyType(specs),
        feature_order=tuple(feature_order),
        interaction_specs=MappingProxyType(interaction_specs),
        interaction_order=tuple(interaction_order),
        separation_records=tuple(assembly.separation_records),
    )
    compiled.scan_codes.update(assembly.scan_codes)
    return compiled
