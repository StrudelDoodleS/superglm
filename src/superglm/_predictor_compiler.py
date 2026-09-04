"""Family-neutral compilation of one predictor design.

This module owns feature and interaction build state.  It deliberately knows
nothing about responses, distributions, links, or offsets; scalar SuperGLM
adapts those concerns in :mod:`superglm.dm_builder`.

The build body below is lifted verbatim from ``dm_builder.build_design_matrix``
so the two stay diffable: ``compile_predictor_design`` binds the cloned graph
to the same names that function uses, which keeps the region byte-identical
and makes any future drift a one-command comparison.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from superglm._frame import EagerFrame
from superglm.features.categorical import Categorical
from superglm.features.factor_smooth import FactorSmooth
from superglm.features.interaction import (
    CategoricalInteraction,
    SplineCategorical,
    TensorInteraction,
)
from superglm.features.ordered_categorical import (
    OrderedCategorical,
    resolve_interaction_parent_of,
)
from superglm.features.piecewise import Piecewise
from superglm.features.polynomial import Polynomial
from superglm.features.spline import _SplineBase
from superglm.group_matrix import DesignMatrix, GroupMatrix
from superglm.types import DiscreteTensorBuildResult, FeatureSpec, GroupInfo, GroupSlice


@dataclass(frozen=True)
class CompiledPredictorDesign:
    """Owned numerical and learned feature state for one predictor."""

    design: DesignMatrix
    groups: tuple[GroupSlice, ...]
    specs: Mapping[str, FeatureSpec]
    feature_order: tuple[str, ...]
    interaction_specs: Mapping[str, Any]
    interaction_order: tuple[str, ...]
    # Terms the build-time separation scan must look at, in build order.  The
    # scan itself needs the response, so it stays with the caller that owns
    # one; this module only records what it saw while the specs learned their
    # levels.
    separation_records: tuple[tuple, ...] = ()


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
) -> CompiledPredictorDesign:
    """Compile an owned design without resolving scalar likelihood concerns.

    ``geometry_weight`` carries the caller's model-geometry rule: the weights
    learned geometry follows, which is an all-rows indicator (or ``None``)
    when it must follow physical rows rather than weight mass, and
    ``sample_weight`` otherwise.  ``physical_rows`` states that same rule as
    the boolean the hosted specs are stamped with.  Both are decided by the
    caller's weight contract, so the caller owns them and this module stays
    family neutral. ``polynomial_weight`` is separate because the two callers
    intentionally differ: distributional predictors classify a learned
    Polynomial QR as data-derived geometry, while scalar SuperGLM preserves
    its historical likelihood-weight standardization.
    ``categorical_reporting_weight`` and ``ordered_reporting_weight`` are
    likewise explicit: scalar reporting keeps
    selecting ``base='most_exposed'`` by likelihood/sample mass while the
    ordered inner basis consumes its caller-selected geometry stream.

    ``separation_boundaries`` is the caller's response-boundary tuple, empty
    when no scan is wanted; the terms it makes this build record come back on
    ``CompiledPredictorDesign.separation_records`` for the caller to scan once
    it pairs them with a response.  ``group_pricing`` and ``alias_prune`` are
    passed through to the dimension pricing and the categorical-interaction
    pruning exactly as ``build_design_matrix`` documents them.

    ``own_specs`` selects who holds the learned state.  The default clones the
    whole graph, which a frozen predictor needs because it may not write to the
    specs its caller handed it.  Scalar SuperGLM passes ``False`` and keeps
    ``build_design_matrix``'s documented in-place contract, which also spares it
    a full spec deepcopy on every REML rebuild and every CV fold.
    """
    # dm_builder imports this module at module scope, so these must stay inside
    # the function body.  Hoisting them to the header is a partially-initialized
    # -module ImportError on `import superglm`, and because they arrive in one
    # statement a single missing name takes the whole block down with it.
    from superglm.dm_builder import (
        _discretize_spline_column,
        _priced_group_dimension,
        _process_info,
        add_interaction,
        resolve_discrete_n_bins,
        should_discretize,
        should_discretize_factor_smooth,
        should_discretize_spline_categorical_interaction,
        should_discretize_tensor_interaction,
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
        # The parameters are annotated with the widest types a caller may
        # hand over, but declining ownership asks for more than reading:
        # the build writes learned state back through these containers and
        # empties ``pending_interactions``, so ``own_specs=False`` requires
        # the caller's own mutable ``dict``/``list``.  That is what
        # ``dm_builder.build_design_matrix`` -- the only non-owning caller --
        # declares and passes; the casts state the extra requirement here
        # rather than narrowing the signature the owning path does not need.
        specs = cast("dict[str, FeatureSpec]", specs)
        feature_order = cast("list[str]", feature_order)
        interaction_specs = cast("dict[str, Any]", interaction_specs)
        interaction_order = cast("list[str]", interaction_order)
        pending_interactions = cast("list[tuple[str, str]]", pending_interactions)

    group_matrices: list[GroupMatrix] = []
    col_offset = 0
    groups: list[GroupSlice] = []
    separation_records: list[tuple] = []

    for name in feature_order:
        spec = specs[name]
        # The physical-rows rule reaches HOSTED model geometry too: an
        # OrderedCategorical hands its caller-selected stream to the inner
        # basis, and a hosted Piecewise's int-mode placement and most_exposed
        # base are as much model geometry as the numeric term's. Polynomial's
        # stream stays explicit and separate because scalar and distributional
        # callers intentionally assign its learned QR to different geometries.
        if isinstance(spec, OrderedCategorical):
            spec._inner_geometry_physical_rows = physical_rows
        x_col = X.column_array(name)
        if separation_boundaries and isinstance(spec, Categorical):
            # Scanned after the loops, once the spec has learned its levels
            # and the GroupSlices exist for the penalty exemption.
            separation_records.append(("cat", name, spec, x_col))
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
        use_discrete = should_discretize(spec, model_discrete)
        B_unique = None
        bin_idx = None
        exposure_agg = None

        constraint_kind = getattr(spec, "constraint_kind", getattr(spec, "monotone", None))
        constraint_mode = getattr(
            spec, "constraint_mode", getattr(spec, "monotone_mode", "postfit")
        )

        _scop_discrete = (
            use_discrete
            and constraint_kind is not None
            and constraint_mode == "fit"
            and hasattr(spec, "_build_scop_reparameterization")
        )

        if use_discrete:
            omega, n_cols_penalty, projection_penalty = spec.build_knots_and_penalty(
                x_col, feature_build_weight
            )
            n_bins_feat = resolve_discrete_n_bins(name, spec, n_bins_config)
            bin_centers, bin_idx = _discretize_spline_column(
                x_col,
                n_bins_feat,
                feature_build_weight,
            )
            B_unique = spec._raw_basis_matrix(bin_centers)
            exposure_agg = np.bincount(bin_idx, weights=sample_weight, minlength=len(bin_centers))

            # ── QP monotone constraint metadata (coefficient-space, survives discretization) ──
            constraints = None
            monotone_engine = None
            raw_to_solver_map = None
            if (
                constraint_kind is not None
                and constraint_mode == "fit"
                and hasattr(spec, "_build_monotone_constraints_raw")
            ):
                cs_raw = spec._build_monotone_constraints_raw()
                constraints = (
                    cs_raw.compose(projection_penalty) if projection_penalty is not None else cs_raw
                )
                monotone_engine = "qp"
                raw_to_solver_map = projection_penalty

            if _scop_discrete:
                # ── SCOP monotone + discrete: bin-level centered design ──
                from superglm.solvers.scop import build_scop_reparam, build_scop_solver_reparam

                q_raw = spec._n_basis
                raw_reparam = build_scop_reparam(
                    q_raw,
                    kind=constraint_kind,
                    knots=spec._knots,
                    degree=spec.degree,
                    domain=(spec._lo, spec._hi),
                )
                X_sigma_unique = B_unique @ raw_reparam.Sigma  # (n_bins, K)

                # Center in model-geometry mass: physical retained rows for a
                # prior contract, literal expanded-row mass for frequency.
                if feature_build_weight is None:
                    bin_geometry_mass = np.bincount(
                        bin_idx,
                        minlength=len(bin_centers),
                    )
                else:
                    bin_geometry_mass = np.bincount(
                        bin_idx,
                        weights=feature_build_weight,
                        minlength=len(bin_centers),
                    )
                drop_dim = 1
                X_shape_unique = X_sigma_unique[:, drop_dim:]
                col_means = (X_shape_unique.T @ bin_geometry_mass) / np.sum(
                    bin_geometry_mass,
                    dtype=np.float64,
                )
                X_centered_unique = X_shape_unique - col_means  # (n_bins, q_eff)

                # Store on spec for predict-time transform
                spec._scop_Sigma = raw_reparam.Sigma
                setattr(spec, "_scop_null_dim", drop_dim)
                spec._scop_col_means = col_means

                solver_reparam = build_scop_solver_reparam(
                    q_raw,
                    kind=constraint_kind,
                    knots=spec._knots,
                    degree=spec.degree,
                    domain=(spec._lo, spec._hi),
                )
                S_scop = solver_reparam.penalty_matrix()
                q_eff = X_centered_unique.shape[1]

                infos = [
                    GroupInfo(
                        columns=X_centered_unique,  # bin-level centered design
                        n_cols=q_eff,
                        penalty_matrix=S_scop,
                        reparametrize=False,
                        penalized=True,
                        scop_reparameterization=solver_reparam,
                        monotone_engine="scop",
                    )
                ]

            elif getattr(spec, "select", False):
                n_null = 1
                n_range = spec._U_range.shape[1]
                n_combined = n_null + n_range
                U_combined = np.hstack([spec._U_null, spec._U_range])
                Z = projection_penalty  # constraint projection

                omega_null = np.zeros((n_combined, n_combined))
                omega_null[:n_null, :n_null] = np.eye(n_null)

                components: list[tuple[str, np.ndarray]] = [("null", omega_null)]

                if len(spec._m_orders) > 1:
                    # Multi-m: project each per-order penalty into combined basis
                    U_null_c = (
                        spec._U_null
                        if Z is None
                        else np.linalg.lstsq(Z, spec._U_null, rcond=None)[0]
                    )
                    U_range_c = (
                        spec._U_range
                        if Z is None
                        else np.linalg.lstsq(Z, spec._U_range, rcond=None)[0]
                    )
                    U_combined_c = np.hstack([U_null_c, U_range_c])
                    for order in spec._m_orders:
                        omega_raw_j = spec._build_penalty_for_order(order)
                        _, omega_c_j, _, _ = spec._apply_constraints(None, omega_raw_j)
                        omega_combined_j = U_combined_c.T @ omega_c_j @ U_combined_c
                        components.append((f"d{order}", omega_combined_j))
                else:
                    omega_wiggle = np.zeros((n_combined, n_combined))
                    omega_wiggle[n_null:, n_null:] = spec._omega_range
                    components.append(("wiggle", omega_wiggle))

                penalty_matrix = sum(omega for _, omega in components)
                infos = [
                    GroupInfo(
                        columns=None,
                        n_cols=n_combined,
                        penalty_matrix=penalty_matrix,
                        reparametrize=True,
                        penalized=True,
                        projection=U_combined,
                        penalty_components=components,
                        component_types={"null": "selection"},
                        constraints=constraints,
                        monotone_engine=monotone_engine,
                        raw_to_solver_map=raw_to_solver_map,
                    ),
                ]
            else:
                infos = [
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
                    )
                ]
        else:
            try:
                # Capture build-time warnings so they can be re-emitted with
                # the feature name.  The errors below already get this
                # treatment; the warnings (Piecewise int-mode collapse and
                # thin segments above all -- the failure modes most likely to
                # reach production silently) were unattributable the moment a
                # model held two terms of the same spec type.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if isinstance(spec, OrderedCategorical):
                        result = spec._build_with_geometry(
                            x_col,
                            reporting_weight=ordered_reporting_weight,
                            geometry_weight=feature_build_weight,
                        )
                    else:
                        result = spec.build(x_col, sample_weight=feature_build_weight)
            except ValueError as err:
                # Name the failing term: spec-level guards (e.g. Polynomial's
                # distinct-support and rank checks) cannot know the column name.
                raise ValueError(f"Feature {name!r}: {err}") from err
            for captured in caught:
                warnings.warn(
                    f"Feature {name!r}: {captured.message}",
                    category=captured.category,
                    # Two frames up is the model fit call that named the
                    # feature -- the place a user can act on the warning.
                    stacklevel=2,
                )
            infos = result if isinstance(result, list) else [result]

        # Resolve lambda_policies from the spec onto each GroupInfo.
        # build() does this internally, but the discrete path constructs
        # GroupInfo manually and needs an explicit resolution step.
        #
        # For single-penalty discrete groups, build_penalty_components uses
        # the multi-penalty path (omega_components with "wiggle" suffix) when
        # penalty_components is set. We need to ensure penalty_components is
        # populated for single-penalty terms with lambda_policy, matching
        # what build() does for non-discrete terms.
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
                B_unique=B_unique,
                bin_idx=bin_idx,
                sample_weight=sample_weight,
                exposure_agg=exposure_agg,
                lambda2=lambda2,
            )
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
            subgroup_suffix = f":{info.subgroup_name}" if info.subgroup_name else ""
            priced, penalty_dim = _priced_group_dimension(info, n_cols, group_pricing)
            groups.append(
                GroupSlice(
                    name=f"{name}{subgroup_suffix}",
                    start=col_offset,
                    end=col_offset + n_cols,
                    weight=np.sqrt(priced),
                    penalty_dim=penalty_dim,
                    penalized=info.penalized,
                    feature_name=name,
                    subgroup_type=info.subgroup_name,
                    constraints=info.constraints,
                    monotone_engine=info.monotone_engine,
                    scop_reparameterization=info.scop_reparameterization,
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
    _next_tensor_id = 0

    for iname in interaction_order:
        ispec = interaction_specs[iname]
        interaction_build_weight = (
            geometry_weight
            if isinstance(ispec, FactorSmooth | SplineCategorical | TensorInteraction)
            else sample_weight
        )
        p1, p2 = ispec.parent_names
        spec1, x1 = resolve_interaction_parent_of(ispec, specs.get(p1), X.column_array(p1))
        spec2, x2 = resolve_interaction_parent_of(ispec, specs.get(p2), X.column_array(p2))
        if separation_boundaries and isinstance(ispec, CategoricalInteraction):
            separation_records.append(("inter", iname, spec1, spec2, x1, x2, p1, p2))
        if spec1 is specs.get(p1) and spec2 is specs.get(p2):
            parent_specs = specs
        else:
            parent_specs = {**specs, p1: spec1, p2: spec2}
        use_discrete_tensor = should_discretize_tensor_interaction(ispec, specs, model_discrete)
        use_discrete_spline_cat = should_discretize_spline_categorical_interaction(
            ispec,
            specs,
            model_discrete,
        )
        use_discrete_factor_smooth = should_discretize_factor_smooth(
            ispec,
            model_discrete,
        )
        B_unique_inter = None
        bin_idx_inter = None
        exposure_agg_inter = None
        tensor_build: DiscreteTensorBuildResult | None = None
        tensor_id = -1
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
            tensor_id = _next_tensor_id
            _next_tensor_id += 1
        elif use_discrete_spline_cat:
            n_bins_spline = resolve_discrete_n_bins(p1, specs[p1], n_bins_config)
            result = ispec.build_discrete(
                x1,
                x2,
                parent_specs,
                n_bins_spline,
                sample_weight=interaction_build_weight,
            )
        else:
            build_kwargs: dict[str, Any] = {"sample_weight": interaction_build_weight}
            if getattr(ispec, "accepts_alias_prune_flag", False):
                build_kwargs["alias_prune"] = alias_prune
            result = ispec.build(x1, x2, parent_specs, **build_kwargs)

        pi_kwargs = dict(
            B_unique=B_unique_inter,
            bin_idx=bin_idx_inter,
            sample_weight=sample_weight,
            exposure_agg=exposure_agg_inter,
            lambda2=lambda2,
            tensor_build=tensor_build,
            tensor_id=tensor_id,
        )

        if isinstance(result, list):
            has_subgroups = any(info.subgroup_name is not None for info in result)
            if has_subgroups:
                r_inv_parts_i: list[NDArray] = []
                for info in result:
                    gm, r_inv, n_cols = _process_info(info, **pi_kwargs)
                    if r_inv is not None:
                        r_inv_parts_i.append(r_inv)

                    group_matrices.append(gm)
                    subgroup_suffix = f":{info.subgroup_name}" if info.subgroup_name else ""
                    priced, penalty_dim = _priced_group_dimension(info, n_cols, group_pricing)
                    groups.append(
                        GroupSlice(
                            name=f"{iname}{subgroup_suffix}",
                            start=col_offset,
                            end=col_offset + n_cols,
                            weight=np.sqrt(priced),
                            penalty_dim=penalty_dim,
                            penalized=info.penalized,
                            feature_name=iname,
                            subgroup_type=info.subgroup_name,
                            constraints=info.constraints,
                            monotone_engine=info.monotone_engine,
                        )
                    )
                    col_offset += n_cols

                if r_inv_parts_i and hasattr(ispec, "set_reparametrisation"):
                    ispec.set_reparametrisation(np.hstack(r_inv_parts_i))
            else:
                # Per-level groups (SplineCategorical, PolynomialCategorical)
                r_inv_dict: dict[str, NDArray] = {}
                for level, info in zip(ispec._non_base, result):
                    gm, r_inv, n_cols = _process_info(info, **pi_kwargs)
                    if r_inv is not None:
                        r_inv_dict[level] = r_inv

                    group_matrices.append(gm)
                    priced, penalty_dim = _priced_group_dimension(info, n_cols, group_pricing)
                    groups.append(
                        GroupSlice(
                            name=f"{iname}[{level}]",
                            start=col_offset,
                            end=col_offset + n_cols,
                            weight=np.sqrt(priced),
                            penalty_dim=penalty_dim,
                            penalized=True,
                            feature_name=iname,
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
            gm, r_inv, n_cols = _process_info(result, **pi_kwargs)
            priced, penalty_dim = _priced_group_dimension(result, n_cols, group_pricing)
            g_new = GroupSlice(
                name=iname,
                start=col_offset,
                end=col_offset + n_cols,
                weight=np.sqrt(priced),
                penalty_dim=penalty_dim,
                penalized=True,
                feature_name=iname,
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

    validate_term_name_namespace(specs, interaction_specs)
    validate_fitted_group_names(groups)
    dm = DesignMatrix(group_matrices, n, col_offset)
    return CompiledPredictorDesign(
        design=dm,
        groups=tuple(groups),
        specs=MappingProxyType(specs),
        feature_order=tuple(feature_order),
        interaction_specs=MappingProxyType(interaction_specs),
        interaction_order=tuple(interaction_order),
        separation_records=tuple(separation_records),
    )
