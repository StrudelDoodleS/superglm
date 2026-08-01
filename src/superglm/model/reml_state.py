"""Internal REML post-fit state synchronization helpers."""

from __future__ import annotations

import numpy as np

from superglm.group_matrix import (
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)

_SSP_LIKE = (
    SparseSSPGroupMatrix
    | SplineCategoricalGroupMatrix
    | DiscretizedSplineCategoricalGroupMatrix
    | DiscretizedSSPGroupMatrix
)


def update_reml_r_inv(model, reml_groups, lambdas) -> None:
    """Update spec R_inv for predict/reconstruct after REML convergence."""
    from superglm.model.report_ops import feature_groups

    for _, g in reml_groups:
        spec = model._specs.get(g.feature_name)
        if spec is not None and hasattr(spec, "set_reparametrisation"):
            fgroups = feature_groups(model, g.feature_name)
            r_inv_parts = []
            for fg in fgroups:
                fg_idx = next(i for i, gg in enumerate(model._groups) if gg.name == fg.name)
                fg_gm = model._dm.group_matrices[fg_idx]
                if isinstance(fg_gm, _SSP_LIKE):
                    r_inv_parts.append(fg_gm.R_inv)
            if r_inv_parts:
                spec.set_reparametrisation(
                    np.hstack(r_inv_parts) if len(r_inv_parts) > 1 else r_inv_parts[0]
                )

    for iname in model._interaction_order:
        ispec = model._interaction_specs[iname]
        if not hasattr(ispec, "set_reparametrisation"):
            continue
        fgroups = feature_groups(model, iname)

        def _has_lambda(fg):
            if fg.name in lambdas:
                return True
            return any(k.startswith(f"{fg.name}:") for k in lambdas)

        if not any(_has_lambda(fg) for fg in fgroups):
            continue
        if len(fgroups) > 1:
            if any(fg.subgroup_type is not None for fg in fgroups):
                r_inv_parts = []
                for fg in fgroups:
                    fg_idx = next(i for i, gg in enumerate(model._groups) if gg.name == fg.name)
                    fg_gm = model._dm.group_matrices[fg_idx]
                    if isinstance(fg_gm, _SSP_LIKE):
                        r_inv_parts.append(fg_gm.R_inv)
                if r_inv_parts:
                    ispec.set_reparametrisation(np.hstack(r_inv_parts))
            else:
                # The bracketed token in a group's DISPLAY name is str(level),
                # so a non-string level arrives here stringified -- an integer
                # level 2 as "2".  score() looks the ORIGINAL key up, so keying
                # this dict on the parsed text silently drops the fitted
                # reparametrisation for every non-string level and predict()
                # then disagrees with its own fit.  Resolve the token back
                # through the spec's own level list; unrecognised tokens keep
                # the parsed form, which is what a string-levelled factor
                # already produced.
                originals = {str(lv): lv for lv in getattr(ispec, "_non_base", ())}
                r_inv_dict = {}
                for fg in fgroups:
                    fg_idx = next(i for i, gg in enumerate(model._groups) if gg.name == fg.name)
                    fg_gm = model._dm.group_matrices[fg_idx]
                    if isinstance(fg_gm, _SSP_LIKE):
                        level = fg.name.split("[")[1].rstrip("]") if "[" in fg.name else fg.name
                        r_inv_dict[originals.get(level, level)] = fg_gm.R_inv
                if r_inv_dict:
                    ispec.set_reparametrisation(r_inv_dict)
        else:
            fg = fgroups[0]
            fg_idx = next(i for i, gg in enumerate(model._groups) if gg.name == fg.name)
            fg_gm = model._dm.group_matrices[fg_idx]
            if isinstance(fg_gm, _SSP_LIKE):
                ispec.set_reparametrisation(fg_gm.R_inv)
