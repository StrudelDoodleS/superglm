"""Model-bound calls into the REML derivative engines, for tests.

Each helper takes a fitted model and forwards its design, family, link and
penalties to the engine in ``superglm.reml`` that production reaches directly.
"""

from __future__ import annotations

from superglm.reml.gradient import reml_direct_gradient as _reml_direct_gradient
from superglm.reml.gradient import reml_direct_hessian as _reml_direct_hessian
from superglm.reml.objective import reml_laml_objective as _reml_laml_objective
from superglm.reml.w_derivatives import reml_w_correction as _reml_w_correction
from superglm.solvers.dispersion import model_weight_semantics


def reml_w_correction(
    model,
    pirls_result,
    XtWX_S_inv,
    lambdas,
    reml_groups,
    penalty_caches,
    sample_weight,
    offset_arr,
    w_correction_order=1,
):
    """W(rho) correction for REML derivatives (first- or second-order)."""
    return _reml_w_correction(
        model._dm,
        model._link,
        model._groups,
        pirls_result,
        XtWX_S_inv,
        lambdas,
        reml_groups,
        penalty_caches,
        sample_weight,
        offset_arr,
        model._distribution,
        w_correction_order=w_correction_order,
    )


def reml_laml_objective(
    model, y, result, lambdas, sample_weight, offset_arr, XtWX=None, penalty_caches=None
):
    """Laplace REML/LAML objective up to additive constants."""
    reml_result = getattr(model, "_reml_result", None)
    scop_states = None if reml_result is None else getattr(reml_result, "scop_states", None)
    return _reml_laml_objective(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        y,
        result,
        lambdas,
        sample_weight,
        offset_arr,
        XtWX=XtWX,
        penalty_caches=penalty_caches,
        reml_penalties=getattr(model, "_reml_penalties", None),
        scop_states=scop_states,
        weight_semantics=model_weight_semantics(model),
    )


def reml_direct_gradient(
    model,
    result,
    XtWX_S_inv,
    lambdas,
    reml_groups,
    penalty_ranks,
    phi_hat=1.0,
    *,
    inverse_phi=None,
):
    """Partial gradient of the LAML objective w.r.t. log-lambdas (fixed W)."""
    return _reml_direct_gradient(
        model._dm.group_matrices,
        result,
        XtWX_S_inv,
        lambdas,
        reml_groups,
        penalty_ranks,
        phi_hat=phi_hat,
        inverse_phi=inverse_phi,
        reml_penalties=getattr(model, "_reml_penalties", None),
    )


def reml_direct_hessian(
    model,
    XtWX_S_inv,
    lambdas,
    reml_groups,
    gradient,
    penalty_ranks,
    penalty_caches=None,
    pirls_result=None,
    n_obs=0,
    phi_hat=1.0,
    dH_extra=None,
    dH2_cross=None,
    *,
    inverse_phi=None,
    d_inverse_phi_d_penalized_deviance=None,
):
    """Outer Hessian of the REML criterion w.r.t. log-lambdas."""
    return _reml_direct_hessian(
        model._dm.group_matrices,
        model._distribution,
        XtWX_S_inv,
        lambdas,
        reml_groups,
        gradient,
        penalty_ranks,
        penalty_caches=penalty_caches,
        pirls_result=pirls_result,
        n_obs=n_obs,
        phi_hat=phi_hat,
        inverse_phi=inverse_phi,
        d_inverse_phi_d_penalized_deviance=d_inverse_phi_d_penalized_deviance,
        dH_extra=dH_extra,
        dH2_cross=dH2_cross,
        reml_penalties=getattr(model, "_reml_penalties", None),
    )
