"""Internal REML fit finalization helpers."""

from __future__ import annotations

import time as _time

from superglm.distributions import clip_mu
from superglm.links import stabilize_eta
from superglm.model.reml_state import update_reml_r_inv
from superglm.reml.penalty_algebra import build_penalty_context, build_penalty_matrix
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.pirls import PIRLSResult


def restore_qp_group_state(model, qp_saved_state) -> None:
    """Restore monotone-engine/constraint state for QP passthrough groups."""
    for gi, engine, constraints in qp_saved_state:
        model._groups[gi].monotone_engine = engine
        model._groups[gi].constraints = constraints


def compute_profiled_phi(model, *, y, lambdas, reml_penalties, pirls_result) -> float:
    """Return REML-profiled phi for estimated-scale families."""
    scale_known = getattr(model._distribution, "scale_known", True)
    if scale_known:
        return 1.0

    p_dim = model._dm.p
    S_final = build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        lambdas,
        p_dim,
        reml_penalties=reml_penalties,
    )
    pq_final = float(pirls_result.beta @ S_final @ pirls_result.beta)
    from superglm.reml.penalty_algebra import compute_total_penalty_rank

    M_p = compute_total_penalty_rank(reml_penalties)
    return float(max((pirls_result.deviance + pq_final) / max(len(y) - M_p, 1.0), 1e-10))


def maybe_qp_passthrough_refit(
    model,
    *,
    qp_passthrough: bool,
    qp_saved_state,
    y,
    sample_weight,
    offset_arr,
    lambdas,
    pirls_result,
    max_pirls_iter,
    pirls_tol,
    reml_penalties,
):
    """Run the constrained post-REML refit for QP passthrough flows when needed."""
    if not qp_passthrough:
        return pirls_result

    restore_qp_group_state(model, qp_saved_state)
    qp_refit, _ = fit_irls_direct(
        X=model._dm,
        y=y,
        weights=sample_weight,
        family=model._distribution,
        link=model._link,
        groups=model._groups,
        lambda2=lambdas,
        offset=offset_arr,
        beta_init=pirls_result.beta,
        intercept_init=float(pirls_result.intercept),
        max_iter=max_pirls_iter,
        tol=pirls_tol,
        convergence="deviance",
        reml_penalties=reml_penalties,
    )
    return qp_refit


def finalize_reml_fit(
    model,
    *,
    best,
    use_direct: bool,
    reml_groups,
    reml_penalties,
    y,
    sample_weight,
    offset,
    offset_arr,
    max_pirls_iter,
    pirls_tol,
    qp_passthrough: bool,
    qp_saved_state,
    profile: dict,
    total_start: float,
    compute_fit_stats,
):
    """Finalize model state after a successful REML optimization run."""
    model._result = best.pirls_result
    model._reml_lambdas = best.lambdas

    if not use_direct:
        reml_penalties, _, _ = build_penalty_context(model._dm.group_matrices, reml_groups)
    model._reml_penalties = reml_penalties
    model._reml_result = best
    lambdas = best.lambdas
    n_reml_iter = best.n_reml_iter
    converged = best.converged

    phi_fixed = compute_profiled_phi(
        model,
        y=y,
        lambdas=lambdas,
        reml_penalties=reml_penalties,
        pirls_result=best.pirls_result,
    )

    final_pirls = maybe_qp_passthrough_refit(
        model,
        qp_passthrough=qp_passthrough,
        qp_saved_state=qp_saved_state,
        y=y,
        sample_weight=sample_weight,
        offset_arr=offset_arr,
        lambdas=lambdas,
        pirls_result=best.pirls_result,
        max_pirls_iter=max_pirls_iter,
        pirls_tol=pirls_tol,
        reml_penalties=reml_penalties,
    )

    corrected = PIRLSResult(
        beta=final_pirls.beta,
        intercept=final_pirls.intercept,
        n_iter=final_pirls.n_iter,
        deviance=final_pirls.deviance,
        converged=final_pirls.converged,
        phi=phi_fixed,
        effective_df=final_pirls.effective_df,
    )
    model._result = corrected
    model._reml_result.pirls_result = corrected

    update_reml_r_inv(model, reml_groups, lambdas)

    profile["total_s"] = _time.perf_counter() - total_start
    profile["n_reml_iter"] = n_reml_iter
    profile["converged"] = converged
    model._reml_profile = profile

    eta = model._dm.matvec(model._result.beta) + model._result.intercept
    if offset is not None:
        eta = eta + offset
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)

    model._fit_stats = compute_fit_stats(
        y, mu, sample_weight, offset, model._distribution, model._link, model._result.phi
    )

    meta = {"method": "fit_reml", "discrete": model._discrete}
    if qp_passthrough:
        meta["lambda_strategy"] = "qp_passthrough"
    model._last_fit_meta = meta

    restore_qp_group_state(model, qp_saved_state)
    return lambdas, n_reml_iter, converged
