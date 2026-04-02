"""Benchmark: quantify W(rho) first-order approximation error.

Compares the analytic total gradient (partial + first-order W correction)
against the full outer finite-difference gradient (re-solve PIRLS at
perturbed rho, evaluate objective) to measure how much the dropped
second-order terms from Wood (2011) Appendix C actually matter.

Usage:
    uv run python scratch/benchmark_w_correction_error.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.distributions import Gamma, NegativeBinomial, Poisson, Tweedie
from superglm.features.spline import CubicRegressionSpline
from superglm.group_matrix import DiscretizedSSPGroupMatrix, SparseSSPGroupMatrix
from superglm.reml import build_penalty_caches
from superglm.solvers.irls_direct import _build_penalty_matrix, fit_irls_direct
from superglm.tweedie_profile import generate_tweedie_cpg


def setup_model(family_name: str, seed: int = 42):
    """Build a fitted model with two CRS splines, matching test_reml_fd pattern."""
    rng = np.random.default_rng(seed)
    n = 2000
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    mu = np.exp(0.5 + np.sin(2 * np.pi * x1) + 0.5 * x2)

    if family_name == "poisson":
        y = rng.poisson(mu).astype(float)
        family_obj = Poisson()
    elif family_name == "gamma":
        y = rng.gamma(shape=5.0, scale=mu / 5.0)
        y = np.maximum(y, 1e-4)
        family_obj = Gamma()
    elif family_name == "tweedie":
        y = generate_tweedie_cpg(n, mu, phi=1.5, p=1.5, rng=rng)
        y = np.maximum(y, 0.0)
        family_obj = Tweedie(p=1.5)
    elif family_name == "nb2":
        theta = 5.0
        y = rng.negative_binomial(n=theta, p=theta / (theta + mu)).astype(float)
        family_obj = NegativeBinomial(theta=theta)
    else:
        raise ValueError(family_name)

    df = pd.DataFrame({"x1": x1, "x2": x2})
    m = SuperGLM(
        features={
            "x1": CubicRegressionSpline(n_knots=8),
            "x2": CubicRegressionSpline(n_knots=8),
        },
        family=family_obj,
    )
    m.fit(df, y)

    sample_weight = np.ones(n)
    offset_arr = np.zeros(n)
    lambdas = {"x1": 10.0, "x2": 0.5}

    reml_groups = []
    penalty_ranks = {}
    for i, (gm, g) in enumerate(zip(m._dm.group_matrices, m._groups)):
        if g.penalized and isinstance(gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
            reml_groups.append((i, g))
            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
            eigv = np.linalg.eigvalsh(omega_ssp)
            penalty_ranks[g.name] = float(np.sum(eigv > 1e-8 * max(eigv.max(), 1e-12)))

    penalty_caches = build_penalty_caches(m._dm.group_matrices, reml_groups)

    pirls_result, XtWX_S_inv, XtWX = fit_irls_direct(
        X=m._dm,
        y=y,
        weights=sample_weight,
        family=m._distribution,
        link=m._link,
        groups=m._groups,
        lambda2=lambdas,
        offset=offset_arr,
        return_xtwx=True,
    )

    p_dim = XtWX.shape[0]
    S = _build_penalty_matrix(m._dm.group_matrices, m._groups, lambdas, p_dim)
    pq = float(pirls_result.beta @ S @ pirls_result.beta)
    M_p = sum(c.rank for c in penalty_caches.values())
    phi_hat = 1.0
    if not getattr(m._distribution, "scale_known", True):
        phi_hat = max((pirls_result.deviance + pq) / max(n - M_p, 1.0), 1e-10)

    return (
        m,
        y,
        sample_weight,
        offset_arr,
        lambdas,
        reml_groups,
        penalty_ranks,
        penalty_caches,
        pirls_result,
        XtWX_S_inv,
        XtWX,
        phi_hat,
        n,
    )


def benchmark_family(family_name: str):
    """Compute and report W(rho) approximation error for one family."""
    print(f"\n{'=' * 70}")
    print(f"  Family: {family_name}")
    print(f"{'=' * 70}")

    (
        m,
        y,
        sample_weight,
        offset_arr,
        lambdas,
        reml_groups,
        penalty_ranks,
        penalty_caches,
        pirls_result,
        XtWX_S_inv,
        XtWX,
        phi_hat,
        n,
    ) = setup_model(family_name)

    group_names = [g.name for _, g in reml_groups]

    # --- Partial gradient (fixed W) ---
    grad_partial = m._reml_direct_gradient(
        pirls_result,
        XtWX_S_inv,
        lambdas,
        reml_groups,
        penalty_ranks,
        phi_hat=phi_hat,
    )

    # --- W correction (first-order) ---
    w_corr = m._reml_w_correction(
        pirls_result,
        XtWX_S_inv,
        lambdas,
        reml_groups,
        penalty_caches,
        sample_weight,
        offset_arr,
    )
    if w_corr is not None:
        grad_total = grad_partial + w_corr[0]
        w_correction_values = w_corr[0]
    else:
        grad_total = grad_partial.copy()
        w_correction_values = np.zeros_like(grad_partial)

    # --- Outer finite-difference gradient (re-solve PIRLS at perturbed rho) ---
    eps = 1e-5
    fd_grad = np.zeros(len(reml_groups))

    for i, name in enumerate(group_names):
        rho_base = np.log(lambdas[name])
        objs = {}
        for sign in [+1, -1]:
            lam_pert = lambdas.copy()
            lam_pert[name] = np.exp(rho_base + sign * eps)
            r_pert, _, xtwx_pert = fit_irls_direct(
                X=m._dm,
                y=y,
                weights=sample_weight,
                family=m._distribution,
                link=m._link,
                groups=m._groups,
                lambda2=lam_pert,
                offset=offset_arr,
                beta_init=pirls_result.beta,
                intercept_init=pirls_result.intercept,
                return_xtwx=True,
            )
            objs[sign] = m._reml_laml_objective(
                y,
                r_pert,
                lam_pert,
                sample_weight,
                offset_arr,
                XtWX=xtwx_pert,
                penalty_caches=penalty_caches,
            )
        fd_grad[i] = (objs[1] - objs[-1]) / (2 * eps)

    # --- Report ---
    print(f"\n  {'Component':<12} {'Partial':>12} {'W Corr':>12} {'Total':>12} {'FD (outer)':>12}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")
    for i, name in enumerate(group_names):
        print(
            f"  {name:<12} {grad_partial[i]:>12.6f} {w_correction_values[i]:>12.6f} "
            f"{grad_total[i]:>12.6f} {fd_grad[i]:>12.6f}"
        )

    print("\n  Error analysis:")
    print(
        f"  {'Component':<12} {'|Total-FD|':>12} {'|Partial-FD|':>12} "
        f"{'Rel(Total)':>12} {'Rel(Partial)':>12} {'Improvement':>12}"
    )
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")

    for i, name in enumerate(group_names):
        abs_err_total = abs(grad_total[i] - fd_grad[i])
        abs_err_partial = abs(grad_partial[i] - fd_grad[i])
        fd_mag = abs(fd_grad[i])
        rel_total = abs_err_total / fd_mag if fd_mag > 1e-12 else float("nan")
        rel_partial = abs_err_partial / fd_mag if fd_mag > 1e-12 else float("nan")

        if abs_err_partial > 1e-12:
            improvement = 1.0 - abs_err_total / abs_err_partial
        else:
            improvement = 0.0

        print(
            f"  {name:<12} {abs_err_total:>12.2e} {abs_err_partial:>12.2e} "
            f"{rel_total:>11.4%} {rel_partial:>11.4%} {improvement:>11.2%}"
        )

    # --- Summary verdict ---
    max_rel_err = 0.0
    for i in range(len(reml_groups)):
        fd_mag = abs(fd_grad[i])
        if fd_mag > 1e-12:
            rel = abs(grad_total[i] - fd_grad[i]) / fd_mag
            max_rel_err = max(max_rel_err, rel)

    if max_rel_err < 0.01:
        verdict = "EXCELLENT (<1%)"
    elif max_rel_err < 0.05:
        verdict = "GOOD (<5%)"
    else:
        verdict = f"MODERATE ({max_rel_err:.1%})"

    print(f"\n  Max relative error of total gradient vs outer FD: {max_rel_err:.4%}")
    print(f"  Verdict: {verdict}")
    print("  (Error reflects dropped second-order dW/drho terms)")

    return family_name, max_rel_err


def main():
    print("=" * 70)
    print("  W(rho) First-Order Approximation Error Benchmark")
    print("  Comparing analytic total gradient vs outer FD gradient")
    print("  (measures how much dropped 2nd-order terms matter)")
    print("=" * 70)

    families = ["poisson", "gamma", "tweedie", "nb2"]
    results = []

    for fam in families:
        try:
            results.append(benchmark_family(fam))
        except Exception as e:
            print(f"\n  ERROR for {fam}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Family':<12} {'Max Rel Error':>14} {'Verdict':>20}")
    print(f"  {'-' * 12} {'-' * 14} {'-' * 20}")
    for fam, err in results:
        if err < 0.01:
            verdict = "<1% (excellent)"
        elif err < 0.05:
            verdict = "<5% (good)"
        else:
            verdict = f"{err:.1%} (moderate)"
        print(f"  {fam:<12} {err:>13.4%} {verdict:>20}")

    print("\n  Conclusion: If all families show <5% error, the first-order")
    print("  W(rho) correction is sufficient and second-order terms are")
    print("  not worth implementing.")
    print()


if __name__ == "__main__":
    main()
