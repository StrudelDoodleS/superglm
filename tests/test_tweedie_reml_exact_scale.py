"""Exact Tweedie REML scale profile fixtures (2026-08-20 audit, finding A).

v0.28.0 substituted a Gaussian-shaped scale profile
``0.5*(n - Mp)*log(Dp)`` for Tweedie inside the REML criterion, charging
every zero row a ``log(phi)`` the exact compound Poisson-gamma saturated
likelihood does not contain (a zero row is an atom, phi-free). These tests
pin the exact Wood Eq. (4) profile against the mgcv 1.9.3 oracle values
recorded in ``docs/audit/2026-08-20-distribution-estimation/README.md``
(sections 2.4-2.5); mgcv was run strictly as a black-box oracle on the
committed fixture CSVs. Every test fails against v0.28.0.

Verification is by *equivalence* (shipped criterion vs the exact criterion
and mgcv's REML answer), not by accuracy against the generating truth.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import Tweedie
from superglm.features import Categorical, RandomEffect
from superglm.features.spline import CubicRegressionSpline
from superglm.profiling.tweedie import generate_tweedie_cpg

FIXTURES = Path(__file__).parent / "fixtures"

# mgcv 1.9.3 gam(y ~ u + s(r, bs="re"), family=Tweedie(p, link="log"),
# method="REML") on the committed CSVs; tau^2 is the squared s(r) vcomp sd.
MGCV_IDENT_P15 = {"tau2": 0.06582, "edf_re": 23.4}
MGCV_FLAT_P15_LOWZERO = {"tau2": 0.00662, "edf_re": 18.2}
MGCV_FLAT_P18 = {"tau2": 8.99e-4, "edf_re": 0.79}


def _random_effect_frame(csv_name: str):
    data = pd.read_csv(FIXTURES / csv_name)
    frame = pd.DataFrame(
        {
            "u": pd.Categorical(data["u"].astype(str)),
            "r": pd.Categorical(data["r"].astype(str)),
        }
    )
    return frame, data["y"].to_numpy(dtype=np.float64)


def _fixed_lambda_evaluation(model, y, lambdas, warm=None):
    """One fixed-lambda coefficient fit plus its REML objective evaluation."""
    from superglm.model.reml_setup import collect_reml_groups
    from superglm.reml.objective import reml_laml_objective
    from superglm.reml.penalty_algebra import build_penalty_context, build_penalty_matrix
    from superglm.solvers.irls_direct import fit_irls_direct

    n = len(y)
    weights = np.ones(n)
    offset = np.zeros(n)
    reml_groups = collect_reml_groups(model._groups, model._dm.group_matrices)
    penalties, _caches, _ranks = build_penalty_context(model._dm.group_matrices, reml_groups)
    S = build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        lambdas,
        model._dm.p,
        reml_penalties=penalties,
    )
    fit_kwargs = {} if warm is None else {"beta_init": warm[0], "intercept_init": warm[1]}
    result, _, XtWX = fit_irls_direct(
        X=model._dm,
        y=y,
        weights=weights,
        family=model._distribution,
        link=model._link,
        groups=model._groups,
        lambda2=lambdas,
        offset=offset,
        return_xtwx=True,
        S_override=S,
        reml_penalties=penalties,
        **fit_kwargs,
        weight_semantics="prior",
    )
    objective_kwargs = {}
    if getattr(result, "log_det_H", None) is not None:
        objective_kwargs = {
            "log_det_H": result.log_det_H,
            "hessian_rank": result.reml_hessian_rank,
        }
    evaluation = reml_laml_objective(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        y,
        result,
        lambdas,
        weights,
        offset,
        XtWX=XtWX,
        S_override=S,
        reml_penalties=penalties,
        return_evaluation=True,
        **objective_kwargs,
        weight_semantics="prior",
    )
    return result, evaluation


def _criterion_phi(evaluation, n):
    """The dispersion the evaluated criterion itself pairs with.

    With the exact profile this is the profiled scale optimum; v0.28.0's
    reduced criterion pairs with the implicit Dp/(n - Mp), which this helper
    falls back to so the fixture measures a value (not plumbing) there.
    """
    profiled = evaluation.profiled_scale
    if profiled is not None:
        return float(profiled.phi)
    return float(evaluation.penalized_deviance / max(n - float(evaluation.penalty_nullity), 1.0))


class TestRandomEffectEquivalenceAgainstMgcv:
    def test_identified_cell_matches_mgcv(self):
        """ident_p15 (tau=0.3, phi=3, p=1.5, 54% zeros): identified cell.

        v0.28.0: tau^2 = 0.0667 (1.3% off mgcv) and RE edf 23.7; the exact
        criterion lands on mgcv's 0.06582 / 23.4 (audit section 2.5).
        """
        frame, y = _random_effect_frame("re_ident_p15.csv")
        model = SuperGLM(
            features={"u": Categorical(), "r": RandomEffect()},
            family=Tweedie(p=1.5),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(frame, y)
        lambdas = dict(model._reml_result.lambdas)
        _, evaluation = _fixed_lambda_evaluation(model, y, lambdas)
        tau2 = _criterion_phi(evaluation, len(y)) / lambdas["r"]
        assert tau2 == pytest.approx(MGCV_IDENT_P15["tau2"], rel=0.005)
        assert float(model._group_edf["r"]) == pytest.approx(MGCV_IDENT_P15["edf_re"], abs=0.25)

    def test_low_zero_flat_cell_matches_mgcv(self):
        """flat_p15_lowzero (tau=0.08, phi=0.5, 3% zeros): flat direction.

        v0.28.0: tau^2 = 0.00517 (22% below mgcv) and RE edf 14.4; the exact
        criterion lands on mgcv's 0.00662 / 18.2 (audit section 2.5).
        """
        frame, y = _random_effect_frame("re_flat_p15_lowzero.csv")
        model = SuperGLM(
            features={"u": Categorical(), "r": RandomEffect()},
            family=Tweedie(p=1.5),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(frame, y)
        lambdas = dict(model._reml_result.lambdas)
        _, evaluation = _fixed_lambda_evaluation(model, y, lambdas)
        tau2 = _criterion_phi(evaluation, len(y)) / lambdas["r"]
        assert tau2 == pytest.approx(MGCV_FLAT_P15_LOWZERO["tau2"], rel=0.05)
        assert float(model._group_edf["r"]) == pytest.approx(
            MGCV_FLAT_P15_LOWZERO["edf_re"], abs=0.75
        )

    def test_flat_heavy_cell_criterion_keeps_an_interior_component(self):
        """flat_p18 (tau=0.08, phi=3, p=1.8, 22% zeros): the categorical case.

        The reduced v0.28.0 criterion runs this RandomEffect to the boundary
        (its argmin sits at the top of this lambda window with tau^2 ~ 4e-5
        and RE edf 0.03) while the exact criterion and mgcv keep a small
        interior component (tau^2 ~ 7.3e-4 vs mgcv 8.99e-4, RE edf 0.64 vs
        0.79) — an 18-22x boundary-vs-interior error (audit section 2.5).
        ``fit_reml`` currently *refuses* this design via mode certification
        on both versions, so the equivalence is asserted on the criterion's
        own argmin, exactly as the audit measured it.
        """
        frame, y = _random_effect_frame("re_flat_p18.csv")
        model = SuperGLM(
            features={"u": Categorical(), "r": RandomEffect()},
            family=Tweedie(p=1.8),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model._build_design_matrix(frame, y, None, None)

        def argmin_scan(grid):
            values, outputs, warm = [], [], None
            for log10_lam in grid:
                result, evaluation = _fixed_lambda_evaluation(
                    model, y, {"r": 10.0**log10_lam}, warm
                )
                warm = (result.beta, result.intercept)
                values.append(evaluation.value)
                outputs.append((result, evaluation))
            best = int(np.argmin(values))
            return best, outputs[best]

        coarse = np.linspace(0.5, 5.0, 19)
        best, _ = argmin_scan(coarse)
        assert 0 < best < len(coarse) - 1, "criterion argmin ran to the lambda boundary"
        refine = np.linspace(coarse[best] - 0.25, coarse[best] + 0.25, 9)
        best_fine, (result, evaluation) = argmin_scan(refine)
        lam = 10.0 ** refine[best_fine]
        tau2 = _criterion_phi(evaluation, len(y)) / lam
        # Wide band: the cell is genuinely flat (mgcv's own CI is wide); the
        # assertion is interior-with-the-right-magnitude vs 4e-5-at-boundary.
        assert 3.5e-4 < tau2 < 1.6e-3
        # Unpenalized fixed part is intercept + 5 contrasts of u; the rest is
        # the RandomEffect's edf, which the reduced criterion shrank to 0.03.
        assert float(result.effective_df) - 6.0 > 0.3


class TestSmoothTermMovesToExactOptimum:
    def test_smooth_edf_matches_exact_criterion(self):
        """Probe A1 sharp cell (p=1.5, phi=2, 32% zeros), CRS(10), n=1200.

        v0.28.0 selects log10(lambda) = -1.913 with total edf 7.13; the
        exact criterion's optimum is -1.973 with total edf 7.32 (audit
        section 2.4).
        """
        rng = np.random.default_rng(11)
        x = rng.uniform(0, 1, 1200)
        mu_true = np.exp(0.3 + 1.0 * np.sin(2 * np.pi * x))
        y = generate_tweedie_cpg(1200, mu_true, phi=2.0, p=1.5, rng=rng)
        model = SuperGLM(
            features={"x": CubicRegressionSpline(n_knots=10)},
            family=Tweedie(p=1.5),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(pd.DataFrame({"x": x}), y)
        assert float(model.result.effective_df) == pytest.approx(7.32, abs=0.10)
        log10_lam = float(np.log10(dict(model._reml_result.lambdas)["x"]))
        assert log10_lam == pytest.approx(-1.973, abs=0.04)


class TestScaleProfileUnit:
    def test_profiler_matches_the_saturated_identity(self):
        """The profiler equals a brute-force minimization of the exact form.

        The exact saturated log-likelihood is reconstructed independently via
        the mu-free identity l_sat(phi) = l(y, mu, w, phi) + D/(2 phi)
        (verified to 5e-13 across all six families in audit probe A0), so
        this pins the profiler to the repo's own density machinery.
        """
        from scipy.optimize import minimize_scalar

        from superglm.reml.scale import (
            prepare_tweedie_reml_scale_data,
            profile_tweedie_reml_scale,
        )

        rng = np.random.default_rng(7)
        n = 400
        mu = np.exp(0.2 + rng.normal(0.0, 0.4, n))
        y = generate_tweedie_cpg(n, mu, phi=1.7, p=1.4, rng=rng)
        assert np.any(y == 0.0), "fixture must exercise the zero atom"
        weights = np.full(n, 1.3)
        distribution = Tweedie(p=1.4)
        deviance = float(np.sum(weights * distribution.deviance_unit(y, np.maximum(mu, 1e-8))))
        penalized_deviance, nullity = deviance + 3.0, 2.0

        def exact_shape(log_phi):
            phi = float(np.exp(log_phi))
            saturated = distribution.log_likelihood(y, np.maximum(mu, 1e-8), weights, phi) + float(
                np.sum(weights * distribution.deviance_unit(y, np.maximum(mu, 1e-8)))
            ) / (2.0 * phi)
            return (
                penalized_deviance / (2.0 * phi)
                - saturated
                - 0.5 * nullity * np.log(2.0 * np.pi * phi)
            )

        brute = minimize_scalar(
            exact_shape, bounds=(-10, 10), method="bounded", options={"xatol": 1e-10}
        )
        data = prepare_tweedie_reml_scale_data(y, weights, 1.4, weight_semantics="prior")
        profiled = profile_tweedie_reml_scale(data, penalized_deviance, nullity)
        assert profiled.criterion == pytest.approx(float(brute.fun), abs=1e-7)
        assert profiled.phi == pytest.approx(float(np.exp(brute.x)), rel=1e-6)
        # Zero rows are an atom: dropping them must not change the saturated
        # sum at any phi.
        positive = y > 0.0
        data_positive_only = prepare_tweedie_reml_scale_data(
            y[positive], weights[positive], 1.4, weight_semantics="prior"
        )
        for phi in (0.5, 1.7, 4.0):
            assert data.saturated_log_likelihood(phi) == pytest.approx(
                data_positive_only.saturated_log_likelihood(phi), rel=1e-12
            )
        # The d(1/phi)/d(Dp) contract against a symmetric difference of the
        # re-profiled optimum.
        step = penalized_deviance * 1e-3
        xi_hi = profile_tweedie_reml_scale(data, penalized_deviance + step, nullity)
        xi_lo = profile_tweedie_reml_scale(data, penalized_deviance - step, nullity)
        finite_difference = (xi_hi.inverse_phi - xi_lo.inverse_phi) / (2.0 * step)
        assert profiled.d_inverse_phi_d_penalized_deviance == pytest.approx(
            finite_difference, rel=1e-3
        )

    def test_sparse_positive_boundary_admits_power_dependent_profiles(self):
        """The finite-profile test must use the Tweedie tail, not a Gaussian one.

        A positive saturated row's density decays as phi**(-1/(p-1)) at large
        phi, so the criterion's upper-tail slope is n_positive/(p-1) - Mp/2.
        The first cut of the profiler tested 2*n_positive <= Mp (a 1/phi
        tail - the very substitution the profiler exists to remove), which
        rejected valid sparse-positive profiles: at p=1.5 one positive row
        against nullity 3 has upper-tail slope +0.5 and an interior optimum,
        yet raised ValueError.
        """
        from superglm.reml.scale import (
            prepare_tweedie_reml_scale_data,
            profile_tweedie_reml_scale,
        )

        data = prepare_tweedie_reml_scale_data(
            np.array([1.3]), np.array([1.0]), 1.5, weight_semantics="prior"
        )
        profiled = profile_tweedie_reml_scale(data, 4.0, 3.0)
        assert np.isfinite(profiled.phi) and profiled.phi > 0.0
        assert np.isfinite(profiled.criterion)
        # Genuinely degenerate: 2*n_positive <= (p-1)*Mp has no interior
        # optimum and must still refuse.
        with pytest.raises(ValueError, match="no finite interior optimum"):
            profile_tweedie_reml_scale(data, 4.0, 9.0)

    def test_profile_optimum_is_a_root_of_the_analytic_score(self):
        """phi-hat must be pinned by the score's zero, not bracket placement.

        A bounded scalar minimizer leaves O(xatol) freedom in where inside
        its final bracket it stops, and which side it stops on flips on
        machine-classed summation rounding - measured ~2e-8 placement
        scatter in log phi across trivially equivalent solver windows, which
        downstream gradient differencing amplified ~2500x into a 2e-6
        machine-dependent Hessian discrepancy on CI. The polished optimum is
        a root of the analytic profile score: placement freedom gone, and
        the returned phi is identical across solver windows to floating
        precision.
        """
        from superglm.reml import scale as scale_module
        from superglm.reml.scale import (
            prepare_tweedie_reml_scale_data,
            profile_tweedie_reml_scale,
        )

        rng = np.random.default_rng(17)
        n = 300
        mu = np.exp(0.2 + rng.normal(0.0, 0.4, n))
        y = generate_tweedie_cpg(n, mu, phi=1.2, p=1.5, rng=rng)
        weights = np.ones(n)
        data = prepare_tweedie_reml_scale_data(y, weights, 1.5, weight_semantics="prior")
        penalized_deviance, nullity = 400.0, 2.0
        profiled = profile_tweedie_reml_scale(data, penalized_deviance, nullity)
        log_phi = float(np.log(profiled.phi))
        score = (
            -0.5 * penalized_deviance * float(np.exp(-log_phi))
            + data.saturated_nll_log_phi_score(profiled.phi)
            - 0.5 * nullity
        )
        # Score residual at the published optimum: the bounded minimizer
        # alone leaves |score| ~ curvature * placement ~ 1e-5; the root
        # polish leaves evaluation roundoff.
        assert abs(score) < 1e-8
        # Placement invariance across a trivially shifted solver window.
        original_window = scale_module._TWEEDIE_LOG_PHI_WINDOW
        try:
            scale_module._TWEEDIE_LOG_PHI_WINDOW = original_window + 3.0e-7
            shifted = profile_tweedie_reml_scale(
                prepare_tweedie_reml_scale_data(y, weights, 1.5, weight_semantics="prior"),
                penalized_deviance,
                nullity,
            )
        finally:
            scale_module._TWEEDIE_LOG_PHI_WINDOW = original_window
        assert abs(float(np.log(shifted.phi)) - log_phi) < 1e-12

    def test_custom_estimated_scale_family_warns_on_the_fallback(self):
        """A custom scale_known=False family must warn, not substitute silently."""
        from types import SimpleNamespace

        from superglm.features import Numeric
        from superglm.model.reml_setup import collect_reml_groups
        from superglm.reml.objective import reml_laml_objective
        from superglm.reml.penalty_algebra import build_penalty_context, build_penalty_matrix
        from superglm.solvers.irls_direct import fit_irls_direct

        rng = np.random.default_rng(3)
        n = 200
        x = rng.uniform(0, 1, n)
        y = np.exp(0.4 * x) + rng.normal(0, 0.1, n)
        model = SuperGLM(
            features={"x": CubicRegressionSpline(n_knots=6), "z": Numeric()},
            family="gaussian",
        )
        frame = pd.DataFrame({"x": x, "z": rng.uniform(0, 1, n)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model._build_design_matrix(frame, y, None, None)
        weights = np.ones(n)
        offset = np.zeros(n)
        lambdas = {"x": 1.0}
        reml_groups = collect_reml_groups(model._groups, model._dm.group_matrices)
        penalties, _caches, _ranks = build_penalty_context(model._dm.group_matrices, reml_groups)
        S = build_penalty_matrix(
            model._dm.group_matrices,
            model._groups,
            lambdas,
            model._dm.p,
            reml_penalties=penalties,
        )
        result, _, XtWX = fit_irls_direct(
            X=model._dm,
            y=y,
            weights=weights,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            lambda2=lambdas,
            offset=offset,
            return_xtwx=True,
            S_override=S,
            reml_penalties=penalties,
            weight_semantics="prior",
        )
        custom_family = SimpleNamespace(scale_known=False)
        with pytest.warns(UserWarning, match="Gaussian-shaped scale profile"):
            reml_laml_objective(
                model._dm,
                custom_family,
                model._link,
                model._groups,
                y,
                result,
                lambdas,
                weights,
                offset,
                XtWX=XtWX,
                S_override=S,
                reml_penalties=penalties,
                weight_semantics="prior",
            )
