"""Import surface compatibility tests.

Verifies that all documented import paths remain working throughout the
src/ restructure.  Old paths survive via package __init__.py re-exports
or thin DeprecationWarning stubs.  New canonical paths are added as each
phase lands.
"""


# ── Old paths (must keep working after moves) ──────────────────


def test_reml_imports():
    from superglm.reml import REMLResult  # noqa: F401


def test_inference_imports():
    from superglm.inference import (  # noqa: F401
        InteractionInference,
        SmoothCurve,
        SplineMetadata,
        TermInference,
    )


def test_metrics_imports():
    from superglm.metrics import ModelMetrics  # noqa: F401


def test_summary_imports():
    from superglm.summary import ModelSummary  # noqa: F401


def test_tweedie_profile_imports():
    from superglm.tweedie_profile import (  # noqa: F401
        TweedieProfileResult,
        estimate_phi,
        estimate_tweedie_p,
        generate_tweedie_cpg,
        tweedie_logpdf,
    )


def test_nb_profile_imports():
    from superglm.nb_profile import NBProfileResult, estimate_nb_theta  # noqa: F401


def test_diagnostics_imports():
    from superglm.diagnostics import (  # noqa: F401
        SplineRedundancyReport,
        spline_redundancy,
        term_drop_diagnostics,
        term_importance,
    )


def test_discretize_imports():
    from superglm.discretize import (  # noqa: F401
        DiscretizationResult,
        discretization_impact,
    )


def test_model_tests_imports():
    from superglm.model_tests import (  # noqa: F401
        DispersionTestResult,
        ScoreTestZIResult,
        VuongTestResult,
        ZeroInflationResult,
        dispersion_test,
        score_test_zi,
        vuong_test,
        zero_inflation_index,
    )


def test_davies_imports():
    from superglm.davies import psum_chisq, satterthwaite  # noqa: F401


def test_wood_pvalue_imports():
    from superglm.wood_pvalue import wood_test_smooth  # noqa: F401


def test_multi_penalty_imports():
    from superglm.multi_penalty import SimilarityTransformResult  # noqa: F401


def test_reml_optimizer_imports():
    from superglm.reml_optimizer import run_reml_once  # noqa: F401


def test_validation_imports():
    from superglm.validation import (  # noqa: F401
        DoubleLiftChartResult,
        LiftChartResult,
        LorenzCurveResult,
        LossRatioChartResult,
        double_lift_chart,
        lift_chart,
        lorenz_curve,
        loss_ratio_chart,
    )


# ── Top-level public API ───────────────────────────────────────


def test_toplevel_reexports():
    """Everything in __all__ is importable from the superglm namespace."""
    import superglm

    for name in superglm.__all__:
        assert hasattr(superglm, name), f"superglm.{name} not accessible"


# ── New canonical paths (added as each phase lands) ────────────


def test_reml_result_canonical():
    from superglm.reml.result import PenaltyCache, REMLResult, _map_beta_between_bases  # noqa: F401


def test_reml_penalty_algebra_canonical():
    from superglm.reml.penalty_algebra import (  # noqa: F401
        build_penalty_caches,
        build_penalty_components,
        cached_logdet_s_plus,
        compute_logdet_s_derivatives,
        compute_logdet_s_plus,
        compute_total_penalty_rank,
    )


def test_reml_optimizer_canonical():
    from superglm.reml.direct import optimize_direct_reml  # noqa: F401
    from superglm.reml.discrete import optimize_discrete_reml_cached_w  # noqa: F401
    from superglm.reml.efs import optimize_efs_reml  # noqa: F401
    from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian  # noqa: F401
    from superglm.reml.objective import reml_laml_objective  # noqa: F401
    from superglm.reml.runner import run_reml_once  # noqa: F401
    from superglm.reml.w_derivatives import (  # noqa: F401
        compute_d2W_deta2,
        compute_dW_deta,
        reml_w_correction,
    )


def test_reml_multi_penalty_canonical():
    from superglm.reml.multi_penalty import (  # noqa: F401
        SimilarityTransformResult,
        logdet_s_gradient,
        logdet_s_hessian,
        similarity_transform_logdet,
    )


def test_inference_term_canonical():
    from superglm.inference.term import (  # noqa: F401
        _VALID_CENTERING,
        InteractionInference,
        SmoothCurve,
        SplineMetadata,
        TermInference,
        _recenter_term,
        _resolve_group_lambda,
        _safe_exp,
        compute_coef_covariance,
        feature_se_from_cov,
        spline_group_enrichment,
        term_inference,
    )


def test_inference_metrics_canonical():
    from superglm.inference.metrics import ModelMetrics  # noqa: F401


def test_inference_coef_tables_canonical():
    from superglm.inference.coef_tables import (  # noqa: F401
        build_basis_detail,
        build_coef_rows,
    )


def test_inference_summary_canonical():
    from superglm.inference.summary import (  # noqa: F401
        ModelSummary,
        _BasisDetailRow,
        _CoefRow,
        _compute_coef_stats,
    )


def test_inference_covariance_canonical():
    from superglm.inference.covariance import (  # noqa: F401
        _penalised_xtwx_inv,
        _penalised_xtwx_inv_gram,
        _second_diff_penalty,
    )


def test_profiling_tweedie_canonical():
    from superglm.profiling.tweedie import TweedieProfileResult, estimate_tweedie_p  # noqa: F401


def test_profiling_nb_canonical():
    from superglm.profiling.nb import NBProfileResult, estimate_nb_theta  # noqa: F401


def test_stats_model_tests_canonical():
    from superglm.stats.model_tests import (  # noqa: F401
        DispersionTestResult,
        ScoreTestZIResult,
        VuongTestResult,
        ZeroInflationResult,
        dispersion_test,
        score_test_zi,
        vuong_test,
        zero_inflation_index,
    )


def test_stats_davies_canonical():
    from superglm.stats.davies import psum_chisq, satterthwaite  # noqa: F401


def test_stats_wood_pvalue_canonical():
    from superglm.stats.wood_pvalue import wood_test_smooth  # noqa: F401


def test_diagnostics_spline_checks_canonical():
    from superglm.diagnostics.spline_checks import (  # noqa: F401
        SplineRedundancyReport,
        spline_redundancy,
    )


def test_diagnostics_term_diagnostics_canonical():
    from superglm.diagnostics.term_diagnostics import (  # noqa: F401
        term_drop_diagnostics,
        term_importance,
    )


def test_diagnostics_discretize_canonical():
    from superglm.diagnostics.discretize import (  # noqa: F401
        DiscretizationResult,
        discretization_impact,
    )
