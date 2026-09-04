"""Public and lower-level contracts for multi-predictor distributional models."""

_EXPORTS = {
    "GammaLS": ("superglm.distributional.families.gamma", "GammaLS"),
    "GaussianLS": ("superglm.distributional.families.gaussian", "GaussianLS"),
    "GeneralizedGammaLSS": (
        "superglm.distributional.families.generalized_gamma",
        "GeneralizedGammaLSS",
    ),
    "GeneralizedParetoLSS": (
        "superglm.distributional.families.generalized_pareto",
        "GeneralizedParetoLSS",
    ),
    "LogNormalLS": ("superglm.distributional.families.log_normal", "LogNormalLS"),
    "NegativeBinomialLS": (
        "superglm.distributional.families.negative_binomial",
        "NegativeBinomialLS",
    ),
    "NegativeBinomialPoissonBoundaryError": (
        "superglm.distributional.families.negative_binomial",
        "NegativeBinomialPoissonBoundaryError",
    ),
    "TweedieLSS": ("superglm.distributional.families.tweedie", "TweedieLSS"),
    "TwoPieceLogNormalLSS": (
        "superglm.distributional.families.two_piece",
        "TwoPieceLogNormalLSS",
    ),
    "TwoPieceNormalLSS": (
        "superglm.distributional.families.two_piece",
        "TwoPieceNormalLSS",
    ),
    "Predictor": ("superglm.distributional.predictor", "Predictor"),
    "SuperLSSTrainingTelemetry": (
        "superglm.distributional.api",
        "SuperLSSTrainingTelemetry",
    ),
    "COMPLETE_OBSERVATION": ("superglm.distributional.family", "COMPLETE_OBSERVATION"),
    "ConfigurableDistributionalFamily": (
        "superglm.distributional.family",
        "ConfigurableDistributionalFamily",
    ),
    "DefaultPredictionFamily": (
        "superglm.distributional.family",
        "DefaultPredictionFamily",
    ),
    "DistributionFunctionFamily": (
        "superglm.distributional.family",
        "DistributionFunctionFamily",
    ),
    "DistributionalFamily": ("superglm.distributional.family", "DistributionalFamily"),
    "ExpectedInformationFamily": (
        "superglm.distributional.family",
        "ExpectedInformationFamily",
    ),
    "ExpectedShortfallFamily": (
        "superglm.distributional.family",
        "ExpectedShortfallFamily",
    ),
    "FamilyCapabilities": ("superglm.distributional.family", "FamilyCapabilities"),
    "FamilyLikelihoodPlan": ("superglm.distributional.family", "FamilyLikelihoodPlan"),
    "FiniteDifferenceDirection": (
        "superglm.distributional.smoothing.endpoint_direction",
        "FiniteDifferenceDirection",
    ),
    "finite_difference_curvature_direction": (
        "superglm.distributional.smoothing.endpoint_direction",
        "finite_difference_curvature_direction",
    ),
    "FitFailureDiagnosingFamily": (
        "superglm.distributional.family",
        "FitFailureDiagnosingFamily",
    ),
    "InitialParameterState": ("superglm.distributional.family", "InitialParameterState"),
    "LikelihoodPlanValidatingFamily": (
        "superglm.distributional.family",
        "LikelihoodPlanValidatingFamily",
    ),
    "NaturalLikelihoodEvaluation": (
        "superglm.distributional.family",
        "NaturalLikelihoodEvaluation",
    ),
    "ObservationContract": ("superglm.distributional.family", "ObservationContract"),
    "ParameterSpec": ("superglm.distributional.family", "ParameterSpec"),
    "ParameterSupport": ("superglm.distributional.family", "ParameterSupport"),
    "PredictorCurvatureDirectionalFamily": (
        "superglm.distributional.family",
        "PredictorCurvatureDirectionalFamily",
    ),
    "ResponseBoundaryFamily": ("superglm.distributional.family", "ResponseBoundaryFamily"),
    "LegacyPowerWeightArtifactError": (
        "superglm.distributional.weights",
        "LegacyPowerWeightArtifactError",
    ),
    "LikelihoodWeightError": ("superglm.distributional.weights", "LikelihoodWeightError"),
    "ResolvedLikelihoodWeights": (
        "superglm.distributional.weights",
        "ResolvedLikelihoodWeights",
    ),
    "UnsupportedLikelihoodContractError": (
        "superglm.distributional.weights",
        "UnsupportedLikelihoodContractError",
    ),
    "WeightContract": ("superglm.distributional.weights", "WeightContract"),
    "WeightProvenance": ("superglm.distributional.weights", "WeightProvenance"),
    "resolve_likelihood_weights": (
        "superglm.distributional.weights",
        "resolve_likelihood_weights",
    ),
    "validated_derivative_order": (
        "superglm.distributional.family",
        "validated_derivative_order",
    ),
    "validated_parameter_matrix": (
        "superglm.distributional.family",
        "validated_parameter_matrix",
    ),
    # The inference suite: the posterior primitive, the residual and check
    # payloads, the scores, the per-parameter term inference and the surfaces.
    # Every payload is a frozen dataclass with a ``to_json()``; every builder
    # takes the fitted model, not the facade.
    "PosteriorDraws": ("superglm.distributional.posterior", "PosteriorDraws"),
    "posterior_bounds": ("superglm.distributional.posterior", "posterior_bounds"),
    "posterior_covariance": ("superglm.distributional.posterior", "posterior_covariance"),
    "posterior_draws": ("superglm.distributional.posterior", "posterior_draws"),
    "posterior_parameters": ("superglm.distributional.posterior", "posterior_parameters"),
    "posterior_predictive": ("superglm.distributional.posterior", "posterior_predictive"),
    "resolve_quantity": ("superglm.distributional.posterior", "resolve_quantity"),
    "simultaneous_critical_value": (
        "superglm.distributional.posterior",
        "simultaneous_critical_value",
    ),
    "ResidualSet": ("superglm.distributional.residuals", "ResidualSet"),
    "compute_residuals": ("superglm.distributional.residuals", "compute_residuals"),
    "replication_sample": ("superglm.distributional.residuals", "replication_sample"),
    "residual_values": ("superglm.distributional.residuals", "residual_values"),
    "QQPayload": ("superglm.distributional.checks.qq", "QQPayload"),
    "order_statistic_grid": ("superglm.distributional.checks.qq", "order_statistic_grid"),
    "qq_payload": ("superglm.distributional.checks.qq", "qq_payload"),
    "WormPanel": ("superglm.distributional.checks.worm", "WormPanel"),
    "WormPayload": ("superglm.distributional.checks.worm", "WormPayload"),
    "q_statistics": ("superglm.distributional.checks.worm", "q_statistics"),
    "worm_payload": ("superglm.distributional.checks.worm", "worm_payload"),
    "PITPayload": ("superglm.distributional.checks.pit", "PITPayload"),
    "pit_payload": ("superglm.distributional.checks.pit", "pit_payload"),
    "BinnedCheck": ("superglm.distributional.checks.binned", "BinnedCheck"),
    "BinnedCheck2D": ("superglm.distributional.checks.binned", "BinnedCheck2D"),
    "binned_check": ("superglm.distributional.checks.binned", "binned_check"),
    "binned_check_2d": ("superglm.distributional.checks.binned", "binned_check_2d"),
    "ActualExpected": ("superglm.distributional.checks.calibration", "ActualExpected"),
    "CalibrationPayload": ("superglm.distributional.checks.calibration", "CalibrationPayload"),
    "ReliabilityCurve": ("superglm.distributional.checks.calibration", "ReliabilityCurve"),
    "actual_expected_check": (
        "superglm.distributional.checks.calibration",
        "actual_expected_check",
    ),
    "calibration_payload": ("superglm.distributional.checks.calibration", "calibration_payload"),
    "reliability_curve": ("superglm.distributional.checks.calibration", "reliability_curve"),
    "crps": ("superglm.distributional.checks.scores", "crps"),
    "crps_closed_form": ("superglm.distributional.checks.scores", "crps_closed_form"),
    "crps_numeric": ("superglm.distributional.checks.scores", "crps_numeric"),
    "has_closed_form_crps": ("superglm.distributional.checks.scores", "has_closed_form_crps"),
    "log_score": ("superglm.distributional.checks.scores", "log_score"),
    "score_table": ("superglm.distributional.checks.scores", "score_table"),
    "threshold_weighted_crps": (
        "superglm.distributional.checks.scores",
        "threshold_weighted_crps",
    ),
    "Comparison": ("superglm.distributional.checks.compare", "Comparison"),
    "MurphyPayload": ("superglm.distributional.checks.compare", "MurphyPayload"),
    "compare_models": ("superglm.distributional.checks.compare", "compare_models"),
    "murphy_diagram": ("superglm.distributional.checks.compare", "murphy_diagram"),
    "grouped_ratio": ("superglm.distributional.checks._aggregate", "grouped_ratio"),
    "ParameterTermEffect": ("superglm.distributional.terms", "ParameterTermEffect"),
    "TermTest": ("superglm.distributional.terms", "TermTest"),
    "summary_table": ("superglm.distributional.terms", "summary_table"),
    "term_effect": ("superglm.distributional.terms", "term_effect"),
    "term_test": ("superglm.distributional.terms", "term_test"),
    "DensityFan": ("superglm.distributional.surfaces", "DensityFan"),
    "Histogram": ("superglm.distributional.surfaces", "Histogram"),
    "Portfolio": ("superglm.distributional.surfaces", "Portfolio"),
    "RiskCurves": ("superglm.distributional.surfaces", "RiskCurves"),
    "Spread": ("superglm.distributional.surfaces", "Spread"),
    "density_fan": ("superglm.distributional.surfaces", "density_fan"),
    "parameter_spread": ("superglm.distributional.surfaces", "parameter_spread"),
    "portfolio": ("superglm.distributional.surfaces", "portfolio"),
    "risk_curves": ("superglm.distributional.surfaces", "risk_curves"),
    "AtomFamily": ("superglm.distributional.family", "AtomFamily"),
    "PriorWeightedDistributionFunctionFamily": (
        "superglm.distributional.family",
        "PriorWeightedDistributionFunctionFamily",
    ),
    "PriorWeightedExpectedShortfallFamily": (
        "superglm.distributional.family",
        "PriorWeightedExpectedShortfallFamily",
    ),
    "PriorWeightedVarianceFamily": (
        "superglm.distributional.family",
        "PriorWeightedVarianceFamily",
    ),
    "VarianceFamily": ("superglm.distributional.family", "VarianceFamily"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    import importlib

    module_name, attribute_name = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
