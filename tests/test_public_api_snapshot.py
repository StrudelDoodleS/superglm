"""The root export list is deliberate.

``superglm.__all__`` is the reviewed public surface: what the API reference
presents as the package's own names and what ``from superglm import *`` hands
a user. A change to it is an API decision, so edit ``PUBLIC_API`` in the same
pull request, where a reviewer sees the name as a diff line rather than inside
a large commit. The rule applied when the list was trimmed: a name is exported
when the documented user path writes it, or when a public function or method
returns or accepts it. Objects the library builds on the user's behalf stay
importable from the root and from a public module, and the mapping below pins
both promises to the paths the release notes name.
"""

from __future__ import annotations

import importlib

import superglm

PUBLIC_API = [
    "Adaptive",
    "BSplineSmooth",
    "Binomial",
    "BoundInteraction",
    "BoundPredictor",
    "BoundTerm",
    "Categorical",
    "CauchitLink",
    "CloglogLink",
    "Constraint",
    "CrossValidationResult",
    "CubicRegressionSpline",
    "DiscretizationResult",
    "DispersionTestResult",
    "DoubleLiftChartResult",
    "FactorSmooth",
    "FactorSmoothResult",
    "FitDiagnosticReport",
    "FractionalFrequencyWeightWarning",
    "Gamma",
    "GammaLS",
    "Gaussian",
    "GaussianLS",
    "GeneralizedGammaLSS",
    "GeneralizedParetoLSS",
    "GroupElasticNet",
    "GroupLasso",
    "IdentityLink",
    "InteractionInference",
    "InverseLink",
    "InverseSquaredLink",
    "LambdaPolicy",
    "LevelGrouping",
    "LiftChartResult",
    "LogLink",
    "LogNormalLS",
    "LogitLink",
    "LorenzCurveResult",
    "LossRatioChartResult",
    "ModelMetrics",
    "ModelSummary",
    "NBProfileResult",
    "NBThetaBoundWarning",
    "NaturalSpline",
    "NegativeBinomial",
    "NegativeBinomialLS",
    "NegativeBinomialLink",
    "Numeric",
    "OrderedCategorical",
    "PSpline",
    "PathResult",
    "Piecewise",
    "Poisson",
    "Polynomial",
    "PowerLink",
    "Predictor",
    "PriorWeightLatticeWarning",
    "ProbitLink",
    "PublicationModeError",
    "REMLResult",
    "RandomEffect",
    "RandomEffectResult",
    "RatingTableBaseNotRepresentableError",
    "Ridge",
    "ScoreTestZIResult",
    "SeparationError",
    "SeparationWarning",
    "SparseGroupLasso",
    "Spline",
    "SplineRedundancyReport",
    "SqrtLink",
    "SuperGLM",
    "SuperGLMClassifier",
    "SuperGLMRegressor",
    "SuperLSS",
    "TermInference",
    "Tweedie",
    "TweedieLSS",
    "TweedieProfileCIDetails",
    "TweedieProfileResult",
    "TwoPieceLogNormalLSS",
    "TwoPieceNormalLSS",
    "VuongTestResult",
    "ZeroInflationResult",
    "bind_predictor",
    "cat",
    "collapse_levels",
    "cross_validate",
    "discretization_impact",
    "dispersion_test",
    "double_lift_chart",
    "estimate_nb_theta",
    "estimate_phi",
    "estimate_tweedie_p",
    "export_rating_tables",
    "families",
    "generate_tweedie_cpg",
    "interaction",
    "lift_chart",
    "lorenz_curve",
    "loss_ratio_chart",
    "n_knots_from_k",
    "plot_term_comparison",
    "psum_chisq",
    "re",
    "s",
    "satterthwaite",
    "score_test_zi",
    "term",
    "ti",
    "tweedie_logpdf",
    "vuong_test",
    "warmup",
    "wood_test_smooth",
    "zero_inflation_index",
]

IMPORTABLE_NOT_EXPORTED = {
    "CategoricalInteraction": "superglm.features",
    "ConstraintSpec": "superglm.features",
    "LinearConstraintSet": "superglm.types",
    "MonotoneRepairResult": "superglm.constraints",
    "MonotoneRepairer": "superglm.constraints",
    "NumericCategorical": "superglm.features",
    "NumericInteraction": "superglm.features",
    "PolynomialCategorical": "superglm.features",
    "PolynomialInteraction": "superglm.features",
    "SmoothCurve": "superglm.inference",
    "SplineCategorical": "superglm.features",
    "SplineMetadata": "superglm.inference",
    "TensorInteraction": "superglm.features.interaction",
    "TweedieProfileCIDensityProvenance": "superglm.profiling",
    "TweedieProfileCIEndpoint": "superglm.profiling",
    "TweedieProfileCIEvaluation": "superglm.profiling",
}


def test_root_exports_match_the_reviewed_list() -> None:
    added = sorted(set(superglm.__all__) - set(PUBLIC_API))
    removed = sorted(set(PUBLIC_API) - set(superglm.__all__))
    assert (added, removed) == ([], []), f"exports added {added}, removed {removed}"


def test_reviewed_lists_are_sorted_and_disjoint() -> None:
    assert PUBLIC_API == sorted(set(PUBLIC_API))
    assert list(IMPORTABLE_NOT_EXPORTED) == sorted(IMPORTABLE_NOT_EXPORTED)
    assert set(PUBLIC_API).isdisjoint(IMPORTABLE_NOT_EXPORTED)


def test_trimmed_names_stay_importable_from_the_root_and_a_public_module() -> None:
    missing = [name for name in IMPORTABLE_NOT_EXPORTED if not hasattr(superglm, name)]
    assert missing == [], f"trimmed names no longer reachable from superglm: {missing}"
    assert set(IMPORTABLE_NOT_EXPORTED).isdisjoint(superglm.__all__)
    for name, path in IMPORTABLE_NOT_EXPORTED.items():
        module = importlib.import_module(path)
        assert getattr(module, name, None) is getattr(superglm, name), (
            f"{name} is not importable from {path} as the same object"
        )
