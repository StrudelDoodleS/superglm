"""The root export list is deliberate.

``superglm.__all__`` is the reviewed public surface: what the API reference
presents as the package's own names and what ``from superglm import *`` hands
a user. A change to it is an API decision, so edit ``PUBLIC_API`` in the same
pull request, where a reviewer sees the name as a diff line rather than inside
a large commit. The rule applied when the list was trimmed: a name is exported
when the documented user path writes it or receives it from a public method;
objects the library builds on the user's behalf stay importable from the root
and from their home modules but are not exports, and the second list pins that.
"""

from __future__ import annotations

import importlib

import superglm

PUBLIC_API = [
    "Adaptive",
    "BSplineSmooth",
    "Binomial",
    "BoundPredictor",
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

IMPORTABLE_NOT_EXPORTED = [
    "BoundInteraction",
    "BoundTerm",
    "CategoricalInteraction",
    "ConstraintSpec",
    "LinearConstraintSet",
    "MonotoneRepairResult",
    "MonotoneRepairer",
    "NumericCategorical",
    "NumericInteraction",
    "PolynomialCategorical",
    "PolynomialInteraction",
    "SmoothCurve",
    "SplineCategorical",
    "SplineMetadata",
    "TensorInteraction",
    "TweedieProfileCIDensityProvenance",
    "TweedieProfileCIDetails",
    "TweedieProfileCIEndpoint",
    "TweedieProfileCIEvaluation",
]


def test_root_exports_match_the_reviewed_list() -> None:
    added = sorted(set(superglm.__all__) - set(PUBLIC_API))
    removed = sorted(set(PUBLIC_API) - set(superglm.__all__))
    assert (added, removed) == ([], []), f"exports added {added}, removed {removed}"


def test_reviewed_lists_are_sorted_and_disjoint() -> None:
    assert PUBLIC_API == sorted(set(PUBLIC_API))
    assert IMPORTABLE_NOT_EXPORTED == sorted(set(IMPORTABLE_NOT_EXPORTED))
    assert set(PUBLIC_API).isdisjoint(IMPORTABLE_NOT_EXPORTED)


def test_trimmed_names_stay_importable_from_the_root() -> None:
    missing = [name for name in IMPORTABLE_NOT_EXPORTED if not hasattr(superglm, name)]
    assert missing == [], f"trimmed names no longer reachable from superglm: {missing}"
    assert set(IMPORTABLE_NOT_EXPORTED).isdisjoint(superglm.__all__)
    for name in IMPORTABLE_NOT_EXPORTED:
        obj = getattr(superglm, name)
        home = importlib.import_module(obj.__module__)
        assert getattr(home, obj.__name__) is obj, f"{name} is not the object from {obj.__module__}"
