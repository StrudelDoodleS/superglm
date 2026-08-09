"""SuperGLM: Penalised GLMs for insurance pricing.

Core API:
    from superglm import SuperGLM, Spline, Categorical, Numeric

    model = SuperGLM(
        penalty="group_lasso", selection_penalty=0.01,
        features={
            "driver_age": Spline(kind="ps", k=14),
            "region": Categorical(base="most_exposed"),
            "density": Numeric(),
        },
    )
    model.fit(X, y, sample_weight=weights)

sklearn-compatible API:
    from superglm import SuperGLMRegressor

    model = SuperGLMRegressor(
        penalty="group_lasso", selection_penalty=0.01,
        spline_features=["driver_age"],
    )
    model.fit(X, y, sample_weight=weights)
"""

from superglm import families
from superglm.constraints import MonotoneRepairer, MonotoneRepairResult
from superglm.diagnostics.discretize import DiscretizationResult, discretization_impact
from superglm.diagnostics.spline_checks import SplineRedundancyReport
from superglm.distributions import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson, Tweedie
from superglm.export import export_rating_tables
from superglm.features.categorical import Categorical
from superglm.features.constraint import Constraint, ConstraintSpec
from superglm.features.factor_smooth import FactorSmooth
from superglm.features.grouping import LevelGrouping, collapse_levels
from superglm.features.interaction import (
    CategoricalInteraction,
    NumericCategorical,
    NumericInteraction,
    PolynomialCategorical,
    PolynomialInteraction,
    SplineCategorical,
    TensorInteraction,
)
from superglm.features.numeric import Numeric
from superglm.features.ordered_categorical import OrderedCategorical
from superglm.features.polynomial import Polynomial
from superglm.features.random_effect import RandomEffect
from superglm.features.spline import (
    BSplineSmooth,
    CubicRegressionSpline,
    NaturalSpline,
    PSpline,
    Spline,
    n_knots_from_k,
)
from superglm.inference.factor_smooths import FactorSmoothResult
from superglm.inference.metrics import ModelMetrics
from superglm.inference.random_effects import RandomEffectResult
from superglm.inference.summary import ModelSummary
from superglm.inference.term import InteractionInference, SmoothCurve, SplineMetadata, TermInference
from superglm.links import (
    CauchitLink,
    CloglogLink,
    IdentityLink,
    InverseLink,
    InverseSquaredLink,
    LogitLink,
    LogLink,
    NegativeBinomialLink,
    PowerLink,
    ProbitLink,
    SqrtLink,
)
from superglm.model import PathResult, SuperGLM
from superglm.model.profile_ops import PublicationModeError
from superglm.model_selection import CrossValidationResult, cross_validate
from superglm.penalties.flavors import Adaptive
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.penalties.sparse_group_lasso import SparseGroupLasso
from superglm.plotting import plot_term_comparison
from superglm.profiling.nb import NBProfileResult, estimate_nb_theta
from superglm.profiling.tweedie import (
    TweedieProfileCIDensityProvenance,
    TweedieProfileCIDetails,
    TweedieProfileCIEndpoint,
    TweedieProfileCIEvaluation,
    TweedieProfileResult,
    estimate_phi,
    estimate_tweedie_p,
    generate_tweedie_cpg,
    tweedie_logpdf,
)
from superglm.reml import REMLResult
from superglm.sklearn import SuperGLMClassifier, SuperGLMRegressor
from superglm.stats.davies import psum_chisq, satterthwaite
from superglm.stats.model_tests import (
    DispersionTestResult,
    ScoreTestZIResult,
    VuongTestResult,
    ZeroInflationResult,
    dispersion_test,
    score_test_zi,
    vuong_test,
    zero_inflation_index,
)
from superglm.stats.wood_pvalue import wood_test_smooth
from superglm.types import LambdaPolicy, LinearConstraintSet
from superglm.validation import (
    DoubleLiftChartResult,
    LiftChartResult,
    LorenzCurveResult,
    LossRatioChartResult,
    double_lift_chart,
    lift_chart,
    lorenz_curve,
    loss_ratio_chart,
)

__all__ = [
    "families",
    "SuperGLM",
    "PathResult",
    "PublicationModeError",
    "DiscretizationResult",
    "discretization_impact",
    "export_rating_tables",
    "ModelMetrics",
    "ModelSummary",
    "RandomEffectResult",
    "SuperGLMRegressor",
    "SuperGLMClassifier",
    "Poisson",
    "Gaussian",
    "Gamma",
    "Binomial",
    "NegativeBinomial",
    "Tweedie",
    "LogLink",
    "LogitLink",
    "IdentityLink",
    "ProbitLink",
    "CloglogLink",
    "CauchitLink",
    "InverseLink",
    "InverseSquaredLink",
    "SqrtLink",
    "PowerLink",
    "NegativeBinomialLink",
    "Spline",
    "PSpline",
    "BSplineSmooth",
    "NaturalSpline",
    "CubicRegressionSpline",
    "Categorical",
    "OrderedCategorical",
    "Constraint",
    "ConstraintSpec",
    "FactorSmooth",
    "FactorSmoothResult",
    "RandomEffect",
    "LevelGrouping",
    "collapse_levels",
    "Numeric",
    "Polynomial",
    "SplineCategorical",
    "PolynomialCategorical",
    "NumericCategorical",
    "CategoricalInteraction",
    "NumericInteraction",
    "PolynomialInteraction",
    "TensorInteraction",
    "GroupElasticNet",
    "GroupLasso",
    "SparseGroupLasso",
    "Ridge",
    "Adaptive",
    "NBProfileResult",
    "REMLResult",
    "LambdaPolicy",
    "LinearConstraintSet",
    "estimate_nb_theta",
    "estimate_tweedie_p",
    "TweedieProfileCIDetails",
    "TweedieProfileCIDensityProvenance",
    "TweedieProfileCIEndpoint",
    "TweedieProfileCIEvaluation",
    "TweedieProfileResult",
    "tweedie_logpdf",
    "estimate_phi",
    "generate_tweedie_cpg",
    "psum_chisq",
    "satterthwaite",
    "wood_test_smooth",
    "n_knots_from_k",
    "TermInference",
    "SmoothCurve",
    "InteractionInference",
    "SplineMetadata",
    "MonotoneRepairResult",
    "MonotoneRepairer",
    "SplineRedundancyReport",
    "cross_validate",
    "CrossValidationResult",
    "plot_term_comparison",
    # Validation toolkit
    "LiftChartResult",
    "DoubleLiftChartResult",
    "LorenzCurveResult",
    "LossRatioChartResult",
    "lift_chart",
    "double_lift_chart",
    "lorenz_curve",
    "loss_ratio_chart",
    # Model adequacy tests
    "ZeroInflationResult",
    "ScoreTestZIResult",
    "DispersionTestResult",
    "VuongTestResult",
    "zero_inflation_index",
    "score_test_zi",
    "dispersion_test",
    "vuong_test",
]
__version__ = "0.22.0"
