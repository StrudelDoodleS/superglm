"""
SuperGLM: Penalised GLMs for insurance pricing.

Core API (auto-detect):
    from superglm import SuperGLM

    model = SuperGLM(
        penalty="group_lasso", selection_penalty=0.01,
        splines=["driver_age"],
    )
    model.fit(X, y, sample_weight=weights)

Core API (explicit):
    from superglm import SuperGLM, Spline, Categorical, Numeric

    model = SuperGLM(
        penalty="group_lasso", selection_penalty=0.01,
        features={
            "driver_age": Spline(kind="bs", k=14),
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
from superglm.cv import CVResult
from superglm.davies import psum_chisq, satterthwaite
from superglm.diagnostics import SplineRedundancyReport
from superglm.discretize import DiscretizationResult, discretization_impact
from superglm.distributions import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson, Tweedie
from superglm.features.categorical import Categorical
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
from superglm.features.spline import (
    BasisSpline,
    CubicRegressionSpline,
    NaturalSpline,
    Spline,
    n_knots_from_k,
)
from superglm.inference import InteractionInference, SmoothCurve, SplineMetadata, TermInference
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
from superglm.metrics import ModelMetrics
from superglm.model import PathResult, SuperGLM
from superglm.model_selection import CrossValidationResult, cross_validate
from superglm.nb_profile import NBProfileResult, estimate_nb_theta
from superglm.penalties.flavors import Adaptive
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.penalties.sparse_group_lasso import SparseGroupLasso
from superglm.reml import REMLResult
from superglm.sklearn import SuperGLMClassifier, SuperGLMRegressor
from superglm.summary import ModelSummary
from superglm.tweedie_profile import (
    TweedieProfileResult,
    estimate_phi,
    estimate_tweedie_p,
    generate_tweedie_cpg,
    tweedie_logpdf,
)
from superglm.wood_pvalue import wood_test_smooth

__all__ = [
    "families",
    "SuperGLM",
    "PathResult",
    "CVResult",
    "DiscretizationResult",
    "discretization_impact",
    "ModelMetrics",
    "ModelSummary",
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
    "BasisSpline",
    "NaturalSpline",
    "CubicRegressionSpline",
    "Categorical",
    "OrderedCategorical",
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
    "estimate_nb_theta",
    "estimate_tweedie_p",
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
]
__version__ = "0.3.1"
