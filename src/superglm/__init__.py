"""
SuperGLM: Penalised GLMs for insurance pricing.

Core API (auto-detect):
    from superglm import SuperGLM

    model = SuperGLM(
        penalty="group_lasso", lambda1=0.01,
        splines=["driver_age"],
    )
    model.fit(X, y, exposure=w)

Core API (explicit):
    from superglm import SuperGLM, Spline, Categorical, Numeric

    model = SuperGLM(
        penalty="group_lasso", lambda1=0.01,
        features={
            "driver_age": Spline(kind="bs", k=14),
            "region": Categorical(base="most_exposed"),
            "density": Numeric(),
        },
    )
    model.fit(X, y, exposure=w)

sklearn-compatible API:
    from superglm import SuperGLMRegressor

    model = SuperGLMRegressor(
        penalty="group_lasso", lambda1=0.01,
        spline_features=["driver_age"],
    )
    model.fit(X, y, sample_weight=exposure)
"""

from superglm.cv import CVResult
from superglm.davies import psum_chisq, satterthwaite
from superglm.discretize import DiscretizationResult, discretization_impact
from superglm.distributions import Gamma, NegativeBinomial, Poisson, Tweedie
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
from superglm.features.polynomial import Polynomial
from superglm.features.spline import (
    BasisSpline,
    CubicRegressionSpline,
    NaturalSpline,
    Spline,
    n_knots_from_k,
)
from superglm.links import LogLink
from superglm.metrics import ModelMetrics
from superglm.model import PathResult, SuperGLM
from superglm.nb_profile import NBProfileResult, estimate_nb_theta
from superglm.penalties.flavors import Adaptive
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.penalties.sparse_group_lasso import SparseGroupLasso
from superglm.plotting import plot_interaction, plot_relativities
from superglm.reml import REMLResult
from superglm.sklearn import SuperGLMRegressor
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
    "SuperGLM",
    "PathResult",
    "CVResult",
    "DiscretizationResult",
    "discretization_impact",
    "ModelMetrics",
    "ModelSummary",
    "SuperGLMRegressor",
    "Poisson",
    "Gamma",
    "NegativeBinomial",
    "Tweedie",
    "LogLink",
    "Spline",
    "BasisSpline",
    "NaturalSpline",
    "CubicRegressionSpline",
    "Categorical",
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
    "plot_interaction",
    "plot_relativities",
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
]
__version__ = "0.1.0"
