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
            "driver_age": Spline(n_knots=10, penalty="ssp"),
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

from superglm.discretize import DiscretizationResult, discretization_impact
from superglm.distributions import Gamma, NegativeBinomial, Poisson, Tweedie
from superglm.links import LogLink
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.polynomial import Polynomial
from superglm.features.spline import Spline
from superglm.metrics import ModelMetrics
from superglm.model import PathResult, SuperGLM
from superglm.summary import ModelSummary
from superglm.plotting import plot_relativities
from superglm.penalties.flavors import Adaptive
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.penalties.sparse_group_lasso import SparseGroupLasso
from superglm.sklearn import SuperGLMRegressor
from superglm.nb_profile import NBProfileResult, estimate_nb_theta
from superglm.tweedie_profile import (
    TweedieProfileResult,
    estimate_phi,
    estimate_tweedie_p,
    generate_tweedie_cpg,
    tweedie_logpdf,
)

__all__ = [
    "SuperGLM",
    "PathResult",
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
    "Categorical",
    "Numeric",
    "Polynomial",
    "GroupLasso",
    "SparseGroupLasso",
    "Ridge",
    "Adaptive",
    "plot_relativities",
    "NBProfileResult",
    "estimate_nb_theta",
    "estimate_tweedie_p",
    "TweedieProfileResult",
    "tweedie_logpdf",
    "estimate_phi",
    "generate_tweedie_cpg",
]
__version__ = "0.1.0"
