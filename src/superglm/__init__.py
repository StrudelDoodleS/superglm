"""
SuperGLM: Penalised GLMs for insurance pricing.

sklearn-compatible API:
    from superglm import SuperGLMRegressor, GroupLasso, Adaptive

    model = SuperGLMRegressor(
        family="poisson",
        spline_features=["driver_age"],
        penalty=GroupLasso(lambda1=0.1, flavor=Adaptive()),
    )
    model.fit(X, y, sample_weight=exposure)

Core API:
    from superglm import SuperGLM, Poisson, Spline, Categorical, Numeric, GroupLasso

    model = SuperGLM(family=Poisson(), penalty=GroupLasso(lambda1=0.1))
    model.add_feature("driver_age", Spline(n_knots=15))
    model.add_feature("region", Categorical(base="most_exposed"))
    model.add_feature("density", Numeric())
    model.fit(X, y, exposure=w)
"""

from superglm.distributions import Gamma, Poisson, Tweedie
from superglm.features.categorical import Categorical
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.model import SuperGLM
from superglm.penalties.flavors import Adaptive
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.penalties.sparse_group_lasso import SparseGroupLasso
from superglm.sklearn import SuperGLMRegressor

__all__ = [
    "SuperGLM",
    "SuperGLMRegressor",
    "Poisson",
    "Gamma",
    "Tweedie",
    "Spline",
    "Categorical",
    "Numeric",
    "GroupLasso",
    "SparseGroupLasso",
    "Ridge",
    "Adaptive",
]
__version__ = "0.1.0"
