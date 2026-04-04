"""REML smoothing parameter estimation.

Estimates per-term smoothing parameters (lambda_j) from the data. The
direct ``lambda1=0`` path optimizes a Laplace-approximate REML/LAML
criterion over log-lambdas, while the mixed penalized-selection path
retains the Wood (2011) fixed-point update around PIRLS.

Coexists with group lasso: REML controls within-group smoothness
(per-term lambda_j), group lasso controls between-group selection
(lambda1). They are orthogonal.

References
----------
- Wood (2011): Fast stable restricted maximum likelihood and marginal
  likelihood estimation of semiparametric generalized linear models.
  JRSS-B 73(1), 3-36.
- Wood (2017): Generalized Additive Models, 2nd ed., Ch 6.2.
- Wood & Fasiolo (2017): A generalized Fellner-Schall method for smoothing
  parameter optimization. Biometrics 73(4), 1071-1081.

# Internal submodules: import siblings directly, not through this __init__.
"""

from superglm.reml.direct import optimize_direct_reml  # noqa: F401
from superglm.reml.discrete import (  # noqa: F401
    optimize_discrete_reml_cached_w,
)
from superglm.reml.efs import optimize_efs_reml  # noqa: F401
from superglm.reml.gradient import (  # noqa: F401
    reml_direct_gradient,
    reml_direct_hessian,
)
from superglm.reml.multi_penalty import (  # noqa: F401
    SimilarityTransformResult,
    logdet_s_gradient,
    logdet_s_hessian,
    similarity_transform_logdet,
)
from superglm.reml.objective import reml_laml_objective  # noqa: F401
from superglm.reml.penalty_algebra import (  # noqa: F401
    build_penalty_caches,
    build_penalty_components,
    cached_logdet_s_plus,
    compute_logdet_s_derivatives,
    compute_logdet_s_plus,
    compute_total_penalty_rank,
)
from superglm.reml.result import PenaltyCache, REMLResult, _map_beta_between_bases  # noqa: F401
from superglm.reml.runner import run_reml_once  # noqa: F401
from superglm.reml.w_derivatives import (  # noqa: F401
    compute_d2W_deta2,
    compute_dW_deta,
    reml_w_correction,
)
