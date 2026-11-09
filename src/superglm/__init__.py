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

import typing as _typing

from superglm import _numba_cache
from superglm._lazy import attach as _attach

if _typing.TYPE_CHECKING:
    from superglm import families
    from superglm._tweedie import NearPoissonDispersionError, generate_tweedie_cpg, tweedie_logpdf
    from superglm.constraints import (
        MonotoneRepairer as MonotoneRepairer,
    )
    from superglm.constraints import (
        MonotoneRepairResult as MonotoneRepairResult,
    )
    from superglm.diagnostics.convergence import ConvergenceWarning
    from superglm.diagnostics.discretize import DiscretizationResult, discretization_impact
    from superglm.diagnostics.fit_cost import SlowFitWarning
    from superglm.diagnostics.fit_report import FitDiagnosticReport
    from superglm.diagnostics.separation import SeparationError, SeparationWarning
    from superglm.diagnostics.spline_checks import SplineRedundancyReport
    from superglm.distributional.api import SuperLSS
    from superglm.distributional.binding import BoundPredictor, bind_predictor
    from superglm.distributional.families.dirichlet import DirichletLSS
    from superglm.distributional.families.gamma import GammaLS
    from superglm.distributional.families.gaussian import GaussianLS
    from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
    from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
    from superglm.distributional.families.log_logistic import LogLogisticLSS
    from superglm.distributional.families.log_normal import LogNormalLS
    from superglm.distributional.families.lomax import LomaxLSS
    from superglm.distributional.families.negative_binomial import NegativeBinomialLS
    from superglm.distributional.families.tweedie import TweedieLSS
    from superglm.distributional.families.two_piece import (
        TwoPieceLogNormalLSS,
        TwoPieceNormalLSS,
    )
    from superglm.distributional.predictor import Predictor
    from superglm.distributions import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson, Tweedie
    from superglm.export import RatingTableBaseNotRepresentableError, export_rating_tables
    from superglm.features._spline_config import SCOPRetiredWarning, SplineOptionWarning
    from superglm.features._spline_ranges import PolynomialRange
    from superglm.features.categorical import Categorical
    from superglm.features.constraint import Constraint, ConstraintRules
    from superglm.features.constraint import ConstraintSpec as ConstraintSpec
    from superglm.features.factor_smooth import FactorSmooth, FactorSmoothGroupTermWarning
    from superglm.features.grouping import LevelGrouping, collapse_levels
    from superglm.features.interaction import (
        CategoricalInteraction as CategoricalInteraction,
    )
    from superglm.features.interaction import (
        NumericCategorical as NumericCategorical,
    )
    from superglm.features.interaction import (
        NumericInteraction as NumericInteraction,
    )
    from superglm.features.interaction import (
        PolynomialCategorical as PolynomialCategorical,
    )
    from superglm.features.interaction import (
        PolynomialInteraction as PolynomialInteraction,
    )
    from superglm.features.interaction import (
        SplineCategorical as SplineCategorical,
    )
    from superglm.features.interaction import (
        TensorInteraction as TensorInteraction,
    )
    from superglm.features.numeric import Numeric
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.piecewise import Piecewise
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
    from superglm.inference._term_types import (
        InteractionInference,
        TermInference,
    )
    from superglm.inference._term_types import (
        SmoothCurve as SmoothCurve,
    )
    from superglm.inference._term_types import (
        SplineMetadata as SplineMetadata,
    )
    from superglm.inference.factor_smooths import FactorSmoothResult
    from superglm.inference.metrics import ModelMetrics
    from superglm.inference.random_effects import RandomEffectResult
    from superglm.inference.summary import ModelSummary
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
    from superglm.model.input_validation import (
        FractionalFrequencyWeightWarning,
        PriorWeightLatticeWarning,
    )
    from superglm.model.profile_ops import PublicationModeError
    from superglm.model_selection import CrossValidationResult, cross_validate
    from superglm.penalties.flavors import Adaptive
    from superglm.penalties.group_elastic_net import GroupElasticNet
    from superglm.penalties.group_lasso import GroupLasso
    from superglm.penalties.ridge import Ridge
    from superglm.penalties.sparse_group_lasso import SparseGroupLasso
    from superglm.plotting import plot_term_comparison
    from superglm.profiling.nb import NBProfileResult, NBThetaBoundWarning
    from superglm.profiling.tweedie import TweedieProfileResult
    from superglm.reml import REMLResult
    from superglm.reml.identified import WeakIdentificationWarning
    from superglm.reml.observed_geometry import (
        ObservedModeNotCertifiedError,
        ObservedModeNotConvergedError,
    )
    from superglm.sklearn import SuperGLMClassifier, SuperGLMRegressor
    from superglm.solvers.irls_direct import StructuredSolverError
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
    from superglm.structure import FeatureStructure, Structure, StructureError, read_structure
    from superglm.terms import (
        BoundInteraction,
        BoundTerm,
        RedundantInteractionWarning,
        cat,
        interaction,
        re,
        s,
        term,
        ti,
    )
    from superglm.types import LambdaPolicy
    from superglm.types import LinearConstraintSet as LinearConstraintSet
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


# Importing superglm imports none of this: a name is read from the module that defines it, and a
# submodule imported, on first access (``superglm._lazy``), so a service that scores a fitted model
# does not load plotting, diagnostics or scikit-learn. ``__all__`` and every importable name are as
# before; tests/test_import_compat.py keeps this table equal to the imports above.
_LAZY_EXPORTS: dict[str, tuple[str, ...]] = {
    "superglm._tweedie": ("NearPoissonDispersionError", "generate_tweedie_cpg", "tweedie_logpdf"),
    "superglm.constraints": ("MonotoneRepairResult", "MonotoneRepairer"),
    "superglm.diagnostics.convergence": ("ConvergenceWarning",),
    "superglm.diagnostics.discretize": ("DiscretizationResult", "discretization_impact"),
    "superglm.diagnostics.fit_cost": ("SlowFitWarning",),
    "superglm.diagnostics.fit_report": ("FitDiagnosticReport",),
    "superglm.diagnostics.separation": ("SeparationError", "SeparationWarning"),
    "superglm.diagnostics.spline_checks": ("SplineRedundancyReport",),
    "superglm.distributional.api": ("SuperLSS",),
    "superglm.distributional.binding": ("BoundPredictor", "bind_predictor"),
    "superglm.distributional.families.dirichlet": ("DirichletLSS",),
    "superglm.distributional.families.gamma": ("GammaLS",),
    "superglm.distributional.families.gaussian": ("GaussianLS",),
    "superglm.distributional.families.generalized_gamma": ("GeneralizedGammaLSS",),
    "superglm.distributional.families.generalized_pareto": ("GeneralizedParetoLSS",),
    "superglm.distributional.families.log_normal": ("LogNormalLS",),
    "superglm.distributional.families.lomax": ("LomaxLSS",),
    "superglm.distributional.families.log_logistic": ("LogLogisticLSS",),
    "superglm.distributional.families.negative_binomial": ("NegativeBinomialLS",),
    "superglm.distributional.families.tweedie": ("TweedieLSS",),
    "superglm.distributional.families.two_piece": ("TwoPieceLogNormalLSS", "TwoPieceNormalLSS"),
    "superglm.distributional.predictor": ("Predictor",),
    "superglm.distributions": (
        "Binomial",
        "Gamma",
        "Gaussian",
        "NegativeBinomial",
        "Poisson",
        "Tweedie",
    ),
    "superglm.export": ("RatingTableBaseNotRepresentableError", "export_rating_tables"),
    "superglm.features._spline_config": ("SCOPRetiredWarning", "SplineOptionWarning"),
    "superglm.features._spline_ranges": ("PolynomialRange",),
    "superglm.features.categorical": ("Categorical",),
    "superglm.features.constraint": ("Constraint", "ConstraintRules", "ConstraintSpec"),
    "superglm.features.factor_smooth": ("FactorSmooth", "FactorSmoothGroupTermWarning"),
    "superglm.features.grouping": ("LevelGrouping", "collapse_levels"),
    "superglm.features.interaction": (
        "CategoricalInteraction",
        "NumericCategorical",
        "NumericInteraction",
        "PolynomialCategorical",
        "PolynomialInteraction",
        "SplineCategorical",
        "TensorInteraction",
    ),
    "superglm.features.numeric": ("Numeric",),
    "superglm.features.ordered_categorical": ("OrderedCategorical",),
    "superglm.features.piecewise": ("Piecewise",),
    "superglm.features.polynomial": ("Polynomial",),
    "superglm.features.random_effect": ("RandomEffect",),
    "superglm.features.spline": (
        "BSplineSmooth",
        "CubicRegressionSpline",
        "NaturalSpline",
        "PSpline",
        "Spline",
        "n_knots_from_k",
    ),
    "superglm.inference._term_types": (
        "InteractionInference",
        "SmoothCurve",
        "SplineMetadata",
        "TermInference",
    ),
    "superglm.inference.factor_smooths": ("FactorSmoothResult",),
    "superglm.inference.metrics": ("ModelMetrics",),
    "superglm.inference.random_effects": ("RandomEffectResult",),
    "superglm.inference.summary": ("ModelSummary",),
    "superglm.links": (
        "CauchitLink",
        "CloglogLink",
        "IdentityLink",
        "InverseLink",
        "InverseSquaredLink",
        "LogLink",
        "LogitLink",
        "NegativeBinomialLink",
        "PowerLink",
        "ProbitLink",
        "SqrtLink",
    ),
    "superglm.model": ("PathResult", "SuperGLM"),
    "superglm.model.input_validation": (
        "FractionalFrequencyWeightWarning",
        "PriorWeightLatticeWarning",
    ),
    "superglm.model.profile_ops": ("PublicationModeError",),
    "superglm.model_selection": ("CrossValidationResult", "cross_validate"),
    "superglm.penalties.flavors": ("Adaptive",),
    "superglm.penalties.group_elastic_net": ("GroupElasticNet",),
    "superglm.penalties.group_lasso": ("GroupLasso",),
    "superglm.penalties.ridge": ("Ridge",),
    "superglm.penalties.sparse_group_lasso": ("SparseGroupLasso",),
    "superglm.plotting": ("plot_term_comparison",),
    "superglm.profiling.nb": ("NBProfileResult", "NBThetaBoundWarning"),
    "superglm.profiling.tweedie": ("TweedieProfileResult",),
    "superglm.reml": ("REMLResult",),
    "superglm.reml.identified": ("WeakIdentificationWarning",),
    "superglm.reml.observed_geometry": (
        "ObservedModeNotCertifiedError",
        "ObservedModeNotConvergedError",
    ),
    "superglm.sklearn": ("SuperGLMClassifier", "SuperGLMRegressor"),
    "superglm.solvers.irls_direct": ("StructuredSolverError",),
    "superglm.stats.davies": ("psum_chisq", "satterthwaite"),
    "superglm.stats.model_tests": (
        "DispersionTestResult",
        "ScoreTestZIResult",
        "VuongTestResult",
        "ZeroInflationResult",
        "dispersion_test",
        "score_test_zi",
        "vuong_test",
        "zero_inflation_index",
    ),
    "superglm.stats.wood_pvalue": ("wood_test_smooth",),
    "superglm.structure": ("FeatureStructure", "Structure", "StructureError", "read_structure"),
    "superglm.terms": (
        "BoundInteraction",
        "BoundTerm",
        "RedundantInteractionWarning",
        "cat",
        "interaction",
        "re",
        "s",
        "term",
        "ti",
    ),
    "superglm.types": ("LambdaPolicy", "LinearConstraintSet"),
    "superglm.validation": (
        "DoubleLiftChartResult",
        "LiftChartResult",
        "LorenzCurveResult",
        "LossRatioChartResult",
        "double_lift_chart",
        "lift_chart",
        "lorenz_curve",
        "loss_ratio_chart",
    ),
}

# A process with no directory Numba can write its compile cache to gets a private temporary one;
# this runs before anything imports Numba (``superglm._numba_cache``).
_numba_cache.ensure_cache_location()

__getattr__, __dir__ = _attach(__name__, __path__, _LAZY_EXPORTS)


def _warmup_scalar_observations() -> None:
    """Warm bounded NB/beta and compound-gamma core operands (orders 0–2).

    Warm the compiled gamma-window compositions and Tweedie remainder
    admission as well as their incomplete-gamma operands. NB's normalized
    recurrence and beta fraction are separate from its complete-density route.
    Optional third/fourth derivatives and endpoint fallbacks remain first-use.
    """
    import numpy as np

    from superglm.distributional.families.negative_binomial import NegativeBinomialLS
    from superglm.distributional.families.tweedie import TweedieLSS
    from superglm.distributional.family import bounded_observation
    from superglm.distributional.observation import ResponseBounds
    from superglm.distributional.weights import WeightContract, resolve_likelihood_weights

    response = np.array([1.0, 2.0, 3.0])
    bounds = ResponseBounds.from_arrays(np.zeros(3), np.full(3, 5.0), np.array([-1, 0, 1]))
    for family, parameters in (
        (NegativeBinomialLS(), [2.0, 0.7]),
        (TweedieLSS(), [2.0, 1.2, 1.5]),
    ):
        theta = np.tile(parameters, (3, 1))
        theta.setflags(write=False)
        for semantics in ("prior", "frequency"):
            weights = resolve_likelihood_weights(
                np.full(3, 2.0), n_observations=3, contract=WeightContract(semantics=semantics)
            )
            plan = family.bind_likelihood(response, weights, bounded_observation(bounds))
            for order in (0, 1, 2):
                family.evaluate_natural(response, theta, plan, derivative_order=order)
        # Public window calls add the non-normalized NB dispatcher; infinite
        # and narrow compound windows exercise the gamma tail/quadrature route.
        family.log_window_probability(
            np.array([-np.inf, 0.0, 1.0]),
            np.array([0.0, np.inf, 1.0 + 1e-6]),
            theta,
        )


def _warmup_dirichlet() -> None:
    """Warm the complete-composition row and contracted curvature routes."""
    import numpy as np

    from superglm.distributional.families.dirichlet import DirichletLSS
    from superglm.distributional.family import COMPLETE_OBSERVATION
    from superglm.distributional.kernels._dirichlet_numba import _fsum
    from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
    from superglm.links import LogLink

    family = DirichletLSS(("a", "b", "c"))
    response = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    weights = resolve_likelihood_weights(
        np.ones(2), n_observations=2, contract=WeightContract(semantics="frequency")
    )
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    alpha = np.array([[0.2, 2.0, 4.0], [20.0, 1.0, 0.7]])
    for readonly in (False, True):
        alpha.setflags(write=not readonly)
        for order in (0, 1, 2):
            family.evaluate_natural(response, alpha, plan, derivative_order=order)
    eta, direction = np.log(alpha), np.full_like(alpha, 0.2)
    links = [LogLink()] * 3
    family.predictor_curvature_directional_derivative(response, eta, direction, links, plan)
    family.predictor_curvature_second_directional_derivative(
        response, eta, direction, direction, links, plan
    )
    # Cached batches need no recompilation, but their Python callee dispatcher
    # can remain empty. Warm the standalone reduction for the import census.
    _fsum(np.array([1.0, 2.0**-53, 2.0**-100]))


def warmup() -> None:
    """Compile superglm's model-fitting kernels ahead of the first fit.

    Use it where compiling ahead pays: a cluster's start-up script, an image
    build or a benchmark, with ``NUMBA_CACHE_DIR`` on persistent storage so the
    compiled kernels outlive the process. Otherwise do not call it: a first fit
    compiles only the kernels it needs and saves them to Numba's cache, and
    later sessions load them from there.

    It compiles the kernels of every family and route, ``RandomEffect`` and
    ``FactorSmooth`` terms included, and the two basis kernels a prediction of
    8,192 rows or more reads. Cost depends on the platform and cache contents.
    A smaller prediction compiles nothing, so a service of quotes does not need
    it. Kernels few fits reach compile on first use
    instead and then join the cache: the third and fourth derivatives a
    ``SuperLSS`` fit needs only at a smoothing-parameter endpoint, a fit's
    diagnostics, and a few rarely run fallbacks. Numba keys its cache to the
    CPU model, so a cache built on one node type compiles again on another.
    See the how-to *Starting fast on ephemeral clusters*.
    """
    from superglm._group_matrix._group_matrix_kernels import _warmup_group_matrix_kernels
    from superglm._group_matrix._group_matrix_support import _warmup_support_scans
    from superglm._group_matrix._lookup_tables import _warmup_lookup_tables
    from superglm._tweedie_series import warmup as _warmup_tweedie_series
    from superglm.distributional.kernels._declared import _warmup_rows as _warmup_declared_rows
    from superglm.distributional.kernels._declared_bounds import _warmup_declared_bounds
    from superglm.distributional.kernels.gamma import _warmup_gamma
    from superglm.distributional.kernels.generalized_gamma import _warmup_generalized_gamma
    from superglm.distributional.kernels.negative_binomial import _warmup_negative_binomial
    from superglm.distributional.kernels.tweedie import _warmup_tweedie
    from superglm.distributional.solver._joint_tables import _warmup_joint_tables
    from superglm.distributional.solver._ordinary_packing import _warmup_ordinary_packing
    from superglm.distributional.solver.derivatives import _warmup_chain_rule
    from superglm.features._basis_kernels import _warmup_basis_kernels
    from superglm.features.interaction import _warmup_row_kron
    from superglm.reml._compensated import _warmup_compensated
    from superglm.reml._leverage_rows import _warmup_leverage_rows
    from superglm.solvers._exact_sums import _warmup_exact_sums
    from superglm.solvers._structured.block_leaves import _warmup_factor_smooth
    from superglm.solvers._structured.leaf_kernels import _warmup_leaf_kernels

    _warmup_tweedie()
    _warmup_tweedie_series()
    _warmup_group_matrix_kernels()
    _warmup_compensated()
    _warmup_exact_sums()
    _warmup_ordinary_packing()
    _warmup_chain_rule()
    _warmup_joint_tables()
    _warmup_support_scans()
    _warmup_row_kron()
    _warmup_basis_kernels()
    _warmup_lookup_tables()
    _warmup_leverage_rows()
    _warmup_gamma()
    _warmup_declared_bounds()
    _warmup_declared_rows()
    _warmup_negative_binomial()
    _warmup_scalar_observations()
    _warmup_dirichlet()
    _warmup_generalized_gamma()
    _warmup_leaf_kernels()
    _warmup_factor_smooth()
    from superglm._numba_compile import forget_compiles

    forget_compiles()


# The reviewed public surface, pinned by tests/test_public_api_snapshot.py.
# Objects the library builds on the user's behalf (auto-detected interaction
# types, constraint machinery, term-inference parts) are importable, not exported.
__all__ = [
    "families",
    "warmup",
    "SuperGLM",
    "SuperLSS",
    "BoundPredictor",
    "bind_predictor",
    "BoundInteraction",
    "BoundTerm",
    "term",
    "s",
    "cat",
    "re",
    "ti",
    "interaction",
    "GaussianLS",
    "DirichletLSS",
    "GammaLS",
    "GeneralizedGammaLSS",
    "GeneralizedParetoLSS",
    "LogNormalLS",
    "LomaxLSS",
    "LogLogisticLSS",
    "NegativeBinomialLS",
    "TweedieLSS",
    "TwoPieceLogNormalLSS",
    "TwoPieceNormalLSS",
    "Predictor",
    "PathResult",
    "NearPoissonDispersionError",
    "PublicationModeError",
    "DiscretizationResult",
    "discretization_impact",
    "FitDiagnosticReport",
    "export_rating_tables",
    "RatingTableBaseNotRepresentableError",
    "Structure",
    "FeatureStructure",
    "StructureError",
    "read_structure",
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
    "PolynomialRange",
    "Categorical",
    "OrderedCategorical",
    "Constraint",
    "ConstraintRules",
    "FactorSmooth",
    "FactorSmoothResult",
    "RandomEffect",
    "LevelGrouping",
    "collapse_levels",
    "Numeric",
    "Piecewise",
    "Polynomial",
    "GroupElasticNet",
    "GroupLasso",
    "SparseGroupLasso",
    "Ridge",
    "Adaptive",
    "NBProfileResult",
    "NBThetaBoundWarning",
    "FractionalFrequencyWeightWarning",
    "PriorWeightLatticeWarning",
    "REMLResult",
    "LambdaPolicy",
    "TweedieProfileResult",
    "tweedie_logpdf",
    "generate_tweedie_cpg",
    "psum_chisq",
    "satterthwaite",
    "wood_test_smooth",
    "n_knots_from_k",
    "TermInference",
    "InteractionInference",
    "ConvergenceWarning",
    "FactorSmoothGroupTermWarning",
    "RedundantInteractionWarning",
    "SCOPRetiredWarning",
    "SplineOptionWarning",
    "SeparationError",
    "SeparationWarning",
    "SlowFitWarning",
    "WeakIdentificationWarning",
    "StructuredSolverError",
    "ObservedModeNotCertifiedError",
    "ObservedModeNotConvergedError",
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
__version__ = "0.42.0"
