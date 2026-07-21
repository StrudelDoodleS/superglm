"""Import surface compatibility tests.

Verifies that the currently supported public import surfaces remain
importable after the src/ cleanup. These tests cover canonical package
entry points and submodule import paths that the codebase still treats as
supported.
"""

import inspect
import subprocess
import sys

# ── Old paths (must keep working after moves) ──────────────────


def test_reml_imports():
    from superglm.reml import REMLResult  # noqa: F401


def test_inference_imports():
    from superglm.inference import (  # noqa: F401
        InteractionInference,
        SmoothCurve,
        SplineMetadata,
        TermInference,
    )


def test_diagnostics_imports():
    from superglm.diagnostics import (  # noqa: F401
        SplineRedundancyReport,
        spline_redundancy,
        term_drop_diagnostics,
        term_importance,
    )


def test_validation_imports():
    from superglm.validation import (  # noqa: F401
        DoubleLiftChartResult,
        LiftChartResult,
        LorenzCurveResult,
        LossRatioChartResult,
        double_lift_chart,
        lift_chart,
        lorenz_curve,
        loss_ratio_chart,
    )


# ── Top-level public API ───────────────────────────────────────


def test_toplevel_reexports():
    """Everything in __all__ is importable from the superglm namespace."""
    import superglm

    for name in superglm.__all__:
        assert hasattr(superglm, name), f"superglm.{name} not accessible"


def test_public_model_signatures_do_not_expose_private_frame_adapter():
    from superglm import SuperGLM

    for name, method in inspect.getmembers(SuperGLM, inspect.isfunction):
        if not name.startswith("_"):
            assert "EagerFrame" not in str(inspect.signature(method)), name


def test_public_plotting_signatures_do_not_expose_private_frame_adapter():
    import superglm.plotting as plotting

    for name in plotting.__all__:
        assert "EagerFrame" not in str(inspect.signature(getattr(plotting, name))), name


def test_pandas_fit_does_not_import_optional_polars_backend():
    script = r"""
import importlib.abc
import importlib.util
import sys

class RejectPolars(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "polars" or fullname.startswith("polars."):
            raise AssertionError(f"unexpected optional import: {fullname}")
        return None

sys.meta_path.insert(0, RejectPolars())

# Tabmat probes optional dataframe packages with find_spec before importing
# them. Simulate the answer from an environment where Polars is not installed;
# the rejecting finder above still fails any actual import attempt.
real_find_spec = importlib.util.find_spec
def optional_polars_is_absent(name, package=None):
    if name == "polars" or name.startswith("polars."):
        return None
    return real_find_spec(name, package)
importlib.util.find_spec = optional_polars_is_absent

import numpy as np
import pandas as pd
from superglm import Numeric, SuperGLM

X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
y = np.array([0.1, 1.1, 2.1, 3.1])
model = SuperGLM(
    family="gaussian",
    selection_penalty=0.0,
    features={"x": Numeric()},
).fit(X, y)
prediction = model.predict(X)
assert prediction.shape == (4,)
assert not any(name == "polars" or name.startswith("polars.") for name in sys.modules)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


# ── Supported canonical paths ───────────────────────────────────


def test_reml_result_canonical():
    from superglm.reml.result import PenaltyCache, REMLResult, _map_beta_between_bases  # noqa: F401


def test_reml_penalty_algebra_canonical():
    from superglm.reml.penalty_algebra import (  # noqa: F401
        build_penalty_caches,
        build_penalty_components,
        cached_logdet_s_plus,
        compute_logdet_s_derivatives,
        compute_logdet_s_plus,
        compute_total_penalty_rank,
    )


def test_reml_optimizer_canonical():
    from superglm.reml.direct import optimize_direct_reml  # noqa: F401
    from superglm.reml.discrete import optimize_discrete_reml_cached_w  # noqa: F401
    from superglm.reml.efs import optimize_efs_reml  # noqa: F401
    from superglm.reml.gradient import reml_direct_gradient, reml_direct_hessian  # noqa: F401
    from superglm.reml.objective import reml_laml_objective  # noqa: F401
    from superglm.reml.runner import run_reml_once  # noqa: F401
    from superglm.reml.w_derivatives import (  # noqa: F401
        compute_d2W_deta2,
        compute_dW_deta,
        reml_w_correction,
    )


def test_reml_multi_penalty_canonical():
    from superglm.reml.multi_penalty import (  # noqa: F401
        SimilarityTransformResult,
        logdet_s_gradient,
        logdet_s_hessian,
        similarity_transform_logdet,
    )


def test_inference_term_canonical():
    from superglm.inference.term import (  # noqa: F401
        _VALID_CENTERING,
        InteractionInference,
        SmoothCurve,
        SplineMetadata,
        TermInference,
        _recenter_term,
        _resolve_group_lambda,
        _safe_exp,
        compute_coef_covariance,
        feature_se_from_cov,
        spline_group_enrichment,
        term_inference,
    )


def test_inference_metrics_canonical():
    from superglm.inference.metrics import ModelMetrics  # noqa: F401


def test_inference_coef_tables_canonical():
    from superglm.inference.coef_tables import (  # noqa: F401
        build_basis_detail,
        build_coef_rows,
    )


def test_inference_summary_canonical():
    from superglm.inference.summary import (  # noqa: F401
        ModelSummary,
        _BasisDetailRow,
        _CoefRow,
        _compute_coef_stats,
    )


def test_inference_covariance_canonical():
    from superglm.inference.covariance import (  # noqa: F401
        _penalised_xtwx_inv,
        _penalised_xtwx_inv_gram,
        _second_diff_penalty,
    )


def test_profiling_tweedie_canonical():
    from superglm.profiling.tweedie import TweedieProfileResult, estimate_tweedie_p  # noqa: F401


def test_profiling_nb_canonical():
    from superglm.profiling.nb import NBProfileResult, estimate_nb_theta  # noqa: F401


def test_stats_model_tests_canonical():
    from superglm.stats.model_tests import (  # noqa: F401
        DispersionTestResult,
        ScoreTestZIResult,
        VuongTestResult,
        ZeroInflationResult,
        dispersion_test,
        score_test_zi,
        vuong_test,
        zero_inflation_index,
    )


def test_stats_davies_canonical():
    from superglm.stats.davies import psum_chisq, satterthwaite  # noqa: F401


def test_stats_wood_pvalue_canonical():
    from superglm.stats.wood_pvalue import wood_test_smooth  # noqa: F401


def test_diagnostics_spline_checks_canonical():
    from superglm.diagnostics.spline_checks import (  # noqa: F401
        SplineRedundancyReport,
        spline_redundancy,
    )


def test_diagnostics_term_diagnostics_canonical():
    from superglm.diagnostics.term_diagnostics import (  # noqa: F401
        term_drop_diagnostics,
        term_importance,
    )


def test_diagnostics_discretize_canonical():
    from superglm.diagnostics.discretize import (  # noqa: F401
        DiscretizationResult,
        discretization_impact,
    )
