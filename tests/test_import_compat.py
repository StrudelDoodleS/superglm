"""Import surface compatibility tests.

Verifies that the currently supported public import surfaces remain
importable after the src/ cleanup. These tests cover canonical package
entry points and submodule import paths that the codebase still treats as
supported.
"""

import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

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


def test_public_model_kernel_warmup_is_lazy_complete_and_idempotent() -> None:
    script = r"""
import inspect

import numpy as np
import superglm
import superglm._group_matrix._group_matrix_kernels as group_kernels
import superglm._tweedie_series as tweedie_series
import superglm.distributional.kernels._tweedie_numba as tweedie_numba
import superglm.reml._compensated as compensated
from numba.core.registry import CPUDispatcher
from superglm.reml import multi_penalty

dispatchers = {}
for module in (tweedie_numba, group_kernels):
    dispatchers.update(
        {
            f"{module.__name__}.{name}": value
            for name, value in vars(module).items()
            if isinstance(value, CPUDispatcher)
            and value.py_func.__module__ == module.__name__
        }
    )
# The series' per-term and per-row helpers are only called from compiled code,
# so their dispatchers never gain a signature; the entry point carries them.
dispatchers["superglm._tweedie_series._series_moments_kernel"] = (
    tweedie_series._series_moments_kernel
)
for name in ("_dot2_quadratic_form", "_dot2_selected", "_dot2_value"):
    dispatchers[f"{compensated.__name__}.{name}"] = getattr(compensated, name)
assert dispatchers
assert inspect.signature(superglm.warmup).parameters == {}
assert all(not dispatcher.nopython_signatures for dispatcher in dispatchers.values())

def signatures():
    return {
        name: tuple(dispatcher.nopython_signatures)
        for name, dispatcher in dispatchers.items()
    }

def readonly(*arrays):
    for array in arrays:
        array.setflags(write=False)
    return arrays

superglm.warmup()
compiled = signatures()
assert all(compiled.values()), [name for name, signatures in compiled.items() if not signatures]

# The SuperLSS penalty value passes one operand form, whatever its coefficients' flags.
from superglm.distributional.solver.solver import _half_penalty_quadratic, _penalty_entries

# np.nonzero's index layout differs with the nonzero count: none, one and several.
for diagonal, expected in (([0.0, 0.0, 0.0], 0.0), ([0.0, 2.0, 0.0], 4.0), ([0.0, 2.0, 3.0], 17.5)):
    penalty = np.diag(diagonal)
    entries = _penalty_entries(penalty)
    for coefficients in (np.array([1.0, 2.0, 3.0]), readonly(np.array([1.0, 2.0, 3.0]))[0]):
        assert _half_penalty_quadratic(penalty, coefficients, entries) == expected
assert signatures() == compiled, "the penalty value compiled a new layout after public warmup"

for values in (
    np.ones((3, 4)),
    np.asfortranarray(np.ones((3, 4))),
    np.ones((6, 4))[::2, ::2],
    np.ones((3, 4))[::-1],
    np.ones(9)[::2, None],
):
    frozen = values.view()
    frozen.setflags(write=False)
    for operand in (values, frozen):
        assert group_kernels._tensor_operand_in_reassociation_range(operand)
        assert group_kernels._operand_exponent_bounds(operand) == (0, 1)
assert signatures() == compiled, "range checks compiled new layouts after public warmup"

frozen_response, frozen_mean = readonly(np.array([0.0, 1.0, 2.5]), np.ones(3))
assert np.all(np.isfinite(superglm.tweedie_logpdf(frozen_response, frozen_mean, 1.0, 1.5)))

values = np.array([1.0, 2.0], dtype=np.float64)
codes = np.array([0, 1], dtype=np.intp)
csr_indices = np.array([0, 1], dtype=np.int32)
csr_indptr = np.array([0, 1, 2], dtype=np.int32)
cell_ptr, cell_order = group_kernels._cell_csr(codes, codes, 2, 2)
frozen_codes, = readonly(codes.copy())
for first in (codes, frozen_codes):
    for second in (codes, frozen_codes):
        assert group_kernels._cell_csr_matches(cell_ptr, cell_order, first, second, 2, 2)
group_kernels._csr_weighted_gram(
    values, csr_indices, csr_indptr, values, 2, absolute_weights=True
)
dense_small, = readonly(np.eye(2, dtype=np.float64))
group_kernels._dense_small_weighted_moments(dense_small, values, values)
group_kernels._factor_smooth_csr_dense_cross(
    values, csr_indices, csr_indptr, codes, values, dense_small, 2, 2
)
group_kernels._factor_smooth_support_dense_cross(
    np.eye(2, dtype=np.float64), codes, codes, values, dense_small, 2
)
group_kernels._factor_smooth_support_dense_cell_aggregates(
    codes, codes, values, dense_small, 2, 2
)

row_patterns, unique_codes, offsets, pair_left, pair_right, pair_offsets, right_sizes = readonly(
    np.array([0, 1], dtype=np.int32),
    np.array([[0, 0], [1, 1]], dtype=np.int32),
    np.array([0, 2, 4], dtype=np.intp),
    codes[:1],
    codes[1:],
    np.array([0, 4], dtype=np.intp),
    np.array([2], dtype=np.intp),
)
group_kernels._pattern_support_summaries(
    row_patterns,
    unique_codes,
    values,
    values,
    offsets,
    pair_left,
    pair_right,
    pair_offsets,
    right_sizes,
)
assert signatures() == compiled

# Dot2 refinement passes one operand form, whatever its callers' layouts and flags.
rng = np.random.default_rng(0)
left, right = rng.normal(size=(3, 300)), rng.normal(size=(300, 2))
for first in (left, readonly(left.copy())[0], np.asfortranarray(left)):
    for second in (right, readonly(right.copy())[0]):
        multi_penalty._refine_product(first, second, *multi_penalty._matmul_enclosed(first, second))
        multi_penalty._compensated_dot(first[0], second[:, 0])
        multi_penalty._compensated_dot(first[0], np.ascontiguousarray(second[:, 1]))
assert signatures() == compiled

superglm.warmup()
assert signatures() == compiled
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_public_warmup_leaves_the_structured_engine_kernels_to_their_first_use() -> None:
    """``superglm.warmup()`` costs what master's does: the structured engine's
    kernels (leaf pass, fs block leaves, sz balance tree, mode score) compile or
    load from numba's cache on a fit's first call, so a process pays only for
    the kernels its model class uses.  Warming all of them added 0.26-0.39 s to
    every process, more than a small sz fit's whole first-use cost.  Asserted on
    compiled signatures, not time.
    """
    script = r"""
import importlib

import superglm
from numba.core.registry import CPUDispatcher

modules = (
    "superglm.solvers._structured.leaf_kernels",
    "superglm.solvers._structured.block_leaves",
    "superglm.solvers._structured.balance_tree",
    "superglm.solvers.mode_score",
)
dispatchers = {}
for name in modules:
    module = importlib.import_module(name)
    dispatchers.update(
        {
            f"{name}.{attribute}": value
            for attribute, value in vars(module).items()
            if isinstance(value, CPUDispatcher) and value.py_func.__module__ == name
        }
    )
assert len(dispatchers) > 20, sorted(dispatchers)
superglm.warmup()
compiled = sorted(name for name, value in dispatchers.items() if value.nopython_signatures)
assert not compiled, compiled
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_an_exception_raised_by_a_root_export_is_catchable_from_the_root():
    """A caller must be able to catch what the top-level API can raise.

    ``export_rating_tables`` is exported from the package root, so a user
    following the documented surface writes ``from superglm import
    export_rating_tables`` and never imports ``superglm.export`` at all.  If the
    exception it raises lives only on the submodule, that user cannot name the
    failure mode in an ``except`` clause without reaching past the API they were
    given.  Asserted as a rule over the raising functions rather than as one
    hard-coded pair, so the next exception added to this module is covered by
    the same check instead of needing a new test.
    """
    import superglm
    from superglm import export as export_module

    assert "export_rating_tables" in superglm.__all__
    raisable = [
        name
        for name in export_module.__all__
        if isinstance(getattr(export_module, name), type)
        and issubclass(getattr(export_module, name), Exception)
    ]
    assert raisable, "superglm.export exports no exception, so this rule pins nothing"

    for name in raisable:
        assert hasattr(superglm, name), (
            f"{name} can be raised through the root-exported export_rating_tables "
            f"but is not importable from superglm"
        )
        assert name in superglm.__all__, f"{name} is reachable but undocumented in __all__"
        assert getattr(superglm, name) is getattr(export_module, name), (
            f"superglm.{name} is a different object from superglm.export.{name}, so "
            "an except clause written against one would not catch the other"
        )


def test_the_structured_solver_error_is_catchable_from_the_root():
    """A structured fit that cannot proceed raises ``StructuredSolverError``
    (no other solver is tried); a caller catches it as ``superglm.StructuredSolverError``
    or, as before it existed, as ``numpy.linalg.LinAlgError``."""
    import numpy as np

    import superglm
    from superglm.solvers import irls_direct

    assert "StructuredSolverError" in superglm.__all__
    assert superglm.StructuredSolverError is irls_direct.StructuredSolverError
    assert issubclass(superglm.StructuredSolverError, np.linalg.LinAlgError)


def test_public_model_signatures_do_not_expose_private_frame_adapter():
    from superglm import SuperGLM

    for name, method in inspect.getmembers(SuperGLM, inspect.isfunction):
        if not name.startswith("_"):
            assert "EagerFrame" not in str(inspect.signature(method)), name


def test_public_plotting_signatures_do_not_expose_private_frame_adapter():
    import superglm.plotting as plotting

    for name in plotting.__all__:
        assert "EagerFrame" not in str(inspect.signature(getattr(plotting, name))), name


def test_parallel_conftest_hook_loads_every_warning_class_module() -> None:
    """Under ``-n``, the conftest leaves xdist no warning module to import.

    pytest-xdist rebuilds a worker's warning by importing the warning class's
    module in a receiver thread. ``pytest_configure`` in tests/conftest.py
    imports superglm on the main thread first when workers will run. That
    protects the receiver threads only while the import loads every module
    defining a warning class. Serial runs must not pay for the import.
    """
    root = Path(__file__).parents[1]
    sources = {
        path: path.read_text(encoding="utf-8") for path in (root / "src" / "superglm").rglob("*.py")
    }
    modules = sorted(
        ".".join(path.relative_to(root / "src").with_suffix("").parts).removesuffix(".__init__")
        for path, text in sources.items()
        # Parse only files that can define one: parsing all of them takes 1.5 s.
        if "Warning" in text
        and any(
            isinstance(node, ast.ClassDef)
            and any(
                getattr(base, "id", getattr(base, "attr", "")).endswith("Warning")
                for base in node.bases
            )
            for node in ast.walk(ast.parse(text))
        )
    )
    assert modules
    script = f"""
import sys, types
import tests.conftest as conftest

def configure(**option):
    conftest.pytest_configure(types.SimpleNamespace(option=types.SimpleNamespace(**option)))

configure()  # CI: pytest-xdist is not installed, so there is no numprocesses option
configure(numprocesses=None)  # xdist installed, no -n
assert "superglm" not in sys.modules, "a serial run imported superglm"
configure(numprocesses=2)
print([module for module in {modules!r} if module not in sys.modules])
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")])),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


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

    # ``superglm.reml.runner.run_reml_once`` was here until the dead covariance
    # chain was deleted (PR "Delete the covariance chain no production fit
    # reaches"). Its whole module is gone: the only inbound chain ended at
    # ``SuperGLM._run_reml_once``, which nothing called, and a full-suite
    # runtime trace recorded zero production frames on it. This entry is
    # dropped deliberately -- it is a supported-surface removal, not an
    # oversight, and it belongs in 0.30.0's changelog.
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
    # ``compute_coef_covariance`` was listed here until the dead covariance
    # chain was deleted (PR "Delete the covariance chain no production fit
    # reaches").  It recomputed W from scratch and called
    # ``_penalised_xtwx_inv_gram``; ``model/state_ops.coef_covariance``
    # superseded it, and nothing in ``src/`` called it.  Dropping it from this
    # contract is a deliberate supported-surface removal for 0.30.0.
    from superglm.inference.term import (  # noqa: F401
        _VALID_CENTERING,
        InteractionInference,
        SmoothCurve,
        SplineMetadata,
        TermInference,
        _recenter_term,
        _resolve_group_lambda,
        _safe_exp,
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
    # ``_penalised_xtwx_inv`` and ``_penalised_xtwx_inv_gram`` were listed here
    # until the dead covariance chain was deleted (PR "Delete the covariance
    # chain no production fit reaches").  Neither had a production caller: the
    # gram form was reached only from ``compute_coef_covariance`` and
    # ``reml/runner.py``, both themselves dead, and the dense form was a
    # test-only oracle.  Their supported-surface removal lands in 0.30.0.
    # ``_active_penalty_matrix`` is the live half of that assembly and is added
    # here in their place, since ``inference/metrics.py`` and
    # ``model/state_ops.py`` both import it across package boundaries.
    from superglm.inference.covariance import (  # noqa: F401
        _active_penalty_matrix,
        _second_diff_penalty,
    )


def test_profiling_tweedie_canonical():
    from superglm.profiling.tweedie import TweedieProfileResult  # noqa: F401


def test_profiling_nb_canonical():
    from superglm.profiling.nb import NBProfileResult, NBThetaBoundWarning  # noqa: F401


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
