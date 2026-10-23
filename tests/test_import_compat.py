"""Import surface compatibility tests.

Verifies that the currently supported public import surfaces remain
importable after the src/ cleanup. These tests cover canonical package
entry points and submodule import paths that the codebase still treats as
supported.
"""

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import superglm

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


# Every numba kernel in the package compiles in ``superglm.warmup()``, is
# inlined into its callers, or is listed here: a kernel left to the first fit
# that reaches it, with the reason.  tests/_numba_dispatchers.py finds every
# kernel, so one added without that decision fails the test below.
_STRUCTURED = "superglm.solvers._structured."
_GROUP = "superglm._group_matrix._group_matrix_kernels."
_TWEEDIE = "superglm.distributional.kernels._tweedie_numba."
_NB = "superglm.distributional.kernels._negative_binomial_numba."
_FIRST_USE: dict[str, str] = {
    # Measured over cold first fits on the freMTPL2 book: 71 on 2026-10-20
    # (every family and link, SuperGLM and SuperLSS, exact and binned,
    # RandomEffect and FactorSmooth terms) and 17 more on 2026-10-21 for the
    # routes those missed (continuous covariates, tensors over disjoint and
    # P-spline margins, a decomposed and a lone tensor, factor smooths on the
    # Gram backend and beside numerics only): which fits reach each kernel,
    # its compile time on top of warmup(), and its load from a filled cache.
    **dict.fromkeys(
        [
            *(
                _TWEEDIE + name
                for name in (
                    "_polygamma_012",
                    "_third_term_channels",
                    "_series_third_cumulants",
                    "_tweedie_third_row",
                    "_tweedie_hessian_direction_rows",
                    "_polygamma_0123",
                    "_fourth_term_channels",
                    "_series_fourth_cumulants",
                    "_tweedie_fourth_row",
                    "_tweedie_fourth_rows",
                    "_positive_closed",
                    "_zero_closed",
                    "_zeta_derivatives",
                    "_term_derivatives",
                    "_series_cumulants",
                )
            ),
            *(
                _NB + name
                for name in (
                    "_log_third_rows",
                    "_log_fourth_rows",
                    "_log_third_row",
                    "_log_fourth_row",
                    "_tail_moment_one",
                    "_tail_moment_two",
                    "_tail_moment_three",
                )
            ),
            "superglm.distributional.kernels.gamma._small_shape_jdd_series",
        ],
        "a SuperLSS endpoint step's third and fourth derivatives: none of 26 SuperLSS fits "
        "reached them (3 with a pure-noise smooth); 7 s to compile after warmup(), 0.06 s "
        "from cache",
    ),
    **dict.fromkeys(
        (
            _STRUCTURED + "balance_tree._leaf_path_row_forms",
            _STRUCTURED + "block_leaves._level_row_forms",
        ),
        "a structured FactorSmooth fit's hat diagonal (Cook's distance, standardized "
        "residuals): diagnostics, not the fit",
    ),
    **dict.fromkeys(
        (
            _STRUCTURED + "block_leaves._moment_segments",
            _STRUCTURED + "block_leaves._close_levels",
        ),
        "the signed moment segments of a binned FactorSmooth term and the close of a stored "
        "level stack: no census fit reaches them (fs or sz, Fisher or signed, exact or binned)",
    ),
    _STRUCTURED + "leaf_kernels._sparse_rows": (
        "a RandomEffect leaf pass whose border has a row its tables cannot code: none in the census"
    ),
    _GROUP + "_operand_exponent_bounds": (
        "the non-float64 scan: fits pass float64 operands to _float64_operand_exponent_bounds"
    ),
    **dict.fromkeys(
        (
            _GROUP + "_dense_small_weighted_moments",
            _GROUP + "_factor_smooth_support_dense_cell_aggregates",
            _GROUP + "_factor_smooth_csr_dense_cross",
            _GROUP + "_factor_smooth_support_dense_cross",
        ),
        "a FactorSmooth moment system built without its leaf layout, or crossed with a dense "
        "term of two or more columns: no census fit reaches either (fs or sz, exact or binned, "
        "beside numerics, splines or a polynomial)",
    ),
    _GROUP + "_fused_2d_bincount_2": (
        "a discrete tensor's two weight grids on the chunked moment route, which the raw-moment "
        "and pattern routes now serve: no census fit reaches it"
    ),
    "superglm.solvers._exact_sums.scaled_exact_sum": (
        "the overflow fallback: only a sum whose products or partials leave float64's range "
        "reaches it (0.6 s)"
    ),
    "superglm.distributional.solver._small_group_panels._checked_factor_scatter": (
        "only an explicit panel budget renders a FactorSmooth group; no public fit sets one (0.4 s)"
    ),
    "superglm.plotting.diagnostics._lowess_core": "a diagnostic plot's smoother, not a fit (2.2 s)",
}

_WARMUP_CENSUS = r"""
import json

import numpy as np
import superglm
from superglm._group_matrix import _group_matrix_algebra as group_algebra
from superglm._group_matrix import _group_matrix_kernels as group_kernels
from superglm._group_matrix import _group_matrix_support as support_scans
from superglm.features import interaction
from superglm.reml import multi_penalty
from tests._numba_dispatchers import is_inlined, package_dispatchers, signatures

dispatchers, problems = package_dispatchers()
at_import = signatures(dispatchers)
superglm.warmup()
warmed = signatures(dispatchers)


def readonly(*arrays):
    for array in arrays:
        array.setflags(write=False)
    return arrays


# The row Kronecker build and the support scans, through their production entry
# points: a first interaction fit must reuse the signatures warmup compiled.
import scipy.sparse as sp

basis = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]]))
interaction._row_kron(basis, basis)
row_hashes = support_scans._row_hashes(basis, 2)
_, first_rows, row_groups = np.unique(row_hashes, return_index=True, return_inverse=True)
assert support_scans._verified_representatives(basis, first_rows, row_groups, 2) is not None

# The SuperLSS penalty value passes one operand form, whatever its coefficients' flags.
from superglm.distributional.solver.solver import _half_penalty_quadratic, _penalty_entries

# np.nonzero's index layout differs with the nonzero count: none, one and several.
for diagonal, expected in (([0.0, 0.0, 0.0], 0.0), ([0.0, 2.0, 0.0], 4.0), ([0.0, 2.0, 3.0], 17.5)):
    penalty = np.diag(diagonal)
    entries = _penalty_entries(penalty)
    for coefficients in (np.array([1.0, 2.0, 3.0]), readonly(np.array([1.0, 2.0, 3.0]))[0]):
        assert _half_penalty_quadratic(penalty, coefficients, entries) == expected

# Every layout a caller holds reaches the range scans through ``_native``, as
# their call sites pass it: C, Fortran and strided, writable or read-only.
for values in (
    np.ones((3, 4)),
    np.asfortranarray(np.ones((3, 4))),
    np.ones((6, 4))[::2, ::2],
    np.ones((3, 4))[::-1],
    np.ones(9)[::2, None],
):
    for operand in (values, *readonly(values.copy() if values.flags.owndata else values.view())):
        assert group_kernels._tensor_operand_in_reassociation_range(
            group_kernels._scan_form(operand)
        )
        assert group_algebra._operand_exponent_bounds(operand) == (0, 1)

# A weight kernel's call site hands it a read-only C operand (``_native``),
# whatever the caller holds: SuperGLM's weights, SuperLSS's read-only channel
# columns (binned) and strided ones (exact).
from superglm._group_matrix._group_matrix_core import _bincount_levels
from superglm.group_matrix import _disc_disc_2d_hist

four = np.array([0, 1, 1, 0], dtype=np.intp)
for weights in (np.ones(4), np.ones(8)[::2], *readonly(np.ones(4), np.ones(8)[::2])):
    np.testing.assert_array_equal(_bincount_levels(four, weights, 2), [2.0, 2.0])
    joint = _disc_disc_2d_hist(four, four, weights, 2, 2)
    np.testing.assert_array_equal(joint, np.diag([2.0, 2.0]))

frozen_response, frozen_mean = readonly(np.array([0.0, 1.0, 2.5]), np.ones(3))
assert np.all(np.isfinite(superglm.tweedie_logpdf(frozen_response, frozen_mean, 1.0, 1.5)))

values = np.array([1.0, 2.0], dtype=np.float64)
codes = np.array([0, 1], dtype=np.intp)
csr_indices = np.array([0, 1], dtype=np.int32)
csr_indptr = np.array([0, 1, 2], dtype=np.int32)
cell_ptr, cell_order = group_kernels._cell_csr(codes, codes, 2, 2)
assert group_kernels._cell_csr_matches(cell_ptr, cell_order, codes, codes, 2, 2)
group_kernels._csr_weighted_gram(
    values, csr_indices, csr_indptr, group_kernels._native(values), 2, absolute_weights=True
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

# Dot2 refinement passes one operand form, whatever its callers' layouts and flags.
rng = np.random.default_rng(0)
left, right = rng.normal(size=(3, 300)), rng.normal(size=(300, 2))
for first in (left, readonly(left.copy())[0], np.asfortranarray(left)):
    for second in (right, readonly(right.copy())[0]):
        multi_penalty._refine_product(first, second, *multi_penalty._matmul_enclosed(first, second))
        multi_penalty._compensated_dot(first[0], second[:, 0])
        multi_penalty._compensated_dot(first[0], np.ascontiguousarray(second[:, 1]))

entry_points = signatures(dispatchers)
superglm.warmup()
rewarmed = signatures(dispatchers)

# First fits: a binned SuperGLM fit and binned Tweedie and negative binomial
# SuperLSS fits, each with a spline, a categorical and a RandomEffect of 300
# levels (the structured engine's leaf pass).
import warnings

import pandas as pd
from superglm import cat, re, s

warnings.simplefilter("ignore")
rng = np.random.default_rng(7)
n = 4000
frame = pd.DataFrame({
    "x": rng.uniform(0, 1, n),
    "g": rng.choice(list("pqrs"), n),
    "u": rng.integers(0, 300, n).astype(str),
})
mu = np.exp(np.sin(3 * frame.x) + rng.normal(0, 0.3, 300)[frame.u.astype(int)])
counts = rng.poisson(mu).astype(np.float64)
amounts = rng.gamma(1.5, mu / 1.5) * (rng.uniform(size=n) < 0.6)
terms = (s("x", kind="cr", k=8), cat("g"), re("u"))
superglm.SuperGLM(family="poisson", discrete=True, terms=list(terms)).fit_reml(frame, counts)
tweedie = superglm.TweedieLSS()
model = superglm.SuperLSS(tweedie, tweedie.mu(*terms), tweedie.phi(), tweedie.p(), discrete=True)
model.fit_reml(frame, amounts)
negative_binomial = superglm.NegativeBinomialLS()
model = superglm.SuperLSS(
    negative_binomial, negative_binomial.mean(*terms), negative_binomial.theta(), discrete=True
)
model.fit_reml(frame, counts)

# Binned factor smooths of both bases on a 50-level group: the structured
# engine's FactorSmooth leaf pass, which warmup compiles by hand.
from superglm import FactorSmooth
from superglm import interaction as term_interaction

frame["h"] = rng.integers(0, 50, n).astype(str)
for smooths in (
    [term_interaction(FactorSmooth("x", group="h", basis="fs", k=5))],
    [s("x", kind="ps", k=7, m=2), term_interaction(FactorSmooth("x", group="h", basis="sz", kind="ps", k=5, m=2))],
):
    superglm.SuperGLM(family="poisson", discrete=True, terms=smooths).fit_reml(frame, counts)

# An exact gamma (log link) fs fit: observed curvature takes the signed half of
# the leaf pass (the signed leaf segments and trials), and an exact term its
# sorted CSR basis rows, raw moments and compensated level sums.
exact = frame.iloc[:2000]
superglm.SuperGLM(
    family="gamma", terms=[term_interaction(FactorSmooth("x", group="h", basis="fs", k=5))]
).fit_reml(exact, rng.gamma(2.0, mu.to_numpy()[:2000] / 2.0))
print(json.dumps({
    "problems": problems,
    "inlined": sorted(name for name, value in dispatchers.items() if is_inlined(value)),
    "at_import": {name: found for name, found in at_import.items() if found},
    "warmed": warmed,
    "entry_points": entry_points,
    "rewarmed": rewarmed,
    "fitted": signatures(dispatchers),
}))
"""


def _changed(before: dict, after: dict) -> dict:
    return {name: after[name] for name in after if after[name] != before.get(name)}


@pytest.mark.slow
def test_public_warmup_compiles_every_kernel_but_the_first_use_allowlist(tmp_path) -> None:
    """After ``superglm.warmup()`` a first fit compiles only the kernels in ``_FIRST_USE``.

    On an empty numba cache a kernel compiles the kernels it calls, so after
    warmup each of the package's dispatchers has a signature, is inlined
    (``inline="always"``: it never compiles on its own), or is listed in
    ``_FIRST_USE``.  A listed kernel that warmup compiles, or that no longer
    exists, fails as well, so the list cannot rot.  The production entry points
    then run with the layouts and flags their callers pass and must reuse
    warmup's signatures, a second warmup compiles nothing, and first fits
    (SuperGLM, and Tweedie and negative binomial SuperLSS, binned, with a
    spline, a categorical and a structured RandomEffect; binned ``fs`` and
    ``sz`` factor smooths; and an exact gamma ``fs`` fit, which takes the
    signed rows) compile nothing outside ``_FIRST_USE``.  ``warmup()`` stays a bare call: import compiles nothing.
    """
    assert inspect.signature(superglm.warmup).parameters == {}
    completed = subprocess.run(
        [sys.executable, "-c", _WARMUP_CENSUS],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba-cache")),
        timeout=1800,  # a cold warmup takes about a minute; this bounds a hang
    )
    assert completed.returncode == 0, completed.stderr
    census = json.loads(completed.stdout)
    assert not census["problems"], census["problems"]
    assert not census["at_import"], f"importing superglm compiled {sorted(census['at_import'])}"
    warmed, inlined = census["warmed"], set(census["inlined"])
    unclassified = sorted(
        name
        for name, found in warmed.items()
        if not found and not {name} & (inlined | _FIRST_USE.keys())
    )
    assert not unclassified, (
        f"warmup() compiles none of {unclassified}: warm it, or list it in _FIRST_USE with the reason"
    )
    stale = sorted(
        name for name in _FIRST_USE if name not in warmed or warmed[name] or name in inlined
    )
    assert not stale, (
        f"_FIRST_USE lists kernels warmup() compiles or the package no longer has: {stale}"
    )
    compiled_inline = sorted(name for name in inlined if warmed[name])
    assert not compiled_inline, (
        f"warmup() compiled the inlined {compiled_inline} on its own: numba caches no inlined "
        "kernel, so it would compile again in every process"
    )
    assert not _changed(warmed, census["entry_points"]), "a first fit's operands compiled anew"
    assert not _changed(warmed, census["rewarmed"]), "a second warmup() compiled anew"
    first_fits = sorted(set(_changed(census["rewarmed"], census["fitted"])) - _FIRST_USE.keys())
    assert not first_fits, f"a first fit after warmup() compiled {first_fits}"


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


def test_the_observed_mode_errors_are_catchable_from_the_root():
    """A REML fit through observed geometry can raise ``ObservedModeNotConvergedError``
    (0.36.0 notes), a subclass of ``ObservedModeNotCertifiedError``; both are named
    from ``superglm`` and are the classes ``observed_geometry`` raises (#449)."""
    import superglm
    from superglm import ObservedModeNotCertifiedError, ObservedModeNotConvergedError
    from superglm.reml import observed_geometry

    for name in ("ObservedModeNotCertifiedError", "ObservedModeNotConvergedError"):
        assert name in superglm.__all__
    assert ObservedModeNotCertifiedError is observed_geometry.ObservedModeNotCertifiedError
    assert ObservedModeNotConvergedError is observed_geometry.ObservedModeNotConvergedError
    assert issubclass(ObservedModeNotConvergedError, ObservedModeNotCertifiedError)
    assert issubclass(ObservedModeNotCertifiedError, RuntimeError)


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
