"""The nested factor forms only the work a caller reads (perf scout findings F2, F4, F5).

These are cost-model assertions -- which quantities exist after which call,
which arrays are shared rather than copied -- beside the numerical checks
they need: a lazily formed quantity must equal what the eager factor
formed, and a solve through the factor must meet the backward-error
certificate ``tests/test_nested_schur_factor.py`` holds every solve to.
Each test fails under the mutation named in its docstring.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from superglm import Categorical, Numeric, RandomEffect, SuperGLM
from superglm.solvers._structured.nested import _frozen, _sealed, _weighted_scatter
from tests.test_nested_schur_factor import _augmented_case, _certified_solve


def test_a_solve_only_factor_never_forms_the_explicit_border_inverse():
    """F4: a PIRLS iterate only solves, so the border inverse is formed on first read.

    ``solve_data`` goes through the Cholesky factor itself (backward stable,
    Higham 2002, chapter 10) and meets the same certificate as the full
    solve; the explicit ``Q^+`` appears only when a caller asks for it.
    Fails with the border inverse and ``_Q_inverse`` formed at construction.
    """
    refs, factor, penalized, *_ = _augmented_case("F1")
    gamma = 4 * (refs["gamma_tree"] + refs["gamma_border"])
    rhs = np.random.default_rng(11).normal(size=(refs["p"], 2))
    solution = factor.solve_data(rhs)
    source = factor._border.source
    assert "inverse_scaled" not in source.__dict__
    assert "_Q_inverse" not in factor.__dict__ and "_Q_inverse_data" not in factor.__dict__
    allowance = _certified_solve(penalized, solution, rhs, refs["H_f"], refs["Hinv_f"], gamma)
    assert np.all(np.abs(solution - refs["Hinv_f"] @ rhs) <= allowance)
    # the explicit inverse, formed on demand, serves the full solve
    full = factor.solve(rhs)
    assert "inverse_scaled" in source.__dict__ and "_Q_inverse" in factor.__dict__
    allowance = _certified_solve(penalized, full, rhs, refs["H_f"], refs["Hinv_f"], gamma)
    assert np.all(np.abs(full - refs["Hinv_f"] @ rhs) <= allowance)


def test_the_root_scatter_reads_the_cells_instead_of_scanning_the_means():
    """F2: the leaf means' one-hot block is nonzero only on the layout's cells.

    The pattern is a superset of the nonzeros (a cell whose mean is an exact
    zero stays in it), and a product over it sums the same nonzero terms plus
    exact zeros, so the scatter and its mass are bitwise those of the scan.
    Fails with the pattern taken from anything but the cells (the scan's
    result must be contained in it) or with its extra entries mishandled.
    """
    rng = np.random.default_rng(4)
    n, K, H = 3000, 60, 25
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "g": [f"g{c:02d}" for c in rng.integers(0, K, n)],
            "h": [f"h{c:02d}" for c in rng.integers(0, H, n)],
            "c": [f"c{c}" for c in rng.integers(0, 4, n)],
        }
    )
    y = rng.poisson(np.exp(0.2 * frame["x"].to_numpy())).astype(float)
    weights = np.ones(n)
    weights[:40] = 0.0  # rows with no weight leave exact-zero means on their cells
    model = SuperGLM(
        family="poisson",
        features={"x": Numeric(), "c": Categorical(), "g": RandomEffect(), "h": RandomEffect()},
        selection_penalty=0,
        direct_solve="structured",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(frame, y, sample_weight=weights, max_reml_iter=2)
    leaf = model._linear_system_state.system.operator.leaf
    assert leaf.indicator is not None and leaf.indicator.any()
    rows, positions = leaf.indicator_pattern
    block = leaf.mean[:, leaf.indicator]
    scanned = set(zip(*np.nonzero(block != 0.0), strict=True))
    assert scanned <= set(zip(rows.tolist(), positions.tolist(), strict=True))
    weight = rng.normal(size=len(leaf.weight))
    reference = _weighted_scatter(leaf.mean, weight, leaf.indicator)
    patterned = _weighted_scatter(leaf.mean, weight, leaf.indicator, pattern=(rows, positions))
    for left, right in zip(reference, patterned, strict=True):
        assert np.array_equal(left, right)


def test_statistics_this_module_just_built_are_not_copied_again():
    """F5: a sealed fresh array, or an earlier ``_frozen`` result, is kept as it is.

    ``augmented()`` shares the source statistics' frozen arrays and seals its
    own new ones, so the ``(K, q)`` means are copied once (by the column
    stack that adds the intercept) instead of twice.  Anything writable, or a
    view of another array, is still copied.  Fails with ``_frozen`` copying
    unconditionally.
    """
    sealed = _sealed(np.arange(6.0))
    assert _frozen(sealed, np.float64) is sealed
    writable = np.arange(6.0)
    assert _frozen(writable, np.float64) is not writable
    view = sealed[1:]
    assert _frozen(view, np.float64) is not view
    _, _, penalized, *_ = _augmented_case("F1")
    data = penalized.data
    augmented = data.augmented()
    assert augmented.leaf.weight is data.leaf.weight
    assert not augmented.leaf.mean.flags.writeable and augmented.leaf.mean.base is None
