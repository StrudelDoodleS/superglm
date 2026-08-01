"""Guards for how a rank-deficient block picks its representative columns.

The choice itself is a labelling convention -- which of a set of aliased
columns gets the reproducible zero coefficient -- but it used to be made by
walking every prefix of the block and taking that prefix's whole spectrum,
one eigendecomposition per candidate column.  That is quartic in the block
width and it runs only when the block is rank deficient, so a design that
was a few columns short of full rank cost hundreds of times one of the same
width that was not.

What is pinned here is both halves: that the cheap route picks the SAME
columns as the prefix walk it replaced, and that it no longer pays an
eigendecomposition per column to do it.
"""

from __future__ import annotations

import numpy as np
import pytest

from superglm.solvers import rank as rank_module
from superglm.solvers.rank import _earliest_representatives, decompose_gram


def _prefix_walk(gram: np.ndarray, rank: int, cutoff: float) -> np.ndarray | None:
    """The selection as it was made before: one spectrum per candidate column."""
    selected: list[int] = []
    for candidate in range(gram.shape[0]):
        trial = selected + [candidate]
        principal = gram[np.ix_(trial, trial)]
        if int(np.count_nonzero(np.linalg.eigvalsh(principal) > cutoff)) > len(selected):
            selected.append(candidate)
        if len(selected) == rank:
            break
    return np.asarray(selected, dtype=int) if len(selected) == rank else None


def _aliased_gram(rng: np.random.Generator, width: int, true_rank: int):
    """A Gram whose columns include exact copies and exact sums of earlier ones."""
    basis = rng.normal(size=(width, true_rank))
    coefficients = np.zeros((true_rank, width))
    coefficients[:, :true_rank] = np.eye(true_rank)
    for column in range(true_rank, width):
        source = int(rng.integers(0, column))
        if rng.random() < 0.5:
            coefficients[:, column] = coefficients[:, source]
        else:
            other = int(rng.integers(0, column))
            coefficients[:, column] = coefficients[:, source] + coefficients[:, other]
    design = basis @ coefficients[:, rng.permutation(width)]
    return design, design.T @ design


@pytest.mark.parametrize("seed", [11, 2029, 77_003])
def test_selection_matches_the_prefix_walk_it_replaced(seed: int) -> None:
    """The cheap route is the same convention, not a new one.

    Which column carries the zero is user visible, so this has to be an
    identity across the whole family, not a claim that both answers are
    "equally valid bases".
    """
    rng = np.random.default_rng(seed)
    compared = 0
    for _ in range(120):
        width = int(rng.integers(4, 26))
        design, gram = _aliased_gram(rng, width, int(rng.integers(2, width)))
        values, vectors = np.linalg.eigh(gram)
        cutoff = max(float(values[-1]), 0.0) * width * np.finfo(float).eps * 16
        keep = values > cutoff
        certified = int(keep.sum())
        if not 0 < certified < width:
            continue
        expected = _prefix_walk(gram, certified, cutoff)
        produced = _earliest_representatives(vectors[:, ~keep], certified)
        if expected is None:
            continue
        assert produced is not None
        assert np.array_equal(produced, expected)
        # the property the convention exists to guarantee
        assert np.linalg.matrix_rank(design[:, produced]) == certified
        compared += 1
    assert compared > 40, f"only {compared} deficient blocks exercised"


def test_a_component_of_numerical_dust_is_not_treated_as_a_dependency() -> None:
    """The null basis arrives from an eigendecomposition, so zeros are not zero.

    A component that is mathematically absent comes back around 1e-12 on a
    singular system.  Thresholding against the matrix maximum rather than each
    vector's own norm pivots on that dust and returns a selection that is not
    full rank -- which is what an absolute cutoff did here.
    """
    width = 8
    null = np.zeros((width, 1))
    null[[0, 2, 5], 0] = [0.4, -0.7, 0.6]
    null[7, 0] = 1e-12  # dust, not a dependency
    null /= np.linalg.norm(null)

    produced = _earliest_representatives(null, width - 1)
    assert produced is not None
    # column 5 carries the last genuine component, so it is the one dropped
    assert 5 not in produced.tolist()
    assert 7 in produced.tolist()


def test_a_deficient_block_does_not_pay_an_eigendecomposition_per_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cost guard, pinned by call count rather than by wall time.

    Before, selection cost one `eigvalsh` per candidate column and ran only on
    the deficient path, so a 36-level pair with a dozen unidentifiable columns
    spent minutes where the same width fully identified spent seconds.  Counting
    calls says that directly and does not turn into a flaky timing assertion on
    a loaded machine.
    """
    rng = np.random.default_rng(4)
    width = 60
    _, gram = _aliased_gram(rng, width, 45)

    calls = 0
    original = np.linalg.eigvalsh

    def counting_eigvalsh(matrix):
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(rank_module.np.linalg, "eigvalsh", counting_eigvalsh)
    decomposition = decompose_gram(gram)

    assert decomposition.rank < width, "fixture must exercise the deficient path"
    # one spectrum for the block itself; the prefix walk needed one per column
    assert calls <= 2, f"{calls} eigvalsh calls for a width-{width} block"

    # On this block the prefix walk did not merely cost more -- it spent 60
    # spectra and then failed its own count check, so the decomposition fell
    # back to `gram_eigh` and no representative basis was recovered at all.
    # Reading the selection off the null basis succeeds where the walk's
    # matrix-wide cutoff mis-rejected a column on a small prefix.
    assert decomposition.method == "pivoted_cholesky"
