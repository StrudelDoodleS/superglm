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

Index order is only a convention, though, and the rest of this file pins the
price it is allowed to charge.  Where the earliest independent columns happen
to be two near-duplicates, keeping them is a real numerical cost paid on every
downstream solve -- and one no rank test can detect, because the block is
positive definite, Cholesky accepts it, and its smallest eigenvalue is above
the very cutoff that decided the rank.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import scipy.linalg

from superglm.solvers import rank as rank_module
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    _achievable_amplification,
    _conditioned_representatives,
    _earliest_representatives,
    _eigensolver_relative_bar,
    _leverage_pivot_representatives,
    _principal_block_condition,
    _selection_amplification,
    decompose_factor,
    decompose_gram,
    needs_factor_certification,
)


def _roundoff_gamma(operations: int) -> float:
    """Standard ``gamma_k`` bound for ``operations`` binary64 roundings."""
    accumulated = float(operations) * np.finfo(float).eps
    if not 0.0 <= accumulated < 1.0:
        raise ValueError("roundoff bound requires operations * eps < 1")
    return accumulated / (1.0 - accumulated)


def _symmetric_boundary_allowance(block: np.ndarray, cutoff: float) -> float:
    """Width-scaled backward-error band for a smallest-eigenvalue comparison."""
    scale = max(
        float(np.linalg.norm(block, ord=np.inf)),
        abs(cutoff),
        np.finfo(float).tiny,
    )
    return _roundoff_gamma(64 * max(1, len(block))) * scale


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
    candidate_subsets: list[tuple[int, int] | None] = []
    original_eigh = np.linalg.eigh
    original_eigvalsh = np.linalg.eigvalsh
    original_scipy_eigvalsh = scipy.linalg.eigvalsh

    def counting_eigh(matrix):
        nonlocal calls
        calls += 1
        return original_eigh(matrix)

    def counting_eigvalsh(matrix, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_eigvalsh(matrix, *args, **kwargs)

    def counting_scipy_eigvalsh(matrix, *args, **kwargs):
        nonlocal calls
        calls += 1
        subset = kwargs.get("subset_by_index")
        candidate_subsets.append(None if subset is None else tuple(subset))
        return original_scipy_eigvalsh(matrix, *args, **kwargs)

    monkeypatch.setattr(rank_module.np.linalg, "eigh", counting_eigh)
    monkeypatch.setattr(rank_module.np.linalg, "eigvalsh", counting_eigvalsh)
    monkeypatch.setattr(rank_module.scipy.linalg, "eigvalsh", counting_scipy_eigvalsh)
    decomposition = decompose_gram(gram)

    assert decomposition.rank < width, "fixture must exercise the deficient path"
    # One spectrum decides the full block and at most two smallest-eigenvalue
    # solves certify the fixed candidate set.  The prefix walk needed one whole
    # spectrum per column.
    assert calls <= 3, f"{calls} symmetric eigenvalue solves for a width-{width} block"
    assert candidate_subsets
    assert all(subset == (0, 0) for subset in candidate_subsets), (
        f"candidate certificates requested non-minimal spectra: {candidate_subsets}"
    )

    # On this block the prefix walk did not merely cost more -- it spent 60
    # spectra and then failed its own count check, so the decomposition fell
    # back to `gram_eigh` and no representative basis was recovered at all.
    # Reading the selection off the null basis succeeds where the walk's
    # matrix-wide cutoff mis-rejected a column on a small prefix.
    assert decomposition.method == "pivoted_cholesky"


def _equilibrated_geometry(gram: np.ndarray):
    """The decomposition's own equilibrated matrix, null basis, rank and cutoff."""
    scale = np.sqrt(np.diag(gram))
    equilibrated = gram / np.outer(scale, scale)
    equilibrated = 0.5 * (equilibrated + equilibrated.T)
    values, vectors = np.linalg.eigh(equilibrated)
    values = np.maximum(values, 0.0)
    # Mirrors the module: `gram_rcond` floored at the eigensolver's bar.  This
    # helper drives the shipped-walk comparison, so a cutoff of its own would
    # be comparing the walk against a rule nothing applies -- issue #356.
    cutoff = max(SHARED_RANK_POLICY.gram_rcond, _eigensolver_relative_bar(len(values))) * max(
        float(values[-1]), 0.0
    )
    retained = values > cutoff
    return equilibrated, vectors[:, ~retained], int(retained.sum()), cutoff


def _shipped_prefix_walk(equilibrated: np.ndarray, rank: int, cutoff: float):
    """The selection as master shipped it, against the WHOLE matrix's cutoff.

    ``_prefix_walk`` above takes its cutoff from the caller so the convention
    can be compared on its own terms.  This one reproduces what the released
    code actually did -- walk the equilibrated matrix testing every prefix
    against the matrix-wide cutoff -- because that, not the convention, is what
    the policy version has to distinguish.
    """
    selected: list[int] = []
    for candidate in range(equilibrated.shape[0]):
        trial = selected + [candidate]
        principal = equilibrated[np.ix_(trial, trial)]
        if int(np.count_nonzero(np.linalg.eigvalsh(principal) > cutoff)) > len(selected):
            selected.append(candidate)
        if len(selected) == rank:
            break
    return np.asarray(selected, dtype=int) if len(selected) == rank else None


def test_the_policy_version_records_that_the_deficient_answer_changed() -> None:
    """Identical thresholds, different retained representation -- hence version 2.

    The rank cutoff decides HOW MANY directions are retained.  It does not
    decide which columns represent them, nor whether a representative basis is
    recovered at all, and both of those moved on this branch while every
    threshold stayed exactly where it was.

    **VERSION 3 IS THE OTHER KIND OF CHANGE AND THIS FIXTURE MEASURES IT
    EXACTLY.**  The paragraph below recorded ``cutoff / (m eps ||A||) =
    0.3333`` as a defect being tolerated; issue #356 closed it, and the ratio
    is now 1.0 by construction because the cutoff IS that bar.  Nothing this
    test asserts moved -- rank 2, selection ``[0, 2]``, method
    ``pivoted_cholesky`` -- because the fixture's computed zero is at
    ``-6.661e-16``, which the old cut cleared by 1.50x and the bar clears by
    2.00x, on the same side both times.  The 2x2 prefix's margin falls from
    1.37e+07 to **4.56e+06**, which is still six orders clear of any
    eigensolver difference.  So this test pins the version and NOT the
    threshold, which is the distinction the paragraph below is about.

    The half checked HERE is the one a single design can pin: at ``eps=1e-4``
    the shipped walk keeps the near-duplicate pair and the branch does not.
    The SELECTION is far from any boundary -- the 2x2 prefix's smaller
    eigenvalue is 6.0823e-09 against a cutoff of 4.445e-16, a margin of
    1.37e+07 -- so no eigensolver difference can move it.  ``eps=5e-8`` gives a
    margin of only 3.0 and is deliberately not used.

    The RANK is a different matter and is asserted separately below, on
    purpose.  This design is exactly rank 2, so its third eigenvalue is a
    computed zero sitting at roughly the cutoff itself: measured
    ``cutoff / (m * eps * ||A||) = 0.3333``, i.e. the cutoff WAS BELOW what a
    symmetric eigensolver resolves; version 3 raised it to exactly that bar
    and the ratio is now 1.0.  Every test of this code path shares that
    property -- a rank-deficient matrix has no other kind of zero -- and it is
    empirically stable for a fixed matrix, holding on this machine at 1 and 16
    threads and on CI's 3.12 runner across five values of eps.  What is NOT
    stable is a CHAIN of such comparisons, which is why the walk's own failure
    mode is tested as a rate instead.  The rank is asserted first so that if a
    driver ever does move it, the failure says so instead of surfacing as a
    confusing selection mismatch.
    """
    assert SHARED_RANK_POLICY.version == 3

    design = _near_alias(1e-4)
    gram = design.T @ design
    decomposition = decompose_gram(gram)
    assert decomposition.rank == 2, (
        f"fixture is no longer rank 2 (got {decomposition.rank}); the computed zero "
        "has crossed the cutoff and this design can no longer pin a selection"
    )

    equilibrated, _, rank, cutoff = _equilibrated_geometry(gram)
    walked = _shipped_prefix_walk(equilibrated, rank, cutoff)
    assert walked is not None and walked.tolist() == [0, 1]

    assert decomposition.active_columns.tolist() == [0, 2]
    assert decomposition.policy_version == 3


def test_the_walk_fails_where_the_read_off_succeeds_at_a_stable_rate() -> None:
    """The other half of the version's justification, as a RATE not an outcome.

    Whether the shipped walk fills its set on any ONE block is not a stable
    fact.  The walk tests each prefix against the whole matrix's cutoff,
    ``eps * ||A||``, and by Cauchy interlacing a prefix's spectrum is bounded
    by the full matrix's -- so the walk only fails when a genuine direction is
    represented in the prefix at roughly the cutoff itself.  A symmetric
    eigensolver resolves eigenvalues to about ``m * eps * ||A||``, so that
    comparison sits below the resolution of the thing computing it, and the
    per-block answer moves with the LAPACK kernel.  It did: this assertion was
    first written against one block, passed here and failed on CI's 3.12 runner
    with the same numpy 2.4.2 and scipy 1.17.1.

    The RATE was stable, because it is a property of the algorithm rather than
    of one rounding decision.  Measured over 958 deficient blocks across 16
    seeds, the walk returned nothing on 126 of them (13.2%), with per-seed
    failures from 5 to 11 out of about 60.  Reading the null basis produces a
    candidate in those cases; the absolute conditioning backstop now keeps it
    only when its principal block is usable.  On this eight-seed regression
    battery 58 of 60 candidates stayed spectral and two were safe to recover.

    **ALL OF THAT IS THE VERSION-2 READING AND IT NO LONGER REPRODUCES,
    BECAUSE VERSION 3 REMOVED THE CAUSE THE FIRST PARAGRAPH NAMES.**  The rate
    is now zero and the assertions below say so; the measurement, the numbers,
    and why the threshold is an equality rather than a floor are all recorded
    at the assertion itself.  The paragraph above is kept because it is the
    diagnosis that turned out to be right, not because it is the current
    behaviour.

    The exactness assertion is the one that carries no margin at all: across
    all 958 blocks there was NOT ONE where both filled a set and the sets
    differed.  That is the equivalence `b2de09d` claims, and it is what makes
    one version bump sufficient rather than a licence.  Under version 3 it is
    also the whole test, and it covers every block instead of the 87.5% where
    both used to resolve.
    """
    agreed = 0
    walk_failed = 0
    candidate_recovered = 0
    recovered = 0
    kept_spectral = 0
    for seed in (4, 11, 2029, 77_003, 5, 6, 7, 8):
        rng = np.random.default_rng(seed)
        for _ in range(60):
            width = int(rng.integers(6, 40))
            _, gram = _aliased_gram(rng, width, int(rng.integers(2, width)))
            equilibrated, null, rank, cutoff = _equilibrated_geometry(gram)
            if not 0 < rank < width:
                continue
            walked = _shipped_prefix_walk(equilibrated, rank, cutoff)
            read_off = _earliest_representatives(null, rank)
            if walked is None:
                walk_failed += 1
                if read_off is not None:
                    candidate_recovered += 1
                    if decompose_gram(gram).method == "pivoted_cholesky":
                        recovered += 1
                    else:
                        kept_spectral += 1
                continue
            assert read_off is not None
            # no margin: the conventions are identical wherever both resolve
            assert np.array_equal(read_off, walked)
            agreed += 1

    assert agreed > 200, f"only {agreed} blocks resolved by both"
    # **THE RATE IS NOW ZERO, AND THAT IS VERSION 3 REMOVING THE CAUSE THIS
    # DOCSTRING ALREADY NAMED -- ISSUE #356.**  The paragraph above says the
    # walk fails because its comparison "sits below the resolution of the
    # thing computing it".  Version 3 raised the cutoff to exactly that
    # resolution, and the failures stopped: measured on this eight-seed
    # battery, 60 of 479 deficient blocks (12.5%) under the version-2 cutoff
    # against **0 of 480** under the bar, with `agreed` going from 419 to
    # **480** -- every block, not 87.5% of them.  Swept over seven
    # `OPENBLAS_CORETYPE` microkernels at one thread the new figures are
    # identical on all seven: 480 agreed, 0 failed, 0 disagreed.  The old
    # per-block identity moved with the kernel and the new one does not.
    #
    # So the assertion is inverted rather than deleted.  `walk_failed >= 10`
    # was pinning a defect rate; what is worth pinning now is that the two
    # conventions agree EVERYWHERE, which is the stronger form of the
    # equivalence `b2de09d` claimed and the reason one version bump was
    # sufficient.  If the walk starts failing again, the cutoff has fallen
    # back beneath the eigensolver.
    assert walk_failed == 0, (
        f"the walk failed on {walk_failed} blocks; under version 3 its prefix "
        "comparison is at the eigensolver's resolution and should not fail at "
        "all, so this means the Gram cutoff has dropped below the bar again"
    )
    assert candidate_recovered == 0
    assert recovered + kept_spectral == candidate_recovered


def _near_alias(eps: float, rows: int = 200, seed: int = 7) -> np.ndarray:
    """Three normalised columns with ``c1 = c0 - eps*c2``, so ``c2`` is in span(c0, c1).

    ``c2`` genuinely IS a combination of the two columns before it, so index
    order is entitled to drop it -- but only with a multiplier of ``1/eps``,
    which is exactly the conditioning that multiplier costs.
    """
    rng = np.random.default_rng(seed)
    first = rng.normal(size=rows)
    third = rng.normal(size=rows)
    design = np.column_stack([first, first - eps * third, third])
    return design / np.linalg.norm(design, axis=0)


def test_the_rank_cutoff_cannot_see_a_near_alias_but_the_certificate_can() -> None:
    """The measurement that rules out certifying against the decomposition cutoff.

    At ``eps=5e-8`` the earliest selection ``[0, 1]`` genuinely clears the
    decomposition cutoff in factor space.  Its factor condition is nonetheless
    about ``3.6e7``.  That is below ``warning_condition``, so neither policy
    constant separates it from an ordinary block.  Characterise that fact with
    its singular values: the equivalent smallest eigenvalue of a formed Gram
    lies at cancellation scale, and its reported sign relative to the cutoff
    is not portable evidence.

    What does separate it is that this is a property of the SELECTION.  The
    amplification the choice adds is 2.5634e+07 where a rank-revealing choice
    provably reaches 1.7321, and the alternative selection is sitting right
    there at condition 1.0299.
    """
    design = _near_alias(5e-8)
    gram = design.T @ design
    scale = np.sqrt(np.diag(gram))
    equilibrated = gram / np.outer(scale, scale)
    values, vectors = np.linalg.eigh(equilibrated)
    cutoff = SHARED_RANK_POLICY.gram_rcond * max(float(values[-1]), 0.0)
    retained = values > cutoff
    rank = int(retained.sum())
    null = vectors[:, ~retained]

    earliest = _earliest_representatives(null, rank)
    assert earliest is not None
    assert earliest.tolist() == [0, 1]

    candidate_singular = scipy.linalg.svdvals(design[:, earliest])
    assert candidate_singular[-1] ** 2 > cutoff
    assert candidate_singular[0] / candidate_singular[-1] < SHARED_RANK_POLICY.warning_condition

    amplification = _selection_amplification(null, earliest)
    achievable = _achievable_amplification(*null.shape)
    assert amplification > 1e7 > achievable

    chosen = _conditioned_representatives(null, rank)
    assert chosen is not None
    assert chosen.tolist() == [0, 2]
    assert _selection_amplification(null, chosen) <= achievable


@pytest.mark.parametrize("eps", [2e-8, 5e-8, 1e-6, 1e-4, 1e-2])
def test_a_near_alias_is_never_handed_on_as_two_near_duplicate_columns(eps: float) -> None:
    """Both decomposition paths, across the scale where index order goes wrong.

    Index order picks ``[0, 1]`` at every one of these, at conditions
    9.0668e+07, 3.6267e+07, 1.8134e+06, 1.8133e+04 and 1.8128e+02 -- the
    condition tracks ``1/eps``, because that is the multiplier index order
    accepted when it declared ``c2`` redundant.  Certifying the selection picks
    ``[0, 2]`` at 1.0299 throughout.
    """
    design = _near_alias(eps)

    from_gram = decompose_gram(design.T @ design)
    from_factor = decompose_factor(design)

    for decomposition in (from_gram, from_factor):
        assert decomposition.rank == 2
        assert decomposition.active_columns.tolist() == [0, 2]
        selected = design[:, decomposition.active_columns]
        assert np.linalg.cond(selected) < 2.0


def test_reselection_leaves_an_exact_alias_convention_where_it_was() -> None:
    """One near alias must not relabel the exact aliases sharing its block.

    Columns 0 and 1 are a 5e-8 near alias; columns 2 and 4 are an exact
    duplicate pair.  Index order alone selects ``[0, 1, 2]`` at condition
    3.6532e+07.  Re-selecting by complete pivoting would answer ``[1, 3, 4]``,
    which fixes the conditioning but surrenders the EARLIER member of the exact
    pair for no reason.  The leverage pivot answers ``[0, 2, 3]``: the near
    alias gives up its second column, the exact pair still gives up its later
    one, and the condition is 1.1309 either way.
    """
    rng = np.random.default_rng(7)
    first = rng.normal(size=200)
    third = rng.normal(size=200)
    shared = rng.normal(size=200)
    design = np.column_stack([first, first - 5e-8 * third, shared, third, shared])
    design = design / np.linalg.norm(design, axis=0)

    from_gram = decompose_gram(design.T @ design)
    from_factor = decompose_factor(design)

    for decomposition in (from_gram, from_factor):
        assert decomposition.active_columns.tolist() == [0, 2, 3]
        assert np.linalg.cond(design[:, decomposition.active_columns]) < 2.0


@pytest.mark.parametrize(("width", "nullity"), [(6, 2), (8, 2), (8, 3), (12, 3), (12, 4), (16, 5)])
def test_the_selection_does_not_move_when_the_null_basis_is_rotated(
    width: int,
    nullity: int,
) -> None:
    """A null SPACE has no canonical basis, so no rule may read one.

    ``eigh`` and ``svd`` return an arbitrary orthonormal basis of the null
    space; ``N @ Q`` spans the same subspace for any orthogonal ``Q``.  A rule
    that reads individual components of ``N`` is therefore reading a coordinate
    the eigensolver was free to choose, and can answer differently for the same
    design on a different LAPACK build -- which is the class of defect that
    already cost this branch a CI failure once.

    This is a SWEEP rather than a fixture on purpose.  The rule this replaces
    was invariant on the first 6x2 subspace tried and moved on 58 of 400; at
    12x4 it moved on 224 of 400.  One fixture would have passed against the
    broken rule at every one of these shapes.

    ALL THREE selectors are covered, not just the fallback, and the composite
    is swept in BOTH of its call forms.  Since ``c78edd7`` the solver calls
    ``_conditioned_representatives`` with a ``block_condition`` scorer -- both
    production sites pass ``_principal_block_condition`` over the equilibrated
    Gram -- so the production form is exercised here with a scorer over a Gram
    built behind the subspace, fixed while the basis rotates, exactly as
    production's Gram is.  The bare form stays swept too: it is the
    documented-weaker fallback for callers that cannot supply a scorer.
    Guarding only the fallback would pin the minority path.
    """
    rng = np.random.default_rng(20260802 + width * 100 + nullity)
    rank = width - nullity
    checked = 0
    for _ in range(60):
        basis, _ = np.linalg.qr(rng.normal(size=(width, nullity)))
        # A fixed anisotropic Gram BEHIND the subspace: its null space is
        # exactly ``span(basis)``, so the production scorer judges a real
        # principal block, and it does not move when the basis does -- in
        # production LAPACK re-picks the basis while the Gram stays what the
        # design says.
        full, _ = np.linalg.qr(np.hstack([basis, rng.normal(size=(width, rank))]))
        complement = full[:, nullity:]
        gram = complement @ np.diag(10.0 ** rng.uniform(-3.0, 3.0, size=rank)) @ complement.T
        answers: dict[str, set] = {
            "leverage": set(),
            "earliest": set(),
            "conditioned_bare": set(),
            "conditioned_production": set(),
        }
        amplifications = set()
        for _rotation in range(5):
            rotation, _ = np.linalg.qr(rng.normal(size=(nullity, nullity)))
            rotated = basis @ rotation
            # same subspace, different basis
            assert np.allclose(rotated.T @ rotated, np.eye(nullity), atol=1e-11)
            for name, selected in (
                ("leverage", _leverage_pivot_representatives(rotated, rank)),
                ("earliest", _earliest_representatives(rotated, rank)),
                ("conditioned_bare", _conditioned_representatives(rotated, rank)),
                (
                    "conditioned_production",
                    _conditioned_representatives(
                        rotated,
                        rank,
                        block_condition=lambda keep: _principal_block_condition(gram, keep),
                    ),
                ),
            ):
                answers[name].add(None if selected is None else tuple(selected.tolist()))
                if name == "leverage" and selected is not None:
                    amplifications.add(round(_selection_amplification(rotated, selected), 9))
        for name, seen in answers.items():
            assert len(seen) == 1, f"{name} moved under rotation: {sorted(seen)}"
        assert len(amplifications) <= 1, f"amplification moved: {sorted(amplifications)}"
        checked += 1
    assert checked == 60


def _straddling_subspace(scale: float, width: int, nullity: int) -> np.ndarray:
    """A null basis whose LAST column carries leverage ``scale**2``, split evenly.

    Split evenly is the point: for leverage ``s**2`` the largest single
    component lies in ``[s / sqrt(k), s]`` depending on the basis, so a rule
    reading components sees ``s / sqrt(k)`` here and ``s`` after a rotation that
    concentrates it on one row.
    """
    rows = np.zeros((nullity, width))
    for index in range(nullity):
        rows[index, index] = 1.0
        rows[index, width - 1] = scale / np.sqrt(nullity)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    return rows.T


# 1.0 is deliberately absent: there the leverage lands on the comparison it is
# tested against -- measured ratio 1.000000000 -- so which side it falls is
# decided by rounding in the normalisation, not by the rule.  The nearest
# cases kept, 0.9 and 1.2, sit at 0.81x and 1.44x the threshold.
@pytest.mark.parametrize("factor", [0.5, 0.9, 1.2, 1.35, 1.6, 2.0, 10.0])
def test_a_component_straddling_the_floor_does_not_decide_by_basis(factor: float) -> None:
    """The adversarial case for the floor, which a random sweep will not produce.

    ``_earliest_representatives`` used to ask ``max |N[i, j]| > sqrt(eps)``.  For
    a column of leverage ``s**2`` that maximum ranges over ``[s / sqrt(k), s]``
    across bases, so every ``s`` in ``(sqrt(eps), sqrt(k) * sqrt(eps))`` was
    dust for one basis and a dependency for another.  Constructed rather than
    sampled: the component form moved on 455 of 4000 subspaces built inside that
    band and carried ``_conditioned_representatives`` with it on 21, while a
    uniform sweep of 400 random subspaces per shape never produced one.

    Leverage is a property of the subspace, so the answer is now decided at
    ``s == sqrt(eps)`` regardless of basis -- and the decision is still the
    earliest-column convention, just made on a quantity a rotation cannot move.
    """
    floor = float(np.sqrt(np.finfo(float).eps))
    width, nullity = 5, 2
    null = _straddling_subspace(factor * floor, width, nullity)
    assert np.allclose(null.T @ null, np.eye(nullity), atol=1e-12)

    # A fixed Gram behind the subspace, so the composite is exercised in the
    # production call form (with a principal-block scorer) as well as the bare
    # fallback form -- the adversarial basis must decide neither.
    rng = np.random.default_rng(20260802)
    full, _ = np.linalg.qr(np.hstack([null, rng.normal(size=(width, width - nullity))]))
    complement = full[:, nullity:]
    gram = complement @ np.diag(10.0 ** rng.uniform(-2.0, 2.0, size=width - nullity)) @ complement.T

    seen: dict[str, set] = {
        "earliest": set(),
        "conditioned_bare": set(),
        "conditioned_production": set(),
    }
    for angle in np.linspace(0.0, np.pi / 2, 9):
        cos, sin = np.cos(angle), np.sin(angle)
        rotated = null @ np.array([[cos, -sin], [sin, cos]])
        for name, selected in (
            ("earliest", _earliest_representatives(rotated, width - nullity)),
            ("conditioned_bare", _conditioned_representatives(rotated, width - nullity)),
            (
                "conditioned_production",
                _conditioned_representatives(
                    rotated,
                    width - nullity,
                    block_condition=lambda keep: _principal_block_condition(gram, keep),
                ),
            ),
        ):
            seen[name].add(None if selected is None else tuple(selected.tolist()))
    for name, answers in seen.items():
        assert len(answers) == 1, f"{name} moved under rotation at {factor}x: {sorted(answers)}"

    # and the threshold sits where the leverage says, not where a basis says
    # every kept factor is clear of the threshold, so the expected side is not
    # itself a knife-edge call
    assert abs(factor - 1.0) > 0.05
    expected_reached = factor > 1.0
    assert (seen["earliest"] == {(0, 2, 3)}) is expected_reached, seen["earliest"]


def test_an_anisotropic_block_is_judged_on_its_own_condition_not_on_a_bound() -> None:
    """A smaller amplification does not mean a better-conditioned block.

    `_selection_amplification` bounds one end of a ratio:
    `sigma_min(X[:, keep]) >= sigma_rank(X) * sigma_min(N_R)` puts a floor under
    `sigma_min`, while `sigma_max` of the selected block also moves with the
    selection.  So the bound improves a WORST CASE and says nothing about a
    specific one, and on anisotropic factors the two orderings disagree.

    Measured before the fix, over 9,654 rank-deficient 3x4 blocks whose columns
    span three orders of magnitude: the routine switched on 9,615 of them and
    24 of those switches went to a block whose actual principal condition was
    worse, by up to 1.34x.  Deciding on the principal block itself takes that to
    0 of 9,615.

    The bound is still the trigger, and being a bound is what makes it a safe
    one -- it cannot miss a genuinely ill-conditioned block.
    """
    rng = np.random.default_rng(20260802)
    switched = 0
    for _ in range(3000):
        base = rng.normal(size=(3, 3)) @ np.diag(10.0 ** rng.uniform(-3, 3, size=3))
        combination = base @ rng.normal(size=(3, 1))
        factor = np.hstack([base, combination])[:, rng.permutation(4)]
        if not np.all(np.isfinite(factor)) or np.min(np.linalg.norm(factor, axis=0)) == 0:
            continue
        equilibrated = factor / np.linalg.norm(factor, axis=0)
        singular, right = np.linalg.svd(equilibrated, full_matrices=True)[1:]
        rank = int(np.count_nonzero(singular > SHARED_RANK_POLICY.factor_rcond * singular[0]))
        if rank != 3 or right.T[:, rank:].shape[1] != 1:
            continue
        earliest = _earliest_representatives(right.T[:, rank:], rank)
        decomposition = decompose_factor(factor)
        if earliest is None or decomposition.pivots is None:
            continue
        chosen = np.asarray(decomposition.active_columns)
        if np.array_equal(chosen, earliest):
            continue
        switched += 1
        gram = equilibrated.T @ equilibrated
        assert _principal_block_condition(gram, chosen) <= _principal_block_condition(
            gram, earliest
        ), f"switched {earliest.tolist()} -> {chosen.tolist()} and made conditioning worse"
    assert switched > 500, f"only {switched} switches exercised"


def test_a_catastrophic_representative_keeps_the_spectral_decomposition() -> None:
    """The better representative may still be too ill-conditioned to use.

    This deterministic factor has five resolvable directions in ten columns,
    so the rank decision is not the finding.  The best representative
    recovered from that subspace nevertheless has a principal condition above
    ``severe_condition``.  Cholesky accepts it, but using its inverse degrades
    ``||G G+ G - G|| / ||G||`` by more than an order against the spectral
    route.

    Both Gram and factor entry points must therefore keep their spectral solve
    when every representative crosses the existing policy boundary.  Before
    the guard the Gram route returned ``pivoted_cholesky`` and the factor route
    retained a representative Cholesky.

    **THE FIFTH DIRECTION IS SCALED BECAUSE VERSION 3 MADE THE ORIGINAL ONE
    UNRESOLVABLE, AND THAT IS THE POINT OF THE SCALING RATHER THAN A TUNING
    CONVENIENCE -- ISSUE #356.**  As written, this fixture's fifth
    equilibrated eigenvalue was "over nine times the rank cutoff", which was
    true against ``eps`` and is **0.933x of ``eigh``'s own bar** at width 10
    -- inside the resolution, so the Gram could not certify five directions
    and only appeared to because the cutoff was beneath what computed it.
    Measured over seven ``OPENBLAS_CORETYPE`` microkernels at one thread it
    ran 0.925x to 0.933x of that bar: stable, and stably UNRESOLVED.  Under
    version 3 the Gram reports rank 4 with ``resolution_limited`` set and
    defers, and ``decompose_factor`` -- which sees the same direction
    **7.0e+06 bars clear** of the SVD's resolution -- still returns 5.  That
    is the policy working, not the fixture breaking.

    Scaling row 4 by 1.5 lifts the direction to **2.112x of the version-3
    cutoff** so the rank is once again not the finding, and it is the largest
    lever available: the retained direction's margin and the representative's
    condition are the SAME quantity, ``sigma_5``, so buying margin sells
    conditioning.  Measured across those seven kernels the pair is
    BIT-IDENTICAL -- 2.112x and 1.426x of ``severe_condition`` on every one --
    but the condition no longer clears ``2 * severe_condition``, and the
    assertion below is relaxed to ``severe_condition`` with the 1.426x it
    actually has.  A finer scan found no scaling that keeps both above 2x:
    1.05 gives 3.199x of the condition on a rank margin of 1.029x, which is
    inside the kernel spread and would be a coin flip.
    """
    factor = np.array(
        [
            [
                -1.4204301211290749e-01,
                -2.5873822242371208e-01,
                1.5740498006169482e-01,
                -3.2443078518196217e-01,
                3.4305646778417126e-01,
                3.0590973081712630e-01,
                3.3430221916331854e-01,
                -4.4277422710063674e-01,
                3.9605603593480526e-01,
                -3.2697213310199175e-01,
            ],
            [
                -1.4034348780716716e-03,
                -3.9863698585931411e-03,
                8.5455453261010193e-03,
                5.4316892080185407e-03,
                5.6656444296932741e-03,
                -1.9447946708481215e-03,
                3.5368614362713398e-04,
                -3.0066715031898941e-03,
                -4.4835270016177261e-03,
                5.6156764250026156e-03,
            ],
            [
                6.9319678575304608e-05,
                -6.0991039198378666e-05,
                -1.1675888718037742e-04,
                8.4194381828303351e-05,
                3.2176165170890973e-06,
                -1.4392784464196176e-05,
                -2.7530888866140620e-05,
                -1.0632775056803482e-04,
                5.4963770666537858e-05,
                5.0725644573084028e-05,
            ],
            [
                2.3968048307093101e-07,
                -2.4095324875802135e-06,
                4.8462225601029339e-07,
                3.7813479919608278e-08,
                -4.5914829320947197e-07,
                -2.3358925374420366e-07,
                -1.1222695801511232e-06,
                1.2516255206841181e-06,
                5.8810904031770385e-07,
                -8.3188832876315809e-07,
            ],
            [
                -5.9721165754639350e-09,
                1.2439191372448542e-08,
                2.0532201416823470e-08,
                1.6836840331938923e-08,
                -2.2371985141641281e-08,
                -5.4180803721079223e-09,
                -2.8869823907668855e-10,
                -6.4023463289349279e-09,
                2.7442053176671418e-08,
                -9.9745758716737190e-10,
            ],
        ]
    )
    # See the docstring: as delivered the fifth direction is 0.933x of the
    # eigensolver's bar and so is not a rank the Gram may claim.
    factor[4, :] = factor[4, :] * 1.5

    gram = factor.T @ factor
    scale = np.sqrt(np.diag(gram))
    equilibrated = gram / np.outer(scale, scale)
    equilibrated = 0.5 * (equilibrated + equilibrated.T)
    values, vectors = np.linalg.eigh(equilibrated)
    cutoff = max(SHARED_RANK_POLICY.gram_rcond, _eigensolver_relative_bar(len(values))) * float(
        values[-1]
    )
    retained = values > cutoff
    assert int(retained.sum()) == 5
    # 2.112x on all seven kernels, so this clears by 1.41x.  The 8.0 it used
    # to carry was against a cutoff a factor of ten smaller.
    assert float(values[retained][0]) > 1.5 * cutoff

    selected = _conditioned_representatives(
        vectors[:, ~retained],
        5,
        block_condition=lambda keep: _principal_block_condition(equilibrated, keep),
    )
    assert selected is not None
    # 1.426x on all seven kernels.  See the docstring for why this cannot be
    # 2x and carry a usable rank margin at the same time.
    assert _principal_block_condition(equilibrated, selected) > SHARED_RANK_POLICY.severe_condition

    gram_decomposition = decompose_gram(gram)
    factor_decomposition = decompose_factor(factor)

    assert gram_decomposition.method == "gram_eigh"
    assert gram_decomposition.cholesky_factor is None
    assert factor_decomposition.method == "qr_svd"
    assert factor_decomposition.cholesky_factor is None
    for decomposition in (gram_decomposition, factor_decomposition):
        residual = np.linalg.norm(gram @ decomposition.pseudo_inverse() @ gram - gram)
        residual /= np.linalg.norm(gram)
        assert residual < 0.02


def _angled_column_geometry(sin_squared_multiple: float):
    """Ten copies of ``e1`` plus one column at an angle, with exact 2x2 predictions.

    The closed forms avoid asking a width-11 Gram eigensolver to choose the
    sign of cancellation-scale differences.
    """
    eps = np.finfo(float).eps
    sin_squared = sin_squared_multiple * eps
    sin = np.sqrt(sin_squared)
    cos = np.sqrt(1.0 - sin_squared)
    factor = np.column_stack(
        [
            np.tile(np.array([[1.0], [0.0]]), 10),
            np.array([cos, sin]),
        ]
    )
    full_trace = 11.0
    full_discriminant = np.hypot(10.0 + cos**2 - sin_squared, 2.0 * cos * sin)
    full_maximum = 0.5 * (full_trace + full_discriminant)
    full_minimum = 10.0 * sin_squared / full_maximum
    candidate_maximum = 1.0 + cos
    candidate_minimum = sin_squared / candidate_maximum
    return factor, full_maximum, full_minimum, candidate_maximum, candidate_minimum


def test_a_representative_must_clear_the_full_gram_rank_cutoff() -> None:
    """Local conditioning cannot replace the cutoff that certified the rank.

    The eleventh column makes an angle whose ``sin(theta)**2`` puts the second
    full-Gram eigenvalue above the rank cutoff while every two-column
    principal representative -- one copy of ``e1`` and the angled column --
    has a smaller eigenvalue BELOW that same cutoff.  Its local condition is
    comfortably inside the severe cap, so the condition-only guard would
    accept it; the cutoff rule is what must reject it.

    **THE ANGLE IS SET FROM THE GRAM'S CUTOFF, AND THAT IS NOW A DIFFERENT
    NUMBER FROM THE FACTOR'S -- ISSUE #356.**  This used to read
    ``sin(theta)**2 = 16 eps``, chosen so that ONE angle put the representative
    below the cutoff on both routes at once.  Version 3 cut that tie: the Gram
    cutoff is ``n eps lambda_max`` and the factor's is still
    ``eps lambda_max``, a factor of ``n = 11`` apart, and the window this
    construction offers is only ``(10/11) / (1/2) = 1.818x`` wide.  Rejection
    on the Gram needs ``candidate_min`` in ``(66.6 eps, 121 eps)``; rejection
    on the factor needs it below ``11 eps``.  Those do not intersect, so no
    single angle exercises both and the routes get one arm each.  That is not
    a fixture defect -- it is the ``n``-fold divergence the version created,
    made visible.

    This arm is the Gram's, at ``180 eps``: ``full_min`` is 1.352x of the
    cutoff and ``candidate_min`` is 0.744x, both BIT-IDENTICAL across seven
    ``OPENBLAS_CORETYPE`` microkernels at one thread, so the two margins are
    1.35x and 1.34x with no measured spread.  The factor legitimately ACCEPTS
    a representative here, which the sibling test below is the arm for.
    """
    (
        factor,
        _full_maximum,
        full_minimum,
        candidate_maximum,
        candidate_minimum,
    ) = _angled_column_geometry(180.0)
    gram = factor.T @ factor
    cutoff = max(SHARED_RANK_POLICY.gram_rcond, _eigensolver_relative_bar(11)) * _full_maximum

    assert full_minimum > cutoff
    assert candidate_minimum < cutoff
    assert candidate_maximum / candidate_minimum < SHARED_RANK_POLICY.severe_condition

    gram_decomposition = decompose_gram(gram)
    assert gram_decomposition.rank == 2
    assert gram_decomposition.cholesky_factor is None
    assert gram_decomposition.active_columns.tolist() == list(range(11))
    assert gram_decomposition.method == "gram_eigh"


def test_the_factor_resolves_an_angle_the_gram_cannot_and_still_refuses_the_block() -> None:
    """The factor arm of the split above, on the angle this fixture always had.

    At ``sin(theta)**2 = 16 eps`` the second direction is 5.683e-08 of the
    design's scale, which is **7.0e+06 times** the SVD's own resolution and
    1.15x above ``factor_rcond`` -- resolved, retained, and its two-column
    representative correctly refused because ``candidate_min`` is below the
    FACTOR cutoff.  That half is unchanged by version 3.

    The Gram sees the same direction at ``lambda_min / lambda_max`` of
    ``1.32 eps``, which is **0.1202x** of ``eigh``'s bar at width 11.  It
    cannot resolve it, so it no longer claims rank 2 -- under version 2 it did,
    by comparing against a cutoff a factor of 11 beneath what computed it, and
    getting the right answer for no reason available to it.  What it does now
    is report rank 1, set ``resolution_limited``, and say through
    ``needs_factor_certification`` that the caller must go to the factor --
    which returns the 2 the Gram could not certify.

    So the rank is still recovered; it is recovered by the route that can see
    it.  That is the whole of #356 on one fixture, and it is asserted here
    rather than described.
    """
    factor, _, _, _, _ = _angled_column_geometry(16.0)
    gram = factor.T @ factor

    factor_decomposition = decompose_factor(factor)
    assert factor_decomposition.rank == 2
    assert factor_decomposition.method == "qr_svd"
    assert factor_decomposition.cholesky_factor is None
    assert factor_decomposition.active_columns.tolist() == list(range(11))
    assert not needs_factor_certification(factor_decomposition)

    gram_decomposition = decompose_gram(gram)
    assert gram_decomposition.rank == 1, (
        "the Gram is certifying a direction at 0.12x of its eigensolver's bar "
        "again; the version-3 cutoff has stopped binding"
    )
    assert gram_decomposition.resolution_limited
    assert needs_factor_certification(gram_decomposition)


@pytest.mark.parametrize(
    ("entrypoint", "rho", "equality_rcond"),
    [
        ("gram", 0.7991251286200829, 0.07342454214376197),
        ("factor", 0.6632179475290408, 0.36250186569360726),
    ],
)
@pytest.mark.parametrize("relation", ["exactly_equal", "just_below", "clearly_above"])
def test_the_full_cutoff_certificate_has_a_conservative_floating_boundary(
    entrypoint: str,
    rho: float,
    equality_rcond: float,
    relation: str,
) -> None:
    """Equality and an indistinguishable one-ULP tightening remain spectral.

    A bare Cholesky of ``B - cutoff*I`` accepted both public-path equality
    fixtures: its final diagonal was about ``1e-8`` even though the shifted
    block was semidefinite.  The one-ULP stricter Gram fixture was accepted too.
    This pins the numerical postcondition on both coordinate systems:

    * Gram rank compares principal eigenvalues directly with ``cutoff``;
    * factor rank compares them with the SQUARED singular-value cutoff.

    ``clearly_above`` sits about ``1e-10`` relative above the boundary, far
    outside the backward-error allowance, and proves the certificate is not a
    blanket spectral fallback.
    """
    factor = np.array(
        [
            [1.0, rho, 1.0],
            [0.0, np.sqrt(1.0 - rho**2), 0.0],
        ]
    )
    gram = factor.T @ factor
    if relation == "exactly_equal":
        rcond = equality_rcond
    elif relation == "just_below":
        rcond = float(np.nextafter(equality_rcond, np.inf))
    else:
        rcond = equality_rcond * (1.0 - 1e-10)

    if entrypoint == "gram":
        policy = replace(SHARED_RANK_POLICY, gram_rcond=rcond)
        decomposition = decompose_gram(gram, policy=policy)
        candidate_cutoff = decomposition.cutoff
    else:
        policy = replace(SHARED_RANK_POLICY, factor_rcond=rcond)
        decomposition = decompose_factor(factor, policy=policy)
        candidate_cutoff = decomposition.cutoff**2

    equilibrated = factor / np.linalg.norm(factor, axis=0)
    candidate_block = (equilibrated.T @ equilibrated)[:2, :2]
    candidate_minimum = float(np.linalg.eigvalsh(candidate_block)[0])
    boundary_allowance = _symmetric_boundary_allowance(candidate_block, candidate_cutoff)
    assert decomposition.rank == 2
    if relation == "clearly_above":
        assert candidate_minimum > candidate_cutoff + boundary_allowance
    else:
        assert abs(candidate_minimum - candidate_cutoff) <= boundary_allowance

    if relation == "clearly_above":
        assert decomposition.active_columns.tolist() == [0, 1]
        assert decomposition.cholesky_factor is not None
    else:
        assert decomposition.active_columns.tolist() == [0, 1, 2]
        assert decomposition.cholesky_factor is None


def test_the_factor_cutoff_squared_public_repro_stays_spectral() -> None:
    """Pin the reported factor boundary as an unresolved numerical band."""
    rho = 0.7991251286200829
    factor = np.array(
        [
            [1.0, rho, 1.0],
            [0.0, np.sqrt(1.0 - rho**2), 0.0],
        ]
    )
    policy = replace(SHARED_RANK_POLICY, factor_rcond=0.27096963325022605)
    decomposition = decompose_factor(factor, policy=policy)
    equilibrated = factor / np.linalg.norm(factor, axis=0)
    candidate_block = (equilibrated.T @ equilibrated)[:2, :2]
    candidate_minimum = float(np.linalg.eigvalsh(candidate_block)[0])
    boundary_allowance = _symmetric_boundary_allowance(
        candidate_block,
        decomposition.cutoff**2,
    )

    assert abs(candidate_minimum - decomposition.cutoff**2) <= boundary_allowance
    assert decomposition.rank == 2
    assert decomposition.active_columns.tolist() == [0, 1, 2]
    assert decomposition.cholesky_factor is None


@pytest.mark.slow
def test_a_tall_valid_representative_stores_its_compact_factor_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid factor representative must not solve a tall-accumulated Gram.

    The selected normalized columns genuinely clear the factor cutoff by
    about 1.61x.  Their directly formed tall Gram lies at cancellation scale:
    different BLAS implementations report different signs and magnitudes for
    its smallest eigenvalue.  The previous factor-space admissibility fix
    selected ``[0, 1]`` correctly but then stored that non-authoritative
    Gram's Cholesky.  Where it factorised, its pseudo-inverse differed from the
    selected factor's SVD reference by more than its own norm.

    The representative's condition, solve factor, and optional factor-RHS
    solve must all come from the compact SVD coordinates.  Coefficients are
    intentionally ill-conditioned here, so reconstruction, prediction, and
    normwise backward error are the solve oracles.  The SVD spy retains the
    authoritative production coordinates; every reference built from them is
    width-bounded.
    """
    n = 500_000
    delta = np.sqrt(9.66 * np.finfo(float).eps)
    first = np.ones(n)
    contrast = np.ones(n)
    contrast[n // 2 :] = -1.0
    factor = np.column_stack([first, first + delta * contrast, first])

    svd_shapes: list[tuple[int, ...]] = []
    qr_shapes: list[tuple[int, ...]] = []
    original_svd = np.linalg.svd
    original_numpy_qr = np.linalg.qr
    original_scipy_qr = scipy.linalg.qr
    authoritative_svd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def recording_svd(matrix, *args, **kwargs):
        nonlocal authoritative_svd
        svd_shapes.append(matrix.shape)
        result = original_svd(matrix, *args, **kwargs)
        if matrix.shape == factor.shape:
            assert authoritative_svd is None
            authoritative_svd = result
        return result

    def recording_numpy_qr(matrix, *args, **kwargs):
        qr_shapes.append(matrix.shape)
        return original_numpy_qr(matrix, *args, **kwargs)

    def recording_scipy_qr(matrix, *args, **kwargs):
        qr_shapes.append(matrix.shape)
        return original_scipy_qr(matrix, *args, **kwargs)

    monkeypatch.setattr(rank_module.np.linalg, "svd", recording_svd)
    monkeypatch.setattr(rank_module.np.linalg, "qr", recording_numpy_qr)
    monkeypatch.setattr(rank_module.scipy.linalg, "qr", recording_scipy_qr)
    decomposition = decompose_factor(factor, retain_factor_solve=True)

    assert decomposition.rank == 2
    assert decomposition.active_columns.tolist() == [0, 1]
    assert decomposition.cholesky_factor is not None
    selected = factor[:, decomposition.active_columns]
    selected_scale = decomposition.column_scale[decomposition.active_columns]
    assert authoritative_svd is not None
    _, authoritative_singular, authoritative_right = authoritative_svd
    spectral_factor = authoritative_singular[:, None] * authoritative_right
    compact_selected = spectral_factor[:, decomposition.active_columns]
    _, reference_upper = original_scipy_qr(
        compact_selected,
        mode="economic",
        check_finite=False,
    )
    row_signs = np.where(np.diag(reference_upper) < 0.0, -1.0, 1.0)
    reference_upper = row_signs[:, None] * reference_upper
    compact_rows = compact_selected.shape[0]
    width = selected.shape[1]
    compact_bound = _roundoff_gamma(64 * compact_rows)
    reconstruction_bound = _roundoff_gamma(n * width + 32 * (compact_rows + width))
    backward_bound = _roundoff_gamma(3 * n + 32 * (compact_rows + width))
    assert max(compact_bound, reconstruction_bound, backward_bound) < 1.0

    lower = decomposition.cholesky_factor
    assert np.array_equal(lower, np.tril(lower))
    assert np.all(np.diag(lower) > 0.0)
    geometry_error = np.linalg.norm(lower - reference_upper.T)
    assert geometry_error / np.linalg.norm(compact_selected) <= compact_bound
    reference_singular = scipy.linalg.svdvals(compact_selected)
    assert reference_singular[-1] ** 2 > decomposition.cutoff**2

    response = selected @ np.array([0.75, -0.4])
    actual_beta = decomposition.solve_factor_rhs(response)
    actual_prediction = selected @ actual_beta[decomposition.active_columns]
    residual_norm = np.linalg.norm(actual_prediction - response)
    response_norm = np.linalg.norm(response)
    assert residual_norm / response_norm <= backward_bound
    backward_scale = np.linalg.norm(selected_scale) * np.linalg.norm(
        actual_beta[decomposition.active_columns]
    )
    backward_scale += response_norm
    assert residual_norm / backward_scale <= backward_bound
    assert decomposition.factor_rhs_left_basis is not None
    assert decomposition.factor_rhs_triangular is not None
    np.testing.assert_allclose(
        decomposition.factor_rhs_triangular,
        lower.T * selected_scale[None, :],
        rtol=_roundoff_gamma(16 * width),
        atol=0.0,
    )
    reconstruction_error = np.linalg.norm(
        decomposition.factor_rhs_left_basis @ decomposition.factor_rhs_triangular - selected
    )
    assert reconstruction_error / np.linalg.norm(selected_scale) <= reconstruction_bound

    assert svd_shapes.count((n, 3)) == 1
    assert all(shape == (n, 3) or max(shape) <= 3 for shape in svd_shapes), svd_shapes
    assert qr_shapes
    assert all(max(shape) <= 3 for shape in qr_shapes), qr_shapes


@pytest.mark.slow
def test_a_tall_factor_certificate_does_not_trust_its_formed_gram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observation-row accumulation cannot certify factor-space rank.

    This is the practical version of the reviewer's three-million-row public
    repro.  The selected factor's analytic geometry is below the factor cutoff,
    but the smallest eigenvalue of its formed tall Gram is cancellation-scale:
    BLAS implementations have reported both a large positive inflation and a
    negative value.  Neither direction is authoritative.

    Factor rank is authoritative in the SVD coordinates already computed by
    ``decompose_factor``.  Certification must stay there, never form the
    observation-row Gram, and retain the spectral result.  A production-local
    NumPy proxy and tracked ndarray reject operator, ``matmul``, ``dot`` and
    ``einsum`` equivalents while allowing width-bounded compact products.
    """
    n = 1_000_000
    delta = np.sqrt(5.0 * np.finfo(float).eps)
    first = np.ones(n)
    contrast = np.ones(n)
    contrast[n // 2 :] = -1.0
    factor = np.column_stack([first, first + delta * contrast, first])

    svd_shapes: list[tuple[int, ...]] = []
    width = factor.shape[1]
    wrapped_input = False
    rejected_routes: list[str] = []

    def reject_matrix_product(left, right, route: str) -> None:
        left_shape = getattr(left, "shape", ())
        right_shape = getattr(right, "shape", ())
        tracked = isinstance(left, RejectObservationGram) or isinstance(
            right, RejectObservationGram
        )
        if (
            tracked
            and len(left_shape) == len(right_shape) == 2
            and left_shape[1] == n
            and right_shape[0] == n
        ):
            rejected_routes.append(route)
            raise AssertionError(f"{route} formed an observation-row Gram")

    class RejectObservationGram(np.ndarray):
        def __matmul__(self, other):
            reject_matrix_product(self, other, "operator @")
            return super().__matmul__(other)

        def __rmatmul__(self, other):
            reject_matrix_product(other, self, "operator @")
            return super().__rmatmul__(other)

        def dot(self, other, out=None):
            reject_matrix_product(self, other, "ndarray.dot")
            return super().dot(other, out=out)

    class LinalgProxy:
        def __getattr__(self, name):
            return getattr(np.linalg, name)

        def svd(self, matrix, *args, **kwargs):
            svd_shapes.append(matrix.shape)
            return np.linalg.svd(matrix, *args, **kwargs)

    class NumpyProxy:
        linalg = LinalgProxy()

        def __getattr__(self, name):
            return getattr(np, name)

        def asarray(self, values, *args, **kwargs):
            nonlocal wrapped_input
            result = np.asarray(values, *args, **kwargs)
            if isinstance(values, RejectObservationGram):
                wrapped_input = wrapped_input or values is tracked_factor
                return result.view(RejectObservationGram)
            return result

        def matmul(self, left, right, *args, **kwargs):
            reject_matrix_product(left, right, "np.matmul")
            return np.matmul(left, right, *args, **kwargs)

        def dot(self, left, right, *args, **kwargs):
            reject_matrix_product(left, right, "np.dot")
            return np.dot(left, right, *args, **kwargs)

        def einsum(self, subscripts, *operands, **kwargs):
            normalized = subscripts.replace(" ", "")
            if (
                normalized == "ni,nj->ij"
                and len(operands) == 2
                and all(
                    isinstance(operand, RejectObservationGram)
                    and operand.ndim == 2
                    and operand.shape[0] == n
                    for operand in operands
                )
            ):
                rejected_routes.append("np.einsum")
                raise AssertionError("np.einsum formed an observation-row Gram")
            return np.einsum(subscripts, *operands, **kwargs)

    tracked_factor = factor.view(RejectObservationGram)
    numpy_proxy = NumpyProxy()
    monkeypatch.setattr(rank_module, "np", numpy_proxy)
    decomposition = decompose_factor(tracked_factor)

    assert wrapped_input
    assert rejected_routes == []
    assert isinstance(
        numpy_proxy.asarray(tracked_factor, dtype=float),
        RejectObservationGram,
    )
    tracked_equilibrated = (factor / np.linalg.norm(factor, axis=0)).view(RejectObservationGram)
    forbidden_products = [
        lambda: tracked_equilibrated.T @ tracked_equilibrated,
        lambda: numpy_proxy.matmul(tracked_equilibrated.T, tracked_equilibrated),
        lambda: numpy_proxy.dot(tracked_equilibrated.T, tracked_equilibrated),
        lambda: tracked_equilibrated.T.dot(tracked_equilibrated),
        lambda: numpy_proxy.einsum(
            "ni,nj->ij",
            tracked_equilibrated,
            tracked_equilibrated,
        ),
    ]
    for product in forbidden_products:
        with pytest.raises(AssertionError, match="observation-row Gram"):
            product()
    assert rejected_routes == [
        "operator @",
        "np.matmul",
        "np.dot",
        "ndarray.dot",
        "np.einsum",
    ]

    compact = np.arange(1.0, 1.0 + 2 * width).reshape(width, 2)
    tracked_compact = compact.view(RejectObservationGram)
    expected_compact_gram = compact.T @ compact
    allowed_products = [
        tracked_compact.T @ tracked_compact,
        numpy_proxy.matmul(tracked_compact.T, tracked_compact),
        numpy_proxy.dot(tracked_compact.T, tracked_compact),
        tracked_compact.T.dot(tracked_compact),
        numpy_proxy.einsum("ni,nj->ij", tracked_compact, tracked_compact),
    ]
    for product in allowed_products:
        np.testing.assert_array_equal(product, expected_compact_gram)
    assert len(rejected_routes) == len(forbidden_products)

    root = np.sqrt(1.0 + delta**2)
    # ``1 - 1/root`` is itself a cancellation-scale subtraction.  This
    # algebraically identical form retains the analytic positive numerator.
    candidate_minimum = float(delta**2 / (root * (1.0 + root)))

    assert decomposition.rank == 2
    assert candidate_minimum < decomposition.cutoff**2
    assert decomposition.active_columns.tolist() == [0, 1, 2]
    assert decomposition.cholesky_factor is None
    assert svd_shapes.count((n, 3)) == 1
    assert all(shape == (n, 3) or max(shape) <= 3 for shape in svd_shapes), svd_shapes


def test_an_admissible_alternate_beats_a_lower_condition_invalid_candidate() -> None:
    """Cutoff validity is decided before condition chooses a representative.

    The leverage candidate ``[0, 2, 3]`` has the lower local condition, but its
    smallest eigenvalue lies below the custom full-Gram cutoff.  The
    index-order candidate ``[0, 1, 2]`` has a slightly higher condition and
    clears that cutoff by a wide numerical margin.  Post-checking only the
    lower-condition answer discarded both and fell back to the spectral route;
    both entry points must instead retain the admissible candidate.
    """
    factor = np.array(
        [
            [-0.5525048221180157, 0.08966203774796645, -2.0430742443075296, -0.36329582968552987],
            [0.19404755577846758, -0.20224797449056428, -0.7447708921482363, 0.22898354867544107],
            [-0.9133692339628716, 0.1252808907553562, -0.5264798799375914, 1.62687955362161],
        ]
    )
    equilibrated = factor / np.linalg.norm(factor, axis=0)
    gram = equilibrated.T @ equilibrated
    valid = np.array([0, 1, 2])
    invalid = np.array([0, 2, 3])
    valid_minimum = float(np.linalg.eigvalsh(gram[np.ix_(valid, valid)])[0])
    invalid_minimum = float(np.linalg.eigvalsh(gram[np.ix_(invalid, invalid)])[0])
    cutoff = 0.5 * (valid_minimum + invalid_minimum)
    full_maximum = float(np.linalg.eigvalsh(gram)[-1])

    assert invalid_minimum < cutoff < valid_minimum
    assert _principal_block_condition(gram, invalid) < _principal_block_condition(gram, valid)

    gram_policy = replace(SHARED_RANK_POLICY, gram_rcond=cutoff / full_maximum)
    factor_policy = replace(
        SHARED_RANK_POLICY,
        factor_rcond=np.sqrt(cutoff / full_maximum),
    )
    from_gram = decompose_gram(factor.T @ factor, policy=gram_policy)
    from_factor = decompose_factor(factor, policy=factor_policy)
    for decomposition in (from_gram, from_factor):
        assert decomposition.rank == 3
        assert decomposition.active_columns.tolist() == valid.tolist()
        assert decomposition.cholesky_factor is not None


def test_the_certificate_is_the_condition_the_selection_itself_adds() -> None:
    """``1/sigma_min(N_R)`` is exactly the amplification, not a proxy for it.

    Both facts the fix rests on, over 200 random rank-deficient blocks: the
    identity ``1/sigma_min(N_R)**2 == 1 + ||N_K N_R^-1||**2``, and the bound
    ``sigma_min(X[:, keep]) >= sigma_rank(X) * sigma_min(N_R)`` that makes it a
    statement about the selected block rather than about the null basis.
    """
    rng = np.random.default_rng(20260801)
    checked = 0
    for _ in range(200):
        width = int(rng.integers(4, 16))
        true_rank = int(rng.integers(2, width))
        nullity = width - true_rank
        design = rng.normal(size=(60, true_rank)) @ rng.normal(size=(true_rank, width))
        design = design / np.linalg.norm(design, axis=0)
        singular, right = np.linalg.svd(design, full_matrices=True)[1:]
        if singular[true_rank - 1] < 1e-8 * singular[0]:
            continue
        null = right.T[:, true_rank:]
        if null.shape[1] != nullity:
            continue
        rejected = rng.permutation(width)[:nullity]
        keep = np.setdiff1d(np.arange(width), rejected)
        smallest = float(np.linalg.svd(null[rejected, :], compute_uv=False).min())
        if smallest < 1e-10:
            continue
        checked += 1

        amplification = _selection_amplification(null, keep)
        assert amplification == pytest.approx(1.0 / smallest)
        coupling = np.linalg.norm(null[keep, :] @ np.linalg.inv(null[rejected, :]), 2)
        assert amplification**2 == pytest.approx(1.0 + coupling**2, rel=1e-9)

        floor = singular[true_rank - 1] / amplification
        assert float(np.linalg.svd(design[:, keep], compute_uv=False).min()) >= floor * (1 - 1e-9)
    assert checked > 100, f"only {checked} blocks exercised"


@pytest.mark.parametrize("seed", [11, 2029, 77_003])
def test_certification_never_returns_a_worse_conditioned_selection(seed: int) -> None:
    """The certificate may only improve the block, and must leave good ones alone.

    Judged on the PRINCIPAL BLOCK, which is what production supplies a scorer
    for.  The earlier version of this test asserted the amplification fell
    instead, and that is the inference an anisotropic factor breaks: the bound
    can improve while the block it stands for gets worse.  So the guarantee is
    stated where it is true -- the condition of the block handed to the solver
    is never worse than index order's, and a selection the certificate leaves
    alone is one that already cleared the achievable bound.
    """
    rng = np.random.default_rng(seed)
    improved = 0
    untouched = 0
    for _ in range(120):
        width = int(rng.integers(4, 20))
        true_rank = int(rng.integers(2, width))
        coefficients = rng.normal(size=(true_rank, width))
        for column in range(true_rank, width):
            source = int(rng.integers(0, column))
            coefficients[:, column] = coefficients[:, source]
            if rng.random() < 0.5:
                coefficients[:, column] += 10.0 ** rng.uniform(-9, -5) * rng.normal(size=true_rank)
        design = rng.normal(size=(80, true_rank)) @ coefficients[:, rng.permutation(width)]
        design = design / np.linalg.norm(design, axis=0)
        gram = design.T @ design
        scale = np.sqrt(np.diag(gram))
        equilibrated = gram / np.outer(scale, scale)
        values, vectors = np.linalg.eigh(equilibrated)
        cutoff = SHARED_RANK_POLICY.gram_rcond * max(float(values[-1]), 0.0)
        retained = values > cutoff
        rank = int(retained.sum())
        if not 0 < rank < width:
            continue
        null = vectors[:, ~retained]
        earliest = _earliest_representatives(null, rank)
        chosen = _conditioned_representatives(
            null,
            rank,
            block_condition=lambda keep: _principal_block_condition(equilibrated, keep),
        )
        if earliest is None:
            assert chosen is None
            continue
        assert chosen is not None
        # the guarantee, whether or not the selection moved
        assert _principal_block_condition(equilibrated, chosen) <= _principal_block_condition(
            equilibrated, earliest
        )
        if np.array_equal(chosen, earliest):
            untouched += 1
        else:
            improved += 1
    assert improved > 0 and untouched > 0, f"improved={improved} untouched={untouched}"
