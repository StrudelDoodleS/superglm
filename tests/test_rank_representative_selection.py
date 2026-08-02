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

import numpy as np
import pytest
import scipy.linalg

from superglm.solvers import rank as rank_module
from superglm.solvers.rank import (
    SHARED_RANK_POLICY,
    _achievable_amplification,
    _conditioned_representatives,
    _earliest_representatives,
    _leverage_pivot_representatives,
    _principal_block_condition,
    _selection_amplification,
    decompose_factor,
    decompose_gram,
)


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


def _equilibrated_geometry(gram: np.ndarray):
    """The decomposition's own equilibrated matrix, null basis, rank and cutoff."""
    scale = np.sqrt(np.diag(gram))
    equilibrated = gram / np.outer(scale, scale)
    equilibrated = 0.5 * (equilibrated + equilibrated.T)
    values, vectors = np.linalg.eigh(equilibrated)
    values = np.maximum(values, 0.0)
    cutoff = SHARED_RANK_POLICY.gram_rcond * max(float(values[-1]), 0.0)
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

    The half checked HERE is the one a single design can pin: at ``eps=1e-4``
    the shipped walk keeps the near-duplicate pair and the branch does not.
    The SELECTION is far from any boundary -- the 2x2 prefix's smaller
    eigenvalue is 6.0823e-09 against a cutoff of 4.445e-16, a margin of
    1.37e+07 -- so no eigensolver difference can move it.  ``eps=5e-8`` gives a
    margin of only 3.0 and is deliberately not used.

    The RANK is a different matter and is asserted separately below, on
    purpose.  This design is exactly rank 2, so its third eigenvalue is a
    computed zero sitting at roughly the cutoff itself: measured
    ``cutoff / (m * eps * ||A||) = 0.3333``, i.e. the cutoff is BELOW what a
    symmetric eigensolver resolves.  Every test of this code path shares that
    property -- a rank-deficient matrix has no other kind of zero -- and it is
    empirically stable for a fixed matrix, holding on this machine at 1 and 16
    threads and on CI's 3.12 runner across five values of eps.  What is NOT
    stable is a CHAIN of such comparisons, which is why the walk's own failure
    mode is tested as a rate instead.  The rank is asserted first so that if a
    driver ever does move it, the failure says so instead of surfacing as a
    confusing selection mismatch.
    """
    assert SHARED_RANK_POLICY.version == 2

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
    assert decomposition.policy_version == 2


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

    The RATE is stable, because it is a property of the algorithm rather than
    of one rounding decision.  Measured over 958 deficient blocks across 16
    seeds: the walk returned nothing on 126 of them (13.2%), the branch
    recovers a `pivoted_cholesky` representative basis on 42, and per-seed
    failures ranged from 5 to 11 out of ~60.  Identical at 1 and at 16 threads.

    So the thresholds below are set far under what was measured -- roughly a
    sixth of the expected count -- to survive a platform where the rate moves,
    while still failing outright if the walk stops failing.

    The exactness assertion is the one that carries no margin at all: across
    all 958 blocks there was NOT ONE where both filled a set and the sets
    differed.  That is the equivalence `b2de09d` claims, and it is what makes
    one version bump sufficient rather than a licence.
    """
    agreed = 0
    walk_failed = 0
    recovered = 0
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
                if read_off is not None and decompose_gram(gram).method == "pivoted_cholesky":
                    recovered += 1
                continue
            assert read_off is not None
            # no margin: the conventions are identical wherever both resolve
            assert np.array_equal(read_off, walked)
            agreed += 1

    assert agreed > 200, f"only {agreed} blocks resolved by both"
    assert walk_failed >= 10, f"the walk failed on only {walk_failed} blocks"
    assert recovered >= 3, f"the branch recovered a basis on only {recovered}"


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

    At ``eps=5e-8`` the earliest selection ``[0, 1]`` has a smallest
    equilibrated eigenvalue of 1.3323e-15 against a decomposition cutoff of
    4.4447e-16.  It is ABOVE the cutoff: not rank deficient by the standard
    that chose the rank, and Cholesky factorises it without complaint.  Its
    condition is nonetheless 3.6267e+07 in design terms and 1.5012e+15 on the
    equilibrated Gram.  Neither policy constant separates that from an ordinary
    block: the Gram figure falls between `warning_condition` (6.7109e+07) and
    `severe_condition` (4.5036e+15), while the design figure is below the
    warning threshold outright.

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

    block = equilibrated[np.ix_(earliest, earliest)]
    smallest = float(np.linalg.eigvalsh(block)[0])
    assert smallest > cutoff, "the rank cutoff accepts this block, which is the whole problem"
    scipy.linalg.cholesky(block, lower=True)  # and so does Cholesky

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

    ALL THREE selectors are covered, not just the fallback.  The fallback fires
    only above ``_achievable_amplification``; ``_earliest_representatives``
    labels almost every deficient block, and ``_conditioned_representatives`` is
    what the solver actually calls.  Guarding only the fallback would pin the
    minority path.
    """
    rng = np.random.default_rng(20260802 + width * 100 + nullity)
    rank = width - nullity
    checked = 0
    for _ in range(60):
        basis, _ = np.linalg.qr(rng.normal(size=(width, nullity)))
        answers: dict[str, set] = {"leverage": set(), "earliest": set(), "conditioned": set()}
        amplifications = set()
        for _rotation in range(5):
            rotation, _ = np.linalg.qr(rng.normal(size=(nullity, nullity)))
            rotated = basis @ rotation
            # same subspace, different basis
            assert np.allclose(rotated.T @ rotated, np.eye(nullity), atol=1e-11)
            for name, selector in (
                ("leverage", _leverage_pivot_representatives),
                ("earliest", _earliest_representatives),
                ("conditioned", _conditioned_representatives),
            ):
                selected = selector(rotated, rank)
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

    seen: dict[str, set] = {"earliest": set(), "conditioned": set()}
    for angle in np.linspace(0.0, np.pi / 2, 9):
        cos, sin = np.cos(angle), np.sin(angle)
        rotated = null @ np.array([[cos, -sin], [sin, cos]])
        for name, selector in (
            ("earliest", _earliest_representatives),
            ("conditioned", _conditioned_representatives),
        ):
            selected = selector(rotated, width - nullity)
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
