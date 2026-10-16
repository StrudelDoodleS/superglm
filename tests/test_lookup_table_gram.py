"""The centred Gram from term-pair tables (``_group_matrix/_lookup_tables.py``).

A design of lookups and dense columns forms ``X_c' W X_c`` and ``X_c' W z``
from one weighted histogram or accumulator per term pair (Li & Wood 2020,
Algorithm 1), every table centred before its product.  These tests pin it
against the chunked exact route within the bound the module derives in ``u``,
bitwise determinism across worker counts, the dispatch (every centred system a
lookup design builds comes from the tables), and that a design with a group
that is not a lookup keeps today's rungs.
"""

from __future__ import annotations

import numpy as np
import pytest

import superglm._group_matrix._lookup_tables as lookup_tables
from superglm._parallel import parallel_config
from superglm.group_matrix import DenseGroupMatrix
from superglm.solvers.centered_system import build_centered_system
from tests.test_reml_leverage_hessian import _gamma, _poisson_model
from tests.test_reml_leverage_tables import (
    _continuous_model,
    _hand_built_design,
    _hand_built_magnitudes,
    _magnitudes,
)


def _working_rows(n: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 2.0, n), rng.normal(size=n)


@pytest.mark.parametrize("sweep_rows", [1 << 17, 256])
@pytest.mark.parametrize("pair_form", [None, "table", "accumulate"])
@pytest.mark.parametrize("offset", [1.0e3, 1.0e9])
def test_table_gram_matches_the_chunked_route_within_its_bound(
    monkeypatch, offset, pair_form, sweep_rows
):
    """``X_c' W X_c``, ``X_c' W z`` and the centre against the chunked exact route.

    Bound: the tables are within ``gamma_(n + 2m + 2) sum_r W_r M_r M_r'``
    (module docstring; ``M_r`` the row magnitudes of ``_magnitudes``, which
    carry a raw-read group's centre twice and bound the rounding of ``B R``);
    the chunked route forms each centred row and its products in at most
    ``p + 4`` roundings plus the sum over ``n`` rows.  A dropped centring term
    is a relative error of order one here.
    """
    model = _poisson_model(offset)
    dm = model._dm
    monkeypatch.setattr(lookup_tables, "_FORCE_PAIR_FORM", pair_form)
    # 256-row ranges sweep the 1,500 rows in six partial tables, summed in order.
    monkeypatch.setattr(lookup_tables, "_SWEEP_ROWS", sweep_rows)
    dm._structured_layout_cache.pop("lookup_tables", None)
    W, z = _working_rows(dm.n)
    penalty = np.zeros((dm.p, dm.p))
    reference = build_centered_system(dm=dm, W=W, z_off=z, penalty=penalty, _force_chunked=True)
    with lookup_tables.table_gram_config(True):
        table = build_centered_system(dm=dm, W=W, z_off=z, penalty=penalty)
    plan = lookup_tables.lookup_plan(dm)
    assert plan is not None
    assert plan.tab_pairs.size if pair_form != "accumulate" else not plan.tab_pairs.size
    assert plan.acc_pairs.size if pair_form != "table" else not plan.acc_pairs.size

    centre = np.asarray(reference.mean_x)
    magnitude, _, widest = _magnitudes(model, centre)
    m_largest = max(term.raw.shape[0] for term in plan.terms)
    k = 2 * dm.n + 2 * m_largest + dm.p + 4 * widest + 16
    gram_bound = _gamma(k) * (magnitude.T @ (W[:, None] * magnitude))
    z_centred = np.abs(z - reference.mean_z) + abs(reference.mean_z)
    rhs_bound = _gamma(k) * (magnitude.T @ (W * z_centred))

    assert np.all(np.abs(table.data_gram - reference.data_gram) <= gram_bound)
    assert np.all(np.abs(table.rhs - reference.rhs) <= rhs_bound)
    # A weighted mean is within gamma_k sum_r W_r M_r / sum W of its value on either route.
    mean_bound = _gamma(2 * dm.n) * (magnitude.T @ W) / W.sum()
    assert np.all(np.abs(table.mean_x - reference.mean_x) <= mean_bound)
    assert np.max(gram_bound) < 1e-8 * np.max(np.abs(reference.data_gram))


@pytest.mark.parametrize("sweep_rows", [1 << 17, 256])
def test_table_gram_is_bitwise_identical_for_any_worker_count(monkeypatch, sweep_rows):
    """One owner per table and per partial, summed in range order: the same bits at any count.

    At 256 rows a range, each sweep with small tables runs as six partial
    tables on the pool, and every coefficient block's product runs there too.
    """
    monkeypatch.setattr(lookup_tables, "_SWEEP_ROWS", sweep_rows)
    split = []
    sweep_run = lookup_tables._Sweeps.run

    def counted_run(self, job_bytes):
        split.append(len(self.reductions))
        return sweep_run(self, job_bytes)

    monkeypatch.setattr(lookup_tables._Sweeps, "run", counted_run)
    started = []
    pool = lookup_tables.ThreadPoolExecutor

    def spy(*args, **kwargs):
        started.append(kwargs["max_workers"])
        return pool(*args, **kwargs)

    monkeypatch.setattr(lookup_tables, "ThreadPoolExecutor", spy)
    dm = _poisson_model(1.0e3)._dm
    W, z = _working_rows(dm.n)
    penalty = np.eye(dm.p)
    results = set()
    for workers in (1, 2, 3, 8):
        with parallel_config(n_jobs=workers), lookup_tables.table_gram_config(True):
            system = build_centered_system(dm=dm, W=W, z_off=z, penalty=penalty)
        results.add(system.data_gram.tobytes() + system.rhs.tobytes() + system.mean_x.tobytes())

    # pooled at every worker count above one (capped by the jobs available)
    assert started and min(started) >= 2
    assert len(results) == 1
    assert (max(split) > 0) == (sweep_rows < dm.n)


def test_a_lookup_fit_takes_every_centred_gram_from_the_tables(monkeypatch):
    """The dispatch: past the row threshold every centred Gram the fit forms is a table Gram.

    v0.40.0 has no table Gram; its rungs form every one of these.
    """
    built = {"grams": 0, "tables": 0}
    import superglm.solvers.centered_system as centered_system

    system = centered_system.table_centred_system
    build = centered_system.build_centered_system

    def counted_tables(*args, **kwargs):
        built["tables"] += 1
        return system(*args, **kwargs)

    def counted_build(*args, **kwargs):
        # a build handed ``_data`` refreshes its right-hand side on a reused Gram
        built["grams"] += kwargs.get("_data") is None
        return build(*args, **kwargs)

    monkeypatch.setattr(centered_system, "table_centred_system", counted_tables)
    monkeypatch.setattr(centered_system, "build_centered_system", counted_build)
    for module in ("superglm.solvers.irls_direct", "superglm.reml.observed_geometry"):
        monkeypatch.setattr(f"{module}.build_centered_system", counted_build)
    monkeypatch.setattr(lookup_tables, "_MIN_TABLE_GRAM_ROWS", 0)
    monkeypatch.setattr(lookup_tables, "_TABLE_GRAM_COLUMNS_PER_SWEEP", 0)
    _poisson_model(1.0e3)

    assert built["tables"] >= 3
    assert built["tables"] == built["grams"]


def test_a_design_without_lookups_keeps_todays_gram(monkeypatch):
    """Continuous splines are not lookups: no table Gram, whatever the row threshold."""
    monkeypatch.setattr(lookup_tables, "_MIN_TABLE_GRAM_ROWS", 0)
    model = _continuous_model()

    assert lookup_tables.lookup_plan(model._dm) is None
    assert "centered_table_gram_hits" not in model._reml_profile


@pytest.mark.parametrize("pair_form", [None, "table", "accumulate"])
def test_hand_built_dense_groups_match_the_materialised_gram(monkeypatch, pair_form):
    """``X_c' W X_c`` and ``X_c' W z`` against the materialised centred design.

    Two dense groups (widths 2 and 1) put a cross block between dense groups
    (``_dense_moments_sweep`` with ``g != h``) and its mirrored placement under
    a value check.  The reference centres about its own weighted means; both
    routes are within ``gamma_k sum_r W_r M_r M_r'`` (the bound of
    ``test_table_gram_matches_the_chunked_route_within_its_bound``).
    Mutation (fails here): the mirrored dense block written as ``out``.
    """
    from superglm.solvers.centered_system import build_centered_system

    dm = _hand_built_design()
    monkeypatch.setattr(lookup_tables, "_FORCE_PAIR_FORM", pair_form)
    W, z = _working_rows(dm.n)
    with lookup_tables.table_gram_config(True):
        system = build_centered_system(dm=dm, W=W, z_off=z, penalty=np.zeros((dm.p, dm.p)))
        plan = lookup_tables.table_gram_plan(dm)
    assert plan is not None and len(plan.dense) == 2

    design = dm.toarray()
    mean = design.T @ W / W.sum()
    centred = design - mean
    z_centred = z - W @ z / W.sum()
    gram = centred.T @ (W[:, None] * centred)
    rhs = centred.T @ (W * z_centred)
    magnitude, widest = _hand_built_magnitudes(dm, mean)
    m_largest = max(term.raw.shape[0] for term in plan.terms)
    k = 2 * dm.n + 2 * m_largest + dm.p + 4 * widest + 16
    assert np.all(
        np.abs(system.data_gram - gram) <= _gamma(k) * (magnitude.T @ (W[:, None] * magnitude))
    )
    z_magnitude = np.abs(z_centred) + abs(W @ z / W.sum())
    assert np.all(np.abs(system.rhs - rhs) <= _gamma(k) * (magnitude.T @ (W * z_magnitude)))
    assert np.all(np.abs(system.mean_x - mean) <= _gamma(2 * dm.n) * (magnitude.T @ W) / W.sum())


def test_a_table_gram_without_dense_columns_is_exactly_symmetric():
    """Each own-term block is a GEMM, ``(A W)' A``, whose two triangles round apart.

    The tables return a symmetric Gram as every other route does, dense
    columns or not; on the PR head the design without a dense column
    returned the GEMM's two triangles as they rounded.
    """
    dm = _poisson_model(1.0e3)._dm
    lookups = [gm for gm in dm.group_matrices if type(gm) is not DenseGroupMatrix]
    design = type(dm)(lookups, n=dm.n, p=sum(gm.shape[1] for gm in lookups))
    W, z = _working_rows(design.n)
    with lookup_tables.table_gram_config(True):
        plan = lookup_tables.table_gram_plan(design)
        assert plan is not None and not plan.dense
        _, gram, *_ = lookup_tables.table_centred_system(plan, design, W, z, float(W.sum()))
    assert np.array_equal(gram, gram.T)


def test_a_narrow_design_keeps_the_chunked_gram():
    """Below ``_TABLE_GRAM_COLUMNS_PER_SWEEP`` columns per row sweep the rungs are cheaper.

    The hand-built design has 15 sweeps (3 terms, 3 pairs, 6 term-by-dense
    blocks, 3 dense pairs): at 16 columns it keeps the rungs, at 110 it takes
    the tables; its leverages read the tables either way.  On the PR head the
    narrow design took the tables (as the freMTPL2 main-effects model did,
    1.65 times slower per Gram).
    """
    narrow, wide = _hand_built_design(), _hand_built_design(basis=100)
    for dm in (narrow, wide):
        assert lookup_tables.lookup_plan(dm) is not None
        assert lookup_tables.lookup_plan(dm).n_sweeps == 15
    assert narrow.p < lookup_tables._TABLE_GRAM_COLUMNS_PER_SWEEP * 15 <= wide.p
    assert lookup_tables.table_gram_plan(narrow) is None
    assert lookup_tables.table_gram_plan(wide) is lookup_tables.lookup_plan(wide)


def _common_component_design(kind: str, n: int = 20_480, seed: int = 8):
    """A support whose ``B`` carries a large common component ``C``, beside a categorical.

    ``offset``: ``B = 2**40 + hat values`` and a generic ``R``, so every
    column of ``B R`` has a mean near ``2**40 colsum(R)`` and a spread near
    one.  ``cancelling``: ``B = 2**60 + 2**8 m`` (``m`` small integers, exact)
    and ``R`` whose columns sum to zero, so ``R`` cancels ``C`` and ``B R`` is
    small.  Both sit inside the reassociation range (``_ssp_gram_needs_exact``).
    """
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DesignMatrix,
        SupportCompressedSSPGroupMatrix,
    )

    rng = np.random.default_rng(seed)
    support, basis = 60, 6
    if kind == "offset":
        common = 2.0**40
        grid = np.linspace(0.0, 1.0, support)
        knots = np.linspace(0.0, 1.0, basis)
        hats = np.maximum(0.0, 1.0 - np.abs(grid[:, None] - knots[None, :]) * (basis - 1))
        B = common + hats
        R = rng.normal(size=(basis, basis - 1))
    else:
        common = 2.0**60
        B = common + 2.0**8 * rng.integers(0, 20, (support, basis)).astype(np.float64)
        R = np.zeros((basis, basis - 2))
        for j in range(basis - 2):  # 1, 1, -2: each column sums to zero, three terms
            R[j, j], R[j + 1, j], R[j + 2, j] = 1.0, 1.0, -2.0
    spline = SupportCompressedSSPGroupMatrix(B, R, rng.integers(0, support, n))
    categorical = CategoricalGroupMatrix(rng.integers(-1, 3, n), 3)
    groups = [spline, categorical]
    return DesignMatrix(groups, n=n, p=sum(gm.shape[1] for gm in groups)), common


@pytest.mark.parametrize("pair_form", [None, "accumulate"])
@pytest.mark.parametrize("kind", ["offset", "cancelling"])
def test_a_common_component_of_a_support_never_rounds_into_the_gram(monkeypatch, kind, pair_form):
    """The table is anchored before ``R`` applies, so ``C`` enters no product (threads 19, 21).

    The reference centres ``B - C`` (exact: ``B`` lies in ``[C, 2C]``) before
    projecting, so its rows carry no trace of ``C``; both routes are within
    the bound of a direct centred Gram on those rows, ``gamma_k sum_r W_r M_r
    M_r'`` with ``M_r`` the anchored rows' magnitudes ``|B - C| |R|`` plus
    their largest (the anchor and the centre).  On e62f0c85 the table was
    ``fl(B R)``, off by ``u C |R|`` in every row: 1e-4 against a spread near
    one (``offset``: 1.3e-4 of the Gram), 1.2e-2 of the Gram
    (``cancelling``), where the chunked route is within 1e-15.  Mutation:
    without the anchoring the ``cancelling`` cases fail.  The ``offset``
    design's anchor row ``B[a] R`` (near ``2**40 colsum(R)``) dwarfs its rows
    past ``_SEPARATION_LIMIT``, so it now keeps the chunked route (thread 36).
    """
    from superglm._group_matrix._group_matrix_kernels import _ssp_gram_needs_exact
    from superglm.solvers.centered_system import build_centered_system

    dm, common = _common_component_design(kind)
    monkeypatch.setattr(lookup_tables, "_FORCE_PAIR_FORM", pair_form)
    spline, categorical = dm.group_matrices
    assert not _ssp_gram_needs_exact(spline.B_unique, spline.R_inv)
    W, z = _working_rows(dm.n)
    with lookup_tables.table_gram_config(True):
        system = build_centered_system(dm=dm, W=W, z_off=z, penalty=np.zeros((dm.p, dm.p)))
        plan = lookup_tables.table_gram_plan(dm)
    if kind == "offset":
        assert plan is None and lookup_tables.lookup_plan(dm) is None
        m_largest = spline.B_unique.shape[0]
    else:
        assert plan is not None and (pair_form is None or plan.acc_pairs.size)
        m_largest = max(term.raw.shape[0] for term in plan.terms)

    shifted = (spline.B_unique - common) @ spline.R_inv  # exact rows, ``C`` removed first
    rows = np.hstack((shifted[spline.bin_idx], categorical.toarray()))
    mean = rows.T @ W / W.sum()
    centred = rows - mean
    gram = centred.T @ (W[:, None] * centred)
    support_magnitude = np.abs(spline.B_unique - common) @ np.abs(spline.R_inv)
    magnitude = np.hstack(
        (
            support_magnitude[spline.bin_idx] + support_magnitude.max(axis=0),
            np.abs(categorical.toarray()) + 2.0 * np.abs(mean[spline.shape[1] :]),
        )
    )
    k = 2 * dm.n + 2 * m_largest + dm.p + 4 * spline.B_unique.shape[1] + 16
    bound = _gamma(k) * (magnitude.T @ (W[:, None] * magnitude))
    assert np.all(np.abs(system.data_gram - gram) <= bound)


def _counted(monkeypatch, cost: dict) -> None:
    """Add every ``_left`` / ``_right`` call's multiply-adds (``_left_cost``) to ``cost["madds"]``."""
    left, right = lookup_tables._left, lookup_tables._right

    def counted_left(term, A, d, X):
        cost["madds"] += lookup_tables._left_cost(term, X.shape[0], X.shape[1])
        return left(term, A, d, X)

    def counted_right(term, A, d, X):
        cost["madds"] += lookup_tables._left_cost(term, X.shape[0], X.shape[1])
        return right(term, A, d, X)

    monkeypatch.setattr(lookup_tables, "_left", counted_left)
    monkeypatch.setattr(lookup_tables, "_right", counted_right)


@pytest.mark.parametrize("pair_form", [None, "table", "accumulate"])
def test_a_categorical_reads_its_table_as_an_identity(monkeypatch, pair_form):
    """A K-level categorical costs ``O(K^2)`` in coefficient space, as on the chunked route.

    Its own block is diagonal plus rank-one centring (``_identity_gram``),
    and every product with its centred table reads the identity (``_left``,
    ``_right``): on 71e79cab the own block alone was a ``(K+1) x K`` by
    ``K`` GEMM, ``K^3`` multiply-adds per centred system (28 ms at K = 1000
    on 542,406 rows, where the whole chunked build took 54 ms).  The count
    here is of multiply-adds, not time.  The Gram equals the materialised
    centred design's within the bound of a direct centred Gram.
    """
    from superglm.group_matrix import CategoricalGroupMatrix, DesignMatrix

    rng = np.random.default_rng(12)
    n, wide, narrow = 20_480, 400, 30
    groups = [
        CategoricalGroupMatrix(rng.integers(-1, wide, n), wide),
        DenseGroupMatrix(rng.normal(50.0, 2.0, (n, 1))),
        CategoricalGroupMatrix(rng.integers(-1, narrow, n), narrow),
    ]
    dm = DesignMatrix(groups, n=n, p=sum(gm.shape[1] for gm in groups))
    monkeypatch.setattr(lookup_tables, "_FORCE_PAIR_FORM", pair_form)
    own = {"calls": 0}
    identity_gram = lookup_tables._identity_gram

    def counted_own(*args):
        own["calls"] += 1
        return identity_gram(*args)

    monkeypatch.setattr(lookup_tables, "_identity_gram", counted_own)
    cost = {"madds": 0}
    _counted(monkeypatch, cost)
    W, z = _working_rows(dm.n)
    plan = lookup_tables.lookup_plan(dm)
    assert plan is not None
    mean_x, gram, *_ = lookup_tables.table_centred_system(plan, dm, W, z, float(W.sum()))

    assert own["calls"] == 2
    assert cost["madds"] < wide**2 * (narrow + 4)  # one GEMM over the identity is wide**3
    design = dm.toarray()
    mean = design.T @ W / W.sum()
    centred = design - mean
    reference = centred.T @ (W[:, None] * centred)
    magnitude = np.abs(design) + 2.0 * np.abs(mean)
    bound = _gamma(2 * dm.n + 2 * (wide + 1) + dm.p + 16) * (magnitude.T @ (W[:, None] * magnitude))
    assert np.all(np.abs(gram - reference) <= bound)


def test_a_table_pair_takes_the_cheaper_association():
    """``A' T B`` (Gram) and ``A X B'`` (leverage) pick their association by full cost.

    The thread's shapes: ``m_a = 10000, p_a = 100, m_b = 10, p_b = 50``.
    ``A' (T B)`` costs 55 million multiply-adds and ``(A' T) B`` about 10
    million; on 71e79cab the Gram compared ``p_b`` with ``p_a`` alone and
    took the first.  The cost is counted from the calls, not timed.
    """
    rng = np.random.default_rng(3)
    shapes = ((10_000, 100), (10, 50))
    terms, tables = [], []
    for m, p in shapes:
        raw = rng.normal(size=(m, p))
        zeros = np.zeros(m, dtype=np.int32)
        terms.append(lookup_tables._Term(np.arange(p), raw, False, zeros, zeros + p, np.zeros(p)))
        tables.append(raw)
    (ta, tb), (A, B) = terms, tables
    T = rng.uniform(size=(A.shape[0], B.shape[0]))
    with pytest.MonkeyPatch.context() as patch:
        cost = {"madds": 0}
        _counted(patch, cost)
        block = lookup_tables._pair_block(ta, A, np.zeros(100), tb, B, np.zeros(50), T)
    assert cost["madds"] == 100 * 10_000 * 10 + 100 * 10 * 50
    assert np.allclose(block, A.T @ T @ B, rtol=1e-10, atol=1e-10 * np.abs(A.T @ T @ B).max())

    X = rng.normal(size=(100, 50))
    calls = []
    rows_times = lookup_tables._rows_times
    with pytest.MonkeyPatch.context() as patch:

        def counted(term, table, M):
            calls.append(lookup_tables._rows_cost(term, table, M.shape[1]))
            return rows_times(term, table, M)

        patch.setattr(lookup_tables, "_rows_times", counted)
        G = lookup_tables._pair_table(ta, A, tb, B, X)
    first = 10_000 * 100 * 50 + 10 * 50 * 10_000  # (A X) B'
    second = 10 * 50 * 100 + 10_000 * 100 * 10  # A (X B')
    assert sum(calls) == min(first, second)
    assert np.allclose(G, A @ X @ B.T, rtol=1e-10, atol=1e-10 * np.abs(A @ X @ B.T).max())


def test_an_offset_that_dwarfs_its_rows_declines_the_plan():
    """Thread 36: a support around ``2**60`` whose anchor row the centre cannot carry apart.

    ``B = 2**60 + 2**8 m`` with a generic ``R``: the anchored rows ``(B - B[a]) R``
    are near ``2**8``, the anchor row ``B[a] R`` near ``2**60``.  The leverage
    reads a centre ``c = fl(B[a] R + shift)`` back as ``c - B[a] R``, which
    erases the shift there, so the plan is declined (``_SEPARATION_LIMIT``).
    The same rows without the common part take the tables.  On 83d33d9c the
    plan was built.
    """
    from superglm.group_matrix import (
        CategoricalGroupMatrix,
        DesignMatrix,
        SupportCompressedSSPGroupMatrix,
    )

    rng = np.random.default_rng(21)
    n, support, basis = 20_480, 60, 6
    m = 2.0**8 * rng.integers(0, 20, (support, basis)).astype(np.float64)
    R = rng.normal(size=(basis, basis - 1))
    bins = rng.integers(0, support, n)
    categorical = CategoricalGroupMatrix(rng.integers(-1, 3, n), 3)
    plans = []
    for common in (2.0**60, 0.0):
        groups = [SupportCompressedSSPGroupMatrix(common + m, R, bins), categorical]
        dm = DesignMatrix(groups, n=n, p=sum(gm.shape[1] for gm in groups))
        plans.append(lookup_tables.lookup_plan(dm))
    assert plans[0] is None and plans[1] is not None


def test_a_categorical_dominated_by_one_bin_keeps_the_chunked_gram():
    """Thread 37: one bin's weight ``1e16`` times the rest cancels the identity Gram's diagonal.

    ``_identity_gram``'s diagonal ``w_k R_k / S`` is formed with an error of
    ``u w_k``; with the dominant bin at ``1e16`` and the base bin at one
    (zero weight elsewhere) that is all of the curvature, so the call is
    declined and the chunked route builds the system, bit for bit its own.
    Ordinary weights on the same design take the tables.  On 83d33d9c the
    table route returned a rank-zero block.
    """
    from superglm.group_matrix import CategoricalGroupMatrix, DesignMatrix
    from superglm.solvers.centered_system import build_centered_system

    rng = np.random.default_rng(5)
    n, K = 20_000, 6
    codes = rng.integers(-1, K, n)
    dm = DesignMatrix([CategoricalGroupMatrix(codes, K)], n=n, p=K)
    dominated = np.where(codes == 0, 1.0e16 / np.sum(codes == 0), 0.0)
    dominated[codes == -1] = 1.0 / np.sum(codes == -1)
    z, penalty = np.zeros(n), np.eye(K)
    for W, tables in ((dominated, 0), (rng.uniform(0.5, 2.0, n), 1)):
        profile: dict = {}
        with lookup_tables.table_gram_config(True):
            system = build_centered_system(dm=dm, W=W, z_off=z, penalty=penalty, profile=profile)
        assert profile.get("centered_table_gram_hits", 0) == tables
        if not tables:
            chunked = build_centered_system(
                dm=dm, W=W, z_off=z, penalty=penalty, _force_chunked=True
            )
            assert np.array_equal(system.data_gram, chunked.data_gram)
            assert np.all(np.diag(system.data_gram)[1:] == 0.0)  # those levels carry no weight
            assert system.data_gram[0, 0] > 0.0
