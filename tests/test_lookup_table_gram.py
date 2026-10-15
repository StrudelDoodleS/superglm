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
from superglm.solvers.centered_system import build_centered_system
from tests.test_reml_leverage_hessian import _gamma, _poisson_model
from tests.test_reml_leverage_tables import _continuous_model, _magnitudes


def _working_rows(n: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 2.0, n), rng.normal(size=n)


@pytest.mark.parametrize("pair_form", [None, "table", "accumulate"])
@pytest.mark.parametrize("offset", [1.0e3, 1.0e9])
def test_table_gram_matches_the_chunked_route_within_its_bound(monkeypatch, offset, pair_form):
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
    assert np.allclose(table.mean_x, reference.mean_x, rtol=_gamma(2 * dm.n), atol=_gamma(2 * dm.n))
    assert np.max(gram_bound) < 1e-8 * np.max(np.abs(reference.data_gram))


def test_table_gram_is_bitwise_identical_for_any_worker_count(monkeypatch):
    """One owner per table, rows in order: the pooled Gram is the serial one, bit for bit."""
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
    _poisson_model(1.0e3)

    assert built["tables"] >= 3
    assert built["tables"] == built["grams"]


def test_a_design_without_lookups_keeps_todays_gram(monkeypatch):
    """Continuous splines are not lookups: no table Gram, whatever the row threshold."""
    monkeypatch.setattr(lookup_tables, "_MIN_TABLE_GRAM_ROWS", 0)
    model = _continuous_model()

    assert lookup_tables.lookup_plan(model._dm) is None
    assert "centered_table_gram_hits" not in model._reml_profile
