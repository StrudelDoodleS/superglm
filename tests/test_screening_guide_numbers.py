"""The screening guide's worked freMTPL example is a published measurement.

``docs/guide/screening.md`` prints a twelve-row sweep table and then reasons
about individual cells of it: the top row's ``z`` against the measured ``ti``
noise maximum, the ``cat_cat`` row's ``z`` and ``statistic``, how many rows
carry a negative ``z``, and two confirmatory refit gains.  Every one of those
moves when the screen's dispersion contract moves or when a margin's knots
move, and both moved together — the Pearson denominator went from
``count_nonzero(w) - edf`` to ``sum(w) - edf`` (on this book ``sum(w)`` is
roughly half of ``n``, so ``phi`` nearly doubles) and ``quantile_tempered``
knot placement started consuming frequency mass, which shifts the
``BonusMalus`` margin the example is built around.  The prose was rewritten to
describe the new contract; the numbers were not regenerated.

Two layers keep the guide honest:

The ``test_screening_guide_*`` tests read the numbers back out of the published
document — the printed sweep table, the quoted ``phi``, the confirmatory-refit
table, the incomparable-``statistic`` paragraph and the closing reading against
the ``ti`` floor — and compare each to the committed measurement in
``tests/fixtures/screening_guide_fremtpl.json``.  They need no data file, so
they run everywhere.

``test_screening_guide_fixture_matches_the_real_book`` recomputes that
measurement from the freMTPL2 parquet, so the fixture cannot itself go stale
behind a code change.  It skips when the (gitignored) parquet is absent, so it
does not run in CI; read its docstring for what it does and does not prove.

One number the closing paragraph reads against — the ``ti`` null floor, 7.31 —
comes from a *different* measurement, the 160-fit null battery in
``benchmarks/screening_null_floors.py``, which was not regenerated.
``test_screening_guide_ti_floor_survives_the_dispersion_contract_change``
pins the argument for why it did not have to be.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from superglm import Categorical, SuperGLM
from superglm.features.numeric import Numeric
from superglm.features.spline import Spline
from superglm.solvers.dispersion import (
    dispersion_likelihood_size,
    pearson_residual_degrees_of_freedom,
)

from . import _datasets

_ROOT = Path(__file__).resolve().parents[1]
_GUIDE_PATH = _ROOT / "docs/guide/screening.md"
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "screening_guide_fremtpl.json"

_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


@pytest.fixture(scope="module")
def measured() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def guide() -> str:
    return _GUIDE_PATH.read_text(encoding="utf-8")


def _published_sweep_rows(guide_text: str) -> list[dict]:
    """Parse the ``print(table.to_string(index=False))`` output block."""
    match = re.search(
        r"^print\(table\.to_string\(index=False\)\)\n(?P<body>(?:^#.*\n)+)",
        guide_text,
        re.MULTILINE,
    )
    assert match is not None, "the guide no longer prints the sweep table"
    lines = [line.lstrip("#").strip() for line in match.group("body").splitlines()]
    header, *body = [line for line in lines if line]
    assert header.split() == [
        "feature_a",
        "feature_b",
        "kind",
        "statistic",
        "z",
        "edf0",
        "lambda0",
        "n_cells",
        "approx",
    ], header
    rows = []
    for line in body:
        a, b, kind, statistic, z, edf0, lambda0, n_cells, approx = line.split()
        rows.append(
            {
                "feature_a": a,
                "feature_b": b,
                "kind": kind,
                "statistic": float(statistic),
                "z": float(z),
                "edf0": float(edf0),
                "lambda0": float(lambda0),
                "n_cells": int(n_cells),
                "approx": {"True": True, "False": False}[approx],
            }
        )
    return rows


def _search(guide_text: str, pattern: str) -> re.Match:
    match = re.search(pattern, " ".join(guide_text.split()))
    assert match is not None, f"the guide no longer contains {pattern!r}"
    return match


def test_screening_guide_table_matches_the_measured_sweep(guide, measured) -> None:
    """Every printed cell of the worked sweep is the measured value."""
    published = _published_sweep_rows(guide)
    expected = measured["rows"]

    # Order is part of the claim: the guide says the sweep comes back as "one
    # sorted table" and then reads the top two rows as the queue.
    assert [(r["feature_a"], r["feature_b"], r["kind"]) for r in published] == [
        (r["feature_a"], r["feature_b"], r["kind"]) for r in expected
    ]
    for got, want in zip(published, expected, strict=True):
        where = f"{want['feature_a']} x {want['feature_b']}"
        # Printed at six decimals, so half a unit in the last place.
        for column in ("statistic", "z", "edf0"):
            assert got[column] == pytest.approx(want[column], abs=5e-7), f"{where}.{column}"
        assert got["lambda0"] == pytest.approx(want["lambda0"], rel=5e-7), f"{where}.lambda0"
        assert got["n_cells"] == want["n_cells"], where
        assert got["approx"] == want["approx"], where


def test_screening_guide_quotes_the_measured_dispersion(guide, measured) -> None:
    """The example's ``phi`` is the one the documented denominator produces."""
    phi = float(_search(guide, r"`phi` estimated at ([0-9.]+)\)").group(1))
    assert phi == pytest.approx(measured["phi"], abs=5e-3)
    # phi is the Pearson sum over sum(w) - edf, and the exposure mass here is
    # far below the row count; quoting a phi consistent with an n - edf
    # denominator would understate it by roughly a factor of two.
    assert measured["sum_sample_weight"] < 0.6 * measured["n_rows"]

    # The retired publication reconciles exactly, which is what makes this a
    # regeneration rather than a re-measurement of something else: the same
    # Pearson sum over the *retired* count_nonzero(w) - edf denominator is
    # 2.55, the number this guide printed at 37a1c18 and earlier.  If the
    # fixture had been produced from a different sample, a different mains
    # model or a different response encoding, this identity would not close.
    pearson = measured["phi"] * (measured["sum_sample_weight"] - measured["mains_edf"])
    retired = pearson / (measured["n_rows"] - measured["mains_edf"])
    assert retired == pytest.approx(measured["retired_phi_n_minus_edf"], rel=1e-6)
    assert retired == pytest.approx(2.55, abs=5e-3)
    # ... and the guide must no longer publish it as a number of its own.
    assert re.search(r"(?<![0-9.])2\.55(?![0-9])", guide) is None


def test_screening_guide_confirmatory_refit_table_matches_the_measured_gains(
    guide, measured
) -> None:
    """The ``z`` / probe df / refit gain table under the sweep is measured."""
    rows = measured["rows"]

    # The confirmatory-refit table: | pair | kind | z | probe df | refit gain |
    for refit in measured["confirmatory_refits"]:
        pair = f"{refit['feature_a']} x {refit['feature_b']}"
        row = next(
            r
            for r in rows
            if (r["feature_a"], r["feature_b"]) == (refit["feature_a"], refit["feature_b"])
        )
        match = _search(
            guide,
            rf"\| `{re.escape(pair)}` \| `{re.escape(refit['kind'])}` "
            rf"\| (-?[0-9.]+) \| ([0-9]+) \| ([0-9.]+) \|",
        )
        assert float(match.group(1)) == pytest.approx(row["z"], abs=5e-3), f"{pair} z"
        assert int(match.group(2)) == round(row["edf0"]), f"{pair} probe df"
        # Published to one decimal, so a tenth either way.
        assert float(match.group(3)) == pytest.approx(refit["deviance_gain"], abs=0.1), (
            f"{pair} refit gain"
        )

    # "43.9 on 2 df is 22.0 per df against 7.3"
    density = _search(guide, r"([0-9.]+) on ([0-9]+) df is ([0-9.]+) per df against ([0-9.]+)")
    first, second = measured["confirmatory_refits"]
    first_row, second_row = (
        next(
            r
            for r in rows
            if (r["feature_a"], r["feature_b"]) == (refit["feature_a"], refit["feature_b"])
        )
        for refit in (first, second)
    )
    assert float(density.group(1)) == pytest.approx(first["deviance_gain"], abs=0.1)
    assert int(density.group(2)) == round(first_row["edf0"])
    assert float(density.group(3)) == pytest.approx(
        first["deviance_gain"] / first_row["edf0"], abs=0.1
    )
    assert float(density.group(4)) == pytest.approx(
        second["deviance_gain"] / second_row["edf0"], abs=0.1
    )
    # The ranking lesson only holds while the wider block still buys more.
    assert second["deviance_gain"] > first["deviance_gain"]
    assert second_row["z"] < first_row["z"]


def test_screening_guide_incomparable_statistic_claims_match(guide, measured) -> None:
    """The "`statistic` is not comparable" paragraph quotes real cells."""
    rows = measured["rows"]
    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row)

    # "the `cat_cat` row's X is a 208-dimensional block and the `numeric_cat`
    # row's Y a 10-dimensional one"
    cat_cat = by_kind["cat_cat"][0]
    incomparable = _search(
        guide,
        r"the `cat_cat` row's ([0-9.]+) is a ([0-9]+)-dimensional block and the "
        r"`numeric_cat` row's ([0-9.]+) a ([0-9]+)-dimensional one",
    )
    assert float(incomparable.group(1)) == pytest.approx(cat_cat["statistic"], abs=0.5)
    assert int(incomparable.group(2)) == round(cat_cat["edf0"])
    numeric_cat = next(
        r for r in by_kind["numeric_cat"] if round(r["edf0"]) == int(incomparable.group(4))
    )
    assert float(incomparable.group(3)) == pytest.approx(numeric_cat["statistic"], abs=5e-2)

    # "Eight of the twelve rows carry a negative `z`"
    negatives = _search(guide, r"(\w+) of the (\w+) rows carry a negative `z`")
    assert _WORD_NUMBERS[negatives.group(1).lower()] == sum(1 for r in rows if r["z"] < 0.0)
    assert _WORD_NUMBERS[negatives.group(2).lower()] == len(rows)

    # "The `cat_cat` row's -6.51 is the extreme case ... lands at 75"
    extreme = _search(
        guide,
        r"The `cat_cat` row's (-[0-9.]+) is the extreme case .*? lands at ([0-9.]+),",
    )
    assert float(extreme.group(1)) == pytest.approx(cat_cat["z"], abs=5e-3)
    assert float(extreme.group(2)) == pytest.approx(cat_cat["statistic"], abs=0.5)
    assert cat_cat["z"] == min(r["z"] for r in rows)

    # "the `VehBrand x Region` row above reports `edf0 = 208` against a
    # nominal 210" — the rank-deficiency illustration in the budget section
    # reads the same row.
    deficient = _search(
        guide,
        r"the `(\w+) x (\w+)` row above reports `edf0 = ([0-9]+)` against a nominal ([0-9]+)",
    )
    assert (deficient.group(1), deficient.group(2)) == (
        cat_cat["feature_a"],
        cat_cat["feature_b"],
    )
    assert int(deficient.group(3)) == round(cat_cat["edf0"])
    assert int(deficient.group(4)) > round(cat_cat["edf0"])


def test_screening_guide_top_row_is_read_against_the_published_ti_floor(guide, measured) -> None:
    """The closing paragraph compares the real top ``z`` to the real floor."""
    rows = measured["rows"]
    first = measured["confirmatory_refits"][0]

    # "(7.31 for `ti`): 4.40 does not clear it -- and the refit bought 43.9
    # deviance anyway."
    floor = _search(
        guide,
        r"measured noise maximum below \(([0-9.]+) for `ti`\): (-?[0-9.]+) does not clear it "
        r"[^.]*?refit bought ([0-9.]+) deviance",
    )
    top = rows[0]
    assert float(floor.group(1)) == pytest.approx(measured["ti_null_floor_max_z"], abs=5e-3)
    assert float(floor.group(2)) == pytest.approx(top["z"], abs=5e-3)
    assert float(floor.group(3)) == pytest.approx(first["deviance_gain"], abs=0.1)
    # The sentence claims the top row falls short of the floor; if the measured
    # z ever cleared it the paragraph would say the opposite.
    assert top["z"] < measured["ti_null_floor_max_z"]
    # ... and the floor itself is the one published in this same guide.
    assert (
        _search(guide, r"\| `ti` \| 480 \| [0-9.]+ \| [0-9.]+ \| ([0-9.]+) \|").group(1)
        == f"{measured['ti_null_floor_max_z']:.2f}"
    )


def test_screening_guide_ti_floor_survives_the_dispersion_contract_change(guide) -> None:
    """The `ti` floor is a unit-weight measurement, so the new phi cannot move it.

    Everything the worked example prints moved when the screen's Pearson
    denominator moved.  The closing paragraph reads its top row against a
    number from a *different* measurement — the null battery's ``ti`` maximum —
    which was not regenerated.  This pins the argument that it did not have to
    be, entirely from published numbers plus the battery's own construction:

    1. the guide attributes both maxima above 6, the ``ti`` one included, to
       the dispersed Gaussian arm of the battery;
    2. only the Poisson arm of that battery screens with a ``sample_weight``
       at all, and the guide publishes its maximum anywhere as well below the
       ``ti`` floor, so the floor cannot have been Poisson-carried and a
       Poisson row cannot become the maximum by moving *down*;
    3. under unit weights the two denominators are the same number, so no
       unweighted arm's ``z`` moved at all.

    Give the battery's Gaussian arm a ``sample_weight`` and step 2 fails: the
    floor would then be contract-sensitive and would have to be re-measured
    alongside the worked example.
    """
    battery = importlib.import_module("benchmarks.screening_null_floors")
    text = " ".join(guide.split())

    # (1) the published attribution, cross-checked against the per-kind table.
    carried = _search(
        guide,
        r"dispersed Gaussian carries it: [^.]*?both maxima above 6 "
        r"\(([0-9.]+) on `numeric_cat`, ([0-9.]+) on `ti`\)",
    )
    ti_floor = _search(guide, r"\| `ti` \| 480 \| [0-9.]+ \| [0-9.]+ \| ([0-9.]+) \|").group(1)
    numeric_cat_floor = _search(
        guide, r"\| `numeric_cat` \| 960 \| [0-9.]+ \| [0-9.]+ \| ([0-9.]+) \|"
    ).group(1)
    assert carried.group(2) == ti_floor
    assert carried.group(1) == numeric_cat_floor
    assert f"({ti_floor} for `ti`)" in text

    # (2) the battery's weighted arm is the Poisson one, and only that one.
    df, exposure = battery._frame(64, np.random.default_rng(0))
    weighted = {}
    for family in battery.FAMILIES:
        _, weight = battery._null_response(df, exposure, family, np.random.default_rng(0))
        weighted[family] = weight is not None
    assert weighted == {"poisson": True, "gamma": False, "binomial": False, "gaussian": False}
    poisson_max = float(_search(guide, r"\(Poisson at most ([0-9.]+),").group(1))
    assert poisson_max < float(ti_floor)

    # (3) with unit weights the retired and current denominators coincide, so
    #     the Gaussian arm that carries the floor is untouched by the change.
    ones = np.ones(64, dtype=np.float64)
    for semantics in ("prior", "frequency"):
        assert dispersion_likelihood_size(ones, weight_semantics=semantics) == pytest.approx(
            float(np.count_nonzero(ones))
        )
        assert pearson_residual_degrees_of_freedom(
            ones, 4.0, weight_semantics=semantics
        ) == pytest.approx(64.0 - 4.0)


# ── the fixture is anchored to the real book (skips without the parquet) ────


_FREQ_SKIP_REASON = _datasets.skip_reason("freMTPL2freq.parquet")

FREQ_SKIP = pytest.mark.skipif(
    _FREQ_SKIP_REASON is not None,
    reason=_FREQ_SKIP_REASON or "",
)


def _guide_features() -> dict:
    """Exactly the specification published in the guide's fence."""
    return {
        "DrivAge": Spline(kind="ps", n_knots=8),
        "VehAge": Spline(kind="ps", n_knots=12),
        "BonusMalus": Spline(
            kind="ps",
            n_knots=12,
            knot_strategy="quantile_tempered",
            knot_alpha=0.2,
        ),
        "LogDensity": Numeric(),
        "VehBrand": Categorical(),
        "Region": Categorical(),
    }


def _guide_frame(n_rows: int):
    df = _datasets.load_freq().sample(n_rows, random_state=0).reset_index(drop=True)
    df["Exposure"] = df["Exposure"].clip(lower=0.01)
    df["LogDensity"] = np.log1p(df["Density"].to_numpy(dtype=np.float64))
    exposure = df["Exposure"].to_numpy(dtype=np.float64)
    y = df["ClaimNb"].to_numpy(dtype=np.float64) / exposure
    return df, y, exposure


@FREQ_SKIP
def test_screening_guide_fixture_matches_the_real_book(measured) -> None:
    """Regression guard for an ALREADY-SHIPPED fix, plus the fixture's anchor.

    Two honest labels, because this test cannot do what its siblings do.

    *It cannot fail against 37a1c18, by construction.*  The behaviour it pins
    shipped in 67b90f8 ("fix: resolve numerical audit findings"), an ancestor
    of this branch's parent: the screen's Pearson denominator moved from
    ``count_nonzero(w) - edf`` to the fit's own ``sum(w) - edf`` contract, and
    ``quantile_tempered`` knot placement started consuming frequency mass.
    Issue #219 is the docs half of that change — the branch diff touches no
    ``src/`` line — so a docs-only yardstick has nothing here to detect.  What
    it does fail against is a surgical revert of either half of 67b90f8:
    restoring ``n_eff = count_nonzero(weights)`` in ``screen_interactions``
    turns the ``phi`` identity below into 2.554 against 4.821, and restoring
    ``del sample_weight`` / ``spec._place_knots(x)`` in
    ``_spline_build.build_group_info`` moves the ``BonusMalus`` margin and
    breaks the mains ``effective_df`` assertion before the screen is even
    reached.  Both were demonstrated; if a future change makes neither break
    this test, the test has gone inert and should be deleted rather than kept.

    *It is also the fixture's only tie to reality.*  The five
    ``test_screening_guide_*`` tests compare the published document to
    ``tests/fixtures/screening_guide_fremtpl.json``, which would be circular
    if the fixture were back-filled from whatever a fixing agent happened to
    print.  This test refits the real 80,000-row book and rederives every
    committed value, so the fixture is a measurement and not an assertion.
    It skips wherever the gitignored parquet is absent, CI included; run it
    locally to regenerate the fixture after any deliberate contract change.
    """
    df, y, exposure = _guide_frame(measured["n_rows"])
    # The dispersion contract this example documents only bites because the
    # exposure mass is far below the row count: sum(w) - edf is the denominator,
    # not n - edf, so the two differ by roughly a factor of two here.
    assert exposure.sum() == pytest.approx(measured["sum_sample_weight"], rel=1e-6)
    assert exposure.sum() < 0.6 * len(df)

    model = SuperGLM(
        family="poisson",
        features=_guide_features(),
        weight_semantics="frequency",
    )
    model.fit_reml(df, y, sample_weight=exposure)
    assert model._result.deviance == pytest.approx(measured["mains_deviance"], rel=1e-6)
    assert model._result.effective_df == pytest.approx(measured["mains_edf"], rel=1e-5)

    table = model.screen_interactions(df, y, sample_weight=exposure)

    # Name the denominator rather than only the number it produces, so a
    # reverted contract fails as a contract failure and not as a bare number
    # mismatch.  The Pearson sum is recomputed here from the public prediction
    # path, independently of the screen's own internals; only the divisor
    # distinguishes the current reading from the retired one.
    mu = np.asarray(model.predict(df), dtype=np.float64)
    pearson = float(np.sum(exposure * (y - mu) ** 2 / mu))
    edf = float(model._result.effective_df)
    assert float(table.attrs["phi"]) == pytest.approx(pearson / (exposure.sum() - edf), rel=1e-6)
    assert pearson / (len(df) - edf) == pytest.approx(measured["retired_phi_n_minus_edf"], rel=1e-4)
    assert float(table.attrs["phi"]) == pytest.approx(measured["phi"], rel=1e-5)

    expected = measured["rows"]
    assert len(table) == len(expected)
    got = table.reset_index(drop=True)
    assert list(zip(got["feature_a"], got["feature_b"], got["kind"], strict=True)) == [
        (r["feature_a"], r["feature_b"], r["kind"]) for r in expected
    ]
    for position, want in enumerate(expected):
        row = got.iloc[position]
        where = f"{want['feature_a']} x {want['feature_b']}"
        assert float(row["statistic"]) == pytest.approx(want["statistic"], rel=1e-4, abs=1e-4), (
            f"{where}.statistic"
        )
        assert float(row["z"]) == pytest.approx(want["z"], rel=1e-4, abs=1e-4), f"{where}.z"
        assert float(row["edf0"]) == pytest.approx(want["edf0"], abs=1e-2), f"{where}.edf0"
        # lambda0 at a clamped rung is a bracket edge, so it is pinned only to
        # the bracket rather than to the last printed digit.
        assert float(row["lambda0"]) == pytest.approx(want["lambda0"], rel=1e-2, abs=1e-6), (
            f"{where}.lambda0"
        )
        assert int(row["n_cells"]) == want["n_cells"], f"{where}.n_cells"
        assert bool(row["approx"]) == want["approx"], f"{where}.approx"

    for refit in measured["confirmatory_refits"]:
        confirm = SuperGLM(
            family="poisson",
            features=_guide_features(),
            interactions=[(refit["feature_a"], refit["feature_b"])],
            weight_semantics="frequency",
        )
        confirm.fit_reml(df, y, sample_weight=exposure)
        gain = model._result.deviance - confirm._result.deviance
        assert gain == pytest.approx(refit["deviance_gain"], rel=1e-3), (
            f"{refit['feature_a']} x {refit['feature_b']} refit gain"
        )
