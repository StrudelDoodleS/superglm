"""Fetch the CASdatasets 2017 pricing-game training book as one policy-year table.

Writes ``<dest>/cas_pg17.parquet``: one row per policy-year of ``pg17trainpol``
(100,000 rows) with that policy-year's claims from ``pg17trainclaim`` (14,243
claims) summed onto it.  It is the vehicle-hierarchy book for the nested
credibility benchmarks, ``re(vh_make) + re(make_model)`` beside ordinary terms.

Licence.  CASdatasets is GPL (>= 2) per its DESCRIPTION.  This script downloads
only the two serialized data files and loads them with R; it never reads the
package's R source.  The data, and anything derived from them, must never be
committed: ``data/`` is git-ignored and has to stay that way.  The data come
from the pricing game of the French Institute of Actuaries (16 November 2017)
and are documented at
https://dutangc.github.io/CASdatasets/reference/pricingame.html.

Provenance.  Both files are fetched from one commit of dutangc/CASdatasets
rather than from ``master``, so each URL names fixed bytes, and each file is
pinned by SHA-256, measured 2026-09-26.  On that day the git blob ids of the
downloads also matched the ones the GitHub contents API lists for ``master``.
A mismatch is a hard error.

Conversion.  ``Rscript`` loads each ``.rda`` and writes it with ``write.csv``
beside it; polars reads the CSVs back.  Measured on these bytes, the hop is
lossless: every double (``pol_bonus``, ``claim_amount``) round-trips exactly
through write.csv's 15 significant digits, and every string is ASCII.
write.csv quotes every factor value and never quotes ``NA``, so the 66,814
empty ``drv_sex2`` levels (no second driver) stay ``""`` while the one missing
``vh_age`` reads as null.

Assumptions and derived columns:

* The claim table has no ``id_policy``; its policy is ``id_client-id_vehicle``.
  :func:`link_claims` verifies that against every policy row before the join
  on ``(id_policy, id_year)``.
* ``claim_count`` sums ``claim_nb`` (1 on every claim row) and ``claim_amount``
  sums the amounts.  Negative amounts are legal recourse on claims where the
  insured was not liable, per the documentation; they are kept, and counted.
* There is no exposure column.  Every row is one policy for the single coverage
  year ``Year 0``, so ``exposure`` is set to 1.0 per policy-year.  That is an
  assumption, not data: a mid-year start or cancellation is invisible here.
* ``make_model = vh_make + '|' + vh_model``, the second level of the vehicle
  hierarchy under ``vh_make``.
* Nothing is imputed.  The report counts nulls and zeros per vehicle column,
  because a zero ``vh_weight`` or ``vh_value`` is not a physical measurement.

Usage::

    python scripts/fetch_cas_pg17.py --dest data/
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import polars as pl

#: The last commit to touch either data file (2024-05-27).
COMMIT = "ef06f44b8669908925b5a590c8a2313723cf504c"
BASE_URL = f"https://raw.githubusercontent.com/dutangc/CASdatasets/{COMMIT}/data"
OUTPUT_NAME = "cas_pg17.parquet"

#: Arguments: the raw directory, then the names of the data frames to export.
#: Each ``.rda`` must hold exactly the one object its file is named after.
EXPORT_TO_CSV = """
args <- commandArgs(trailingOnly = TRUE)
for (name in args[-1]) {
  env <- new.env()
  stopifnot(identical(load(file.path(args[1], paste0(name, ".rda")), envir = env), name))
  write.csv(env[[name]], file.path(args[1], paste0(name, ".csv")),
            row.names = FALSE, fileEncoding = "UTF-8")
}
"""


class FetchError(RuntimeError):
    """A download missed its pin, or the two tables do not link."""


@dataclass(frozen=True)
class RdaTable:
    """One pinned ``.rda`` file holding one data frame of the same name."""

    name: str
    sha256: str
    size: int
    #: Row count stated by the documentation.
    rows: int
    #: Factors whose levels look numeric, read as text so an INSEE code keeps
    #: its leading zero and a model called ``306`` stays a level.
    code_columns: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.name}.rda"


POLICIES = RdaTable(
    name="pg17trainpol",
    sha256="7b0010b021215cf48cdeeccf1f10d44d2d9f9e4f5b60b28d3704cbf1af373a7b",
    size=3233631,
    rows=100_000,
    code_columns=("pol_insee_code", "vh_model"),
)
CLAIMS = RdaTable(
    name="pg17trainclaim",
    sha256="2eba91dc2243339e5e8fe1c20e36c76d0c8a9c169bb49ce8e4e7ef77c48e492a",
    size=118553,
    rows=14_243,
)


def sha256_of(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_pinned(table: RdaTable, raw_dir: Path) -> None:
    """Leave a copy of *table*'s ``.rda`` matching its pin in *raw_dir*.

    The body lands in a ``.part`` file and moves into place only once it
    matches, so a file under the real name has always been verified.
    """
    path = raw_dir / f"{table.name}.rda"
    if path.exists() and sha256_of(path) == table.sha256:
        print(f"{table.name}: cached copy matches pin {table.sha256[:12]}", flush=True)
        return
    print(f"{table.name}: downloading {table.url}", flush=True)
    partial = path.with_suffix(".rda.part")
    with urllib.request.urlopen(table.url, timeout=300) as response:
        partial.write_bytes(response.read())
    found = sha256_of(partial)
    size = partial.stat().st_size
    if found != table.sha256:
        partial.unlink()
        raise FetchError(
            f"{table.name}: sha256 {found[:12]} at {size} bytes, pinned {table.sha256[:12]} at "
            f"{table.size} bytes. The URL names a fixed commit, so this is a damaged transfer "
            f"or a changed host rather than a new release: re-run, and do not re-pin."
        )
    partial.replace(path)
    print(f"{table.name}: verified sha256 {found[:12]}", flush=True)


def read_export(table: RdaTable, raw_dir: Path) -> pl.DataFrame:
    frame = pl.read_csv(
        raw_dir / f"{table.name}.csv",
        null_values="NA",
        infer_schema_length=None,
        schema_overrides=dict.fromkeys(table.code_columns, pl.String),
    )
    if frame.height != table.rows:
        raise FetchError(f"{table.name}: {frame.height} rows, documented {table.rows}")
    return frame


def link_claims(policies: pl.DataFrame, claims: pl.DataFrame) -> pl.DataFrame:
    """Return *claims* with the ``id_policy`` of the policy each belongs to.

    The claim table carries ``id_client`` and ``id_vehicle`` but no
    ``id_policy``.  That the policy table's ``id_policy`` is exactly
    ``id_client-id_vehicle`` is checked on every row rather than taken from the
    documented format, together with what summing claims onto policy-years
    needs: no claim listed twice, and no claim whose policy-year is missing.
    The policy side's uniqueness is checked by the join itself.
    """
    policy_id = pl.concat_str("id_client", "id_vehicle", separator="-")
    mislabelled = policies.filter(pl.col("id_policy") != policy_id).height
    if mislabelled:
        raise FetchError(f"{mislabelled} policies whose id_policy is not id_client-id_vehicle")
    claims = claims.with_columns(id_policy=policy_id)
    repeated = claims.select(pl.struct("id_policy", "id_year", "id_claim").is_duplicated().sum())
    if repeated.item():
        raise FetchError(f"{repeated.item()} claim rows share an (id_policy, id_year, id_claim)")
    orphans = claims.join(policies, on=["id_policy", "id_year"], how="anti").height
    if orphans:
        raise FetchError(f"{orphans} claims name a policy-year absent from {POLICIES.name}")
    return claims


def policy_years(policies: pl.DataFrame, claims: pl.DataFrame) -> pl.DataFrame:
    """One row per policy-year, in policy order, with its claims summed on."""
    per_policy_year = claims.group_by("id_policy", "id_year").agg(
        claim_count=pl.col("claim_nb").sum(),
        claim_amount=pl.col("claim_amount").sum(),
    )
    return policies.join(
        per_policy_year,
        on=["id_policy", "id_year"],
        how="left",
        validate="1:1",
        maintain_order="left",
    ).with_columns(
        pl.col("claim_count").fill_null(0),
        pl.col("claim_amount").fill_null(0.0),
        exposure=pl.lit(1.0),
        make_model=pl.concat_str("vh_make", "vh_model", separator="|"),
    )


def report(frame: pl.DataFrame, claims: pl.DataFrame) -> None:
    """Print the counts a credibility benchmark on this table depends on."""
    claims_per_model = frame.group_by("make_model").agg(pl.col("claim_count").sum())
    vehicle = [column for column in frame.columns if column.startswith("vh_")]
    measured = [column for column in vehicle if frame.schema[column].is_numeric()]
    total_claims = frame["claim_count"].sum()
    print(f"rows {frame.height}, policy-years with a claim {(frame['claim_count'] > 0).sum()}")
    print(
        f"levels: vh_make {frame['vh_make'].n_unique()}, "
        f"make_model {claims_per_model.height}, "
        f"pol_insee_code {frame['pol_insee_code'].n_unique()}"
    )
    print(
        f"claims {total_claims} over exposure {frame['exposure'].sum():.0f}: "
        f"frequency {total_claims / frame['exposure'].sum():.5f} per policy-year"
    )
    print(f"make_model levels with no claim {(claims_per_model['claim_count'] == 0).sum()}")
    print(
        f"claim rows with a negative amount {(claims['claim_amount'] < 0).sum()}, "
        f"with a zero amount {(claims['claim_amount'] == 0).sum()}; "
        f"policy-years with a negative total {(frame['claim_amount'] < 0).sum()}"
    )
    print(f"vehicle nulls {frame.select(pl.col(vehicle).null_count()).row(0, named=True)}")
    print(f"vehicle zeros {frame.select((pl.col(measured) == 0).sum()).row(0, named=True)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data"),
        help=f"directory to write {OUTPUT_NAME} to",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="directory for the pinned .rda files and their CSV exports (defaults to <dest>/raw)",
    )
    args = parser.parse_args(argv)

    raw_dir = args.raw_dir or args.dest / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        fetch_pinned(POLICIES, raw_dir)
        fetch_pinned(CLAIMS, raw_dir)
        subprocess.run(
            ["Rscript", "-e", EXPORT_TO_CSV, str(raw_dir), POLICIES.name, CLAIMS.name],
            check=True,
        )
        policies = read_export(POLICIES, raw_dir)
        claims = link_claims(policies, read_export(CLAIMS, raw_dir))
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    frame = policy_years(policies, claims)
    out = args.dest / OUTPUT_NAME
    partial = args.dest / f".{OUTPUT_NAME}.partial"
    frame.write_parquet(partial)
    partial.replace(out)
    print(f"wrote {out} ({frame.height} rows, {frame.width} columns)", flush=True)
    report(frame, claims)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
