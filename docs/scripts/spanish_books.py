"""The two public Spanish books behind the held-out smoothing guide, prepared for fitting.

Fetch them first, from the repository root:
    uv run python scripts/fetch_spanish_books.py --dest data/

That script's docstring carries each book's credit line, licence and link.
This module holds the preparation every figure script shares: which rows are
fitted and why, the model of each book, the outer folds, and the score.

Spanish health portfolio (Lledó, Espinosa Adamez and Perez Gimenez, Mendeley
386vmj2tbk v4, CC BY 4.0).  One row per insured person and year, 2017 to 2019;
the count is medical services used, the exposure the year's insured time.  Two
kinds of row are left out: rows with zero exposure, which nonetheless carry
services and cannot be put on a rate, and the dental-only product, which
records almost no medical services and so has no frequency to model.  Ages are
capped at 90 and years with the insurer at 40.  Columns known only after the
cover is written (premium, cost of claims, lapse) are never terms.

Spanish motor book by coverage (Espinosa, Lledó and Atance, Mendeley
sw4jmdb2sm v1, CC BY 4.0).  One row per insured and year, 2022 to 2024; the
count is liability claims, the exposure the liability cover's insured time.
Rows with zero exposure, or without a vehicle value, vehicle age or licence
age, are left out; each continuous factor is capped at fixed limits, the same
in every fold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from superglm import Categorical, Spline

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
STYLE = ROOT / "docs" / "_static" / "superglm.mplstyle"
IMAGES = ROOT / "docs" / "images" / "ncv"
#: Fold results kept between runs; delete it, or pass --refit, after changing the library.
RESULTS = DATA / "ncv_examples"
#: Seed of every outer split.
SEED = 20261126

INK = "#15171C"
RED = "#D6402B"
YELLOW = "#F4B942"
MUTED = "#6B7280"
BLUE = "#2E5AAC"
BAND = "#F3F4F6"


@dataclass
class Book:
    """Rating factors, counts, exposure in years, and the keys rows can be held out by."""

    X: pd.DataFrame
    y: np.ndarray
    exposure: np.ndarray
    keys: dict[str, np.ndarray]
    raw: pd.DataFrame
    #: Rows of the file left out, by reason.
    dropped: dict[str, int]


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise SystemExit(
            f"{path} is missing: run `uv run python scripts/fetch_spanish_books.py --dest data/` first"
        )
    return pd.read_parquet(path)


def load_health() -> Book:
    """The health portfolio's fitted rows; ``keys`` holds ``policy`` (a family or a collective) and ``year``."""
    raw = _read("spanish_health.parquet")
    keep = (raw["exposure_time"] > 0) & (raw["type_product"] != "D")
    d = raw[keep].reset_index(drop=True)
    X = pd.DataFrame(
        {
            "age": d["age"].clip(0, 90).astype(float),
            "seniority": d["seniority_insured"].clip(0, 40).astype(float),
            "gender": d["gender"],
            "product": d["type_product"],
            "policy_type": d["type_policy_dg"],
            "reimbursement": d["reimbursement"],
            "channel": d["distribution_channel"],
            "habitat": d["C_H"].fillna("unknown"),
            "climate": d["C_C"].fillna("unknown"),
        }
    )
    return Book(
        X=X,
        y=d["n_medical_services"].to_numpy(float),
        exposure=d["exposure_time"].to_numpy(float),
        keys={"policy": d["ID_policy"].to_numpy(), "year": d["period"].to_numpy()},
        raw=d,
        dropped={
            "zero_exposure": int((raw["exposure_time"] <= 0).sum()),
            "services_on_zero_exposure": int(
                raw.loc[raw["exposure_time"] <= 0, "n_medical_services"].sum()
            ),
            "dental_only": int(((raw["type_product"] == "D") & (raw["exposure_time"] > 0)).sum()),
        },
    )


def health_terms() -> dict:
    """Use of medical services by age for women and the difference for men, a knot about every
    two years of age (the spline form of a Whittaker-Henderson graduation of an age table), years
    with the insurer, and the product and place factors known when the cover is written."""
    factors = ("product", "policy_type", "reimbursement", "channel", "habitat", "climate")
    return {
        "features": {
            "age": Spline(kind="ps", k=48),
            "seniority": Spline(kind="ps", k=20),
            "gender": Categorical(),
            **{name: Categorical() for name in factors},
        },
        "interactions": [("age", "gender")],
    }


def _lump(values: pd.Series, min_rows: int) -> pd.Series:
    counts = values.value_counts()
    return values.where(values.isin(counts.index[counts >= min_rows]), "OTHER")


def load_motor() -> Book:
    """The motor book's fitted rows for liability claims; ``keys`` holds ``insured`` and ``year``."""
    raw = _read("spanish_motor_coverage.parquet")
    keep = (
        (raw["liability_exposure"] > 0)
        & raw["vehicle_value"].notna()
        & raw["vehicle_age"].notna()
        & raw["age_driving_licence"].notna()
    )
    d = raw[keep].reset_index(drop=True)
    X = pd.DataFrame(
        {
            "driver_age": d["driver_age"].clip(18, 85).astype(float),
            "vehicle_age": d["vehicle_age"].clip(0, 55).astype(float),
            "licence_age": d["age_driving_licence"].clip(18, 50).astype(float),
            "log_value": np.log(d["vehicle_value"].clip(3_000, 100_000)),
            "power_weight": d["power_to_weight_ratio"].clip(5, 22).astype(float),
            "policy_type": d["policy_type"],
            "payment": d["payment_frequency"],
            "bonus": d["bonus_score"],
            "fuel": d["fuel_type"].fillna("unknown"),
            "municipality": d["municipality_type"],
            "area": d["circulation_area"],
            "business": d["business_type"],
            "brand": _lump(d["vehicle_brand"], 2_000),
        }
    )
    return Book(
        X=X,
        y=d["liability_claims"].to_numpy(float),
        exposure=d["liability_exposure"].to_numpy(float),
        keys={"insured": d["insured_id"].to_numpy(), "year": d["year"].to_numpy()},
        raw=d,
        dropped={
            "zero_exposure": int((raw["liability_exposure"] <= 0).sum()),
            "incomplete": int((raw["liability_exposure"] > 0).sum() - keep.sum()),
        },
    )


def motor_terms() -> dict:
    """Every rating factor in the file known before the loss, with room for fine structure in each curve."""
    curves = {
        "driver_age": 25,
        "vehicle_age": 25,
        "licence_age": 20,
        "log_value": 20,
        "power_weight": 20,
    }
    factors = (
        "policy_type",
        "payment",
        "bonus",
        "fuel",
        "municipality",
        "area",
        "business",
        "brand",
    )
    return {
        "features": {
            **{name: Spline(kind="cr", k=k) for name, k in curves.items()},
            **{name: Categorical() for name in factors},
        }
    }


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Unit Poisson deviance of each count ``y`` against its expected count ``mu``."""
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    ratio = np.where(y > 0, y / mu, 1.0)
    return 2.0 * (y * np.log(ratio) - (y - mu))


def poisson_score(model, X, y, *, sample_weight, offset) -> dict:
    """``cross_validate`` scorer: held-out Poisson deviance per row, whatever family the fold was
    fitted under, and the fold's scale (1 for a family whose scale is known)."""
    del sample_weight
    deviance = poisson_deviance(y, model.predict(X, offset=offset))
    return {"poisson_deviance": float(deviance.mean()), "scale": float(model.result.phi)}


def turning_points(curve: np.ndarray) -> int:
    """How many times a curve changes direction."""
    step = np.diff(curve)
    return int(np.sum(step[1:] * step[:-1] < 0))


def number(tag: str, value) -> None:
    """Print one number the guide may quote, on its own tagged line."""
    print(f"NUMBER {tag} {value}", flush=True)


def stored(name: str, refit: bool, compute) -> dict:
    """The fold results stored under ``RESULTS / name``, computed and stored first when absent or ``refit``."""
    path = RESULTS / f"{name}.json"
    if path.exists() and not refit:
        return json.loads(path.read_text(encoding="utf-8"))
    result = compute()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
