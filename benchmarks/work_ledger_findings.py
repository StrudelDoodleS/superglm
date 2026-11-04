"""Run one fit on the freMTPL2 book under the work ledger and print what it did.

    python benchmarks/work_ledger_findings.py CASE [--rows N] [--data-dir DIR] [--json FILE]

Cases (every fit is preceded by an untimed 3,000-row fit that compiles the kernels):

    glm-exact      Poisson GAM, ``fit_reml``, exact Gram builds
    glm-discrete   the same model with ``discrete=True``
    shape-a        configuration A of ``shape_rules_fit_cost.py`` (three monotone rules)
    lss-nb         SuperLSS, negative binomial location-scale, three splines per parameter
    tweedie        Tweedie(1.5) pure premium, claim cost per year of exposure, ``fit_reml``

The ledger lives in ``superglm.diagnostics.work_ledger``; this script only builds
the fits.  Claim counts are recounted from the severity file as in
``examples/fremtpl2_data.py``.  The output is the ledger's table, then one JSON line.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "examples"))

import fremtpl2_data  # noqa: E402
import shape_rules_fit_cost as shape_rules  # noqa: E402

from superglm import Categorical, PSpline, SuperGLM, SuperLSS  # noqa: E402
from superglm.diagnostics.work_ledger import work_ledger  # noqa: E402

WARMUP_ROWS = 3000
CASES = ("glm-exact", "glm-discrete", "shape-a", "lss-nb", "tweedie")


def load_book(data_dir, rows):
    """The recounted book as a frame, optionally its first ``rows`` rows."""
    frame = fremtpl2_data.load_corrected_freq(data_dir)
    frame["Exposure"] = frame["Exposure"].clip(lower=shape_rules.EXPOSURE_FLOOR)
    frame["LogDensity"] = np.log1p(frame["Density"].to_numpy(dtype=np.float64))
    for column in ("DrivAge", "VehAge", "BonusMalus", "VehPower"):
        frame[column] = frame[column].astype(np.float64)
    for column in ("Region", "VehBrand"):
        frame[column] = frame[column].astype(str)
    return frame if rows is None else frame.iloc[:rows].reset_index(drop=True)


def plain_features():
    """The shape-rule configurations' terms, with no rule on any of them."""
    return {
        "DrivAge": PSpline(n_knots=8),
        "VehAge": PSpline(n_knots=12),
        "BonusMalus": PSpline(n_knots=12, knot_strategy="quantile_tempered", knot_alpha=0.2),
        "LogDensity": PSpline(n_knots=8),
        "VehPower": PSpline(n_knots=6),
        "Region": Categorical(),
        "VehBrand": Categorical(),
    }


def glm_fit(discrete, family="poisson", features=plain_features):
    def fit(frame):
        offset = np.log(frame["Exposure"].to_numpy(np.float64))
        y = frame["ClaimNb"].to_numpy(np.float64)
        SuperGLM(family=family, features=features(), discrete=discrete).fit_reml(
            frame, y, offset=offset
        )

    return fit


def tweedie_fit(frame):
    from superglm import families

    exposure = frame["Exposure"].to_numpy(np.float64)
    y = frame["ClaimTotal"].to_numpy(np.float64) / exposure
    SuperGLM(family=families.tweedie(p=1.5), features=plain_features()).fit_reml(
        frame, y, sample_weight=exposure
    )


def shape_a_fit(frame):
    shape_rules.fit_model(shape_rules.build_features("A"), frame, "fit_reml", False)


def lss_fit(frame):
    from superglm.distributional import NegativeBinomialLS, Predictor
    from superglm.distributional.binding import _bind_predictor_template
    from superglm.features import Spline

    family = NegativeBinomialLS()
    columns = ["DrivAge", "VehAge", "BonusMalus"]
    model = SuperLSS(
        family,
        *(
            _bind_predictor_template(
                family, Predictor(p.name, {c: Spline(n_knots=8) for c in columns})
            )
            for p in family.parameters
        ),
    )
    offset = np.log(frame["Exposure"].to_numpy(np.float64))
    model.fit_reml(
        frame[columns],
        frame["ClaimNb"].to_numpy(np.float64),
        offsets={family.parameters[0].name: offset},
    )


RUNNERS = {
    "glm-exact": glm_fit(False),
    "glm-discrete": glm_fit(True),
    "shape-a": shape_a_fit,
    "lss-nb": lss_fit,
    "tweedie": tweedie_fit,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--rows", type=int, default=None, help="use the first ROWS rows")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--json", default=None, help="append the JSON line to this file")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args(argv)
    data_dir = fremtpl2_data.find_data_dir(args.data_dir)
    if args.case == "shape-a":
        frame, _ = shape_rules.load_split(data_dir, args.rows)
    else:
        frame = load_book(data_dir, args.rows)
    run = RUNNERS[args.case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run(frame.iloc[:WARMUP_ROWS])
        started = time.perf_counter()
        with work_ledger() as ledger:
            run(frame)
        wall = time.perf_counter() - started
    print(f"{args.case}: {len(frame)} rows, wall {wall:.1f} s with the ledger on")
    print(ledger.report(top=args.top))
    if ledger.errors or ledger.skipped:
        print("ledger faults:", ledger.errors, "unresolved entry points:", ledger.skipped)
    record = {"case": args.case, "rows": len(frame), "wall_with_ledger_s": wall, **ledger.as_dict()}
    print(json.dumps(record))
    if args.json:
        with open(args.json, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
