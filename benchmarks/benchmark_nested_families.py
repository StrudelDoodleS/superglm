"""Evidence for keeping nested SuperLSS families as their own classes.

Each case fits a specialist family and the general family that nests it on the
same simulated rows, then records wall seconds per fit, the held-out log score
(lower is better) with the paired comparison, and how far the fitted means sit
from each other and from the truth. The prior-weight case fits gamma claim
averages with their claim counts as prior weights, beside the claim-level fit,
and records what the non-reproductive families say when given the same weights.

Run single-threaded through benchboard, e.g.

    benchboard heavy nested-families --cores 1 -- timeout 1800 \
        uv run python benchmarks/benchmark_nested_families.py --out receipt.json

The documentation page ``docs/explanation/nested-distributional-families.md``
quotes this script's receipt.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

import superglm
from superglm import (
    GammaLS,
    GaussianLS,
    GeneralizedGammaLSS,
    LogNormalLS,
    SuperLSS,
    TwoPieceNormalLSS,
    cat,
    s,
)

REGIONS = np.array(list("ABCDEF"))
REGION_EFFECT = dict(zip(REGIONS, [0.0, 0.15, -0.1, 0.3, -0.25, 0.05], strict=True))


def _frame(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": rng.uniform(0.0, 10.0, n),
            "x2": rng.uniform(-1.0, 1.0, n),
            "region": pd.Categorical(rng.choice(REGIONS, n)),
        }
    )


def _log_mean(frame: pd.DataFrame) -> np.ndarray:
    region = frame["region"].astype(str).map(REGION_EFFECT).to_numpy(dtype=float)
    return 7.0 + 0.4 * np.sin(frame["x1"].to_numpy()) + region


def _mean_terms(family):
    return family.mean(s("x1", kind="cr", k=8), cat("region"))


def _location_terms(family):
    return family.location(s("x1", kind="cr", k=8), cat("region"))


def _fit(build: Callable[[], SuperLSS], frame, y, **kwargs) -> tuple[SuperLSS, float, str]:
    start = time.perf_counter()
    try:
        model = build().fit_reml(frame, y, **kwargs)
    except Exception as exc:  # the refusal is the evidence in the prior-weight case
        return None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}"
    return model, time.perf_counter() - start, "ok"


def _held_out(model: SuperLSS, frame, y) -> float:
    return float(np.mean(model.scores(frame, y, which=("log",))["log"]))


def _compare(model: SuperLSS, other: SuperLSS, frame, y) -> dict[str, float]:
    overall = model.compare(other, frame, y, which="log").overall
    return {key: float(value) for key, value in dict(overall).items()}


def _case(name, candidates, train, y_train, test, y_test, true_mean) -> dict:
    """Fit every candidate twice (warm-up compile, then timed) and score it."""
    rows = {}
    fitted = {}
    for label, build in candidates.items():
        _fit(build, train.iloc[:400], y_train[:400])  # compile outside the timing
        model, seconds, status = _fit(build, train, y_train)
        row = {"wall_s": round(seconds, 3), "status": status}
        if model is not None:
            mean = model.predict(test)
            row["held_out_log_score"] = _held_out(model, test, y_test)
            row["mean_rel_error_vs_truth_max"] = float(np.max(np.abs(mean / true_mean - 1)))
            row["parameters_at_first_test_row"] = {
                k: float(v) for k, v in model.predict_parameters(test.iloc[:1]).iloc[0].items()
            }
            fitted[label] = (model, mean)
        rows[label] = row
    labels = list(fitted)
    reference = labels[0]
    for label in labels[1:]:
        model, mean = fitted[label]
        rows[label]["vs_" + reference] = _compare(model, fitted[reference][0], test, y_test)
        rows[label]["mean_rel_diff_vs_" + reference + "_max"] = float(
            np.max(np.abs(mean / fitted[reference][1] - 1))
        )
    print(name, json.dumps(rows, indent=1, default=str), flush=True)
    return rows


def gamma_cases(rng, n):
    train, test = _frame(rng, n), _frame(rng, n)
    out = {}
    for label, log_cv in (
        ("gamma_constant_cv", lambda f: np.full(len(f), np.log(1.2))),
        ("gamma_cv_varies_with_x2", lambda f: 0.5 * f["x2"].to_numpy()),
    ):
        draws = {}
        for part, frame in (("train", train), ("test", test)):
            mean, cv = np.exp(_log_mean(frame)), np.exp(log_cv(frame))
            draws[part] = rng.gamma(1.0 / cv**2, mean * cv**2)
        scale_terms = () if label == "gamma_constant_cv" else ("x2",)
        gamma, gg = GammaLS(), GeneralizedGammaLSS()
        candidates = {
            "GammaLS": lambda: SuperLSS(gamma, _mean_terms(gamma), gamma.scale(*scale_terms)),
            "GG_shape_free": lambda: SuperLSS(
                gg, _mean_terms(gg), gg.scale(*scale_terms), gg.shape()
            ),
        }
        if scale_terms:
            candidates["GG_shape_on_x2"] = lambda: SuperLSS(
                gg, _mean_terms(gg), gg.scale(*scale_terms), gg.shape(*scale_terms)
            )
        out[label] = _case(
            label, candidates, train, draws["train"], test, draws["test"], np.exp(_log_mean(test))
        )
    return out


def lognormal_case(rng, n):
    train, test = _frame(rng, n), _frame(rng, n)
    sigma = {
        k: np.exp(np.log(0.9) + 0.3 * f["x2"].to_numpy())
        for k, f in (("train", train), ("test", test))
    }
    draws = {
        k: np.exp(_log_mean(f) - sigma[k] ** 2 / 2 + sigma[k] * rng.standard_normal(len(f)))
        for k, f in (("train", train), ("test", test))
    }
    ln, gg = LogNormalLS(), GeneralizedGammaLSS()
    candidates = {
        "LogNormalLS": lambda: SuperLSS(ln, _mean_terms(ln), ln.scale("x2")),
        "GG_shape_fixed_0": lambda: SuperLSS(
            gg, _mean_terms(gg), gg.scale("x2"), gg.shape(intercept=False)
        ),
        "GG_shape_free": lambda: SuperLSS(gg, _mean_terms(gg), gg.scale("x2"), gg.shape()),
    }
    return {
        "lognormal": _case(
            "lognormal",
            candidates,
            train,
            draws["train"],
            test,
            draws["test"],
            np.exp(_log_mean(test)),
        )
    }


def gaussian_case(rng, n):
    train, test = _frame(rng, n), _frame(rng, n)
    draws = {
        k: _log_mean(f)
        + np.exp(np.log(0.9) + 0.3 * f["x2"].to_numpy()) * rng.standard_normal(len(f))
        for k, f in (("train", train), ("test", test))
    }
    ga, tp = GaussianLS(), TwoPieceNormalLSS()
    candidates = {
        "GaussianLS": lambda: SuperLSS(ga, _location_terms(ga), ga.scale("x2")),
        "TwoPieceNormal_skew_fixed_0": lambda: SuperLSS(
            tp, _location_terms(tp), tp.scale("x2"), tp.skew(intercept=False)
        ),
        "TwoPieceNormal_skew_free": lambda: SuperLSS(
            tp, _location_terms(tp), tp.scale("x2"), tp.skew()
        ),
    }
    return {
        "gaussian": _case(
            "gaussian", candidates, train, draws["train"], test, draws["test"], _log_mean(test)
        )
    }


def prior_weight_case(rng, n_policies):
    """Claim averages with claim counts as prior weights, against claim-level rows."""
    policies = _frame(rng, n_policies)
    counts = 1 + rng.poisson(1.5, n_policies)
    mean, cv = np.exp(_log_mean(policies)), 1.2
    claims = rng.gamma(1.0 / cv**2, np.repeat(mean, counts) * cv**2)
    averages = np.add.reduceat(claims, np.r_[0, np.cumsum(counts)[:-1]]) / counts
    claim_frame = policies.loc[policies.index.repeat(counts)].reset_index(drop=True)
    gamma, gg, ln = GammaLS(), GeneralizedGammaLSS(), LogNormalLS()
    out = {}
    for label, build, frame, y, weight in (
        (
            "GammaLS_claim_rows",
            lambda: SuperLSS(gamma, _mean_terms(gamma), gamma.scale()),
            claim_frame,
            claims,
            None,
        ),
        (
            "GammaLS_averages_prior_weight",
            lambda: SuperLSS(gamma, _mean_terms(gamma), gamma.scale()),
            policies,
            averages,
            counts.astype(float),
        ),
        (
            "GG_averages_prior_weight",
            lambda: SuperLSS(gg, _mean_terms(gg), gg.scale(), gg.shape()),
            policies,
            averages,
            counts.astype(float),
        ),
        (
            "LogNormalLS_averages_prior_weight",
            lambda: SuperLSS(ln, _mean_terms(ln), ln.scale()),
            policies,
            averages,
            counts.astype(float),
        ),
    ):
        model, seconds, status = _fit(build, frame, y, sample_weight=weight)
        row = {"wall_s": round(seconds, 3), "status": status, "rows": len(y)}
        if model is not None:
            fitted = model.predict_parameters(policies)
            row["mean_rel_error_vs_truth_max"] = float(
                np.max(np.abs(fitted["mean"].to_numpy() / mean - 1))
            )
            row["cv_at_first_policy"] = float(fitted["scale"].iloc[0])
            row["_mean"] = fitted["mean"].to_numpy()
        out[label] = row
    reference = out["GammaLS_claim_rows"].pop("_mean")
    other = out["GammaLS_averages_prior_weight"].pop("_mean")
    out["GammaLS_averages_prior_weight"]["mean_rel_diff_vs_claim_rows_max"] = float(
        np.max(np.abs(other / reference - 1))
    )
    print("prior_weights", json.dumps(out, indent=1), flush=True)
    return {"prior_weights": out}


def small_sample_case(rng, n_train, replicates):
    """Specialist and general family where the restriction holds, at a small sample size."""
    test = pd.DataFrame({"region": pd.Categorical(rng.choice(REGIONS, 5_000))})
    true_mean = np.exp(_log_mean(test.assign(x1=0.0)))
    gamma, ln, gg = GammaLS(), LogNormalLS(), GeneralizedGammaLSS()
    laws = {
        "gamma_cv_1.2": (
            lambda m, size: rng.gamma(1 / 1.44, m * 1.44, size),
            {"GammaLS": lambda: SuperLSS(gamma, gamma.mean(cat("region")), gamma.scale())},
        ),
        "lognormal_sigma_0.9": (
            lambda m, size: np.exp(np.log(m) - 0.405 + 0.9 * rng.standard_normal(size)),
            {"LogNormalLS": lambda: SuperLSS(ln, ln.mean(cat("region")), ln.scale())},
        ),
    }
    out = {}
    for law, (draw, specialist) in laws.items():
        candidates = dict(specialist)
        candidates["GG_shape_free"] = lambda: SuperLSS(
            gg, gg.mean(cat("region")), gg.scale(), gg.shape()
        )
        differences, shapes, errors, failures = [], [], {k: [] for k in candidates}, 0
        y_test = draw(true_mean, len(test))
        for _ in range(replicates):
            train = pd.DataFrame({"region": pd.Categorical(rng.choice(REGIONS, n_train))})
            y = draw(np.exp(_log_mean(train.assign(x1=0.0))), n_train)
            fits = {k: _fit(build, train, y)[0] for k, build in candidates.items()}
            if any(model is None for model in fits.values()):
                failures += 1
                continue
            first, general = fits[next(iter(specialist))], fits["GG_shape_free"]
            differences.append(
                general.compare(first, test, y_test, which="log").overall["mean_diff"]
            )
            shapes.append(float(general.predict_parameters(test.iloc[:1])["shape"].iloc[0]))
            for k, model in fits.items():
                errors[k].append(
                    float(np.sqrt(np.mean((model.predict(test) / true_mean - 1) ** 2)))
                )
        d = np.asarray(differences)
        out[law] = {
            "train_rows": n_train,
            "replicates": replicates,
            "failed_fits": failures,
            "held_out_log_score_gg_minus_specialist_mean": float(d.mean()),
            "held_out_log_score_gg_minus_specialist_se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "gg_worse_in_fraction_of_replicates": float(np.mean(d > 0)),
            "gg_shape_estimate_mean": float(np.mean(shapes)),
            "gg_shape_estimate_sd": float(np.std(shapes, ddof=1)),
            "mean_rms_rel_error": {k: float(np.mean(v)) for k, v in errors.items()},
        }
    print("small_sample", json.dumps(out, indent=1), flush=True)
    return {"small_sample": out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--small-rows", type=int, default=300)
    parser.add_argument("--replicates", type=int, default=40)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    receipt = {
        "superglm": superglm.__version__,
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "rows": args.rows,
        "seed": args.seed,
    }
    for build in (gamma_cases, lognormal_case, gaussian_case):
        receipt.update(build(rng, args.rows))
    receipt.update(prior_weight_case(rng, args.rows // 2))
    receipt.update(small_sample_case(rng, args.small_rows, args.replicates))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(receipt, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
