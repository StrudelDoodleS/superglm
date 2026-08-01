"""Two readings that turn a screening rank into a fitting decision.

`z` says whether a pair carries signal.  It does not say whether refitting that
pair will help, and at wide factors the two answers come apart: table 1 has
spiky truths whose z is comfortably "significant" at every width and whose
fixed refit costs holdout MSE anyway.  A screen that only reports z hands the
caller a number that is correct and points the wrong way.  Table 4 measures the
screen and the refit on ONE train split so the two can be quoted side by side --
which is the only way such a sentence should ever be written.

Two derived numbers close that gap.  Neither is new machinery, but they do not
cost the same to obtain: the first is arithmetic on the returned screening row,
the second needs one pass over the residuals the screen does not hand back.

1. WORTH THRESHOLD.  Mallows' Cp says a term earns its place when its score
   beats twice the df it spends, `T/phi > 2*edf0`.  The evaluation guide already
   notes `gain - 2*edf` as a scoring variant; what is added here is the same
   rule on PSST's own z scale, plus a measurement of where the crossing lands.
   Since `z = (T/phi - edf0) / sqrt(2*edf0)`,

       T/phi > 2*edf0   <=>   z > sqrt(edf0 / 2)

   The bar GROWS with the block's df -- z > 4.95 at 8x8, z > 28.3 at 41x41 --
   which is exactly what a constant cutoff cannot express.  Both sides read the
   SAME `edf0`, and for an unpenalized `cat_cat` that is the block's achieved
   rank, not `(L-1)^2`; this file therefore takes `edf0` off the screening row
   rather than re-deriving it (see `_run_gate_ladder`).

2. CONCENTRATION.  Every chi^2-family score reads only the TOTAL: PSST's `T`,
   FAST's RSS gain, Information Value, mutual information, deviance change.
   None reads the shape, so none can separate 5 live cells from 1681 faintly
   live ones carrying the same total.  The participation ratio of the per-cell
   contributions does:

       P = (sum_c t_c)^2 / sum_c t_c^2,    t_c = n_c * mean_c^2 / phi

   For k independent chi^2_1 contributions E[t] = 1 and E[t^2] = 3, so the null
   sits at P = k/3 for large k.  Reporting `P / (k/3)` gives ~1 for noise AND
   for genuinely diffuse truths, and near zero when a handful of cells carry
   everything.  Unlike the gate, this is NOT arithmetic on the screening row:
   `screen_interactions` returns aggregates (`statistic`, `z`, `edf0`), not the
   per-cell contributions, so `cell_contributions` below recomputes them from
   the mains-model residuals and the joint cell index.

Run:  uv run python benchmarks/screening_worth_gate.py [--reps 3]

Four tables come out, because the threshold alone answers only half of it:

  1. the gate ladder -- table width x effect size, the screen's z, and the
     holdout change from actually refitting.  The question is whether the sign
     of the holdout change flips where `z / sqrt(edf0/2)` crosses 1.  The effect
     sizes at each width were CHOSEN so z straddles the threshold there, which
     locates the crossing but also guarantees a grid no constant cutoff can
     track -- read the margin over fixed cutoffs with that in mind;
  2. concentration at matched z -- spiky and diffuse truths tuned so their z
     values coincide, which is the only honest way to ask whether `P` carries
     information `z` does not;
  3. the sparse payoff -- if `P` says the signal is concentrated, does fitting
     only those cells pay?  Cells are ranked on TRAINING residuals only;
     ranking them on the full sample is the target leakage that makes
     supervised binning look better than it is.  The widest arm is the full
     fixed refit, so the whole row -- sparse arms and full refit alike -- comes
     from one seed and one split;
  4. the same pair through three model classes -- mains, a fixed interaction,
     and the cell partially pooled, plus the screen on that same split.  The
     gate says what NOT to do with a wide pair; this says whether the pair is
     worth having at all in some other class.

Tables 3 and 4 are NOT reported in the evaluation guide.  Figures that used to
stand in their place were not reproducible from this file and were removed
rather than corrected; each 41x41 refit is a tens-of-minutes job and the pair of
tables is a multi-hour one, which is the likeliest reason they went stale.  The
arms are kept, and fixed, so the numbers can be produced properly -- run them
with `--tables 3` / `--tables 4` and publish the command beside the figure.

These are measurements of a simulated Gaussian book, not calibrated quantiles
and not p-values.  `2*edf` is a Gaussian-family argument; the constant has not
been checked for Poisson with exposure.

The default runs all four tables, and the cost is NOT evenly spread: between
them tables 3 and 4 need twelve wide fits -- nine plain `cat_cat` refits (six in
the sparse payoff, three in the model-class table) and three `RandomEffect` fits
on the same 1681-level cell -- and those dominate everything else.  An earlier
"~45 minutes at the defaults" here was never reproducible and has been withdrawn
rather than replaced with a guess: the total has not been re-timed since
`rank.py` stopped choosing alias representatives by prefix walk, which took one
of the nine from 668.55 s to 9.27 s.  Use `--tables` to bound what you are
paying for.  Wall-clock moves several-fold under CPU contention either way; the
holdout columns do not, so do not read the timings as a benchmark of the fitting
paths.
"""

from __future__ import annotations

import argparse
import faulthandler
import math
import signal
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm import SuperGLM
from superglm.features import Categorical
from superglm.features.random_effect import RandomEffect

# widths, and effect sizes chosen so z straddles sqrt(edf0/2) at each width
LADDER: dict[int, tuple[float, ...]] = {
    8: (0.12, 0.23, 0.40, 0.70),
    16: (0.50, 1.00, 1.60, 2.50),
    25: (1.20, 2.50, 4.00, 6.00),
    32: (2.00, 4.00, 6.50, 9.00),
}
# spiky/diffuse magnitudes paired so each row's z matches its partner's
MATCHED: tuple[tuple[str, str, float], ...] = (
    ("noise", "none", 0.0),
    ("spiky 5 cells @ 4.0", "spike", 4.0),
    ("diffuse sd=0.20", "diffuse", 0.20),
    ("spiky 5 cells @ 6.0", "spike", 6.0),
    ("diffuse sd=0.30", "diffuse", 0.30),
    ("spiky 5 cells @ 8.0", "spike", 8.0),
    ("diffuse sd=0.41", "diffuse", 0.41),
)
TOP_M: tuple[int, ...] = (5, 10, 25, 50)
HOT_CELLS = 5
# the three model classes table 4 compares.  `pooled` is load-bearing: it is the
# only arm that shows the pair is worth HAVING once it stops being 1681 free
# cells, and the guide quotes its holdout gain.
SHRINKAGE_ARMS: tuple[str, ...] = ("mains", "fixed", "pooled")


def worth_threshold(edf0: float) -> float:
    """z a pair must clear for a plain fixed refit to pay for its own df.

    `T/phi > 2*edf0` (Mallows' Cp) expressed on the z scale the screen reports.
    `edf0` must be the SAME value the screen normalized z by -- read it off the
    returned row rather than re-deriving it from the factor widths.
    """
    return float(np.sqrt(edf0 / 2.0))


def cell_contributions(
    resid: NDArray, joint: NDArray, n_cells: int, phi: float
) -> tuple[NDArray, int]:
    """Per-cell score contributions ``t_c = n_c * mean_c^2 / phi``.

    Returns the contributions and the number of OCCUPIED cells -- empty cells
    contribute nothing and must not count toward the null, which is why the
    caller gets the occupancy back rather than assuming ``n_cells``.
    """
    count = np.bincount(joint, minlength=n_cells).astype(float)
    total = np.bincount(joint, weights=resid, minlength=n_cells)
    occupied = count > 0
    t = np.zeros(n_cells)
    t[occupied] = total[occupied] ** 2 / count[occupied] / phi
    return t, int(occupied.sum())


def paired_deltas(arm: Sequence[float], mains: Sequence[float]) -> list[float]:
    """Percent holdout change of each replicate against ITS OWN mains fit.

    The three arms of table 4 share a draw and a split within a replicate, so
    the comparison is paired and the pairing is the point: dividing every
    replicate by the mains MEAN puts between-seed baseline variation back into
    a number that is supposed to isolate the model class.  At three replicates
    that is not a rounding difference -- it can invert the sign of an
    individual split, which is exactly what the spread is being read for.
    """
    if len(arm) != len(mains):
        raise ValueError(f"paired arms must be the same length, got {len(arm)} and {len(mains)}")
    return [(a / m - 1.0) * 100.0 for a, m in zip(arm, mains, strict=True)]


def participation_ratio(t: NDArray) -> float:
    """Effective number of contributing cells, ``(sum t)^2 / sum t^2``.

    Equals k when k cells contribute equally and 1 when a single cell carries
    everything, so it reads the shape of a score the total cannot see.
    """
    denom = float(np.sum(np.asarray(t, dtype=float) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(t) ** 2 / denom)


def concentration(t: NDArray, n_occupied: int) -> float:
    """``participation_ratio`` against its null of ``k/3``.

    ~1 means the score is spread as noise spreads it -- which a genuinely
    diffuse truth also does.  Near 0 means a few cells carry the pair.

    LARGE BLOCKS ONLY.  ``k/3`` is the large-``k`` LIMIT, not the finite-sample
    expectation.  For k independent chi^2_1 contributions ``E[(sum t)^2] =
    k^2 + 2k`` against ``E[sum t^2] = 3k``, so the null mean of P is ``(k+2)/3``
    to first order and this ratio sits ABOVE 1 at finite k -- measured and
    pinned in the tests at ~1.39 (k=8), ~1.15 (k=25), ~1.04 (k=100), ~1.003
    (k=1600).  At the bottom it is not a small bias: a single occupied cell
    gives ``P = 1`` exactly and this function returns 3, the value that is
    supposed to mean "as diffuse as noise".  The reading is calibrated for the
    wide blocks it was introduced for; a narrow or thinly occupied block needs
    a finite-``k`` calibration before its value means anything.
    """
    if n_occupied <= 0:
        return float("nan")
    return participation_ratio(t) / (n_occupied / 3.0)


@dataclass(frozen=True)
class PairData:
    frame: pd.DataFrame
    y: NDArray
    joint: NDArray
    n_levels: int
    # the planted interaction surface, carried out so the truth can be COUNTED
    # rather than inferred from a difference of two noise draws
    cell: NDArray


def _make(kind: str, magnitude: float, n_levels: int, n: int, seed: int) -> PairData:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, n_levels, n)
    right = rng.integers(0, n_levels, n)
    eta = rng.normal(scale=0.5, size=n_levels)[left] + rng.normal(scale=0.5, size=n_levels)[right]
    cell = np.zeros((n_levels, n_levels))
    if kind == "spike":
        cell.flat[rng.choice(n_levels * n_levels, size=HOT_CELLS, replace=False)] = magnitude
    elif kind == "diffuse":
        cell[:] = rng.normal(scale=magnitude, size=(n_levels, n_levels))
    elif kind != "none":
        # a typo must not produce a valid-looking null run
        raise ValueError(f"unknown truth kind: {kind!r}")
    y = eta + cell[left, right] + rng.normal(scale=1.0, size=n)
    frame = pd.DataFrame({"g": [f"G{i}" for i in left], "h": [f"H{i}" for i in right]})
    return PairData(
        frame=frame,
        y=y,
        joint=left * n_levels + right,
        n_levels=n_levels,
        cell=cell,
    )


class NonconvergedFitError(RuntimeError):
    """A fit that did not converge, refused before it can reach a table."""


def _require_converged(model: SuperGLM, label: str) -> SuperGLM:
    """Refuse a fit whose own convergence flags say it failed.

    Every number in this file is a mean over replicates, so one failed fit does
    not announce itself -- it moves a column and leaves the table looking
    exactly as valid as before.  That is the same failure `_positive_int`
    exists to prevent, and it deserves the same treatment: stop, rather than
    average it in.

    Both flags are checked because the arms disagree about which one is
    meaningful.  `reml_diagnostics()["enabled"]` is False for `mains` and
    `fixed` -- `fit_reml` finds no REML-eligible group in a pure-`Categorical`
    model and falls back to `fit()` -- so for those the outer flag is `None`
    and only the solver's own convergence exists.  `pooled` estimates a
    variance component and reports both.  Reading only one of them would
    exempt two arms of table 4 from the check, or all of tables 1 and 3.

    This mirrors what the public report shows: `diagnostics()["_model"]`
    reports the REML flag when REML ran and the solver flag otherwise.
    """
    reml = model.reml_diagnostics()
    solver_converged = bool(model.result.converged)
    reml_converged = bool(reml["converged"]) if reml["enabled"] else True
    if solver_converged and reml_converged:
        return model
    raise NonconvergedFitError(
        f"{label}: fit did not converge "
        f"(solver converged={solver_converged}, n_iter={int(model.result.n_iter)}; "
        f"reml enabled={reml['enabled']}, converged={reml['converged']!r}, "
        f"termination={reml.get('termination_reason')!r}). "
        "Refusing to average a failed fit into the table."
    )


def _mains(frame: pd.DataFrame, y: NDArray) -> SuperGLM:
    model = SuperGLM(family="gaussian", features={"g": Categorical(), "h": Categorical()})
    model.fit(frame[["g", "h"]], y)
    return _require_converged(model, "mains")


def _split(n: int, seed: int) -> tuple[NDArray, NDArray]:
    """Half-sample train/test split.

    Every runner calls this with the SAME seed it passed to `_make`, so the
    permutation and the level assignment come off two fresh streams opened on
    one seed.  That is worth a second look and it survives it: the two draws use
    different bounded-integer paths, and fold membership shows no measurable
    dependence on the levels or on any fixed cell set -- pinned in
    `tests/test_screening_worth_gate.py`.
    """
    order = np.random.default_rng(seed).permutation(n)
    return order[: n // 2], order[n // 2 :]


def _run_gate_ladder(reps: int, n: int) -> list[dict[str, object]]:
    """Does `z > sqrt(edf0/2)` predict the sign of the holdout change?

    `edf0` is READ from the same screening row as `z`, not re-derived as
    `(n_levels - 1)**2`.  For an unpenalized `cat_cat` the screen reports the
    block's achieved rank, which drops below the nominal rank whenever a joint
    cell is empty in the training split -- routine at these widths.  The Cp
    identity needs the same `edf0` on both sides of `T/phi > 2*edf0`, so
    normalizing z by the achieved rank and thresholding on the nominal one
    would compare two different quantities.
    """
    rows: list[dict[str, object]] = []
    for n_levels, effects in LADDER.items():
        for effect in effects:
            zs, edfs, deltas = [], [], []
            for rep in range(reps):
                data = _make("spike", effect, n_levels, n, 7000 + rep)
                train, test = _split(n, 7000 + rep)
                dtr, ytr = data.frame.iloc[train], data.y[train]
                dte, yte = data.frame.iloc[test], data.y[test]

                mains = _mains(dtr, ytr)
                screened = mains.screen_interactions(dtr, ytr).iloc[0]
                zs.append(float(screened["z"]))
                edfs.append(float(screened["edf0"]))
                full = SuperGLM(
                    family="gaussian",
                    features={"g": Categorical(), "h": Categorical()},
                    interactions=[("g", "h")],
                )
                full.fit(dtr, ytr)
                _require_converged(full, f"gate ladder L={n_levels} effect={effect} rep={rep}")
                base = float(np.mean((mains.predict(dte) - yte) ** 2))
                with_pair = float(np.mean((full.predict(dte) - yte) ** 2))
                deltas.append((with_pair / base - 1.0) * 100.0)

            # the Cp identity is a statement about ONE fit, so the gate is
            # aggregated on its own scale: the ratio z_i / sqrt(edf0_i / 2) per
            # replicate, averaged.  Thresholding mean(z) against a threshold
            # built from mean(edf0) is a different statement whenever the
            # achieved rank varies between replicates, and it is not the
            # identity this section sells as exact.
            ratios = [z_i / worth_threshold(e_i) for z_i, e_i in zip(zs, edfs, strict=True)]
            ratio = float(np.mean(ratios))
            edf0 = float(np.mean(edfs))
            z = float(np.mean(zs))
            delta = float(np.mean(deltas))
            rows.append(
                {
                    "n_levels": n_levels,
                    "edf0": edf0,
                    "nominal_edf0": float((n_levels - 1) ** 2),
                    "effect": effect,
                    "z": z,
                    "threshold": worth_threshold(edf0),
                    "ratio": ratio,
                    # how many replicates individually cleared their own bar,
                    # so a cell where they disagree cannot hide behind the mean
                    "reps_above": sum(1 for r in ratios if r > 1.0),
                    "reps": len(ratios),
                    "delta_pct": delta,
                    "agrees": (ratio > 1.0) == (delta < 0.0),
                }
            )
    return rows


def _run_concentration(reps: int, n: int, n_levels: int) -> list[dict[str, object]]:
    """Spiky and diffuse truths tuned so their z values coincide.

    `edf0` and the raw `P` are reported alongside the ratio so the guide can
    quote the null (`k/3`) and the gate threshold at this width without
    re-deriving either.  Note the scope: these fit the FULL sample, where the
    ladder and the sparse payoff fit a half-sample train split.
    """
    rows: list[dict[str, object]] = []
    for label, kind, magnitude in MATCHED:
        zs, edfs, ratios, prs, occs = [], [], [], [], []
        for rep in range(reps):
            data = _make(kind, magnitude, n_levels, n, 31337 + rep)
            mains = _mains(data.frame, data.y)
            screened = mains.screen_interactions(data.frame, data.y).iloc[0]
            zs.append(float(screened["z"]))
            edfs.append(float(screened["edf0"]))
            resid = data.y - mains.predict(data.frame)
            # P is scale-free, so phi cancels out of the reported ratio; passing
            # 1.0 keeps that visible rather than implying a calibrated scale
            t, occupied = cell_contributions(resid, data.joint, n_levels * n_levels, 1.0)
            ratios.append(concentration(t, occupied))
            prs.append(participation_ratio(t))
            occs.append(occupied)
            print(f"  {label} rep {rep} done", flush=True)
        edf0 = float(np.mean(edfs))
        rows.append(
            {
                "label": label,
                "z": float(np.mean(zs)),
                "edf0": edf0,
                "threshold": worth_threshold(edf0),
                "occupied": float(np.mean(occs)),
                "participation": float(np.mean(prs)),
                "concentration": float(np.mean(ratios)),
            }
        )
    return rows


def _run_sparse_payoff(reps: int, n: int, n_levels: int) -> list[dict[str, object]]:
    """Top-m cells against the full refit, every arm from ONE seed and split.

    The widest arm is the plain fixed `cat_cat` refit -- the same model class
    table 4 fits, but on THIS row's seed and split, so the full-refit cell of
    each row is measured here rather than imported from a second simulation.
    It has to be the interaction rather than another `hot` factor: at these
    widths the training split leaves cells empty, a level unseen in training is
    a predict-time error rather than a zero, and the `cat_cat` block already
    drops the unidentified columns and predicts those rows off the margins.

    The concentration reported here is computed on the TRAINING residuals that
    ranked the cells, so it labels the same fit its deltas come from -- table 2
    fits the full sample and its value is a different quantity.
    """
    n_cells = n_levels * n_levels
    arms: tuple[int, ...] = (*TOP_M, n_cells)
    rows: list[dict[str, object]] = []
    for kind, magnitude in (("spike", 6.0), ("diffuse", 0.30)):
        acc: dict[int, list[float]] = {m: [] for m in arms}
        used: dict[int, list[float]] = {m: [] for m in arms}
        cons: list[float] = []
        for rep in range(reps):
            data = _make(kind, magnitude, n_levels, n, 31337 + rep)
            train, test = _split(n, 31337 + rep)
            dtr, ytr, jtr = data.frame.iloc[train], data.y[train], data.joint[train]
            dte, yte, jte = data.frame.iloc[test], data.y[test], data.joint[test]

            mains = _mains(dtr, ytr)
            base = float(np.mean((mains.predict(dte) - yte) ** 2))
            resid = ytr - mains.predict(dtr)
            # ranked on TRAINING residuals only -- ranking on the full sample
            # would leak the test rows into the cell choice
            t, occupied = cell_contributions(resid, jtr, n_cells, 1.0)
            cons.append(concentration(t, occupied))
            live = np.flatnonzero(np.bincount(jtr, minlength=n_cells) > 0)
            ranked = np.argsort(t)[::-1]
            mains_edf = float(mains.result.effective_df)

            for m in arms:
                if m == n_cells:
                    model = SuperGLM(
                        family="gaussian",
                        features={"g": Categorical(), "h": Categorical()},
                        interactions=[("g", "h")],
                    )
                    model.fit(dtr, ytr)
                    _require_converged(model, f"sparse payoff {kind} rep={rep} full refit")
                    got = float(np.mean((model.predict(dte) - yte) ** 2))
                else:
                    hot = set(np.intersect1d(ranked[:m], live).tolist())
                    a, b = dtr.copy(), dte.copy()
                    a["hot"] = np.where([j in hot for j in jtr], jtr.astype(str), "other")
                    b["hot"] = np.where([j in hot for j in jte], jte.astype(str), "other")
                    model = SuperGLM(
                        family="gaussian",
                        features={
                            "g": Categorical(),
                            "h": Categorical(),
                            "hot": Categorical(),
                        },
                    )
                    model.fit(a, ytr)
                    _require_converged(model, f"sparse payoff {kind} rep={rep} arm-{m}")
                    got = float(np.mean((model.predict(b) - yte) ** 2))
                acc[m].append((got / base - 1.0) * 100.0)
                # df bought over the mains model, so the sparse arms and the
                # full refit are priced on one scale
                used[m].append(float(model.result.effective_df) - mains_edf)
                print(f"  {kind} rep {rep} arm-{m} done", flush=True)
        for m in arms:
            rows.append(
                {
                    "kind": kind,
                    "top_m": m,
                    "n_cells": n_cells,
                    "extra_edf": float(np.mean(used[m])),
                    "concentration": float(np.mean(cons)),
                    "delta_pct": float(np.mean(acc[m])),
                }
            )
    return rows


def _shrinkage_spec(arm: str) -> tuple[dict[str, object], list[str]]:
    """Model spec for one arm of table 4.

    `fit_reml` finds no REML-eligible group in a pure-`Categorical` model and
    falls back to `fit()`, so `mains` and `fixed` are plain fits and only
    `pooled` estimates a variance component.  That asymmetry IS the comparison:
    the fixed arm spends its df whatever the data says, the pooled arm spends
    what REML thinks the data supports.
    """
    if arm == "mains":
        return {"features": {"g": Categorical(), "h": Categorical()}}, ["g", "h"]
    if arm == "fixed":
        return (
            {
                "features": {"g": Categorical(), "h": Categorical()},
                "interactions": [("g", "h")],
            },
            ["g", "h"],
        )
    if arm == "pooled":
        return (
            {
                "features": {
                    "g": Categorical(),
                    "h": Categorical(),
                    "gh": RandomEffect(),
                }
            },
            ["g", "h", "gh"],
        )
    raise ValueError(f"unknown shrinkage arm: {arm!r}")


def _run_shrinkage(reps: int, n: int, n_levels: int) -> tuple[list[dict[str, object]], dict]:
    """mains vs a fixed cat_cat interaction vs the same cell partially pooled.

    The gate says a wide pair should not be refit as a fixed interaction. This
    asks the follow-on question the gate does not answer: is the pair worth
    having at all, in some other model class?

    The screen is run HERE, on this table's own train split, so the z the guide
    quotes beside the +22% comes from the data that produced it.  Table 2's z at
    the same width is a different quantity -- a different seed, and the full
    sample rather than a half-sample train split.  Per-replicate holdout deltas
    come back too, since the guide quotes their spread.
    """
    acc: dict[str, list[tuple[float, float, float, float, float]]] = {
        arm: [] for arm in SHRINKAGE_ARMS
    }
    zs, edfs = [], []
    for rep in range(reps):
        data = _make("spike", 6.0, n_levels, n, 4242 + rep)
        frame = data.frame.copy()
        frame["gh"] = data.joint.astype(str)
        train, test = _split(n, 4242 + rep)
        ytr, yte = data.y[train], data.y[test]

        for arm in SHRINKAGE_ARMS:
            kwargs, cols = _shrinkage_spec(arm)
            dtr, dte = frame[cols].iloc[train], frame[cols].iloc[test]
            started = time.perf_counter()
            model = SuperGLM(family="gaussian", **kwargs)
            model.fit_reml(dtr, ytr)
            seconds = time.perf_counter() - started
            _require_converged(model, f"shrinkage arm={arm} rep={rep}")
            result = model.result
            acc[arm].append(
                (
                    seconds,
                    float(np.asarray(result.beta).size),
                    float(result.effective_df),
                    float(np.mean((model.predict(dtr) - ytr) ** 2)),
                    float(np.mean((model.predict(dte) - yte) ** 2)),
                )
            )
            if arm == "mains":
                screened = model.screen_interactions(dtr, ytr).iloc[0]
                zs.append(float(screened["z"]))
                edfs.append(float(screened["edf0"]))
            print(f"  rep {rep} {arm} done", flush=True)

    rows: list[dict[str, object]] = []
    for arm in SHRINKAGE_ARMS:
        values = np.array(acc[arm])
        rows.append(
            {
                "model": arm,
                "seconds": float(values[:, 0].mean()),
                "params": float(values[:, 1].mean()),
                "edf": float(values[:, 2].mean()),
                "train": float(values[:, 3].mean()),
                "holdout": float(values[:, 4].mean()),
                "holdout_reps": [float(v) for v in values[:, 4]],
            }
        )
    # every arm in a replicate is fitted on the SAME draw and the SAME split,
    # so the comparison is paired -- price each arm against its own replicate's
    # mains fit rather than against the mains mean
    mains_reps = next(r["holdout_reps"] for r in rows if r["model"] == "mains")
    for row in rows:
        row["delta_reps"] = paired_deltas(row["holdout_reps"], mains_reps)
    # same rule as the ladder: the Cp ratio is per fit, so aggregate it on its
    # own scale rather than dividing a mean z by a mean threshold
    ratios = [z_i / worth_threshold(e_i) for z_i, e_i in zip(zs, edfs, strict=True)]
    edf0 = float(np.mean(edfs))
    screen = {
        "z": float(np.mean(zs)),
        "z_reps": [float(v) for v in zs],
        "edf0": edf0,
        "threshold": worth_threshold(edf0),
        "ratio": float(np.mean(ratios)),
        "reps_above": sum(1 for r in ratios if r > 1.0),
        "reps": len(ratios),
    }
    return rows, screen


def _print_gate_ladder(rows: list[dict[str, object]]) -> None:
    print("\n1. Does z > sqrt(edf0/2) predict the sign of the holdout change?\n")
    print(
        f"{'levels':>7} {'edf0':>7} {'nominal':>8} {'effect':>7} {'z':>8} {'thresh':>7} "
        f"{'ratio':>9} {'above':>6} {'gate':>8} {'holdout':>9}"
    )
    for row in rows:
        gate = "INCLUDE" if row["ratio"] > 1.0 else "exclude"
        flag = "" if row["agrees"] else "   <- disagrees"
        print(
            f"{row['n_levels']:>7} {row['edf0']:>7.1f} {row['nominal_edf0']:>8.0f} "
            f"{row['effect']:>7.2f} "
            f"{row['z']:>8.2f} {row['threshold']:>7.2f} "
            f"{row['ratio']:>9.2f} {row['reps_above']:>3}/{row['reps']:<2} {gate:>8} "
            f"{row['delta_pct']:>+8.1f}%{flag}"
        )
    agree = sum(1 for r in rows if r["agrees"])
    print(f"\n  sqrt(edf0/2) agrees with the holdout sign in {agree}/{len(rows)}")
    # a fixed cutoff fails in BOTH directions; counting only one understates it
    for fixed in (2.0, 3.0, 5.0):
        hit = sum(1 for r in rows if (r["z"] > fixed) == (r["delta_pct"] < 0.0))
        admitted = [r for r in rows if r["z"] > fixed and r["delta_pct"] >= 0.0]
        rejected = [r for r in rows if r["z"] <= fixed and r["delta_pct"] < 0.0]
        print(
            f"    a fixed z > {fixed}: {hit}/{len(rows)}"
            f"   admits harmful z=[{', '.join(f'{r["z"]:.2f}' for r in admitted)}]"
            f"   rejects helpful z=[{', '.join(f'{r["z"]:.2f}' for r in rejected)}]"
        )
    missed = [r for r in rows if not r["agrees"]]
    print(f"    sqrt(edf0/2) misses: {[f'z={r["z"]:.2f} {r["delta_pct"]:+.1f}%' for r in missed]}")


def _print_concentration(rows: list[dict[str, object]], n_levels: int) -> None:
    print(
        f"\n2. Concentration at matched z ({n_levels}x{n_levels}). Rows are paired so "
        "spiky and\n   diffuse truths carry the SAME z -- any separation is "
        "information z lacks.\n   Fitted on the FULL sample, unlike tables 1 and 3.\n"
    )
    print(
        f"{'case':>22} {'z':>8} {'edf0':>8} {'thresh':>7} {'occupied':>9} "
        f"{'k/3':>8} {'P':>9} {'P/(k/3)':>9}"
    )
    for row in rows:
        print(
            f"{row['label']:>22} {row['z']:>8.2f} {row['edf0']:>8.1f} "
            f"{row['threshold']:>7.2f} {row['occupied']:>9.0f} "
            f"{row['occupied'] / 3.0:>8.1f} {row['participation']:>9.1f} "
            f"{row['concentration']:>9.3f}"
        )
    print("\n  ~1.0 = spread as noise spreads it;  << 1.0 = a few cells carry it")


def _print_sparse_payoff(rows: list[dict[str, object]]) -> None:
    print(
        "\n3. If P says concentrated, does fitting only those cells pay?\n"
        "   Cells ranked on training residuals only.  `P/(k/3)` is the TRAINING\n"
        "   -split value, the same fit these deltas come from.\n"
    )
    print(f"{'truth':>10} {'P/(k/3)':>9} {'arm':>10} {'+edf':>8} {'holdout vs mains':>18}")
    for row in rows:
        label = "full refit" if row["top_m"] == row["n_cells"] else f"top-{row['top_m']}"
        print(
            f"{row['kind']:>10} {row['concentration']:>9.3f} {label:>10} "
            f"{row['extra_edf']:>8.1f} {row['delta_pct']:>+17.1f}%"
        )


def _print_shrinkage(rows: list[dict[str, object]], screen: dict, n_levels: int) -> None:
    print(
        f"\n4. Same pair, three model classes ({n_levels}x{n_levels}, 5 cells @ 6.0).\n"
        "   The detection is not in doubt; the question is what to DO with it.\n"
    )
    print(
        f"  screened on this table's own train split: z = {screen['z']:.2f} "
        f"(reps {', '.join(f'{v:.2f}' for v in screen['z_reps'])}), "
        f"edf0 = {screen['edf0']:.1f}, threshold = {screen['threshold']:.2f}, "
        f"ratio = {screen['ratio']:.2f} ({screen['reps_above']}/{screen['reps']} reps above) -> "
        f"{'INCLUDE' if screen['ratio'] > 1.0 else 'exclude'}\n"
    )
    print(
        f"{'model':>8} {'fit':>8} {'params':>8} {'edf':>9} {'train':>8} {'HOLDOUT':>9} {'vs mains':>10}"
    )
    mains = next((r for r in rows if r["model"] == "mains"), None)
    for row in rows:
        # mean of the PAIRED deltas, not a ratio of means -- see `paired_deltas`
        delta = "" if mains is None else f"{np.mean(row['delta_reps']):+9.1f}%"
        print(
            f"{row['model']:>8} {row['seconds']:>7.1f}s {row['params']:>8.0f} "
            f"{row['edf']:>9.1f} {row['train']:>8.4f} {row['holdout']:>9.4f} {delta:>10}"
        )
    if mains is not None:
        for row in rows:
            per_rep = ", ".join(f"{v:+.1f}%" for v in row["delta_reps"])
            print(f"    {row['model']:>8} per-replicate vs ITS OWN mains fit: {per_rep}")
    print(
        "\n  Wall-clock here is indicative only -- it moves several-fold under CPU\n"
        "  contention.  The holdout column does not."
    )


def _tables(value: str) -> tuple[int, ...]:
    """Parse `--tables 3,4` into a validated tuple.

    A subset has to be a first-class option rather than an ad-hoc edit: the
    wide refits run for tens of minutes each, so a guide figure will sometimes
    be refreshed one table at a time, and the command that produced it has to
    be quotable.
    """
    wanted = tuple(int(part) for part in value.replace(",", " ").split())
    if not wanted or any(t not in (1, 2, 3, 4) for t in wanted):
        raise argparse.ArgumentTypeError(f"tables must be drawn from 1,2,3,4; got {value!r}")
    return wanted


def arm_watchdog(seconds: float) -> bool:
    """Make a stuck fit name itself instead of looking merely expensive.

    Every few minutes, dump every thread's stack to stderr without interrupting
    the run.  A frame that repeats across two dumps is the hot spot, and that is
    the whole diagnosis: this file spent an afternoon treating a pathological
    41x41 refit as a big one, and the profile that ended it took twenty minutes
    once someone looked.  `SIGUSR1` dumps on demand for a run already in flight
    (`kill -SIGUSR1 <pid>`).

    Returns whether the periodic dump was armed; `seconds <= 0` disables it.
    """
    if hasattr(faulthandler, "register") and hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr)
    if seconds <= 0:
        return False
    faulthandler.dump_traceback_later(seconds, repeat=True, file=sys.stderr)
    return True


# Smallest width the spiky generator is defined at: it draws HOT_CELLS
# DISTINCT cells from the L x L grid, so L*L must reach HOT_CELLS.  At L = 1
# and 2 `rng.choice` raises "Cannot take a larger sample than population when
# replace is False".
MIN_WIDE_LEVELS = math.ceil(math.sqrt(HOT_CELLS))
# Smallest sample `_split` is defined at: below this the training half is
# empty and the first fit sees no rows at all.
MIN_ROWS = 2


def _at_least(minimum: int, what: str):
    """An argparse type for a dimension with its own floor.

    A single `>= 1` floor was not enough.  `--wide-levels 1` and `2` clear it
    and then fail inside `rng.choice`, and `--n 1` clears it and then hands the
    first fit an empty training split -- both a long way from the flag that
    caused them.  These floors are STRUCTURAL: the smallest values at which the
    generator and the split are defined at all.  They are not a claim that the
    configuration has enough data to fit; that stays data-dependent and is
    checked per table in `_validate_configuration`.
    """

    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"expected an integer >= {minimum}, got {value!r}"
            ) from None
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"{what}, so must be >= {minimum}, got {parsed}")
        return parsed

    return parse


def _validate_configuration(args: argparse.Namespace) -> None:
    """Reject a table configuration that provably cannot run.

    With `n // 2` training rows and `L` levels per factor, `n // 2 < L` leaves
    at least one level absent from the training half by pigeonhole, and a level
    seen only at predict time is a hard error rather than a zero.  So this is a
    NECESSARY condition, exactly derivable, and it catches the configurations
    that otherwise surface far downstream as an unseen-level error naming a
    category rather than a flag.

    It is not sufficient -- a level can still be missing when the inequality
    holds -- and that residue is left to fail loudly at fit time, where the
    message names the actual level.
    """
    train_rows = args.n // 2
    required: list[tuple[str, int]] = []
    if 1 in args.tables:
        required.append(("table 1 (gate ladder)", max(LADDER)))
    for table in (2, 3, 4):
        if table in args.tables:
            required.append((f"table {table}", args.wide_levels))
    for label, levels in required:
        if train_rows < levels:
            raise SystemExit(
                f"--n {args.n} gives {train_rows} training rows, but {label} needs at least "
                f"{levels} (one per level of each factor, or a level is certainly missing "
                f"from the training half and predict fails on it)."
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reps", type=_at_least(1, "a run needs at least one replicate"), default=3
    )
    parser.add_argument(
        "--n",
        type=_at_least(MIN_ROWS, "the split must leave both halves non-empty"),
        default=12_000,
    )
    parser.add_argument(
        "--wide-levels",
        type=_at_least(
            MIN_WIDE_LEVELS, f"the spiky truth plants {HOT_CELLS} distinct cells in L*L"
        ),
        default=41,
    )
    parser.add_argument("--tables", type=_tables, default=(1, 2, 3, 4))
    parser.add_argument(
        "--watchdog",
        type=float,
        default=300.0,
        help="seconds between stack dumps to stderr; 0 disables (default: 300)",
    )
    return parser


def main() -> None:
    # NOT `ignore`: the pooled arm asks REML to estimate a variance component
    # over 1681 levels, and a run that mutes its own convergence warnings
    # reports a number where it should report a problem.  `once` keeps a long
    # run readable without hiding anything that fires.
    warnings.simplefilter("once")
    args = _build_parser().parse_args()
    _validate_configuration(args)
    if arm_watchdog(args.watchdog):
        print(f"watchdog: stack dump to stderr every {args.watchdog:.0f}s", flush=True)
    started = time.perf_counter()

    if 1 in args.tables:
        _print_gate_ladder(_run_gate_ladder(args.reps, args.n))
    if 2 in args.tables:
        _print_concentration(
            _run_concentration(args.reps, args.n, args.wide_levels), args.wide_levels
        )
    if 3 in args.tables:
        _print_sparse_payoff(_run_sparse_payoff(args.reps, args.n, args.wide_levels))
    if 4 in args.tables:
        shrinkage, screen = _run_shrinkage(args.reps, args.n, args.wide_levels)
        _print_shrinkage(shrinkage, screen, args.wide_levels)
    print(f"\ntotal {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
