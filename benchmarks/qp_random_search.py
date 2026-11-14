"""Random small QPs through ``solve_constrained_qp``, each checked against its exact optimum.

    uv run python benchmarks/qp_random_search.py --cases 20000 --seed 1
    uv run python benchmarks/qp_random_search.py --cases 5000 --out failures/

Run it before the first push of any change to ``solvers/constrained_qp.py``
and report its counts in the pull request. Each case is

    minimise 1/2 beta' H beta - g' beta   subject to   A beta >= b

with ``H`` positive definite (some cases nearly singular), duplicated and
nearly parallel rows in ``A``, a feasible start whose rows are tight, nearly
tight or slack, and a random warm working set. A strictly convex QP's optimum
is the equality-constrained minimiser of its own active set, so the minimum
objective over every feasible equality-constrained minimiser (all ``2^m``
working sets, ``m <= 8``) is the exact optimum, found without the solver.

Each case is counted as one of:

- ``ok``: converged, feasible and at the optimum's objective;
- ``wrong``: reported converged but infeasible or above the optimum, a
  silently wrong answer;
- ``unconverged``: reported not converged, which the fit discloses;
- ``error``: raised;
- ``unchecked``: the oracle found no feasible minimiser although the start
  is feasible, so the answer could not be checked.

The exit status is 1 when any case is wrong, raised or unchecked. ``--out``
saves those cases as ``.npz`` files to replay.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np

from superglm.solvers.constrained_qp import solve_constrained_qp

U = np.finfo(np.float64).eps / 2


def _case(rng: np.random.Generator) -> dict[str, np.ndarray | list[int]]:
    p = int(rng.integers(2, 7))
    m = int(rng.integers(2, 9))
    factor = rng.normal(size=(p, p))
    ridge = rng.choice([1e-1, 1e-4, 1e-8], p=[0.7, 0.2, 0.1])
    H = factor @ factor.T + ridge * np.eye(p)
    A = rng.normal(size=(m, p))
    for _ in range(int(rng.integers(0, 3))):  # duplicated or nearly parallel rows
        source, target = rng.integers(m, size=2)
        A[target] = A[source] * (1.0 + rng.choice([0.0, 1e-10, 1e-6]))
    start = rng.normal(size=p) * 0.3
    slack = np.abs(rng.normal(size=m)) * rng.choice([0.0, 1e-12, 0.01, 1.0], size=m)
    active = sorted(rng.choice(m, int(rng.integers(0, min(m, p) + 1)), replace=False).tolist())
    return {
        "H": H,
        "g": rng.normal(size=p) * 3,
        "A": A,
        "b": A @ start - slack,
        "start": start,
        "active": active,
    }


def _objective(H: np.ndarray, g: np.ndarray, beta: np.ndarray) -> float:
    return float(0.5 * beta @ H @ beta - g @ beta)


def _optimum(H: np.ndarray, g: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
    """The least objective over every feasible equality-constrained minimiser."""
    p, m = H.shape[0], A.shape[0]
    best = np.inf
    for size in range(min(m, p) + 1):
        for rows in itertools.combinations(range(m), size):
            rows = list(rows)
            kkt = np.block([[H, A[rows].T], [A[rows], np.zeros((size, size))]])
            rhs = np.concatenate([g, b[rows]])
            beta = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:p]
            # A working set whose rows are inconsistent (a duplicated row with
            # another bound) has no minimiser: least squares returns a point
            # off its rows, which is skipped. A consistent one is feasible to
            # within the accuracy its solve reached on its own rows, since a
            # vertex of nearly parallel rows is solved only that well.
            scale = np.abs(A) @ np.abs(beta) + np.abs(b) + 1.0
            achieved = np.max(np.abs(A[rows] @ beta - b[rows]), initial=0.0)
            if achieved > 1e6 * U * np.max(scale[rows], initial=1.0):
                continue
            if np.all(A @ beta >= b - 1e3 * U * scale - 10.0 * achieved):
                best = min(best, _objective(H, g, beta))
    return best


def _verdict(case: dict, result) -> str:
    H, g, A, b = case["H"], case["g"], case["A"], case["b"]
    if not result.converged:
        return "unconverged"
    beta = result.beta
    row_scale = np.abs(b) + np.abs(A) @ np.abs(beta) + 1.0
    if not np.all(A @ beta >= b - 1e4 * U * row_scale):
        return "wrong"
    optimum = _optimum(H, g, A, b)
    if not np.isfinite(optimum):  # the start is feasible, so the oracle failed
        return "unchecked"
    value = _objective(H, g, beta)
    magnitude = abs(optimum) + abs(_objective(H, np.abs(g), np.abs(beta))) + 1.0
    return "ok" if value <= optimum + 1e-8 * magnitude else "wrong"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--cases", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="directory for wrong or raising cases")
    options = parser.parse_args(argv)
    rng = np.random.default_rng(options.seed)
    counts = dict.fromkeys(("ok", "wrong", "unconverged", "error", "unchecked"), 0)
    for number in range(options.cases):
        case = _case(rng)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = solve_constrained_qp(
                    case["H"],
                    case["g"],
                    case["A"],
                    case["b"],
                    active_set_init=list(case["active"]),
                    beta_start=case["start"].copy(),
                )
            verdict = _verdict(case, result)
        except Exception:
            verdict = "error"
        counts[verdict] += 1
        if verdict in {"wrong", "error", "unchecked"} and options.out is not None:
            options.out.mkdir(parents=True, exist_ok=True)
            arrays = {key: np.asarray(value) for key, value in case.items()}
            np.savez(options.out / f"case-{options.seed}-{number}-{verdict}.npz", **arrays)
    print(" ".join(f"{name}={count}" for name, count in counts.items()))
    return int(counts["wrong"] + counts["error"] + counts["unchecked"] > 0)


if __name__ == "__main__":
    sys.exit(main())
