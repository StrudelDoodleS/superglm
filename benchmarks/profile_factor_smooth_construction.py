"""Profile large-row factor-smooth marginal construction."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import resource
import time

import numpy as np

from superglm import FactorSmooth


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--basis", choices=("fs", "sz"), required=True)
    parser.add_argument("--bins", type=int, required=True)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    x = np.linspace(-2.0, 2.0, args.rows, dtype=np.float64)
    group = np.arange(args.rows, dtype=np.intp) % args.levels
    spec = FactorSmooth(
        "x",
        group="group",
        basis=args.basis,
        k=args.k,
    )

    profile = cProfile.Profile() if args.profile else None
    started = time.perf_counter()
    if profile is None:
        info = spec.build_discrete(x, group, {}, args.bins)
    else:
        info = profile.runcall(spec.build_discrete, x, group, {}, args.bins)
    elapsed = time.perf_counter() - started
    basis_unique = info.factor_smooth_basis_unique
    if basis_unique is None:  # pragma: no cover - discrete construction contract
        raise RuntimeError("FactorSmooth discrete construction did not retain support geometry.")

    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(
        json.dumps(
            {
                "rows": args.rows,
                "basis": args.basis,
                "backend": spec._marginal_build_backend,
                "elapsed_s": elapsed,
                "peak_rss_mib": peak_mib,
                "support_shape": list(basis_unique.shape),
            },
            sort_keys=True,
        )
    )
    if profile is not None:
        pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(30)


if __name__ == "__main__":
    main()
