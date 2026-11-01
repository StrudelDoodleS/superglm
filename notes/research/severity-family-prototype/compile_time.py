"""Generation and numba compile time per declared family, kernel by kernel.

Prototype for notes/research/2026-10-31-severity-family-framework.md; not production code.

    uv run python compile_time.py OUT_DIR

The first call of each ``@njit`` row loop compiles it; the second call's time
on the same ten rows is subtracted.  Primitives are compiled once beforehand,
so the figures are the per-family cost a first fit would pay without a cache.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import declarations as D
import sevgen
import sevprims


def timed(call) -> float:
    start = time.perf_counter()
    call()
    first = time.perf_counter() - start
    start = time.perf_counter()
    call()
    return first - (time.perf_counter() - start)


def main() -> None:
    out_dir = Path(sys.argv[1])
    start = time.perf_counter()
    sevprims.nb_gdev(0.5, 1.0)
    for name in ("nb_softplus", "nb_expit", "nb_log1mexp", "nb_invexpm1", "nb_gnorm"):
        getattr(sevprims, name)(0.5)
    sevprims.nb_psires(1, 0.5)
    sevprims.nb_polygamma(1, 0.5)
    sevprims.nb_digamma(0.5)
    report = {"primitives_seconds": time.perf_counter() - start}
    for declaration in D.FAMILIES:
        kernel, built = sevgen.build(declaration, out_dir / "kernels")
        y = np.full(10, 2.0)
        theta = np.full((10, kernel.k), 1.5)
        w = np.ones(10)
        seconds = {f"density_o{order}": timed(lambda o=order: kernel.evaluate(y, theta, w, o)) for order in (0, 1, 2)}
        seconds["density_d34"] = timed(lambda: kernel.third_fourth(y, theta, w))
        for kind in kernel.derived.tails:
            for order in (0, 2):
                seconds[f"{kind}_o{order}"] = timed(lambda k=kind, o=order: kernel.tail(k, y, theta, o))
        report[declaration.name] = {
            "derive_seconds": built["derive_seconds"],
            "render_seconds": built["render_seconds"],
            "generated_lines": built["generated_lines"],
            "compile_seconds": seconds,
            "compile_total_seconds": sum(seconds.values()),
        }
        print(declaration.name, json.dumps(report[declaration.name]), flush=True)
    (out_dir / "compile_time.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
