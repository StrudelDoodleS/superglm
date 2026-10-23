"""Write the cross-platform SuperLSS fixtures of ``tests/test_superlss_load_portability.py``.

Each fixture is a model saved by one numerical platform that another platform
recomputes differently at load. It needs an x86-64 machine whose NumPy
dispatches the AVX-512 ``log`` loop, run with the 0.41 pre-release build on
the path (commit 66ae870c), whose identifiers still hash the ``log`` carrier:

    PYTHONPATH=<dir>/src python scripts/make_portable_load_fixtures.py \
        tests/fixtures/saved_portable_v041

``gamma_avx512`` is a ``GammaLS`` fit saved with NumPy's AVX-512 ``log`` on
responses where that loop is not correctly rounded but the C library's ``log``
is; ``gamma_libm`` is saved with AVX-512 disabled
(``NPY_DISABLE_CPU_FEATURES``), on responses where the reverse holds, so every
platform recomputes at least one of them differently from its saver.
``face_native`` and ``face_haswell`` are a ``fit_reml`` on an exact penalty
face, saved under the machine's own OpenBLAS kernel and under
``OPENBLAS_CORETYPE=Haswell``; each records face tolerances the other kernel
recomputes an ulp apart. The correctly rounded ``log`` comes from ``decimal``.

Each record holds the ``to_bytes()`` artifact, the rows, the coefficients and
linear predictors the saver predicted, and how it was saved; it is written
``lzma``-compressed.
"""

from __future__ import annotations

import lzma
import math
import os
import pickle
import subprocess
import sys
import warnings
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd

GAMMA_ROWS = 150
NO_AVX512 = {"NPY_DISABLE_CPU_FEATURES": "X86_V4,AVX512_ICL"}
HASWELL = {"OPENBLAS_CORETYPE": "Haswell"}


def _correctly_rounded_log(value: float) -> float:
    getcontext().prec = 60
    return float(Decimal(value).ln())


def _gamma_candidates(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    y = rng.gamma(3.0, np.exp(0.5 + 0.1 * x) / 3.0)
    return x, y


def pick_gamma_rows() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Rows on which exactly one of NumPy's ``log`` and the C library's is correctly rounded."""
    x, y = _gamma_candidates(2_000_000, 41)
    vector = np.log(y)
    scalar = np.array([math.log(value) for value in y])
    disagree = np.flatnonzero(vector != scalar)
    if len(disagree) == 0:
        raise SystemExit("NumPy's log matches the C library here: run on an AVX-512 machine")
    exact = np.array([_correctly_rounded_log(float(y[i])) for i in disagree])
    picks = {
        "gamma_avx512": disagree[(scalar[disagree] == exact) & (vector[disagree] != exact)],
        "gamma_libm": disagree[(vector[disagree] == exact) & (scalar[disagree] != exact)],
    }
    rows = {}
    for name, index in picks.items():
        if len(index) < GAMMA_ROWS:
            raise SystemExit(f"only {len(index)} rows for {name}")
        chosen = index[:GAMMA_ROWS]
        rows[name] = (x[chosen], y[chosen])
    return rows


def _record(model, frame: pd.DataFrame, saved_with: str) -> dict:
    import superglm as sg

    return {
        "version": sg.__version__,
        "saved_with": saved_with,
        "serialized": model.to_bytes(),
        "frame": frame,
        "coefficients": dict(model.coef_),
        "eta": model.predict_link(frame).to_numpy(dtype=np.float64),
    }


def save_gamma(name: str, rows_path: str, out: str) -> None:
    import superglm as sg
    from superglm import s

    data = np.load(rows_path)
    x, y = data["x"], data["y"]
    vector = np.log(y)
    scalar = np.array([math.log(value) for value in y])
    # The saver's log must be the loop the rows were chosen against.
    expect_libm = name == "gamma_libm"
    if bool(np.all(vector == scalar)) != expect_libm or not (
        expect_libm or bool(np.all(vector != scalar))
    ):
        raise SystemExit(f"{name}: NumPy dispatched the wrong log loop")
    frame = pd.DataFrame({"x": x})
    family = sg.GammaLS()
    model = sg.SuperLSS(family, family.mean(s("x", kind="cr", k=5)), family.scale())
    model.fit(frame, y, lambdas={"mean:x#wiggle": 1.0})
    saved_with = "C library log" if expect_libm else "NumPy AVX-512 log"
    _write(out, name, _record(model, frame, saved_with))


def face_case() -> tuple[pd.DataFrame, np.ndarray, list]:
    import superglm as sg
    from superglm import s

    n = 150
    rng = np.random.default_rng(28)
    frame = pd.DataFrame(
        {"x": rng.uniform(0, 10, n), "z": rng.uniform(-2, 2, n), "w": rng.uniform(0, 5, n)}
    )
    y = 1.0 + 0.3 * frame["x"].to_numpy() + 0.2 * frame["z"].to_numpy() + rng.normal(0, 0.5, n)
    family = sg.GaussianLS()
    declarations = [
        family,
        family.location(s("x", kind="cr", k=5), s("z", kind="cr", k=5), s("w", kind="cr", k=5)),
        family.scale(),
    ]
    return frame, y, declarations


def save_face(name: str, out: str) -> None:
    import superglm as sg

    frame, y, declarations = face_case()
    model = sg.SuperLSS(*declarations)
    model.fit_reml(frame, y, retain_rows=False)
    if not model.exact_face_components_:
        raise SystemExit(f"{name}: the fit did not reach an exact penalty face")
    kernel = os.environ.get("OPENBLAS_CORETYPE", "native")
    _write(out, name, _record(model, frame, f"OpenBLAS {kernel} kernel"))


def _write(out: str, name: str, record: dict) -> None:
    os.makedirs(out, exist_ok=True)
    with lzma.open(os.path.join(out, f"{name}.pkl.xz"), "wb", preset=9) as handle:
        pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(name, record["saved_with"])


def _child(env: dict[str, str], *arguments: str) -> None:
    subprocess.run(
        [sys.executable, os.path.abspath(__file__), *arguments],
        check=True,
        env={**os.environ, **env},
    )


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    rows_paths = []
    for name, (x, y) in pick_gamma_rows().items():
        rows_path = os.path.join(out, f"_{name}_rows.npz")
        np.savez(rows_path, x=x, y=y)
        rows_paths.append(rows_path)
        _child(NO_AVX512 if name == "gamma_libm" else {}, "--gamma", name, rows_path, out)
    for rows_path in rows_paths:
        os.remove(rows_path)
    _child({}, "--face", "face_native", out)
    _child(HASWELL, "--face", "face_haswell", out)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    if sys.argv[1] == "--gamma":
        save_gamma(*sys.argv[2:5])
    elif sys.argv[1] == "--face":
        save_face(*sys.argv[2:4])
    else:
        main(sys.argv[1])
