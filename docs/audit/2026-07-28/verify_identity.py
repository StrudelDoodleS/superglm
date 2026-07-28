"""Verify the row-tensor Gram identity against superglm's REAL tensor block.

Two levels are checked separately, because they have different prerequisites:

  L1 (pair compression): A'WA == sum over unique (i1,i2) cells of Wbar[c] * a_c a_c'
      Requires only that rows repeat. No factorization needed.

  L2 (row-tensor / G-operator): A'WA == G(M1)' Wbar G(M2) reshaped
      Requires the marginal factorization A[r] = kron(M1[i1[r]], M2[i2[r]]).

Both are compared against the actual `gm.gram(W)` superglm computes.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from superglm import SuperGLM
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline

DATA = "/home/mhick/python_projects/superglm/data/freMTPL2freq.csv"
N = 100_000
PAIR = ("DrivAge", "BonusMalus")


def load(n_rows):
    df = pd.read_csv(DATA)
    df["ClaimNb"] = np.asarray(df["ClaimNb"], float).clip(0, 4)
    df["Exposure"] = np.asarray(df["Exposure"], float).clip(1e-3, 1.0)
    idx = np.random.default_rng(0).choice(len(df), size=n_rows, replace=False)
    df = df.iloc[idx].reset_index(drop=True)
    cols = ["DrivAge", "VehAge", "BonusMalus", "VehPower"]
    X = df[[*cols, "Area"]].copy()
    for c in cols:
        X[c] = X[c].astype(float)
    X["Area"] = X["Area"].astype(str)
    return X, (df["ClaimNb"] / df["Exposure"]).to_numpy(), df["Exposure"].to_numpy()


def main():
    X, y, w = load(N)
    feats = {c: Spline(kind="ps", k=10) for c in ("DrivAge", "VehAge", "BonusMalus", "VehPower")}
    feats["Area"] = Categorical(base="first")
    m = SuperGLM(family="poisson", selection_penalty=None, discrete=False, features=feats)
    m._add_interaction(*PAIR)
    m._build_design_matrix(X, y, w, None)
    dm = m._dm

    tensor_gm = tensor_name = None
    for g, gm in zip(m._groups, dm.group_matrices, strict=False):
        if "DrivAge" in g.name and "BonusMalus" in g.name:
            tensor_gm, tensor_name = gm, g.name
    if tensor_gm is None:
        for g, gm in zip(m._groups, dm.group_matrices, strict=False):
            print("  group:", g.name, type(gm).__name__, gm.shape)
        raise SystemExit("tensor group not found")

    A = np.asarray(tensor_gm.toarray(), dtype=float)
    n, p_t = A.shape
    B = getattr(tensor_gm, "B", None)
    nnz = int(B.nnz) if B is not None and hasattr(B, "nnz") else int(np.count_nonzero(A))
    print(f"tensor group : {tensor_name}")
    print(f"  class      : {type(tensor_gm).__name__}")
    print(f"  shape      : {A.shape}   (n={n}, p_tensor={p_t})")
    print(f"  stored nnz : {nnz:,}  density={nnz / (n * p_t):.4f}")
    print(f"  dense-equiv: {A.nbytes / 1e6:.1f} MB")

    rng = np.random.default_rng(1)
    W = np.abs(rng.normal(1.0, 0.2, n))

    t0 = time.perf_counter()
    gold = tensor_gm.gram(W)
    t_gold = time.perf_counter() - t0
    print(f"\nsuperglm gm.gram(W)          : {t_gold * 1000:8.1f} ms")

    x1 = X[PAIR[0]].to_numpy()
    x2 = X[PAIR[1]].to_numpy()
    u1, i1 = np.unique(x1, return_inverse=True)
    u2, i2 = np.unique(x2, return_inverse=True)
    m1, m2 = len(u1), len(u2)
    print(f"\nmarginal supports: |{PAIR[0]}|={m1}  |{PAIR[1]}|={m2}  m1*m2={m1 * m2:,}  n={n:,}")
    print(f"  crossover m1*m2 <= n ?  {'YES' if m1 * m2 <= n else 'NO'}")

    # ---- L1: unique-cell compression -------------------------------------
    cell = i1.astype(np.int64) * m2 + i2
    ucell, inv = np.unique(cell, return_inverse=True)
    n_cells = len(ucell)
    print(f"  occupied cells: {n_cells:,}  (compression {n / n_cells:.1f}x)")

    first = np.zeros(n_cells, dtype=np.int64)
    first[inv[::-1]] = np.arange(n - 1, -1, -1)
    A_cells = A[first]

    t0 = time.perf_counter()
    Wbar = np.bincount(inv, weights=W, minlength=n_cells)
    g_l1 = (A_cells * Wbar[:, None]).T @ A_cells
    t_l1 = time.perf_counter() - t0
    err1 = np.max(np.abs(g_l1 - gold)) / max(np.max(np.abs(gold)), 1e-300)
    print(
        f"\nL1 pair-compressed gram      : {t_l1 * 1000:8.1f} ms   rel_err={err1:.2e}   "
        f"speedup={t_gold / t_l1:.1f}x"
    )

    # ---- L2: row-tensor factorization ------------------------------------
    # Recover marginals from the cell basis: A_cells[c] should equal
    # kron(M1[c1], M2[c2]). Recover M1, M2 by SVD of the reshaped cell block.
    c1 = (ucell // m2).astype(np.int64)
    c2 = (ucell % m2).astype(np.int64)
    ok_kron = False
    p1 = p2 = None
    for cand_p1 in range(1, p_t + 1):
        if p_t % cand_p1:
            continue
        cand_p2 = p_t // cand_p1
        blk = A_cells.reshape(n_cells, cand_p1, cand_p2)
        # rank-1 in (p1,p2) for every cell => Kronecker structure
        s = np.linalg.svd(blk[: min(64, n_cells)], compute_uv=False)
        if s.shape[-1] > 1 and np.max(s[:, 1]) <= 1e-9 * max(np.max(s[:, 0]), 1e-300):
            p1, p2, ok_kron = cand_p1, cand_p2, True
            break
    print(
        f"\nKronecker structure detected : {ok_kron}" + (f"  (p1={p1}, p2={p2})" if ok_kron else "")
    )

    if ok_kron:
        M1 = np.zeros((m1, p1))
        M2 = np.zeros((m2, p2))
        seen1, seen2 = set(), set()
        for c in range(n_cells):
            blk = A_cells[c].reshape(p1, p2)
            U, S, Vt = np.linalg.svd(blk)
            a = U[:, 0] * np.sqrt(S[0])
            b = Vt[0] * np.sqrt(S[0])
            if c1[c] not in seen1 and abs(a[np.argmax(np.abs(a))]) > 0:
                M1[c1[c]] = a
                seen1.add(c1[c])
            if c2[c] not in seen2:
                M2[c2[c]] = b
                seen2.add(c2[c])
        # sign/scale gauge is per-cell; only verify the *dense* factored gram path timing
        Wmat = np.zeros((m1, m2))
        np.add.at(Wmat, (i1, i2), W)
        t0 = time.perf_counter()
        # G(M1)' Wbar G(M2)  ==  sum_{p,q} Wbar[p,q] kron(M1[p],M2[q]) kron(M1[p],M2[q])'
        # computed as a 4-index contraction over the (m1,m2) grid
        tmp = np.einsum("pq,pa,pc->acq", Wmat, M1, M1, optimize=True)
        g_l2 = np.einsum("acq,qb,qd->abcd", tmp, M2, M2, optimize=True)
        g_l2 = g_l2.reshape(p1 * p2, p1 * p2)
        t_l2 = time.perf_counter() - t0
        err2 = np.max(np.abs(g_l2 - gold)) / max(np.max(np.abs(gold)), 1e-300)
        print(
            f"L2 row-tensor (einsum) gram  : {t_l2 * 1000:8.1f} ms   rel_err={err2:.2e}   "
            f"speedup={t_gold / t_l2:.1f}x"
        )
        print("  (L2 rel_err is only meaningful if the per-cell sign gauge recovered cleanly)")


if __name__ == "__main__":
    main()
