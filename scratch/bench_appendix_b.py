"""Microbenchmark: Appendix B similarity transform vs simple eigvalsh."""

import time

import numpy as np

from superglm.multi_penalty import (
    logdet_s_gradient,
    logdet_s_hessian,
    similarity_transform_logdet,
)


def _make_psd(q, rank, rng):
    A = rng.standard_normal((q, rank))
    return A @ A.T + 0.01 * np.eye(q)


def bench(q, M, n_reps=200):
    rng = np.random.default_rng(42)
    penalties = [_make_psd(q, q, rng) for _ in range(M)]
    lambdas = np.exp(rng.standard_normal(M))

    # Warm-up
    similarity_transform_logdet(penalties, lambdas)

    # Benchmark similarity_transform_logdet
    t0 = time.perf_counter()
    for _ in range(n_reps):
        result = similarity_transform_logdet(penalties, lambdas)
    t_logdet = (time.perf_counter() - t0) / n_reps

    # Benchmark gradient
    t0 = time.perf_counter()
    for _ in range(n_reps):
        logdet_s_gradient(result, penalties, lambdas)
    t_grad = (time.perf_counter() - t0) / n_reps

    # Benchmark hessian
    t0 = time.perf_counter()
    for _ in range(n_reps):
        logdet_s_hessian(result, penalties, lambdas)
    t_hess = (time.perf_counter() - t0) / n_reps

    # Baseline: simple eigvalsh
    S_total = sum(lam * S for lam, S in zip(lambdas, penalties))
    t0 = time.perf_counter()
    for _ in range(n_reps):
        eigvals = np.linalg.eigvalsh(S_total)
        pos = eigvals[eigvals > 1e-10 * eigvals.max()]
        _ = float(np.sum(np.log(pos)))
    t_eigvalsh = (time.perf_counter() - t0) / n_reps

    print(
        f"  q={q:3d}  M={M}  "
        f"logdet={t_logdet * 1e6:7.0f}us  "
        f"grad={t_grad * 1e6:6.0f}us  "
        f"hess={t_hess * 1e6:6.0f}us  "
        f"eigvalsh={t_eigvalsh * 1e6:6.0f}us  "
        f"ratio={t_logdet / t_eigvalsh:.1f}x"
    )


def main():
    print("Appendix B microbenchmark (all operations on q×q dense matrices)")
    print("=" * 80)
    for q in [10, 20, 40, 60, 80]:
        for M in [2, 4, 8]:
            bench(q, M)
        print()


if __name__ == "__main__":
    main()
