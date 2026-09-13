"""Finite-dimensional factor checks; run with the isolated research environment.

These are the three symbolic checks recorded in the cheap interaction
representation memo. They do not prove arbitrary dimensions or certify
floating-point evaluation.
"""

import hashlib
import json
import platform
from itertools import combinations
from pathlib import Path

import sympy as s


def matrix(prefix, symmetric=False):
    if symmetric:
        a, b, c = s.symbols(f"{prefix}a {prefix}b {prefix}c")
        return s.Matrix([[a, b], [b, c]])
    return s.Matrix(2, 2, lambda i, j: s.Symbol(f"{prefix}{i}{j}"))


# Equation (3): four features, two factors, and signs +1 and -1.
g = s.Matrix(4, 2, lambda j, t: s.Symbol(f"g{j}{t}"))
signs = (1, -1)
pairs = list(combinations(range(4), 2))
direct = sum(signs[t] * g[j, t] * g[k, t] for j, k in pairs for t in range(2))
shared = sum(
    signs[t] * (sum(g[j, t] for j in range(4)) ** 2 - sum(g[j, t] ** 2 for j in range(4))) / 2
    for t in range(2)
)
shared_remainder = s.expand(direct - shared)
assert shared_remainder == 0

# Equation (2): arbitrary 2-by-2 factors, symmetric marginal matrices,
# and directional weights 1 and 3, as in the originally executed check.
U, V = matrix("u"), matrix("v")
K1, K2, G1, G2 = [matrix(prefix, True) for prefix in ("k1", "k2", "m1", "m2")]
C = U * V.T
edge = s.trace(C.T * K1 * C * G2) + 3 * s.trace(C.T * G1 * C * K2)
factor_penalty = s.trace((U.T * K1 * U) * (V.T * G2 * V)) + 3 * s.trace(
    (U.T * G1 * U) * (V.T * K2 * V)
)
edge_remainder = s.expand(edge - factor_penalty)
assert edge_remainder == 0

# Equation (4): all pairs of four features, two signed factors, symmetric
# 2-by-2 marginal factor Grams, and symbolic feature-direction lambdas.
J = s.diag(1, -1)
R = [matrix(f"r{j}", True) for j in range(4)]
B = [matrix(f"b{j}", True) for j in range(4)]
lambdas = s.symbols("l0:4")
penalty_pairs = sum(
    lambdas[j] * s.trace(J * R[j] * J * B[k]) + lambdas[k] * s.trace(J * R[k] * J * B[j])
    for j, k in pairs
)
Bsum = sum(B, s.zeros(2))
penalty_sum = sum(lambdas[j] * s.trace(J * R[j] * J * (Bsum - B[j])) for j in range(4))
aggregate_remainder = s.expand(penalty_pairs - penalty_sum)
assert aggregate_remainder == 0

print(
    json.dumps(
        {
            "python": platform.python_version(),
            "sympy": s.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "signed_four_feature_rank_two_sum_remainder": str(shared_remainder),
            "two_by_two_edge_penalty_remainder": str(edge_remainder),
            "signed_four_feature_rank_two_penalty_sum_remainder": str(aggregate_remainder),
            "scope": (
                "Finite-dimensional symbolic checks, not a general formal proof "
                "or floating-point certification. Symmetry is assumed for marginal "
                "matrices; positivity is unnecessary for these algebraic equalities."
            ),
        },
        indent=2,
    )
)
