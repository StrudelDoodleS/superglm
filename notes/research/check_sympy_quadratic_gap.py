"""Exact symbolic checks; run with the isolated research environment."""

if not __debug__:
    raise RuntimeError("Symbolic check requires enabled assertions; remove -O/PYTHONOPTIMIZE")

import json
import platform

import mpmath
import sympy as s

h, b, x, x_star = s.symbols("h b x x_star", real=True)
scalar = (h * x**2 / 2 - b * x) - (h * x_star**2 / 2 - b * x_star)
scalar_remainder = s.expand((scalar - h * (x - x_star) ** 2 / 2).subs(b, h * x_star))
assert scalar_remainder == 0

a, c, d = s.symbols("a c d", real=True)
H = s.Matrix([[a, c], [c, d]])
v = s.Matrix(s.symbols("x1 x2", real=True))
v_star = s.Matrix(s.symbols("z1 z2", real=True))
rhs = H * v_star


def objective(w):
    return (w.T * H * w)[0] / 2 - (rhs.T * w)[0]


gap = objective(v) - objective(v_star)
error = v - v_star
energy_remainder = s.expand(gap - (error.T * H * error)[0] / 2)
residual = rhs - H * v
inverse_remainder = s.cancel(gap - (residual.T * H.inv() * residual)[0] / 2)
assert energy_remainder == inverse_remainder == 0
print(
    json.dumps(
        {
            "python": platform.python_version(),
            "sympy": s.__version__,
            "mpmath": mpmath.__version__,
            "scalar_gap_remainder": str(scalar_remainder),
            "symmetric_2x2_gap_remainder": str(energy_remainder),
            "symmetric_2x2_residual_gap_remainder": str(inverse_remainder),
            "inverse_assumption": "a*d-c**2 != 0",
        },
        indent=2,
    )
)
