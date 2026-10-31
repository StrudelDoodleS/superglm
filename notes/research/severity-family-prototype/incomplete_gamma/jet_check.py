"""Taylor-mode (jet) differentiation of log P(a,x), log Q(a,x) in the shape a.

Scratch check, not production code. Jets are truncated Taylor coefficient
arrays f[k] = f^{(k)}(a)/k!, k = 0..K. The series for P and Legendre's
continued fraction for Q are evaluated in jet arithmetic with the log
prefactor handled analytically, so log P / log Q never underflow. The
reference is mpmath at 60 digits, differentiated with mpmath.diff.
"""

import math
import sys

import mpmath as mp
import numpy as np
from scipy.special import gammaln, polygamma

K = 4
U = 2.0**-53


def jmul(f, g):
    h = np.zeros(K + 1)
    for k in range(K + 1):
        h[k] = sum(f[i] * g[k - i] for i in range(k + 1))
    return h


def jinv(f):
    g = np.zeros(K + 1)
    g[0] = 1.0 / f[0]
    for k in range(1, K + 1):
        g[k] = -sum(f[i] * g[k - i] for i in range(1, k + 1)) / f[0]
    return g


def jlog(f):
    # (log f)' = f'/f, coefficient recursion
    g = np.zeros(K + 1)
    g[0] = math.log(f[0])
    for k in range(1, K + 1):
        s = k * f[k] - sum(i * g[i] * f[k - i] for i in range(1, k))
        g[k] = s / (k * f[0])
    return g


def jexp(f):
    g = np.zeros(K + 1)
    g[0] = math.exp(f[0])
    for k in range(1, K + 1):
        g[k] = sum(i * f[i] * g[k - i] for i in range(1, k + 1)) / k
    return g


def jlgamma(a0):
    g = np.zeros(K + 1)
    g[0] = gammaln(a0)
    for k in range(1, K + 1):
        g[k] = polygamma(k - 1, a0) / math.factorial(k)
    return g


def jvar(a0):
    f = np.zeros(K + 1)
    f[0] = a0
    f[1] = 1.0
    return f


def log_p_series(a0, x):
    """log P via P = D * sum_n x^n/(a+1)_n, D = x^a e^-x / Gamma(a+1)."""
    A = jvar(a0)
    t = np.zeros(K + 1)
    t[0] = 1.0
    s = t.copy()
    n = 0
    while True:
        n += 1
        den = A.copy()
        den[0] += n
        t = jmul(t, jinv(den)) * x
        s = s + t
        if np.all(np.abs(t) <= U * np.abs(s) + 1e-300) and n > 2:
            break
        if n > 100000:
            raise RuntimeError("series")
    logD = A * math.log(x) - jlgamma(a0 + 1.0)
    logD[0] -= x
    return logD + jlog(s), n


def log_q_cf(a0, x):
    """log Q via Legendre's CF (modified Lentz), Q = x^a e^-x/Gamma(a) * h."""
    A = jvar(a0)
    tiny = 1e-300
    b = -A.copy()
    b[0] += x + 1.0
    c = np.zeros(K + 1)
    c[0] = 1.0 / tiny
    d = jinv(b)
    h = d.copy()
    i = 0
    while True:
        i += 1
        an = -i * (-A)  # -i*(i - a)
        an[0] = -i * (i - a0)
        b = b.copy()
        b[0] += 2.0
        d = jmul(an, d) + b
        if abs(d[0]) < tiny:
            d[0] = tiny
        c = b + jmul(an, jinv(c))
        if abs(c[0]) < tiny:
            c[0] = tiny
        d = jinv(d)
        de = jmul(d, c)
        h = jmul(h, de)
        dm1 = de.copy()
        dm1[0] -= 1.0
        if np.all(np.abs(dm1) <= U) and i > 2:
            break
        if i > 100000:
            raise RuntimeError("cf")
    logpre = A * math.log(x) - jlgamma(a0)
    logpre[0] -= x
    return logpre + jlog(h), i


def jlog1mexp(lf):
    """log(1 - exp(lf)) on jets, for exp(lf[0]) <= ~0.75."""
    e = jexp(lf)
    one_minus = -e
    one_minus[0] = -math.expm1(lf[0])
    g = jlog(one_minus)
    # Maechler (2012): log1p(-exp(l)) for l < -ln 2, log(-expm1(l)) otherwise
    g[0] = math.log1p(-math.exp(lf[0])) if lf[0] < -math.log(2.0) else math.log(-math.expm1(lf[0]))
    return g


def log_pq(a0, x):
    if x < a0 + 1.0:
        lp, n = log_p_series(a0, x)
        return lp, jlog1mexp(lp), ("series", n)
    lq, n = log_q_cf(a0, x)
    return jlog1mexp(lq), lq, ("cf", n)


mp.mp.dps = 60


def ref(a0, x, which):
    xa = mp.mpf(x)
    lo = lambda aa: mp.gammainc(aa, 0, xa, regularized=True)
    up = lambda aa: mp.gammainc(aa, xa, mp.inf, regularized=True)
    p_small = lo(mp.mpf(a0)) < 0.5
    if which == "P":
        f = (lambda aa: mp.log(lo(aa))) if p_small else (lambda aa: mp.log1p(-up(aa)))
    else:
        f = (lambda aa: mp.log1p(-lo(aa))) if p_small else (lambda aa: mp.log(up(aa)))
    out = []
    for k in range(K + 1):
        v = f(mp.mpf(a0)) if k == 0 else mp.diff(f, mp.mpf(a0), k)
        out.append(v / mp.factorial(k))
    return out


POINTS = [
    (0.01, 1e-6), (0.01, 0.5), (0.01, 5.0), (0.01, 50.0),
    (0.1, 1e-8), (0.1, 0.1), (0.1, 3.0), (0.1, 100.0),
    (1.0, 1e-6), (1.0, 1.0), (1.0, 30.0), (1.0, 500.0),
    (10.0, 0.1), (10.0, 8.0), (10.0, 12.0), (10.0, 60.0), (10.0, 800.0),
    (100.0, 10.0), (100.0, 90.0), (100.0, 100.0), (100.0, 110.0), (100.0, 300.0),
    (1000.0, 900.0), (1000.0, 1000.0), (1000.0, 1050.0), (1000.0, 1200.0),
    (1e4, 9800.0), (1e4, 1e4), (1e4, 1e4 + 150.0),
]

if __name__ == "__main__":
    pts = POINTS if len(sys.argv) < 2 else POINTS[: int(sys.argv[1])]
    print(f"{'a':>8} {'x':>9} {'meth':>6} {'n':>5} {'fn':>2} "
          f"{'logF':>12} " + " ".join(f"relerr_d{k}" for k in range(K + 1)))
    worst = np.zeros(K + 1)
    for a0, x in pts:
        lp, lq, (meth, n) = log_pq(a0, x)
        for which, jet in (("P", lp), ("Q", lq)):
            r = ref(a0, x, which)
            errs = []
            for k in range(K + 1):
                rv = float(r[k])
                errs.append(abs(jet[k] - rv) / abs(rv) if rv != 0 else abs(jet[k]))
            worst = np.maximum(worst, errs)
            print(f"{a0:8.3g} {x:9.4g} {meth:>6} {n:5d} {which:>2} "
                  f"{float(r[0]):12.5g} " + " ".join(f"{e:9.1e}" for e in errs))
    print("worst relerr by order:", " ".join(f"{w:.1e}" for w in worst))
