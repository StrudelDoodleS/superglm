"""Row-wise derivative implementations for the gamma log density in (mu, s):
   a = 1/s^2,  l = a log a - a log mu + (a-1) log y - a y/mu - lgamma(a).
Outputs per row: (value, d_mu, d_s, d_mumu, d_mus, d_ss).  float64 only, serial numba."""
import math
import numpy as np
from numba import njit

LOG2PI = math.log(2.0 * math.pi)

# ---------------------------------------------------------------- special functions
@njit(cache=True)
def _digamma_trigamma(x):
    """(psi(x), psi1(x)) for x > 0: recurrence up to x >= 15, then Bernoulli asymptotic series."""
    psi = 0.0
    psi1 = 0.0
    while x < 15.0:
        psi -= 1.0 / x
        psi1 += 1.0 / (x * x)
        x += 1.0
    i = 1.0 / x
    i2 = i * i
    psi += math.log(x) - 0.5 * i - i2 * (1.0/12 - i2 * (1.0/120 - i2 * (1.0/252 - i2 * (1.0/240 - i2 * (1.0/132 - i2 * (691.0/32760 - i2/12.0))))))
    psi1 += i * (1.0 + i * (0.5 + i * (1.0/6 - i2 * (1.0/30 - i2 * (1.0/42 - i2 * (1.0/30 - i2 * (5.0/66 - i2 * (691.0/2730 - i2 * 7.0/6.0))))))))
    return psi, psi1

@njit(cache=True)
def _log_minus_digamma(a):
    """log(a) - psi(a) without cancellation."""
    n = 0.0
    s = 0.0
    x = a
    while x < 15.0:
        s += 1.0 / x
        x += 1.0
        n += 1.0
    i = 1.0 / x
    i2 = i * i
    ser = 0.5 * i + i2 * (1.0/12 - i2 * (1.0/120 - i2 * (1.0/252 - i2 * (1.0/240 - i2 * (1.0/132 - i2 * (691.0/32760 - i2/12.0))))))
    if n > 0.0:
        return -math.log1p(n / a) + ser + s
    return ser

@njit(cache=True)
def _inv_minus_trigamma(a):
    """1/a - psi1(a) without cancellation (negative)."""
    if a < 15.0:
        psi, psi1 = _digamma_trigamma(a)
        return 1.0 / a - psi1
    i = 1.0 / a
    i2 = i * i
    return -i2 * (0.5 + i * (1.0/6 - i2 * (1.0/30 - i2 * (1.0/42 - i2 * (1.0/30 - i2 * (5.0/66 - i2 * (691.0/2730 - i2 * 7.0/6.0)))))))

@njit(cache=True)
def _log_ratio_defect(y, mu):
    """log(r) + 1 - r with r = y/mu: series in x = (y-mu)/mu for |x| < 0.1, direct otherwise
    (x = r - 1 would lose r when r << 1)."""
    x = (y - mu) / mu
    if abs(x) < 0.1:
        t = 0.0
        for k in range(21, 1, -1):          # sum_{k=2}^{21} (-1)^(k+1) x^(k-2) / k
            c = 1.0 / k if k % 2 == 1 else -1.0 / k
            t = c + x * t
        return x * x * t
    r = y / mu
    return math.log(r) + 1.0 - r

@njit(cache=True)
def _stirling_err(a):
    """lgamma(a) - ((a-1/2) log a - a + log(2 pi)/2), a >= 15."""
    i = 1.0 / a
    i2 = i * i
    return i * (1.0/12 - i2 * (1.0/360 - i2 * (1.0/1260 - i2 * (1.0/1680 - i2 * (1.0/1188 - i2 * 691.0/360360.0)))))

# ---------------------------------------------------------------- log densities
@njit(cache=True)
def ld_naive(y, mu, s):
    a = 1.0 / (s * s)
    return a * math.log(a) - a * math.log(mu) + (a - 1.0) * math.log(y) - a * y / mu - math.lgamma(a)

@njit(cache=True)
def ld_stable(y, mu, s):
    """Same density, Loader-style: a*(log r + 1 - r) + log(a/2pi)/2 - stirling_err(a) - log y for a >= 15."""
    a = 1.0 / (s * s)
    if a < 15.0:
        return ld_naive(y, mu, s)
    x = (y - mu) / mu
    return a * _log_ratio_defect(y, mu) + 0.5 * (math.log(a) - LOG2PI) - _stirling_err(a) - math.log(y)

# ---------------------------------------------------------------- closed form
@njit(cache=True)
def closed_naive_row(y, mu, s):
    a = 1.0 / (s * s)
    x = (y - mu) / mu
    psi, psi1 = _digamma_trigamma(a)
    la = math.log(a) + 1.0 - math.log(mu) + math.log(y) - y / mu - psi
    laa = 1.0 / a - psi1
    return _chain(a, s, mu, x, la, laa, ld_naive(y, mu, s))

@njit(cache=True)
def closed_stable_row(y, mu, s):
    a = 1.0 / (s * s)
    x = (y - mu) / mu
    la = _log_minus_digamma(a) + _log_ratio_defect(y, mu)
    laa = _inv_minus_trigamma(a)
    return _chain(a, s, mu, x, la, laa, ld_stable(y, mu, s))

@njit(cache=True)
def _chain(a, s, mu, x, la, laa, v):
    ap = -2.0 / (s * s * s)
    app = 6.0 / (s * s * s * s)
    lmu = a * x / mu
    lamu = x / mu
    lmumu = -a * (1.0 + 2.0 * x) / (mu * mu)
    return (v, lmu, la * ap, lmumu, lamu * ap, laa * ap * ap + la * app)

# ---------------------------------------------------------------- finite differences
@njit(cache=True)
def _cd1_mu(ld, y, mu, s, h):
    return (ld(y, mu + h, s) - ld(y, mu - h, s)) / (2.0 * h)

@njit(cache=True)
def _cd1_s(ld, y, mu, s, h):
    return (ld(y, mu, s + h) - ld(y, mu, s - h)) / (2.0 * h)

@njit(cache=True)
def _cd2_mu(ld, y, mu, s, h, f0):
    return (ld(y, mu + h, s) - 2.0 * f0 + ld(y, mu - h, s)) / (h * h)

@njit(cache=True)
def _cd2_s(ld, y, mu, s, h, f0):
    return (ld(y, mu, s + h) - 2.0 * f0 + ld(y, mu, s - h)) / (h * h)

@njit(cache=True)
def _cd2_mix(ld, y, mu, s, hm, hs):
    return (ld(y, mu + hm, s + hs) - ld(y, mu + hm, s - hs)
            - ld(y, mu - hm, s + hs) + ld(y, mu - hm, s - hs)) / (4.0 * hm * hs)

EPS = np.finfo(np.float64).eps
C1 = EPS ** (1.0 / 3.0)    # central first derivative, O(h^2)
C2 = EPS ** (1.0 / 4.0)    # central second derivative, O(h^2)
R1 = EPS ** (1.0 / 5.0)    # Richardson first derivative, O(h^4)
R2 = EPS ** (1.0 / 6.0)    # Richardson second derivative, O(h^4)

@njit(cache=True)
def fd_row(ld, y, mu, s, scale_aware):
    """13 evaluations of ld.  Steps are C * scale with scale = (mu, s) or (mu*min(s,1), s)."""
    sm = mu * min(s, 1.0) if scale_aware else mu
    ss = s
    f0 = ld(y, mu, s)
    return (f0, _cd1_mu(ld, y, mu, s, C1 * sm), _cd1_s(ld, y, mu, s, C1 * ss),
            _cd2_mu(ld, y, mu, s, C2 * sm, f0), _cd2_mix(ld, y, mu, s, C2 * sm, C2 * ss),
            _cd2_s(ld, y, mu, s, C2 * ss, f0))

@njit(cache=True)
def fd_richardson_row(ld, y, mu, s):
    """25 evaluations: (4 D(h/2) - D(h))/3 for every entry, scale-aware steps."""
    sm = mu * min(s, 1.0)
    ss = s
    f0 = ld(y, mu, s)
    hm1, hs1, hm2, hs2 = R1 * sm, R1 * ss, R2 * sm, R2 * ss
    g_mu = (4.0 * _cd1_mu(ld, y, mu, s, hm1 / 2) - _cd1_mu(ld, y, mu, s, hm1)) / 3.0
    g_s = (4.0 * _cd1_s(ld, y, mu, s, hs1 / 2) - _cd1_s(ld, y, mu, s, hs1)) / 3.0
    h_mm = (4.0 * _cd2_mu(ld, y, mu, s, hm2 / 2, f0) - _cd2_mu(ld, y, mu, s, hm2, f0)) / 3.0
    h_ss = (4.0 * _cd2_s(ld, y, mu, s, hs2 / 2, f0) - _cd2_s(ld, y, mu, s, hs2, f0)) / 3.0
    h_ms = (4.0 * _cd2_mix(ld, y, mu, s, hm2 / 2, hs2 / 2) - _cd2_mix(ld, y, mu, s, hm2, hs2)) / 3.0
    return (f0, g_mu, g_s, h_mm, h_ms, h_ss)

# ---------------------------------------------------------------- array drivers (out is (6, n))
@njit(cache=True)
def run_closed_naive(y, mu, s, out):
    for i in range(y.shape[0]):
        r = closed_naive_row(y[i], mu[i], s[i])
        for k in range(6):
            out[k, i] = r[k]

@njit(cache=True)
def run_closed_stable(y, mu, s, out):
    for i in range(y.shape[0]):
        r = closed_stable_row(y[i], mu[i], s[i])
        for k in range(6):
            out[k, i] = r[k]

@njit(cache=True)
def run_fd_generic(y, mu, s, out):
    for i in range(y.shape[0]):
        r = fd_row(ld_naive, y[i], mu[i], s[i], False)
        for k in range(6):
            out[k, i] = r[k]

@njit(cache=True)
def run_fd_scaled(y, mu, s, out):
    for i in range(y.shape[0]):
        r = fd_row(ld_naive, y[i], mu[i], s[i], True)
        for k in range(6):
            out[k, i] = r[k]

@njit(cache=True)
def run_fd_scaled_stable(y, mu, s, out):
    for i in range(y.shape[0]):
        r = fd_row(ld_stable, y[i], mu[i], s[i], True)
        for k in range(6):
            out[k, i] = r[k]

@njit(cache=True)
def run_fd_richardson(y, mu, s, out):
    for i in range(y.shape[0]):
        r = fd_richardson_row(ld_naive, y[i], mu[i], s[i])
        for k in range(6):
            out[k, i] = r[k]

@njit(cache=True)
def run_fd_richardson_stable(y, mu, s, out):
    for i in range(y.shape[0]):
        r = fd_richardson_row(ld_stable, y[i], mu[i], s[i])
        for k in range(6):
            out[k, i] = r[k]

@njit(cache=True)
def run_value_only_naive(y, mu, s, out):
    """Floor: one log-density evaluation per row (1 lgamma + 3 logs)."""
    for i in range(y.shape[0]):
        out[0, i] = ld_naive(y[i], mu[i], s[i])

NUMBA_IMPLS = {
    "closed_naive": run_closed_naive,
    "closed_stable": run_closed_stable,
    "fd_generic_step": run_fd_generic,
    "fd_scaled_step": run_fd_scaled,
    "fd_scaled_stablevalue": run_fd_scaled_stable,
    "fd_richardson": run_fd_richardson,
    "fd_richardson_stablevalue": run_fd_richardson_stable,
}

def evaluate(name, y, mu, s):
    y, mu, s = (np.ascontiguousarray(v, dtype=np.float64) for v in (y, mu, s))
    out = np.empty((6, y.shape[0]))
    NUMBA_IMPLS[name](y, mu, s, out)
    return out

# ---------------------------------------------------------------- JAX
def make_jax():
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    def logf(theta, y):
        mu, s = theta[0], theta[1]
        a = 1.0 / (s * s)
        return a * jnp.log(a) - a * jnp.log(mu) + (a - 1.0) * jnp.log(y) - a * y / mu - gammaln(a)

    def row(theta, y):
        g = jax.grad(logf)(theta, y)
        H = jax.hessian(logf)(theta, y)
        return jnp.stack([logf(theta, y), g[0], g[1], H[0, 0], H[0, 1], H[1, 1]])

    @jax.jit
    def kernel(mu, s, y):
        theta = jnp.stack([mu, s], axis=1)
        return jax.vmap(row)(theta, y)          # (n, 6)
    return kernel
