"""Part C: complex-step accuracy for the gamma log density in (mu, s); second-derivative options;
which special functions accept complex arguments (scipy, numba)."""
import json, warnings
import numpy as np, mpmath as mp
import scipy.special as sc
from rows import arrays, ROWS
mp.mp.dps = 50
ref = json.load(open("refs.json"))
Rmp = [[mp.mpf(v) for v in r["ref"]] for r in ref]
y, mu, s = arrays()
tags = np.array([r[3] for r in ROWS]); n = len(ROWS)

def f(y, mu, s):                      # works for real or complex mu, s
    a = 1.0 / (s * s)
    return a * np.log(a) - a * np.log(mu) + (a - 1.0) * np.log(y) - a * y / mu - sc.loggamma(a)
def err(x, k):
    return np.array([float(abs((mp.mpf(float(x[i])) - Rmp[i][k]) / Rmp[i][k])) for i in range(n)])
def summary(label, x, k):
    e = err(x, k)
    print("%-52s typ %9.1e  adv %9.1e" % (label, e[tags == "typ"].max(), e[tags == "adv"].max()))
    return e

print("== complex-step gradient (scipy.special.loggamma on complex a) ==")
out = {}
for h in (1e-8, 1e-20, 1e-100, 1e-300):
    for name, scale_mu, scale_s in (("rel", mu, s),):
        hm, hs = h * scale_mu, h * scale_s
        gm = f(y, mu + 1j * hm, s).imag / hm
        gs = f(y, mu, s + 1j * hs).imag / hs
        summary("h=%g*x   d_mu" % h, gm, 1); summary("h=%g*x   d_s " % h, gs, 2)
print("== second derivatives ==")
# (a) complex-step second-derivative formula f'' ~ 2 (f - Re f(x+ih)) / h^2, scan h (relative to a curvature scale)
for k, nm, sc_mu, sc_s in ((3, "d_mumu", mu * np.minimum(s, 1), None), (5, "d_ss", None, s)):
    best = (1e99, None)
    for h in 10.0 ** np.arange(-7, 0, 0.5):
        f0 = f(y, mu, s)
        if k == 3:
            hh = h * sc_mu; d2 = 2 * (f0 - f(y, mu + 1j * hh, s).real) / hh**2
        else:
            hh = h * sc_s; d2 = 2 * (f0 - f(y, mu, s + 1j * hh).real) / hh**2
        e = err(d2, k); worst = e.max()
        if worst < best[0]: best = (worst, h, e[tags == "typ"].max(), e[tags == "adv"].max())
    print("complex-step 2f''/h^2 %-7s best over h: worst-row err %.1e at h=%.0e*scale (typ %.1e adv %.1e)" % ((nm,) + (best[0], best[1], best[2], best[3])))
# (b) finite differences of complex-step gradients (CS gradient is exact to round-off; FD error eps/k + k^2)
def cs_grad(y, mu, s, h=1e-30):
    gm = f(y, mu + 1j * h * mu, s).imag / (h * mu)
    gs = f(y, mu, s + 1j * h * s).imag / (h * s)
    return gm, gs
def hess_fd_cs(y, mu, s, km, ks, rich=False):
    def once(km, ks):
        gm_p, gs_p = cs_grad(y, mu + km, s); gm_m, gs_m = cs_grad(y, mu - km, s)
        hmm = (gm_p - gm_m) / (2 * km); hms1 = (gs_p - gs_m) / (2 * km)
        gm_p2, gs_p2 = cs_grad(y, mu, s + ks); gm_m2, gs_m2 = cs_grad(y, mu, s - ks)
        hss = (gs_p2 - gs_m2) / (2 * ks); hms2 = (gm_p2 - gm_m2) / (2 * ks)
        return hmm, 0.5 * (hms1 + hms2), hss
    if not rich: return once(km, ks)
    a = once(km, ks); b = once(km / 2, ks / 2)
    return tuple((4 * q - p) / 3 for p, q in zip(a, b))
for rich, c in ((False, np.finfo(float).eps ** (1 / 3)), (True, np.finfo(float).eps ** (1 / 5))):
    km, ks = c * mu * np.minimum(s, 1), c * s
    hmm, hms, hss = hess_fd_cs(y, mu, s, km, ks, rich)
    tag = "Richardson" if rich else "central"
    summary("FD of CS gradient (%s, 2x2 grads=%d complex evals) d_mumu" % (tag, 16 if rich else 8), hmm, 3)
    summary("FD of CS gradient (%s)                       d_mus " % tag, hms, 4)
    summary("FD of CS gradient (%s)                       d_ss  " % tag, hss, 5)
print("== hyper-dual numbers (Fike & Alonso 2011): exact second derivatives, 1 evaluation per Hessian entry ==")
class HD:
    """(f0, f1, f2, f12) with e1^2 = e2^2 = (e1 e2)^2 = 0 (arrays)."""
    def __init__(s_, a, b=0.0, c=0.0, d=0.0): s_.a, s_.b, s_.c, s_.d = a, b + 0 * a, c + 0 * a, d + 0 * a
    def _w(s_, o): return o if isinstance(o, HD) else HD(o + 0 * s_.a)
    def __add__(s_, o): o = s_._w(o); return HD(s_.a + o.a, s_.b + o.b, s_.c + o.c, s_.d + o.d)
    __radd__ = __add__
    def __neg__(s_): return HD(-s_.a, -s_.b, -s_.c, -s_.d)
    def __sub__(s_, o): return s_ + (-s_._w(o))
    def __rsub__(s_, o): return s_._w(o) - s_
    def __mul__(s_, o):
        o = s_._w(o)
        return HD(s_.a * o.a, s_.a * o.b + s_.b * o.a, s_.a * o.c + s_.c * o.a,
                  s_.a * o.d + s_.b * o.c + s_.c * o.b + s_.d * o.a)
    __rmul__ = __mul__
    def apply(s_, f0, f1, f2):   # f(x) with derivatives f1, f2 at real part
        return HD(f0, f1 * s_.b, f1 * s_.c, f1 * s_.d + f2 * s_.b * s_.c)
    def inv(s_): return s_.apply(1 / s_.a, -1 / s_.a**2, 2 / s_.a**3)
    def __truediv__(s_, o): return s_ * s_._w(o).inv()
    def __rtruediv__(s_, o): return s_._w(o) * s_.inv()
def hlog(x): return x.apply(np.log(x.a), 1 / x.a, -1 / x.a**2)
def hlgamma(x): return x.apply(sc.gammaln(x.a), sc.digamma(x.a), sc.polygamma(1, x.a))
def hd_f(Y, MU, S):
    A = 1.0 / (S * S)
    return A * hlog(A) - A * hlog(MU) + (A - 1.0) * hlog(Y) - A * Y / MU - hlgamma(A)
Y = HD(y)
H_mm = hd_f(Y, HD(mu, 1.0, 1.0, 0.0), HD(s))              # e1,e2 both on mu
H_ss = hd_f(Y, HD(mu), HD(s, 1.0, 1.0, 0.0))
H_ms = hd_f(Y, HD(mu, 1.0, 0.0, 0.0), HD(s, 0.0, 1.0, 0.0))
summary("hyper-dual d_mu  (e1 part)", H_mm.b, 1); summary("hyper-dual d_s   (e1 part)", H_ss.b, 2)
summary("hyper-dual d_mumu", H_mm.d, 3); summary("hyper-dual d_mus ", H_ms.d, 4); summary("hyper-dual d_ss  ", H_ss.d, 5)
print("   (hyper-dual lgamma needs psi and psi1 from a special-function library, the same two the closed form needs)")

print("== which special functions accept complex arguments in scipy ==")
z = np.array([2.5 + 1e-30j]); xr = np.array([3.0])
cases = {
    "loggamma(z)": lambda: sc.loggamma(z), "gammaln(z)": lambda: sc.gammaln(z), "digamma(z)": lambda: sc.digamma(z),
    "polygamma(1, z)": lambda: sc.polygamma(1, z), "zeta(2, z) (Hurwitz)": lambda: sc.zeta(2, z),
    "gammainc(z, 3.0) (complex shape)": lambda: sc.gammainc(z, xr), "gammaincc(z, 3.0) (complex shape)": lambda: sc.gammaincc(z, xr),
    "gammainc(2.5, z) (complex x)": lambda: sc.gammainc(2.5, z),
    "log_ndtr(z)": lambda: sc.log_ndtr(z), "ndtr(z)": lambda: sc.ndtr(z), "erf(z)": lambda: sc.erf(z), "erfc(z)": lambda: sc.erfc(z),
    "erfcx(z)": lambda: sc.erfcx(z), "log1p(z)": lambda: sc.log1p(z), "expm1(z)": lambda: sc.expm1(z),
    "xlogy(z, 2.0)": lambda: sc.xlogy(z, 2.0), "xlog1py(z, 2.0)": lambda: sc.xlog1py(z, 2.0), "betaln(z, 2.0)": lambda: sc.betaln(z, 2.0),
    "gamma(z)": lambda: sc.gamma(z),
}
scipy_support = {}
for k, g in cases.items():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); r = g()
        ok = np.iscomplexobj(r); scipy_support[k] = "accepts complex" + ("" if ok else " (returns real: imaginary part dropped)")
    except Exception as e:
        scipy_support[k] = "REJECTED: " + type(e).__name__
    print("  %-38s %s" % (k, scipy_support[k]))
# complex step through those that accept it: is d/da loggamma = psi to 1e-15 ?
a0 = np.array([0.04, 1.0, 15.0, 2500.0]); hh = 1e-30
cs = sc.loggamma(a0 + 1j * hh).imag / hh; ex = np.array([float(mp.digamma(mp.mpf(float(v)))) for v in a0])
print("  complex-step d/da loggamma vs mp.digamma, rel err:", np.abs(cs - ex) / np.abs(ex))
# log_ndtr derivative by complex step vs mpmath at tail points
xs = np.array([-30.0, -10.0, -1.0, 0.5, 5.0])
try:
    cs = sc.log_ndtr(xs + 1j * 1e-30).imag / 1e-30
    ex = np.array([float(mp.diff(lambda t: mp.log(mp.ncdf(t)), mp.mpf(float(v)))) for v in xs])
    print("  complex-step d/dx log_ndtr vs mpmath, rel err:", np.abs(cs - ex) / np.abs(ex))
except Exception as e:
    print("  log_ndtr complex step failed:", e)
json.dump(scipy_support, open("complex_scipy_support.json", "w"), indent=1)

print("== numba ==")
import numba, math, cmath
from numba import njit
def try_numba(label, src_fn, *args):
    try:
        r = njit(src_fn)(*args); print("  %-44s OK  -> %s" % (label, r))
    except Exception as e:
        print("  %-44s FAILS (%s)" % (label, type(e).__name__))
try_numba("math.lgamma(float)", lambda x: math.lgamma(x), 2.5)
try_numba("math.lgamma(complex)", lambda x: math.lgamma(x), 2.5 + 1e-30j)
try_numba("cmath.log(complex)", lambda x: cmath.log(x), 2.5 + 1e-30j)
try_numba("np.log(complex)", lambda x: np.log(x), 2.5 + 1e-30j)
try_numba("np.log1p(complex)", lambda x: np.log1p(x), 2.5 + 1e-30j)
try_numba("math.erfc(complex)", lambda x: math.erfc(x), 2.5 + 1e-30j)
try_numba("complex * complex, / complex", lambda x: (x * x) / (x + 1.0), 2.5 + 1e-30j)
print("  hasattr(cmath, 'lgamma') =", hasattr(cmath, "lgamma"), "; hasattr(math, 'digamma') =", hasattr(math, "digamma"), "; math.lgamma exists =", hasattr(math, "lgamma"))
try:
    from numba.extending import get_cython_function_address
    for nm in ("loggamma", "__pyx_fuse_0loggamma", "__pyx_fuse_1loggamma", "digamma", "__pyx_fuse_1digamma"):
        try:
            get_cython_function_address("scipy.special.cython_special", nm); print("  cython_special exports", nm)
        except Exception as e:
            print("  cython_special has no", nm)
except Exception as e:
    print("  get_cython_function_address unavailable:", e)
import importlib.util
print("  numba-scipy installed:", importlib.util.find_spec("numba_scipy") is not None)
import sys; print("  python", sys.version.split()[0], "ctypes complex type:", hasattr(__import__("ctypes"), "c_double_complex"))
