"""Part A facts for JAX: x64 flag semantics, special-function coverage + accuracy, numba interop."""
import json, threading, warnings, time
import numpy as np, mpmath as mp
mp.mp.dps = 50
import jax, jax.numpy as jnp
from jax.scipy import special as jsp
print("jax", jax.__version__)
# ---- float64 flag
print("== x64 flag ==")
print("default jax_enable_x64 =", jax.config.jax_enable_x64)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always"); a = jnp.asarray(np.array([1.0 + 2**-40], dtype=np.float64))
    print("jnp.asarray(float64) without flag ->", a.dtype, "| value kept:", float(a[0]) == 1.0 + 2**-40, "| warnings:", [str(x.message)[:80] for x in w])
ctx_name = None
for nm in ("enable_x64",):
    for mod, label in ((jax, "jax.enable_x64"), (getattr(jax, "experimental", None), "jax.experimental.enable_x64")):
        try:
            cm = getattr(mod, nm); ctx_name = label; break
        except Exception: pass
    if ctx_name: break
print("scoped context manager available:", ctx_name)
def f2(x): return jnp.sum(x * x)
if ctx_name:
    cm = jax.enable_x64
    with cm(True):
        r = jax.jit(f2)(np.array([1.0 + 2**-40]))
        print("inside scoped x64: result dtype", r.dtype, "exact", float(r) == (1.0 + 2**-40) ** 2)
        seen = {}
        t = threading.Thread(target=lambda: seen.update(dt=str(jnp.asarray(np.array([1.0])).dtype))); t.start(); t.join()
        print("other thread inside the scope sees dtype:", seen["dt"])
    print("after scope: dtype", jnp.asarray(np.array([1.0])).dtype)
jax.config.update("jax_enable_x64", True)
print("after jax.config.update('jax_enable_x64', True): float32 user code now promotes differently, e.g.",
      (jnp.asarray(np.float32(1)) + 1.0).dtype, "(python-float weak type) and jnp.arange(3).dtype =", jnp.arange(3).dtype)
# ---- special function coverage
print("== special-function coverage in jax.scipy.special (jax %s) ==" % jax.__version__)
for nm in ("gammaln", "digamma", "polygamma", "gammainc", "gammaincc", "log_ndtr", "ndtr", "ndtri", "xlog1py", "xlogy", "erfc", "erfcx", "betaln", "zeta", "logsumexp", "i0e", "expit", "log_expit"):
    print("  %-10s %s" % (nm, "present" if hasattr(jsp, nm) else "MISSING"))
print("  jnp.log1p / jnp.expm1:", hasattr(jnp, "log1p"), hasattr(jnp, "expm1"))
# ---- accuracy vs mpmath at float64
def rel(x, r): return float(abs((mp.mpf(float(x)) - r) / r)) if r != 0 else float(abs(x))
res = {}
def check(name, fn, ref, pts):
    errs = [(rel(fn(jnp.float64(p)), ref(mp.mpf(p))), p) for p in pts]
    w = max(errs); res[name] = w; print("  %-34s worst rel err %.1e at x=%g" % (name, w[0], w[1]))
pts = [1e-3, 0.04, 0.5, 1.0, 1.4616, 2.5, 15.0, 2500.0, 1e6, 1e12]
print("== accuracy vs mpmath (50 digits) ==")
check("gammaln", jsp.gammaln, mp.loggamma, pts)
check("digamma", jsp.digamma, mp.digamma, pts)
check("polygamma(1)", lambda x: jsp.polygamma(1, x), lambda x: mp.polygamma(1, x), pts)
check("polygamma(2)", lambda x: jsp.polygamma(2, x), lambda x: mp.polygamma(2, x), pts)
check("grad(gammaln)=digamma", jax.grad(jsp.gammaln), mp.digamma, pts)
check("grad(grad(gammaln))=trigamma", jax.grad(jax.grad(jsp.gammaln)), lambda x: mp.polygamma(1, x), pts)
check("log1p", jnp.log1p, mp.log1p, [1e-300, 1e-20, 1e-5, 0.3, 5.0])
check("expm1", jnp.expm1, mp.expm1, [1e-300, 1e-20, 1e-5, 0.3, 5.0])
check("xlog1py(2.5, x)", lambda x: jsp.xlog1py(2.5, x), lambda x: mp.mpf("2.5") * mp.log1p(x), [1e-300, 1e-20, 1e-5, 0.3, 5.0])
check("log_ndtr", jsp.log_ndtr, lambda x: mp.log(mp.ncdf(x)), [-40.0, -20.0, -10.0, -1.0, 0.0, 3.0, 6.0, 10.0])
check("grad(log_ndtr)", jax.grad(jsp.log_ndtr), lambda x: mp.diff(lambda t: mp.log(mp.ncdf(t)), x), [-40.0, -20.0, -10.0, -1.0, 0.0, 3.0, 6.0, 10.0])
check("grad(grad(log_ndtr))", jax.grad(jax.grad(jsp.log_ndtr)), lambda x: mp.diff(lambda t: mp.log(mp.ncdf(t)), x, 2), [-20.0, -10.0, -1.0, 0.0, 3.0, 6.0])
def P(a, x): return mp.gammainc(a, 0, x, regularized=True)
gi_pts = [(0.04, 1e-6), (0.04, 3.0), (0.5, 0.01), (0.5, 30.0), (2.5, 2.5), (15.0, 20.0), (2500.0, 2400.0), (2500.0, 2600.0), (1e5, 1e5)]
def gi_check(name, jf, rf):
    errs = []
    for a_, x_ in gi_pts:
        try:
            v = jf(jnp.float64(a_), jnp.float64(x_)); r = rf(mp.mpf(a_), mp.mpf(x_)); errs.append((rel(v, r), (a_, x_)))
        except Exception as e:
            errs.append((float("nan"), (a_, x_, type(e).__name__)))
    w = max(errs, key=lambda t: (np.isnan(t[0]), t[0])); res[name] = w; print("  %-34s worst rel err %.1e at (a,x)=%s" % (name, w[0], w[1]))
gi_check("gammainc(a,x)", jsp.gammainc, P)
gi_check("gammaincc(a,x)", jsp.gammaincc, lambda a, x: 1 - P(a, x) if x < a else mp.gammainc(a, x, mp.inf, regularized=True))
gi_check("d gammainc / dx (grad argnum 1)", jax.grad(jsp.gammainc, argnums=1), lambda a, x: mp.diff(lambda t: P(a, t), x))
gi_check("d gammainc / da (grad argnum 0)", jax.grad(jsp.gammainc, argnums=0), lambda a, x: mp.diff(lambda t: P(t, x), a))
gi_check("d gammaincc / da", jax.grad(jsp.gammaincc, argnums=0), lambda a, x: mp.diff(lambda t: mp.gammainc(t, x, mp.inf, regularized=True), a))
gi_check("d2 gammainc / da2 (grad of grad)", jax.grad(jax.grad(jsp.gammainc, argnums=0), argnums=0), lambda a, x: mp.diff(lambda t: P(t, x), a, 2))
gi_check("d2 gammainc / da dx", jax.grad(jax.grad(jsp.gammainc, argnums=0), argnums=1), lambda a, x: mp.diff(lambda t, u: P(t, u), (a, x), (1, 1)))
json.dump({k: v[0] if isinstance(v, tuple) else v for k, v in res.items()}, open("jax_special_accuracy.json", "w"), indent=1)
# ---- numba interop
print("== numba interop ==")
from numba import njit
try:
    @njit
    def bad(x): return jnp.sum(x)
    bad(np.ones(3)); print("njit calling a jax function: works")
except Exception as e:
    print("njit calling a jax function fails:", type(e).__name__)
x = np.random.default_rng(0).random(1_000_000)
jx = jnp.asarray(x); back = np.asarray(jx)
print("np.asarray(jax cpu array): writeable =", back.flags.writeable, "| shares memory with jax buffer (zero-copy):", True if not back.flags.owndata else False,
      "| jnp.asarray(np) copies: ", not np.shares_memory(np.asarray(jnp.asarray(x)), x))
