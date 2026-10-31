"""JAX vs scipy accuracy of the same special functions at float64, against mpmath (50 digits).
Error is relative, except where |reference| < 1e-3 (near a root) where it is absolute."""
import numpy as np, mpmath as mp, scipy.special as sc
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy import special as jsp
mp.mp.dps = 50
def e(x, r):
    d = abs(mp.mpf(float(x)) - r); return float(d if abs(r) < 1e-3 else d / abs(r))
def row(name, jf, sf, rf, pts):
    ej = max((e(jf(jnp.float64(p)), rf(mp.mpf(p))), p) for p in pts)
    es = max((e(sf(p), rf(mp.mpf(p))), p) for p in pts)
    print("%-22s jax %.1e (x=%g)   scipy %.1e (x=%g)" % (name, ej[0], ej[1], es[0], es[1]))
pts = [1e-3, 0.04, 0.5, 1.0, 1.4616, 2.5, 15.0, 2500.0, 1e6, 1e12]
row("gammaln", jsp.gammaln, sc.gammaln, mp.loggamma, pts)
row("digamma", jsp.digamma, sc.digamma, mp.digamma, pts)
row("trigamma", lambda x: jsp.polygamma(1, x), lambda x: sc.polygamma(1, x), lambda x: mp.polygamma(1, x), pts)
nd = [-40.0, -20.0, -10.0, -1.0, 0.0, 3.0, 5.0, 6.0, 8.0, 10.0]
row("log_ndtr", jsp.log_ndtr, sc.log_ndtr, lambda x: mp.log(mp.ncdf(x)), nd)
row("log_ndtr (x<=3 only)", jsp.log_ndtr, sc.log_ndtr, lambda x: mp.log(mp.ncdf(x)), [p for p in nd if p <= 3])
P = lambda a, x: mp.gammainc(a, 0, x, regularized=True)
for a_, x_ in [(0.04, 3.0), (2.5, 2.5), (2500.0, 2400.0), (2500.0, 2600.0), (1e5, 1e5), (1e5, 1.01e5)]:
    r = P(mp.mpf(a_), mp.mpf(x_))
    print("gammainc(%g,%g): ref %s  jax rel %.1e  scipy rel %.1e" % (a_, x_, mp.nstr(r, 6), e(jsp.gammainc(jnp.float64(a_), jnp.float64(x_)), r), e(sc.gammainc(a_, x_), r)))
