"""50-digit reference for value, score(mu,s), Hessian(mu mu, mu s, s s) of the gamma log
density in (mu, s). Derivatives come from SymPy diff evaluated in mpmath at 50 digits, and are
cross-checked against mpmath.diff (independent numerical differentiation at 80 digits)."""
import json, sys
import mpmath as mp, sympy as sp
from rows import ROWS
mp.mp.dps = 50
y_, mu_, s_ = sp.symbols("y mu s", positive=True)
a = 1 / s_**2
f = a*sp.log(a) - a*sp.log(mu_) + (a-1)*sp.log(y_) - a*y_/mu_ - sp.loggamma(a)
exprs = [f, sp.diff(f, mu_), sp.diff(f, s_), sp.diff(f, mu_, 2), sp.diff(f, mu_, s_), sp.diff(f, s_, 2)]
fn = [sp.lambdify((y_, mu_, s_), e, modules="mpmath") for e in exprs]
def fmp(y, mu, s):
    a = 1/s**2
    return a*mp.log(a) - a*mp.log(mu) + (a-1)*mp.log(y) - a*y/mu - mp.loggamma(a)
out = []; worst = 0
for mu, cv, r, tag in ROWS:
    y = mp.mpf(mu)*mp.mpf(r); mu = mp.mpf(mu); s = mp.mpf(cv)
    vals = [g(y, mu, s) for g in fn]
    with mp.workdps(80):
        chk = [fmp(y, mu, s),
               mp.diff(lambda m: fmp(y, m, s), mu, 1),
               mp.diff(lambda t: fmp(y, mu, t), s, 1),
               mp.diff(lambda m: fmp(y, m, s), mu, 2),
               mp.diff(lambda m, t: fmp(y, m, t), (mu, s), (1, 1)),
               mp.diff(lambda t: fmp(y, mu, t), s, 2)]
    for k, (v, c) in enumerate(zip(vals[1:], chk[1:])):
        e = abs((v-c)/c); worst = max(worst, e)
        if e > 1e-20: print('mismatch', tag, mu, s, r, k, mp.nstr(v,15), mp.nstr(c,15))
    out.append({"mu": r_ if False else float(mu), "cv": float(s), "ratio": r, "tag": tag, "y": float(y),
                "ref": [mp.nstr(v, 40) for v in vals]})
json.dump(out, open("refs.json", "w"), indent=1)
print("rows", len(out), "max rel disagreement sympy vs mp.diff:", mp.nstr(worst, 3))
