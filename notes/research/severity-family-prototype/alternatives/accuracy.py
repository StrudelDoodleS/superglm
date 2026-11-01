"""Relative error of score/Hessian vs the 50-digit reference, 21 rows."""
import json, sys
import numpy as np, mpmath as mp
mp.mp.dps = 50
import impls
from rows import arrays, ROWS
ref = json.load(open("refs.json"))
R = np.array([[float(mp.mpf(v)) for v in r["ref"]] for r in ref])          # (n,6) rounded refs for sign/magnitude
Rmp = [[mp.mpf(v) for v in r["ref"]] for r in ref]
y, mu, s = arrays()
NAMES = ["value", "d_mu", "d_s", "d_mumu", "d_mus", "d_ss"]
def rel(out):
    E = np.empty((len(ROWS), 6))
    for i in range(len(ROWS)):
        for k in range(6):
            E[i, k] = float(abs((mp.mpf(float(out[k, i])) - Rmp[i][k]) / Rmp[i][k]))
    return E
results = {}
outs = {n: impls.evaluate(n, y, mu, s) for n in impls.NUMBA_IMPLS}
if "--jax" in sys.argv:
    import jax; jax.config.update("jax_enable_x64", True)
    k = impls.make_jax()
    outs["jax_ad_naive"] = np.asarray(k(mu, s, y)).T
tags = np.array([r[3] for r in ROWS])
def fmt(x): return "%.1e" % x
print("max relative error (typical rows: %d, adversarial rows: %d)" % ((tags == "typ").sum(), (tags == "adv").sum()))
print("%-28s %-5s " % ("impl", "set") + " ".join("%9s" % n for n in NAMES))
for name, out in outs.items():
    E = rel(out); results[name] = E.tolist()
    for t in ("typ", "adv"):
        print("%-28s %-5s " % (name, t) + " ".join("%9s" % fmt(E[tags == t, k].max()) for k in range(6)))
print("\nper adversarial row (rows: mu cv y/mu), score+Hessian max rel err per impl")
adv = np.where(tags == "adv")[0]
print("%-22s " % "row" + " ".join("%12s" % n[:12] for n in outs))
for i in adv:
    rr = ROWS[i]
    print("%-22s " % ("%g %g %g" % rr[:3]) + " ".join("%12s" % fmt(rel(o)[i, 1:].max()) for o in outs.values()))
json.dump({"names": NAMES, "rows": [list(r) for r in ROWS], "err": results}, open("accuracy.json", "w"))
