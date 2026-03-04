import numpy as np
import pandas as pd
from superglm.model import SuperGLM
from superglm.features.numeric import Numeric
from superglm.features.categorical import Categorical
from superglm.features.spline import Spline
from superglm.penalties.group_lasso import GroupLasso

rng = np.random.default_rng(42)
n = 3000
driver_age = rng.uniform(18, 85, n)
region = rng.choice(["Paris", "Lyon", "Rural"], n, p=[0.3, 0.3, 0.4])
density = rng.normal(5, 2, n)
true_log_mu = -2.0 + 0.01 * (driver_age - 50)**2 / 100 + (region == "Paris") * 0.3 + 0.05 * density
exposure = rng.uniform(0.3, 1.0, n)
y = rng.poisson(np.exp(true_log_mu) * exposure).astype(float)
X = pd.DataFrame({"driver_age": driver_age, "region": region, "density": density})

model = SuperGLM(family="poisson", penalty=GroupLasso(lambda1=0.01), lambda2=0.1)
model.add_feature("driver_age", Spline(n_knots=10, penalty="ssp"))
model.add_feature("region", Categorical(base="most_exposed"))
model.add_feature("density", Numeric())
model.fit(X, y, exposure=exposure)

print(f"Converged: {model.result.converged} in {model.result.n_iter} iter")
print(f"Deviance: {model.result.deviance:.1f}  Phi: {model.result.phi:.4f}")
s = model.summary()
for name, info in s.items():
    if name != "_model":
        print(f"{name}: active={info['active']} norm={info['group_norm']:.4f} p={info['n_params']}")

cat_r = model.reconstruct_feature("region")
print("Region relativities:")
for lev, rel in sorted(cat_r["relativities"].items()):
    print(f"  {lev}: {rel:.3f}")

sp_r = model.reconstruct_feature("driver_age")
ages, rels = sp_r["x"], sp_r["relativity"]
young = float(np.mean(rels[ages < 25]))
mid = float(np.mean(rels[(ages > 40) & (ages < 60)]))
old = float(np.mean(rels[ages > 75]))
print(f"Age: young={young:.3f} mid={mid:.3f} old={old:.3f}")
print(f"U-shape: {'PASS' if young > mid and old > mid else 'CHECK'}")

num_r = model.reconstruct_feature("density")
print(f"Density coef={num_r['coef_original']:.4f} (true=0.05)")
print("END-TO-END COMPLETE")
