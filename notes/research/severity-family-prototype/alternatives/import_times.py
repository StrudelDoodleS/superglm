"""Cold-interpreter import wall time (fresh subprocess each, OS file cache warm), min and median of 7."""
import subprocess, sys, time, json, statistics
mods = ["numpy", "scipy.special", "numba", "jax", "jax.numpy", "jax.scipy.special"]
res = {}
for m in mods:
    t = []
    for _ in range(7):
        t0 = time.perf_counter(); subprocess.run([sys.executable, "-I", "-c", "import " + m], check=True); t.append(time.perf_counter() - t0)
    res[m] = {"min_s": min(t), "median_s": statistics.median(t)}
    print("%-20s min %.2f s  median %.2f s" % (m, min(t), statistics.median(t)), flush=True)
json.dump(res, open("import_times.json", "w"), indent=1)
