"""ns/row for value + score + Hessian of the gamma log density, 1e6 rows, one thread.
usage: speed.py numba|jax [n] [repeats]"""
import sys, time, json, os
import numpy as np
which = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
rng = np.random.default_rng(1)
mu = np.exp(rng.normal(0, 1, n)); s = rng.uniform(0.3, 2.0, n)
y = rng.gamma(1 / s**2, mu * s**2)
res = {}
def clock(fn):
    wall, cpu = [], []
    for _ in range(reps):
        w0, c0 = time.perf_counter(), time.process_time(); fn()
        wall.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
    return {"ns_row_min": min(wall) / n * 1e9, "ns_row_median": float(np.median(wall)) / n * 1e9,
            "cpu_over_wall": float(np.median(np.array(cpu) / np.array(wall)))}
if which == "numba":
    import impls
    out = np.empty((6, n))
    for name, f in list(impls.NUMBA_IMPLS.items()) + [("value_only_floor", impls.run_value_only_naive)]:
        f(y[:8], mu[:8], s[:8], out[:, :8])                      # compile
        res[name] = clock(lambda: f(y, mu, s, out))
        print(name, res[name], flush=True)
else:
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import impls
    k = impls.make_jax()
    t0 = time.perf_counter(); k(mu[:8], s[:8], y[:8]).block_until_ready(); small = time.perf_counter() - t0
    t0 = time.perf_counter(); k(mu, s, y).block_until_ready(); big = time.perf_counter() - t0
    res["compile_first_call_n8_s"] = small; res["first_call_n1e6_s"] = big
    t0 = time.perf_counter(); k(mu[:1000], s[:1000], y[:1000]).block_until_ready(); res["new_shape_recompile_n1000_s"] = time.perf_counter() - t0
    def full():
        o = k(jnp.asarray(mu), jnp.asarray(s), jnp.asarray(y)); return np.asarray(o)   # (n,6) on host
    res["jax_full_incl_transfer"] = clock(full)
    dm, ds, dy = jnp.asarray(mu), jnp.asarray(s), jnp.asarray(y)
    res["jax_device_only"] = clock(lambda: k(dm, ds, dy).block_until_ready())
    o = k(dm, ds, dy); o.block_until_ready()
    res["transfer_out_np_asarray"] = clock(lambda: np.asarray(o))
    res["transfer_in_3x_asarray"] = clock(lambda: [jnp.asarray(v).block_until_ready() for v in (mu, s, y)])
    res["devices"] = str(jax.devices()); res["jax"] = jax.__version__
    for kk, v in res.items(): print(kk, v, flush=True)
json.dump(res, open("speed_%s.json" % which, "w"), indent=1)
