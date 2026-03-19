# PIRLS/IRLS divergence from extreme working weight ratios

## Problem

When fitting Poisson or other count models with a log link on datasets that have **wide-ranging exposure weights** (e.g. vehicle-years spanning 1 to 83,000), the PIRLS and IRLS-direct solvers can diverge even though statsmodels and mgcv converge fine on the same data.

The root cause is **extreme working weight ratios**. For Poisson + log link, the IRLS working weight is `W_i = weight_i * μ_i`. During early iterations when `η` is far from converged, `stabilize_eta` allows `η ∈ [-20, 20]`, so `μ` ranges from `exp(-20) ≈ 2e-9` to `exp(20) ≈ 5e8`. Combined with exposure weights spanning ~1e5, the W ratio can reach **~1e22**.

This makes `X'WX` catastrophically ill-conditioned for Cholesky decomposition. The Newton updates become garbage, deviance spikes, and the old `dev > dev_prev * 10` guard aborted the fit immediately.

Using `offset = np.log(weight).clip(lower=1e-2)` to pass exposure as an offset (instead of through sample weights) does **not** fix this, because the working weights themselves are the issue — the offset shifts `η` but doesn't prevent transient extreme W ratios.

## What other libraries do

### mgcv

mgcv uses a multi-layered approach across its fitting methods:

#### `gam.fit3` (standard families, REML default)

The main PIRLS engine for `gam()`. Weight handling:

- **`good` logical vector**: Progressively filters observations: first `weights > 0`, then `(weights > 0) & (mu.eta.val != 0)`. Excluded observations don't participate in the WLS solve at all.
- **Fisher vs full Newton**: For canonical links, uses Fisher scoring where `w = (weg * mevg^2) / var.mug`. For non-canonical links, uses full Newton with an `alpha` correction factor:
  ```r
  alpha <- 1 + c*(family$dvar(mug)/var.mug + family$d2link(mug)*mevg)
  alpha[alpha==0] <- .Machine$double.eps  # floor at machine epsilon
  w <- weg*alpha*mevg^2/var.mug
  ```
  The `alpha == 0` floor prevents zero weights from collapsing the Newton system.
- **`use.wy` flag**: Hardcoded to `0` in `gam.fit3` (only activated in `gam.fit4`).

#### `gam.fit4` (extended families)

For extended family objects (e.g. `gevlss`, `ziplss`):

- **`use.wy` flag**: Switches from `sqrt(w) * z` to `w * z` representation when `z` contains non-finite values:
  ```r
  if (sum(!good)) {
    use.wy <- TRUE
    good <- is.finite(w) & is.finite(wz)
    z[!is.finite(z)] <- 0
  } else use.wy <- family$use.wz
  ```
  Some families set `use.wz = TRUE` by default because their second derivative of deviance is near-zero for some data, making `sqrt(w) * z` poorly scaled.

- **Negative weight zeroing**: When the penalized Hessian is indefinite, negative second derivatives are zeroed:
  ```r
  good <- is.finite(dd$Deta2)
  good[good] <- dd$Deta2[good] > 0
  w[!good] <- 0
  ```

#### `gam.fit5` (Hessian preconditioning)

The penalized likelihood engine for doubly-extended families. Applies a relative floor to the **Hessian diagonal** (not IRLS weights directly, but analogous):

```r
D <- diag(Hp)
if (min(D) <= 0) {
  Dthresh <- max(D) * sqrt(.Machine$double.eps)   # ≈ max(D) * 1.5e-8
  if (-min(D) < Dthresh) {
    indefinite <- FALSE
    D[D < Dthresh] <- Dthresh
  } else indefinite <- TRUE
}
```

This floors diagonal elements at `max(D) * 1.5e-8` to prevent ill-conditioned Cholesky decomposition — the same numerical concern as our W ratio problem. When the Hessian is truly indefinite (not just numerically), a ridge penalty `Ip = diag(rank) * sqrt(.Machine$double.eps)` is added.

#### BAM (`bam()` / `bgam.fitd`)

Uses the same weight computation pattern as `gam.fit3`/`gam.fit4`:
```r
w <- (G$w * mu.eta.val^2) / variance(mu)
good <- is.finite(z) & is.finite(w)
w[!good] <- 0
```

No additional weight floor beyond zeroing non-finite values. BAM relies on the same `good` filtering plus the discretized covariate structure for stability.

### statsmodels

statsmodels does **not** clip W, but:
- Uses `lstsq`/`pinv`/`qr` for the WLS solve (more numerically robust than Cholesky)
- Has no hard divergence abort — lets IRLS run to `maxiter`
- For canonical links, uses `1 / (g'(μ)² * V(μ))` which simplifies cleanly (e.g. `W = μ` for Poisson/log)

## Fix applied

Two changes in both `pirls.py` and `irls_direct.py`:

1. **Relative W floor**: After computing `W = weights * (dμ/dη)² / V(μ)`, floor at `W_max * 1e-8`:

   ```python
   w_max = W.max()
   if w_max > 0:
       W = np.maximum(W, w_max * 1e-8)
   ```

   This mirrors mgcv's `gam.fit5` Hessian diagonal floor at `max(D) * sqrt(.Machine$double.eps) ≈ max(D) * 1.5e-8`. With `cond(W) ≤ 1e8` and typical spline bases (`cond(X) ~ 1e3`), `cond(X'WX) ~ 1e8 * 1e6 = 1e14` — tight but within double precision for Cholesky.

2. **Softened divergence check**: Changed `dev > dev_prev * 10` from a hard `break` to a logged info message. The solver now only aborts on non-finite deviance (NaN/Inf). This matches statsmodels behavior and allows early-iteration overshoots to recover.

## Future considerations

- **`good` vector filtering**: mgcv drops zero-weight and non-finite observations from the WLS solve entirely. This is cleaner than flooring — we could adopt this too.
- **`use.wy` flag**: When `sqrt(W)` is poorly scaled, working with `W * z` directly instead of `sqrt(W) * z` avoids amplifying numerical errors. Not currently relevant since our solvers don't use the `sqrt(W)` form.
- **Canonical link simplification**: For canonical links (Poisson/log, Binomial/logit), W simplifies to a well-behaved expression (e.g. `W = μ` for Poisson/log). Detecting canonical links and using the simplified form would avoid the `dmu_deta² / V` computation entirely, eliminating the cancellation that causes extreme ratios.

## References

- mgcv `gam.fit3.r`: `good` observation filtering, `alpha == 0` → machine epsilon
- mgcv `gam.fit4.r`: `use.wy` flag, negative weight zeroing
- mgcv `gam.fit5` (in `gam.fit4.r`): Hessian diagonal floor at `max(D) * sqrt(.Machine$double.eps)`
- statsmodels issue [#4269](https://github.com/statsmodels/statsmodels/issues/4269): canonical link simplification for numerical stability
- Wood, S.N. (2011) "Fast stable restricted maximum likelihood..." JRSS-B 73(1):3-36
- Wood, S.N. (2017) *Generalized Additive Models: An Introduction with R*, 2nd ed. CRC Press.
