<div class="sg-hero">
  <div class="sg-hero__logo">
    <img src="images/logo.png" alt="SuperGLM logo">
  </div>
  <div class="sg-hero__copy">
    <h1>SuperGLM</h1>
    <p>
      Penalised GLMs for insurance pricing, with spline smooths, exact and
      discrete REML, group penalties, interactions, diagnostics, and actuarial
      validation tooling.
    </p>
    <div class="sg-hero__actions">
      <a class="md-button md-button--primary" href="getting-started/quickstart/">Quick Start</a>
      <a class="md-button" href="notebooks/plotting_diagnostics_demo/">Diagnostics Demo</a>
      <a class="md-button" href="api/model/">API Reference</a>
    </div>
  </div>
</div>

## What it covers

<div class="sg-feature-grid">
  <div class="sg-feature-card">
    <h3>Flexible feature types</h3>
    <p>B-splines, natural splines, cubic regression splines, ordered categoricals, numeric, categorical, and interaction terms.</p>
  </div>
  <div class="sg-feature-card">
    <h3>Modern fitting paths</h3>
    <p>Standard penalised fitting, exact REML, and large-<em>n</em> discrete/fREML-style REML for production-scale insurance data.</p>
  </div>
  <div class="sg-feature-card">
    <h3>Diagnostics and validation</h3>
    <p>Quantile-residual diagnostics, Lorenz curves, CAS-style double lift charts, and cross-validation with out-of-fold predictions.</p>
  </div>
  <div class="sg-feature-card">
    <h3>Inference and shrinkage</h3>
    <p>Statsmodels-style summaries, smooth significance tests, confidence bands, group penalties, and mgcv-style double-penalty selection.</p>
  </div>
</div>

## Example plots

Poisson frequency model on French MTPL2 (678k policies), REML smoothness selection.
95% pointwise confidence bands with exposure-weighted density strip and interior knot positions.
Click an image to open it at full size.

| Vehicle Age (`quantile_rows` knots) | Bonus-Malus (`quantile_tempered`, α=0.2) |
|:---:|:---:|
| [![VehAge](images/readme_vehage.png)](images/readme_vehage.png) | [![BonusMalus](images/readme_bonusmalus.png)](images/readme_bonusmalus.png) |
