# Inspecting Results

## Summary table

Statsmodels-style summary with SEs, p-values, and smooth tests:

```python
m = model.metrics(df, y, sample_weight=exposure)
print(m.summary())
```

## Per-term inference

The `TermInference` dataclass holds everything about a single term: grid values, relativities, confidence intervals, spline metadata.

```python
ti = model.term_inference("DrivAge")

ti.x                        # evaluation grid (spline/polynomial) or levels (categorical)
ti.relativity               # exp(f(x)) relativity curve
ti.ci_lower, ti.ci_upper    # pointwise CI bounds
ti.edf                      # effective degrees of freedom
ti.spline                   # SplineMetadata (interior_knots, boundary_knots, basis_dim, ...)
```

## Plotting

All plotting goes through `model.plot()`:

```python
# Single term
model.plot("DrivAge", X=df, sample_weight=exposure)

# All terms in a grid
model.plot(X=df, sample_weight=exposure)

# Subset of terms
model.plot(["DrivAge", "VehAge"], X=df, sample_weight=exposure)

# Interaction
model.plot("DrivAge:Area")
```

Options: `ci` (`"pointwise"`, `"simultaneous"`, `"both"`, `None`, `False`), `show_knots`, `show_density`, `title`, `subtitle`, `engine`.

## Relativity DataFrames

For manual access or export:

```python
rels = model.relativities(with_se=True)
# dict of {feature_name: DataFrame}
```

## Families

| Family | Variance function | Use case |
|--------|------------------|----------|
| `Poisson()` | V(μ) = μ | Claim frequency |
| `NegativeBinomial(theta=1.0)` | V(μ) = μ + μ²/θ | Overdispersed frequency |
| `Gamma()` | V(μ) = μ² | Claim severity |
| `Tweedie(p=1.5)` | V(μ) = μᵖ | Pure premium (frequency × severity) |
