# Installation

## From PyPI

```bash
pip install superglm
```

The normal install includes pandas, and model-data entry points accept either a
`pandas.DataFrame` or an eager Polars DataFrame. Polars is an optional input
backend and is installed separately:

```bash
pip install polars
```

Pass eager frames to SuperGLM. A `polars.LazyFrame` must be materialized by the
caller with `LazyFrame.collect()` before fitting or prediction. SuperGLM keeps
the input backend native while compiling features; it does not convert a whole
Polars frame to pandas. Outputs remain pandas DataFrames where the existing
reporting, inference, plotting-data, and export APIs are table-oriented.

## With optional dependencies

```bash
# Interactive Plotly charts
pip install "superglm[plotting]"

# Benchmarking (glum, statsmodels, pyarrow)
pip install "superglm[bench]"
```

The local model editor and its FastAPI/Uvicorn server are included in the
normal installation.

## Unreleased development version

```bash
pip install "superglm @ git+https://github.com/StrudelDoodleS/superglm.git"
```

## Development install

```bash
git clone https://github.com/StrudelDoodleS/superglm.git
cd superglm
uv sync --extra dev --extra bench --extra plotting
```
