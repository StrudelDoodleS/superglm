# Installation

## From PyPI

```bash
pip install superglm
```

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
