# Installation

## From GitHub

```bash
pip install git+https://github.com/StrudelDoodleS/superglm.git
```

## With optional dependencies

```bash
# Benchmarking (glum, statsmodels, pyarrow)
pip install "superglm[bench] @ git+https://github.com/StrudelDoodleS/superglm.git"

# Everything (dev, bench, interactions)
pip install "superglm[all] @ git+https://github.com/StrudelDoodleS/superglm.git"
```

## Development install

```bash
git clone https://github.com/StrudelDoodleS/superglm.git
cd superglm
pip install -e ".[all]"
```
