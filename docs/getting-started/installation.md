# Installation

## From GitHub

```bash
pip install git+https://github.com/StrudelDoodleS/superglm.git
```

## With optional dependencies

```bash
# scikit-learn compatible estimator
pip install "superglm[sklearn] @ git+https://github.com/StrudelDoodleS/superglm.git"

# Everything (dev, sklearn, plotting, interactions)
pip install "superglm[all] @ git+https://github.com/StrudelDoodleS/superglm.git"
```

## Development install

```bash
git clone https://github.com/StrudelDoodleS/superglm.git
cd superglm
pip install -e ".[all]"
```
