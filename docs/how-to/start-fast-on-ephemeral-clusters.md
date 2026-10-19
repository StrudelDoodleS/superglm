# Starting Fast On Ephemeral Clusters

superglm's numerical kernels are compiled by Numba the first time a process
uses them, and Numba keeps the compiled code in an on-disk cache. On a laptop
that compile happens once. On a cluster that starts from a clean disk every
time, such as a Databricks job cluster, every new cluster compiles again.

Measured on one machine (AMD EPYC 9645, Python 3.13, Numba 0.67), with a
first Tweedie `fit_reml` and `summary` on 20,000 rows:

| | Empty cache | Cache from an earlier cluster |
|---|---|---|
| First Tweedie fit and summary | 4.7 s | 0.6 s |
| `superglm.warmup()` | 50 s | 1.5 s |

A larger model compiles a few more kernels: a first fit on the freMTPL2 book
with four interactions spent 4.6 to 7.7 seconds compiling, depending on family
and route.

## Keep Numba's cache on persistent storage

Set `NUMBA_CACHE_DIR` to a directory that outlives the cluster, before Python
starts, so Numba reads it when it is imported. On Databricks, set it in the
cluster's environment variables to a path on a Unity Catalog volume or other
persistent storage:

```
NUMBA_CACHE_DIR=/Volumes/<catalog>/<schema>/<volume>/numba-cache/superglm-<version>
```

Putting the superglm version in the path gives each release its own cache.

Numba finds its entries by three keys, and each must be the same from one
cluster to the next:

- The path superglm is installed at. Numba names the cache subdirectory after
  the absolute path of each source directory, so install superglm the same way
  on every cluster, for example as a cluster library or from an init script.
  An environment whose path changes per session starts an empty cache.
- The CPU. Entries are compiled for the machine's CPU model and features. A
  cluster on another node type compiles its own entries into the same
  directory; both sets are kept.
- The source files. Numba 0.68 and later key the cache on each file's
  contents, so a fresh install of the same superglm version reuses it. Numba
  0.67 and earlier key it on each file's modification time, and `pip` stamps
  every file with the time of the install, so each new cluster compiles again.
  With those versions, give the installed files a fixed time after installing:

  ```bash
  SUPERGLM_DIR=$(python -c "import os, superglm; print(os.path.dirname(superglm.__file__))")
  find "$SUPERGLM_DIR" -name '*.py' -exec touch -d '2000-01-01 00:00:00' {} +
  ```

  Only do this with a versioned cache directory, as above: Numba does not
  notice when a function in one file calls a changed function in another, and
  a fixed time hides an upgrade from it.

Measured with a fresh `pip install` of the same wheel into the same path,
against a cache an earlier install wrote:

| Numba | Reinstalled, as installed | Reinstalled, times fixed |
|---|---|---|
| 0.67 | compiles again (5.2 s, 25 kernels) | loads (1.0 s) |
| 0.68 | loads (0.8 s) | loads |

Several workers can share one directory: Numba writes each entry to a
temporary file and renames it into place.

## Optionally, warm up in the init script

The cache fills as fits run: a first fit compiles only the kernels it needs
and saves them, so a cluster needs nothing more than the directory above. To
pay the compile before anyone fits, call `superglm.warmup()` once in the
cluster's init script. Outside cluster start-up, an image build or a
benchmark, there is no reason to call it.

A cluster init script that installs superglm, keeps the cache and warms it:

```bash
#!/bin/bash
set -euo pipefail
VERSION=<version>
# The cluster's own Python, so every cluster installs to the same path.
python -m pip install "superglm==$VERSION"
export NUMBA_CACHE_DIR=/Volumes/<catalog>/<schema>/<volume>/numba-cache/superglm-$VERSION
# Numba 0.67 and earlier only: give the installed files a fixed time.
SUPERGLM_DIR=$(python -c "import os, superglm; print(os.path.dirname(superglm.__file__))")
find "$SUPERGLM_DIR" -name '*.py' -exec touch -d '2000-01-01 00:00:00' {} +
python -c "import superglm; superglm.warmup()"
```

Set the same `NUMBA_CACHE_DIR` in the cluster's environment variables, so
notebooks and jobs read the cache the script filled.

`warmup()` compiles the kernels of every family and route: about 50
seconds on an empty cache, once per superglm version and node type, and
about 1.5 seconds against a filled cache. Numba keys its cache to the
CPU model, so a cache built on one node type compiles again on another; the
entries for both are kept side by side. After `warmup()`, a first
fit compiled nothing on any of the freMTPL2 benchmark books (every GLM
family, exact and binned, and the SuperLSS families). The one exception is
by design: the third- and fourth-derivative kernels of a SuperLSS Tweedie or
negative binomial fit's final iterations compile on the first fit that
reaches them, so a process that never fits those families does not pay for
them.
