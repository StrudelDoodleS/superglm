#!/bin/bash
# runs both timing jobs through benchboard, one after the other
cd "$(dirname "$0")"
PY=$HOME/.cache/superglm-ad-alternatives/venv/bin/python
PIN="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false"
benchboard run-timed ad_alt-speed-numba 2 --wait-max 3000 -- $PIN taskset -c 2 $PY -W ignore speed.py numba 1000000 5 > speed_numba.log 2>&1
benchboard run-timed ad_alt-speed-jax 2 --wait-max 3000 -- $PIN taskset -c 2 $PY -W ignore speed.py jax 1000000 5 > speed_jax.log 2>&1
echo done > speed_done.flag
