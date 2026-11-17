#!/usr/bin/env bash
# Shared config definition for the ablation matrix (run_ablation.sh) AND the
# numerics runs (run_numerics*.sh) — review round-1 P1-1: numerics experiments
# must cover the exact same config as the performance experiments.
# config_<name> prints: PYTHONPATH | FS_TREE | EXTRA_ENV  ('|'-separated).
# Host: KunlunXIN P800, card 1 (CARD=1).
FS=/workspace/FlagScale
FS_STOCK=/workspace/FlagScale-stock
FG=/workspace/FlagGems-fixed/src
CONF=/workspace/FlagScale/examples/retrieval/conf/train/bge_m3.yaml

config_native() { echo "$FS_STOCK | $FS_STOCK | "; }
config_gemm()   { echo "$FS_STOCK | $FS_STOCK | XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1"; }
config_gelu()   { echo "$FS:$FG | $FS | FLAGSCALE_ADAMW_FUSED=0"; }
config_glue()   { echo "$FS_STOCK | $FS_STOCK | XMLIR_ENABLE_LINEAR_FC_FUSION=0"; }
config_all()    { echo "$FS:$FG | $FS | FLAGSCALE_ADAMW_FUSED=auto XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1"; }
