#!/usr/bin/env bash
# Single source of truth for ablation configs — consumed by run_ablation.sh,
# run_ablation_matrix.sh and run_numerics_v2.sh (review round-2 P1: the
# "shared config" claim must actually hold).
# config_<name> prints: PYTHONPATH | FS_TREE | EXTRA_ENV  ('|'-separated).
# GEMS_VENDOR only matters when flag_gems is imported (gelu/all); it is inert
# for native/gemm/glue (flag_gems never imported there).
# Host: KunlunXIN P800, card 1 (CARD=1).
FS=/workspace/FlagScale
FS_STOCK=/workspace/FlagScale-stock
FG=/workspace/FlagGems-fixed/src
CONF=/workspace/FlagScale/examples/retrieval/conf/train/bge_m3.yaml
GELU_INCLUDE="gelu,gelu_backward"

config_native() { echo "$FS_STOCK | $FS_STOCK | "; }
config_gemm()   { echo "$FS_STOCK | $FS_STOCK | XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1"; }
config_gelu()   { echo "$FS:$FG | $FS | GEMS_VENDOR=kunlunxin FLAGSCALE_ADAMW_FUSED=0"; }
config_glue()   { echo "$FS_STOCK | $FS_STOCK | XMLIR_ENABLE_LINEAR_FC_FUSION=0"; }
config_all()    { echo "$FS:$FG | $FS | GEMS_VENDOR=kunlunxin FLAGSCALE_ADAMW_FUSED=auto XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1"; }
config_uses_gems() { case $1 in gelu|all) return 0;; *) return 1;; esac; }
