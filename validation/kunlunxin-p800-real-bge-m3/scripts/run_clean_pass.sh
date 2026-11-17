#!/usr/bin/env bash
# CLEAN authoritative pass: everything strictly sequential on card 1, nothing
# else running on the host. Order: segments matrix (5 configs x 3 reps),
# GEMM sweep, trainer-entry runs (4 configs), 2-card FSDP2 (native+all).
set -uo pipefail
EV=/workspace/evidence
export CARD=1
echo "=== clean pass start $(date)"
bash $EV/run_ablation_matrix.sh
echo "=== gemm sweep $(date)"
bash $EV/run_gemm_sweep.sh
echo "=== trainer entry $(date)"
MAX_SAMPLES=600 FLAGSCALE_SAVE_EVERY_EPOCH=false FS_TREE=/workspace/FlagScale-stock bash $EV/train_entry.sh native r1 $EV/results-clean/trainer | tail -3
MAX_SAMPLES=600 FLAGSCALE_SAVE_EVERY_EPOCH=false FS_TREE=/workspace/FlagScale-stock bash $EV/train_entry.sh native r2 $EV/results-clean/trainer | tail -3
MAX_SAMPLES=600 FLAGSCALE_SAVE_EVERY_EPOCH=false FS_TREE=/workspace/FlagScale FS_TREE=/workspace/FlagScale FG_TREE=/workspace/FlagGems-fixed/src USE_FLAGGEMS_CFG=true FLAGGEMS_INCLUDE=gelu,gelu_backward FLAGSCALE_ADAMW_FUSED=0 bash $EV/train_entry.sh gelu r1 $EV/results-clean/trainer | tail -3
MAX_SAMPLES=600 FLAGSCALE_SAVE_EVERY_EPOCH=false FS_TREE=/workspace/FlagScale FG_TREE=/workspace/FlagGems-fixed/src USE_FLAGGEMS_CFG=true FLAGGEMS_INCLUDE=gelu,gelu_backward FLAGSCALE_ADAMW_FUSED=auto XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1 bash $EV/train_entry.sh all r1 $EV/results-clean/trainer | tail -3
MAX_SAMPLES=600 FLAGSCALE_SAVE_EVERY_EPOCH=false FS_TREE=/workspace/FlagScale-stock XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1 bash $EV/train_entry.sh gemm r1 $EV/results-clean/trainer | tail -3
echo "=== 2-card fsdp2 (quiet window) $(date)"
bash $EV/run_fsdp2.sh profile-native
bash $EV/run_fsdp2.sh native
bash $EV/run_fsdp2.sh profile-all
bash $EV/run_fsdp2.sh all
echo "=== clean pass done $(date)"
