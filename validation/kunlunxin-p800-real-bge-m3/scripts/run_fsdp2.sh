#!/usr/bin/env bash
# 2-card FSDP2 boundary measurement (quiet-window protocol, per PR #3 history):
#  1) real trainer run (wall + loss trajectory + checkpoint), native config
#  2) all-fixes trainer run
#  3) profiled single-step harness on both ranks (comm/compute split)
# Any hang/crash is recorded as a platform blocker after a single isolated
# trigger (issue-#5 protocol); no retries under load.
set -uo pipefail
EV=/workspace/evidence
FS=${FS_TREE:-/workspace/FlagScale}
MODE=$1   # native | all | profile-native | profile-all
export TRITON_CACHE_DIR=/workspace/evidence/.triton
export FLAGGEMS_CACHE_DIR=/workspace/evidence/.flaggems
export TORCH_EXTENSIONS_DIR=/workspace/evidence/.torch_ext
export KLX_USE_AUTOTUNE=0 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export FLAGSCALE_DEVICE=cuda FS_PLATFORM=cuda FLAGSCALE_AUTOCAST_DEVICE=cuda
export XPU_ENABLE_PROFILER_TRACING=1 XPU_CUPTI_ENABLE_DEVICE=0
export CUDA_VISIBLE_DEVICES=1,2
unset USE_FLAGGEMS
mkdir -p $EV/logs-clean/fsdp2 $EV/results-clean/fsdp2

case $MODE in
  native)
    export PYTHONPATH=/workspace/FlagScale-stock
    python3 $EV/compose_fsdp2_cfg.py $EV/results-clean/fsdp2/native > /dev/null
    LOG=$EV/logs-clean/fsdp2/native.log
    python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
      -m flagscale.train.train_retrieval --config-file $EV/results-clean/fsdp2/native/config.yaml \
      2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done > "$LOG"
    echo "rc=$?"; grep -E "optimizer_step|validation_loss" "$LOG" | tail -4 | sed 's/^/    /'
    ;;
  all)
    export PYTHONPATH=$FS:/workspace/FlagGems-fixed/src GEMS_VENDOR=kunlunxin
    export FLAGSCALE_ADAMW_FUSED=auto XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1
    export USE_FLAGGEMS_CFG=true FLAGGEMS_INCLUDE=gelu,gelu_backward
    python3 $EV/compose_fsdp2_cfg.py $EV/results-clean/fsdp2/all > /dev/null
    LOG=$EV/logs-clean/fsdp2/all.log
    python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
      -m flagscale.train.train_retrieval --config-file $EV/results-clean/fsdp2/all/config.yaml \
      2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done > "$LOG"
    echo "rc=$?"; grep -E "optimizer_step|validation_loss" "$LOG" | tail -4 | sed 's/^/    /'
    ;;
  profile-native)
    export PYTHONPATH=/workspace/FlagScale-stock
    LOG=$EV/logs-clean/fsdp2/profile-native.log
    python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
      $EV/fsdp2_profile_step.py --out-dir $EV/results-clean/fsdp2/profile-native \
      2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done > "$LOG"
    echo "rc=$?"; grep -E "^rank|saved|warmup" "$LOG" | tail -10 | sed 's/^/    /'
    ;;
  profile-all)
    export PYTHONPATH=$FS:/workspace/FlagGems-fixed/src GEMS_VENDOR=kunlunxin
    export FLAGSCALE_ADAMW_FUSED=auto XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1
    LOG=$EV/logs-clean/fsdp2/profile-all.log
    python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
      $EV/fsdp2_profile_step.py --out-dir $EV/results-clean/fsdp2/profile-all \
      2>&1 | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done > "$LOG"
    echo "rc=$?"; grep -E "^rank|saved|warmup" "$LOG" | tail -10 | sed 's/^/    /'
    ;;
esac
