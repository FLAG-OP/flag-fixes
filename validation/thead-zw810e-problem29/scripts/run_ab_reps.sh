#!/bin/bash
set -u
cd /tmp/opencode/FlagScale-ppu
mkdir -p /tmp/opencode/zw810e-problem29/results/orig_ab_reps
LOG=/tmp/opencode/zw810e-problem29/logs/ab_reps.log
for rep in 1 2 3; do
  for cfg in "native_stock:" "native_fused:FL_ADAMW_FUSED=1" "sel_stock:FL_FLAGGEMS=selective" "sel_fused:FL_FLAGGEMS=selective FL_ADAMW_FUSED=1"; do
    name=${cfg%%:*}; envs=${cfg#*:}
    OUT=/tmp/opencode/zw810e-problem29/results/orig_ab_reps/${name}_r${rep}.json
    [ -f "$OUT" ] && { echo "skip $name r$rep (exists)" >> $LOG; continue; }
    echo "[$(date +%T)] run $name r$rep" >> $LOG
    env PYTHONPATH=/tmp/opencode/FlagScale-ppu:/tmp/opencode/FlagGems-base/src CUDA_VISIBLE_DEVICES=0 FLAGSCALE_DEVICE=cuda $envs \
      timeout 1500 python3 -u examples/retrieval/profile_step_segments_ab.py \
      --conf examples/retrieval/conf/run/bge_m3_ppu.yaml --steps 3 --skip 2 --no-profile --device cuda --out "$OUT" \
      > /tmp/opencode/zw810e-problem29/logs/ab_reps_${name}_r${rep}.log 2>&1
    echo "[$(date +%T)] $name r$rep exit=$?" >> $LOG
  done
done
echo REPS_DONE >> $LOG
