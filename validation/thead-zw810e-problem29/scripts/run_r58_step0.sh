#!/bin/bash
# round-58 item 2 runner: native / fallback-only / real route from step 0, fresh process each, interleaved;
# then the same two relevant groups with --copy-grads none (no host sync between backward and clip)
cd /tmp/opencode/FlagScale-ppu
export PYTHONPATH=/tmp/opencode/FlagScale-ppu:/tmp/opencode/FlagGems-base/src CUDA_VISIBLE_DEVICES=0 FLAGSCALE_DEVICE=cuda
D=/tmp/opencode/zw810e-problem29; CONF=examples/retrieval/conf/run/bge_m3_ppu.yaml
run(){ mode=$1; n=$2; cg=$3; sub=$4; dir=$D/results/$sub/${mode}_$n; mkdir -p $dir
  extra=""; [ "$cg" = host ] && extra="--save-grads $dir"
  timeout 900 python3 examples/retrieval/route_from_step0_consistency.py --conf $CONF --mode $mode --copy-grads $cg $extra --out $dir/run.json > $D/logs/r58_step0_${sub}_${mode}_$n.log 2>&1
  echo "$sub $mode $n exit=$? $(date -u +%H:%M:%S)" >> $D/logs/r58_step0_runner.log; }
for n in 1 2 3 4 5; do
  run native_from_start $n host route_from_step0_r58
  run fallback_from_start $n host route_from_step0_r58
  [ $n -eq 1 ] || run route_from_start $n host route_from_step0_r58      # route_from_start_1 already ran as the probe
done
for n in 1 2 3; do
  run native_from_start $n none route_from_step0_r58_nocopy
  run route_from_start $n none route_from_step0_r58_nocopy
done
echo "ALL_DONE $(date -u +%H:%M:%S)" >> $D/logs/r58_step0_runner.log
