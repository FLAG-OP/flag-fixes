#!/bin/bash
cd /tmp/opencode/FlagScale-ppu
export PYTHONPATH=/tmp/opencode/FlagScale-ppu:/tmp/opencode/FlagGems-base/src CUDA_VISIBLE_DEVICES=0 FLAGSCALE_DEVICE=cuda
D=/tmp/opencode/zw810e-problem29; CONF=examples/retrieval/conf/run/bge_m3_ppu.yaml
run(){ mode=$1; n=$2; dir=$D/results/route_from_step0_r58_nocopy/${mode}_$n; mkdir -p $dir
  timeout 900 python3 examples/retrieval/route_from_step0_consistency.py --conf $CONF --mode $mode --copy-grads none --out $dir/run.json > $D/logs/r58_step0_nocopy_${mode}_$n.log 2>&1
  echo "nocopy $mode $n exit=$? $(date -u +%H:%M:%S)" >> $D/logs/r58_step0_nocopy_runner.log; }
for n in 1 2 3; do run native_from_start $n; run route_from_start $n; done
echo "ALL_DONE $(date -u +%H:%M:%S)" >> $D/logs/r58_step0_nocopy_runner.log
