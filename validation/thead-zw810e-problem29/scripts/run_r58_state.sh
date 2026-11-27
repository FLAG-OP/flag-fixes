#!/bin/bash
# round-58 item 1 runner: five candidates from the same warmup recipe, then merge
cd /tmp/opencode/FlagScale-ppu
export PYTHONPATH=/tmp/opencode/FlagScale-ppu:/tmp/opencode/FlagGems-base/src CUDA_VISIBLE_DEVICES=0 FLAGSCALE_DEVICE=cuda
D=/tmp/opencode/zw810e-problem29; CONF=examples/retrieval/conf/run/bge_m3_ppu.yaml
for C in dw_route prefetch_faithful prefetch_loss_reimpl prefetch_loss_reimpl_run1 h2d_nonblocking; do
  timeout 1500 python3 examples/retrieval/train_state_consistency.py --conf $CONF --candidate $C --out $D/results/train_state_r58.json > $D/logs/r58_tsc_$C.log 2>&1
  echo "$C exit=$? $(date -u +%H:%M:%S)" >> $D/logs/r58_state_runner.log
done
python3 examples/retrieval/train_state_consistency.py --merge --out $D/results/train_state_r58.json >> $D/logs/r58_state_runner.log 2>&1
echo "ALL_DONE $(date -u +%H:%M:%S)" >> $D/logs/r58_state_runner.log
