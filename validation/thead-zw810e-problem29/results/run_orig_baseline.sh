#!/bin/bash
# round-8: wait for downloads, verify, then run the ORIGINAL problem29 profiling
# entry (profile_step_segments.py) on ZW810E with the real bge_m3.yaml config.
set -u
cd /tmp/opencode/zw810e-problem29
LOG=logs/orig_baseline.log
echo "[$(date +%T)] waiting for downloads..." >> $LOG
for i in $(seq 1 120); do
  M=$(stat -c%s /tmp/bge-m3-hf/pytorch_model.bin 2>/dev/null || echo 0)
  DS=$(cat /tmp/dl_data.log 2>/dev/null | grep -c DATA_DONE)
  if [ "$M" -ge 2270000000 ] && [ "$DS" -ge 1 ]; then
    echo "[$(date +%T)] downloads complete: model=$M" >> $LOG
    break
  fi
  sleep 30
done
# verify parquet readable
python3 - <<'EOF' >> $LOG 2>&1
import pyarrow.parquet as pq
t = pq.ParquetFile("/tmp/t2ranking/t2ranking/triplet-15/train-00000-of-00011.parquet")
print("parquet rows:", t.metadata.num_rows, "| cols:", t.schema_arrow.names)
EOF
echo "[$(date +%T)] running profile_step_segments (original entry)..." >> $LOG
export PYTHONPATH=/tmp/opencode/FlagScale-ppu
export CUDA_VISIBLE_DEVICES=0
cd /tmp/opencode/FlagScale-ppu
python3 examples/retrieval/profile_step_segments.py \
  --conf examples/retrieval/conf/run/bge_m3_ppu.yaml \
  --steps 3 --skip 2 --prof-steps 3 --device cuda \
  --out /tmp/opencode/zw810e-problem29/results/ppu_bge_m3_segments_native.json \
  >> $LOG 2>&1
RC=$?
echo "[$(date +%T)] profile_step_segments exit=$RC" >> $LOG
echo BASELINE_DONE >> $LOG
