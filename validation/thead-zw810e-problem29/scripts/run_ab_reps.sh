#!/bin/bash
# round-20 (review bd1eea8 P1): strict standalone validator (nonfinite_param_count
# required ==0, finite fields literal true); skip ONLY if file valid AND manifest
# records a successful (exit 0) run with the CURRENT script SHA; failures propagate.
set -u
REPO=/tmp/opencode/FlagScale-ppu
W=/tmp/opencode/zw810e-problem29
VAL=$W/scripts/validate_ab_json.py
MANIFEST=$W/results/orig_ab_reps_r19/manifest.jsonl
SCRIPT_SHA=$(sha256sum $REPO/examples/retrieval/profile_step_segments_ab.py | cut -d' ' -f1)
mkdir -p "$W/results/orig_ab_reps_r19"
LOG=$W/logs/ab_reps_r19.log
touch "$MANIFEST"
FAILED=0
reused_ok() {  # name rep -> 0 if manifest has exit=0 record with current sha AND file validates
  local name=$1 rep=$2
  grep -q "\"name\":\"$name\",\"rep\":$rep,\"script_sha\":\"$SCRIPT_SHA\",\"exit\":0" "$MANIFEST" || return 1
  python3 "$VAL" "$W/results/orig_ab_reps_r19/${name}_r${rep}.json" > /dev/null 2>&1
}
for rep in 1 2 3; do
  for cfg in "native_stock:" "native_fused:FL_ADAMW_FUSED=1" "sel_stock:FL_FLAGGEMS=selective" "sel_fused:FL_FLAGGEMS=selective FL_ADAMW_FUSED=1"; do
    name=${cfg%%:*}; envs=${cfg#*:}
    OUT=$W/results/orig_ab_reps_r19/${name}_r${rep}.json
    if [ -f "$OUT" ] && reused_ok "$name" "$rep"; then
      echo "skip $name r$rep (valid + manifest exit0 sha-match)" >> $LOG; continue
    fi
    [ -f "$OUT" ] && { echo "re-run $name r$rep (no valid manifest record)" >> $LOG; mv "$OUT" "$OUT.pre_r20"; }
    echo "[$(date +%T)] run $name r$rep" >> $LOG
    env PYTHONPATH=$REPO:/tmp/opencode/FlagGems-base/src CUDA_VISIBLE_DEVICES=0 FLAGSCALE_DEVICE=cuda $envs \
      timeout 1500 python3 -u $REPO/examples/retrieval/profile_step_segments_ab.py \
      --conf $REPO/examples/retrieval/conf/run/bge_m3_ppu.yaml --steps 3 --skip 2 --no-profile --device cuda --out "$OUT" \
      > $W/logs/ab_reps_r19_${name}_r${rep}.log 2>&1
    rc=$?
    if [ $rc -eq 0 ] && python3 "$VAL" "$OUT" > /dev/null 2>&1; then
      printf '{"name":"%s","rep":%d,"script_sha":"%s","exit":0,"ts":"%s"}\n' \
        "$name" "$rep" "$SCRIPT_SHA" "$(date -u +%FT%TZ)" >> "$MANIFEST"
      echo "[$(date +%T)] $name r$rep OK" >> $LOG
    else
      echo "[$(date +%T)] $name r$rep FAILED rc=$rc" >> $LOG; FAILED=$((FAILED+1))
    fi
  done
done
if [ $FAILED -gt 0 ]; then echo "REPS_FAILED count=$FAILED" >> $LOG; exit 1; fi
echo "REPS_DONE script_sha=$SCRIPT_SHA" >> $LOG
