#!/bin/bash
# round-19 fixes (review b5521fb P2):
# - file-exists -> skip ONLY if JSON validates (required records present + finite)
# - child failure propagates: FAILED count > 0 -> REPS_DONE NOT printed, exit 1
# - required records: 20 losses (steps3*accum4*2phase... actual: warmup8+timed12), >=3 grad norms, non-finite must FAIL validation
set -u
REPO=/tmp/opencode/FlagScale-ppu
W=/tmp/opencode/zw810e-problem29
mkdir -p "$W/results/orig_ab_reps_r19"
LOG=$W/logs/ab_reps_r19.log
FAILED=0
validate() {  # 0=ok 1=bad
  python3 - "$1" <<'PY'
import json,sys,math
try:
    d=json.load(open(sys.argv[1])); c=d.get("correctness",{})
    losses=c.get("losses",[]); grads=c.get("grad_norms",[])
    ok = (len(losses)>=20 and len(grads)>=3
          and all(math.isfinite(v) for v in losses)
          and all(math.isfinite(v) for v in grads)
          and c.get("losses_finite",False) is not False
          and (bool(d.get("segments")) or bool(d.get("segments_ms_median"))))
except Exception:
    ok=False
sys.exit(0 if ok else 1)
PY
}
for rep in 1 2 3; do
  for cfg in "native_stock:" "native_fused:FL_ADAMW_FUSED=1" "sel_stock:FL_FLAGGEMS=selective" "sel_fused:FL_FLAGGEMS=selective FL_ADAMW_FUSED=1"; do
    name=${cfg%%:*}; envs=${cfg#*:}
    OUT=$W/results/orig_ab_reps_r19/${name}_r${rep}.json
    if [ -f "$OUT" ]; then
      if validate "$OUT"; then echo "skip $name r$rep (exists+valid)" >> $LOG; continue
      else echo "INVALID existing $name r$rep -> re-run" >> $LOG; mv "$OUT" "$OUT.bad"; fi
    fi
    echo "[$(date +%T)] run $name r$rep" >> $LOG
    env PYTHONPATH=$REPO:/tmp/opencode/FlagGems-base/src CUDA_VISIBLE_DEVICES=0 FLAGSCALE_DEVICE=cuda $envs \
      timeout 1500 python3 -u $REPO/examples/retrieval/profile_step_segments_ab.py \
      --conf $REPO/examples/retrieval/conf/run/bge_m3_ppu.yaml --steps 3 --skip 2 --no-profile --device cuda --out "$OUT" \
      > $W/logs/ab_reps_r19_${name}_r${rep}.log 2>&1
    rc=$?
    if [ $rc -ne 0 ] || ! validate "$OUT"; then
      echo "[$(date +%T)] $name r$rep FAILED rc=$rc" >> $LOG; FAILED=$((FAILED+1))
    else
      echo "[$(date +%T)] $name r$rep OK" >> $LOG
    fi
  done
done
if [ $FAILED -gt 0 ]; then echo "REPS_FAILED count=$FAILED" >> $LOG; exit 1; fi
echo REPS_DONE >> $LOG
