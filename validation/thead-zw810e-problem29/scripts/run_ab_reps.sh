#!/bin/bash
# round-20 (review bd1eea8 P1): strict standalone validator (nonfinite_param_count
# required ==0, finite fields literal true); skip ONLY if file valid AND manifest
# records a successful (exit 0) run with the CURRENT script SHA; failures propagate.
set -u
REPO=/tmp/opencode/FlagScale-ppu
W=/tmp/opencode/zw810e-problem29
VAL=$W/scripts/validate_ab_json.py
BATCH=r20
MANDIR=$W/results/orig_ab_reps_$BATCH
MANIFEST=$MANDIR/manifest.jsonl
SCRIPT_SHA=$(sha256sum $REPO/examples/retrieval/profile_step_segments_ab.py | cut -d' ' -f1)
mkdir -p "$MANDIR"
LOG=$W/logs/ab_reps_r19.log
touch "$MANIFEST"
FAILED=0
recorded_wall() {  # name rep -> wall_ms recorded in manifest for successful run, empty if none
  grep "\"name\":\"$1\",\"rep\":$2,\"script_sha\":\"$SCRIPT_SHA\",\"exit\":0" "$MANIFEST" | tail -1 | python3 -c "import json,sys
line=sys.stdin.read().strip()
print(json.loads(line)[\"wall_ms\"] if line else \"\")" 2>/dev/null
}
expected_env_json() {  # name -> expected env dict for that group
  case $1 in
    native_stock) echo '{"FL_ADAMW_FUSED":"0","FL_FLAGGEMS":""}' ;;
    native_fused) echo '{"FL_ADAMW_FUSED":"1","FL_FLAGGEMS":""}' ;;
    sel_stock)    echo '{"FL_ADAMW_FUSED":"0","FL_FLAGGEMS":"selective"}' ;;
    sel_fused)    echo '{"FL_ADAMW_FUSED":"1","FL_FLAGGEMS":"selective"}' ;;
  esac
}
reused_ok() {  # name rep -> 0 iff v2 record authorizes reuse (sha/exit/env/wall/result-hash all match)
  local name=$1 rep=$2 line
  line=$(grep "\"name\":\"$name\",\"rep\":$rep,\"script_sha\":\"$SCRIPT_SHA\"" "$MANIFEST" | tail -1)
  [ -n "$line" ] || return 1
  python3 "$W/scripts/check_reuse.py" "$line" "$MANDIR/${name}_r${rep}.json" "$SCRIPT_SHA" "$(expected_env_json "$name")" > /dev/null 2>&1 || return 1
  python3 "$VAL" "$MANDIR/${name}_r${rep}.json" > /dev/null 2>&1
}
for rep in 1 2 3; do
  # round-21: explicit BOTH vars per group — no inheritance from parent shell
  for cfg in "native_stock:FL_ADAMW_FUSED=0 FL_FLAGGEMS=" \
             "native_fused:FL_ADAMW_FUSED=1 FL_FLAGGEMS=" \
             "sel_stock:FL_ADAMW_FUSED=0 FL_FLAGGEMS=selective" \
             "sel_fused:FL_ADAMW_FUSED=1 FL_FLAGGEMS=selective"; do
    name=${cfg%%:*}; envs=${cfg#*:}
    EV_FUSED=$(echo "$envs" | tr ' ' '\n' | grep '^FL_ADAMW_FUSED=' | cut -d= -f2)
    EV_GEMS=$(echo "$envs" | tr ' ' '\n' | grep '^FL_FLAGGEMS=' | cut -d= -f2)
    ENVJSON="\"FL_ADAMW_FUSED\":\"${EV_FUSED}\",\"FL_FLAGGEMS\":\"${EV_GEMS}\""
    OUT=$MANDIR/${name}_r${rep}.json
    if [ -f "$OUT" ] && reused_ok "$name" "$rep"; then
      echo "skip $name r$rep (valid + manifest exit0 sha-match)" >> $LOG; continue
    fi
    [ -f "$OUT" ] && { echo "re-run $name r$rep (no valid manifest record)" >> $LOG; mv "$OUT" "$OUT.pre_r20"; }
    echo "[$(date +%T)] run $name r$rep" >> $LOG
    env PYTHONPATH=$REPO:/tmp/opencode/FlagGems-base/src CUDA_VISIBLE_DEVICES=0 FLAGSCALE_DEVICE=cuda $envs \
      timeout 1500 python3 -u $REPO/examples/retrieval/profile_step_segments_ab.py \
      --conf $REPO/examples/retrieval/conf/run/bge_m3_ppu.yaml --steps 3 --skip 2 --no-profile --device cuda --out "$OUT" \
      > $W/logs/ab_reps_${BATCH}_${name}_r${rep}.log 2>&1
    rc=$?
    if [ $rc -eq 0 ] && python3 "$VAL" "$OUT" > /dev/null 2>&1; then
      RS=$(sha256sum "$OUT" | cut -d' ' -f1)
      CONF_SHA=$(sha256sum "$REPO/examples/retrieval/conf/run/bge_m3_ppu.yaml" | cut -d' ' -f1)
      printf '{"name":"%s","rep":%d,"script_sha":"%s","config_sha256":"%s","exit":0,"ts":"%s","env":{%s},"wall_ms":%s,"result_sha256":"%s"}\n' \
        "$name" "$rep" "$SCRIPT_SHA" "$CONF_SHA" "$(date -u +%FT%TZ)" "$ENVJSON" "$(python3 -c "import json;print(json.load(open('$OUT'))['wall_ms'])")" "$RS" >> "$MANIFEST"
      echo "[$(date +%T)] $name r$rep OK" >> $LOG
    else
      echo "[$(date +%T)] $name r$rep FAILED rc=$rc" >> $LOG; FAILED=$((FAILED+1))
    fi
  done
done
if [ $FAILED -gt 0 ]; then echo "REPS_FAILED count=$FAILED" >> $LOG; exit 1; fi
echo "REPS_DONE script_sha=$SCRIPT_SHA" >> $LOG
