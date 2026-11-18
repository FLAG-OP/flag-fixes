#!/usr/bin/env bash
# Clean authoritative pass: strictly sequential single-job stages on card 1,
# with per-stage status collection (review round-3 P1: the previous version
# printed an unconditional success line and always exited 0). FSDP2 stages are
# recorded as BLOCKED (pending platform work F-P800-1) rather than silently
# counted as passed; any non-blocked stage failure makes the whole pass FAIL
# (nonzero exit).
set -uo pipefail
EV=/workspace/evidence
export CARD=1
declare -a STAGE_NAMES STAGE_RESULTS
record () { STAGE_NAMES+=("$1"); STAGE_RESULTS+=("$2"); echo "### stage $1 -> $2"; }

echo "=== clean pass start $(date)"
bash $EV/run_ablation_matrix.sh && record ablation-matrix PASS || record ablation-matrix FAIL

echo "=== gemm sweep $(date)"
bash $EV/run_gemm_sweep.sh && record gemm-sweep PASS || record gemm-sweep FAIL

echo "=== trainer entry $(date)"
MAX_SAMPLES=600 FLAGSCALE_SAVE_EVERY_EPOCH=false FS_TREE=/workspace/FlagScale-stock \
  bash $EV/train_entry.sh native r1 $EV/results-clean/trainer >/dev/null 2>&1 \
  && record trainer-native-r1 PASS || record trainer-native-r1 FAIL
MAX_SAMPLES=600 FLAGSCALE_SAVE_EVERY_EPOCH=false FS_TREE=/workspace/FlagScale-stock \
  bash $EV/train_entry.sh native r2 $EV/results-clean/trainer >/dev/null 2>&1 \
  && record trainer-native-r2 PASS || record trainer-native-r2 FAIL
MAX_SAMPLES=600 FLAGSCALE_SAVE_EVERY_EPOCH=false FS_TREE=/workspace/FlagScale \
  FG_TREE=/workspace/FlagGems-fixed/src USE_FLAGGEMS_CFG=true \
  FLAGGEMS_INCLUDE=gelu,gelu_backward FLAGSCALE_ADAMW_FUSED=0 \
  bash $EV/train_entry.sh gelu r1 $EV/results-clean/trainer >/dev/null 2>&1 \
  && record trainer-gelu PASS || record trainer-gelu FAIL
MAX_SAMPLES=600 FLAGSCALE_SAVE_EVERY_EPOCH=false FS_TREE=/workspace/FlagScale \
  FG_TREE=/workspace/FlagGems-fixed/src USE_FLAGGEMS_CFG=true \
  FLAGGEMS_INCLUDE=gelu,gelu_backward FLAGSCALE_ADAMW_FUSED=auto \
  XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1 \
  bash $EV/train_entry.sh all r1 $EV/results-clean/trainer >/dev/null 2>&1 \
  && record trainer-all PASS || record trainer-all FAIL
MAX_SAMPLES=600 FLAGSCALE_SAVE_EVERY_EPOCH=false FS_TREE=/workspace/FlagScale-stock \
  XMLIR_MATMUL_FAST_MODE=1 XMLIR_MATMUL_FAST_TUNE=1 \
  bash $EV/train_entry.sh gemm r1 $EV/results-clean/trainer >/dev/null 2>&1 \
  && record trainer-gemm PASS || record trainer-gemm FAIL

# FSDP2: blocked on platform issue F-P800-1; failures here are recorded as
# BLOCKED (pending), not FAIL — multi-card completion is not a prerequisite
# for archiving this pass (review round-3).
bash $EV/run_fsdp2.sh profile-native >/dev/null 2>&1 \
  && record fsdp2-profile-native PASS || record fsdp2-profile-native "BLOCKED (F-P800-1)"
bash $EV/run_fsdp2.sh native >/dev/null 2>&1 \
  && record fsdp2-native PASS || record fsdp2-native "BLOCKED (F-P800-1)"
bash $EV/run_fsdp2.sh profile-all >/dev/null 2>&1 \
  && record fsdp2-profile-all PASS || record fsdp2-profile-all "BLOCKED (F-P800-1)"
bash $EV/run_fsdp2.sh all >/dev/null 2>&1 \
  && record fsdp2-all PASS || record fsdp2-all "BLOCKED (F-P800-1)"

echo "=== stage summary ==="
FAILED_COUNT=0
BLOCKED=0
for i in "${!STAGE_NAMES[@]}"; do
  echo "  ${STAGE_RESULTS[$i]}  ${STAGE_NAMES[$i]}"
  if [ "${STAGE_RESULTS[$i]}" = "FAIL" ]; then FAILED_COUNT=$((FAILED_COUNT + 1)); fi
done
BLOCKED=$(printf '%s\n' "${STAGE_RESULTS[@]}" | grep -c BLOCKED || true)

if [ "$FAILED_COUNT" -ne 0 ]; then
  echo "CLEAN PASS FAILED ($FAILED_COUNT failed stage(s); $BLOCKED blocked/pending)"
  exit 1
fi
if [ "$BLOCKED" -gt 0 ]; then
  echo "CLEAN PASS COMPLETE WITH BLOCKED STAGES ($BLOCKED blocked/pending — not counted as passed)"
  exit 0
fi
echo "CLEAN PASS OK (all stages passed)"
