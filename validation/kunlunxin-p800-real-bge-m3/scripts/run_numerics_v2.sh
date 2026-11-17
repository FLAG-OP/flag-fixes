#!/usr/bin/env bash
# Numerics trajectory runs (issue #9 B/E) — configs sourced from the SAME
# definition as the perf ablation (ablation_configs.sh), review round-1 P1-1:
# the all_fixes numerics now carries the exact perf-all env (incl. XMLIR vars).
# Fixed seed 42, 6 optimizer steps on the real data via the entry building
# blocks. Aggregates failures: exits nonzero if any requested run failed.
set -uo pipefail
EV=/workspace/evidence
source $EV/ablation_configs.sh
mkdir -p $EV/results/numerics $EV/logs/numerics $EV/results/numerics/state
export TRITON_CACHE_DIR=/workspace/evidence/.triton
export FLAGGEMS_CACHE_DIR=/workspace/evidence/.flaggems
export TORCH_EXTENSIONS_DIR=/workspace/evidence/.torch_ext
export KLX_USE_AUTOTUNE=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
INCLUDE_GELU="gelu,gelu_backward"
FAILS=()

num () {
  local name=$1 cfgname=$2 save=$3
  IFS='|' read -r pp fs env <<< "$(config_$cfgname)"
  local include=""
  case $cfgname in gelu|all) include=$INCLUDE_GELU;; esac
  echo "=== numerics $name (config=$cfgname env='$env' include='$include' save=$save) $(date +%H:%M:%S)"
  env CUDA_VISIBLE_DEVICES=${CARD:-2} $env \
    PYTHONPATH=$pp \
    python3 $EV/numerics_check.py --label $name --steps 6 \
      --flagscale-root $fs \
      $( [ -n "$include" ] && echo --flaggems-include $include ) \
      $( [ "$save" = "save" ] && echo --save-final $EV/results/numerics/state/$name.pt ) \
      --out $EV/results/numerics/$name.json > $EV/logs/numerics/$name.log 2>&1
  local rc=$?
  echo "rc=$rc"; grep -E "^step|saved" $EV/logs/numerics/$name.log | tail -8 | sed 's/^/    /'
  [ -s $EV/results/numerics/$name.json ] || { echo "MISSING JSON $name"; rc=1; }
  [ $rc -ne 0 ] && FAILS+=("$name")
  return $rc
}

WHAT=${1:-revision}
if [ "$WHAT" = "revision" ]; then
  # round-1 revision: configs that cover the final perf configuration exactly
  num native_r3 native save
  num all_fixes_v2 all save     # = perf all: GELU + adamw(auto) + XMLIR fast GEMM
  num gemm_env_v2 gemm save     # perf D config numerics
else
  num native_r1 native save
  num native_r2 native save
  num gelu_only gelu nosave
  num all_fixes all nosave
fi

echo "=== numerics compare (quantified, provisional criteria)"
if [ -f $EV/results/numerics/native_r3.pt ] && [ -f $EV/results/numerics/all_fixes_v2.pt ]; then
  CUDA_VISIBLE_DEVICES= python3 $EV/numerics_compare.py \
    $EV/results/numerics/native_r3.json $EV/results/numerics/native_r3.pt \
    $EV/results/numerics/all_fixes_v2.json $EV/results/numerics/all_fixes_v2.pt \
    $EV/results/numerics/compare_all_vs_native.json 2>&1 | grep -v "XCCL\|SYMBOL" | tail -40
fi
if [ ${#FAILS[@]} -gt 0 ]; then echo "NUMERICS_FAILED: ${FAILS[*]}"; exit 1; fi
echo NUMERICS_DONE
